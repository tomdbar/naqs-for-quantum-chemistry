import torch
from torch import nn
from abc import ABC, abstractmethod
from enum import Enum
import torch.nn.functional as F

from scipy.special import comb

from collections.abc import Iterable

from src.naqs.network.base import ComplexAutoregressiveMachine_Base, NadeMasking, InputEncoding, ComplexCoord
from src.naqs.network.activations import *

import numpy as np
import itertools
import math

import time

def multinomial_arr(count, p):
    N = len(count)
    assert len(p) == N, "Counts and probs must have same first dimension"

    count = np.copy(count)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1] - 1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count

    return out

class MaxBatchSizeExceededError(Exception):
    pass

class ConvertToAmpPhase(nn.Module):

    def __init__(self,
                 n_out_amp, n_out_phase,
                 amp_activation=SoftmaxLogProbAmps,
                 phase_activation=ScaledSoftSign,
                 combine_amp_phase=True):
        super().__init__()

        self.n_out_amp = n_out_amp
        self.n_out_phase = n_out_phase

        self.amp_activation = amp_activation() if amp_activation is not None else None
        self.phase_activation = phase_activation() if phase_activation is not None else None
        self.combine_amp_phase = combine_amp_phase

    def forward(self, x):
        # amp, phase = torch.split(x, self.n_out, 1)
        amp, phase = torch.split_with_sizes(x, [self.n_out_amp, self.n_out_phase], 1)

        if self.amp_activation is not None:
            amp = self.amp_activation(amp)
        if self.phase_activation is not None:
            phase = self.phase_activation(phase)

        if self.combine_amp_phase:
            return torch.stack([amp, phase], -1)
        else:
            return amp, phase

class OrbitalBlock(nn.Module):

    def __init__(self,
                 num_in=2,
                 n_hid=[],
                 num_out=4,

                 hidden_activation=nn.ReLU,
                 bias=True,
                 batch_norm=True,
                 batch_norm_momentum=0.1,
                 out_activation=None,
                 max_batch_size=250000,
                 ):
        super().__init__()

        self.num_in = num_in
        self.n_hid = n_hid
        self.num_out = num_out

        layer_dims = [num_in] + n_hid + [num_out]
        self.layers = []
        for i, (n_in, n_out) in enumerate(zip(layer_dims, layer_dims[1:])):
            if batch_norm:
                l = [nn.Linear(n_in, n_out, bias=False), nn.BatchNorm1d(n_out, momentum=batch_norm_momentum)]
            else:
                l = [nn.Linear(n_in, n_out, bias=bias)]
            if (hidden_activation is not None) and i < len(layer_dims) - 2:
                l.append(hidden_activation())
            elif (out_activation is not None) and i == len(layer_dims) - 2:
                l.append(out_activation())
            l = nn.Sequential(*l)
            self.layers.append(l)

        self.max_batch_size = max_batch_size

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        if len(x) <= self.max_batch_size:
            return self.layers(x)
        else:
            return torch.cat([self.layers(x_batch) for x_batch in torch.split(x, self.max_batch_size)])
        # return self.layers(x.clamp(min=0))

class OrbitalLUT(nn.Module):

    def __init__(self,
                 num_in=1,
                 dim_vals_in=2,
                 num_out=4,
                 out_activation=None):
        super().__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.dim_vals_in = dim_vals_in

        self._dim_lut = (self.dim_vals_in ** self.num_in, self.num_out)
        self._idx_basis_vec = torch.LongTensor([self.dim_vals_in ** n for n in range(self.num_in)])

        lut = torch.randn(self._dim_lut)
        self.lut = nn.Parameter(lut, requires_grad=True)

        if out_activation is None:
            self.out_activation = out_activation()
        else:
            self.out_activation = out_activation

    @torch.no_grad()
    def _state2idx(self, s):
        return ((s.to(self._idx_basis_vec) > 0).long() * self._idx_basis_vec[:s.shape[-1]]).sum(-1)

    def forward(self, x):
        x_idx = self._state2idx(x)
        out = self.lut[x_idx]
        if self.out_activation:
            out = self.out_activation(out)
        return out

    def __repr__(self):
        str = f"ShellLUT(num_in={self.num_in}, dim_vals_in={self.dim_vals_in}, num_out={self.num_out})"
        str += f" --> lut dim = {self._dim_lut}"
        return str

class ComplexAutoregressiveMachine1D_OrbitalNade(ComplexAutoregressiveMachine_Base):

    def __init__(self,
                 num_qubits,

                 num_lut=0,

                 n_electrons=None,
                 n_alpha_electrons=None,
                 n_beta_electrons=None,
                 # mask_to_restricted_hilbert=False,
                 masking = NadeMasking.PARTIAL,

                 input_encoding=InputEncoding.BINARY,

                 amp_hidden_size=[],
                 amp_hidden_activation=nn.ReLU,
                 amp_bias=True,

                 phase_hidden_size=[],
                 phase_hidden_activation=nn.ReLU,
                 phase_bias=True,

                 combined_amp_phase_blocks = False,

                 use_amp_spin_sym=True,
                 use_phase_spin_sym=True,
                 aggregate_phase=True,

                 amp_batch_norm=False,
                 phase_batch_norm=False,
                 batch_norm_momentum=1,

                 amp_activation=SoftmaxLogProbAmps,
                 phase_activation=None,

                 device=None,
                 out_device="cpu"
                 ):
        amplitude_encoding = None
        if amp_activation is not None:
            amplitude_encoding = amp_activation.amplitude_encoding
        super().__init__(
            device=device,
            out_device=out_device,
            output_coords=ComplexCoord.POLAR,
            amplitude_encoding=amplitude_encoding
        )
        self.input_encoding = input_encoding

        self.amp_batch_norm = amp_batch_norm
        self.phase_batch_norm = phase_batch_norm
        self.batch_norm_momentum = batch_norm_momentum

        self.num_qubits = self.N = num_qubits
        self.num_lut = num_lut

        self.n_tot_up = n_electrons
        self.n_alpha_up = n_alpha_electrons
        self.n_beta_up = n_beta_electrons
        # self.mask_to_restricted_hilbert = mask_to_restricted_hilbert
        self.masking = masking

        self.use_restricted_hilbert, self._alpha_beta_restricted = False, False
        if (self.n_tot_up is not None) and ((self.n_alpha_up is None) and (self.n_beta_up is None)):
            self.use_restricted_hilbert, self._alpha_beta_restricted = True, False
            self.n_tot_down = self.N - self.n_tot_up
            self._min_n_set = min(self.n_tot_up, self.n_tot_down)
            print(f"\tComplexAutoregressiveMachine1D_OrbitalNade configured for f{self.n_tot_up} total electrons.")

        elif (self.n_alpha_up is not None) and (self.n_beta_up is not None):
            self.use_restricted_hilbert, self._alpha_beta_restricted = True, True

            if not isinstance(self.n_alpha_up, Iterable):
                self.n_alpha_up = [self.n_alpha_up]
            if not isinstance(self.n_beta_up, Iterable):
                self.n_beta_up = [self.n_beta_up]
            self.n_alpha_up, self.n_beta_up = np.array(self.n_alpha_up), np.array(self.n_beta_up)
            assert len(self.n_alpha_up)==len(self.n_beta_up), "Possible options for number of alpha/beta electrons do not match."
            self.n_tot_up = self.n_alpha_up[0] + self.n_beta_up[0]
            assert all( (self.n_alpha_up+self.n_beta_up)==self.n_tot_up ), "Possible options for number of alpha/beta electrons do not match."

            self.n_alpha_down = math.ceil(self.N / 2) - self.n_alpha_up
            self.n_beta_down = math.floor(self.N / 2) - self.n_beta_up
            self.n_tot_down = self.n_alpha_down + self.n_beta_down

            self._min_n_set = np.min(np.concatenate([self.n_alpha_up, self.n_beta_up, self.n_alpha_down, self.n_beta_down]))
            print(f"\tComplexAutoregressiveMachine1D_OrbitalNade configured for {self.n_tot_up} total electrons ({self.n_alpha_up}/{self.n_beta_up} spin up/down).")

        if self.use_restricted_hilbert:
            assert (self.amplitude_encoding is AmplitudeEncoding.LOG_AMP), "Restricted hilbert spaces requires AmplitudeEncoding.LOG_AMP."
        else:
            self._min_n_set = 0

        # self.use_spin_sym = use_spin_sym
        self.use_amp_spin_sym = use_amp_spin_sym
        self.use_phase_spin_sym = use_phase_spin_sym

        self.aggregate_phase = aggregate_phase
        self.combined_amp_phase_blocks = combined_amp_phase_blocks
        if self.combined_amp_phase_blocks:
            print("\tUsing combined amplitude and phase blocks:\n\t\t--> defaulting to amp network params for these blocks.")
            if self.use_amp_spin_sym != self.use_phase_spin_sym:
                print("\t\t--> Warning: must use same spin-sym settings for both amplitude and phase when combining them into a single block.")
                print(f"\t\t\t--> setting self.use_amp_spin_sym=self.use_phase_spin_sym={self.use_amp_spin_sym}")
                self.use_phase_spin_sym = self.use_amp_spin_sym
        print("Configuring spin symettries:")
        print(f"\t--> use for amplitude = {self.use_amp_spin_sym}")
        print(f"\t--> use for phase = {self.use_phase_spin_sym}")

        if (self.N % 2 != 0):
            raise ValueError("Symmetric NADE requires an even number of qubits.")

        if (torch.cuda.device_count() >= 1) and (self.device == 'cuda'):
            print(f"GPU found : model --> cuda", end="...")

        # For each shell...
        # If spin sym we need 5 outputs for |00>, |01>==|10>, |11>, |01>, |10>
        # Otherwise we need 4 outputs for |00>, |01>, |10>, |11>.
        self._n_out_amp = 5 if self.use_amp_spin_sym else 4

        # For each shell...
        # If spin sym we need 2 outputs for |00>/|11>, |01>/|10>.
        # Otherwise we need 4 outputs for |00>, |01>, |10>, |11>.
        self._n_out_phase = 3 if self.use_phase_spin_sym else 4

        self.amp_layers, self.phase_layers = [], []
        for n in range(self.N // 2):
            if self.input_encoding is InputEncoding.BINARY:
                n_in = 2*n
                dim_vals_in = 2
            else: # self.input_encoding is InputEncoding.INTEGER
                n_in = n
                dim_vals_in = 3
            n_in = max(1, n_in) # Make sure we have at least one input (i.e. for the first block).

            make_with_phase = self.aggregate_phase or (n == self.N // 2 - 1)
            if make_with_phase and self.combined_amp_phase_blocks:
                # Split output into amp and phase if using combined blocks.
                n_amp_block_out = self._n_out_amp + self._n_out_phase
                out_activation = lambda : ConvertToAmpPhase(n_out_amp=self._n_out_amp,
                                                            n_out_phase=self._n_out_phase,
                                                            amp_activation=None,
                                                            phase_activation=None,
                                                            combine_amp_phase=False)
            else:
                n_amp_block_out = self._n_out_amp
                out_activation = None
            n_amp_block_in = n_in
            # if self.use_restricted_hilbert:
            #     n_amp_block_in += 4

            if n < num_lut:
                # Use an explicit look-up-table (lut) for this qubit.
                amp_i = OrbitalLUT(num_in=n_amp_block_in,
                                   dim_vals_in=dim_vals_in,
                                   num_out=n_amp_block_out,
                                   out_activation=out_activation)
            else:
                amp_i = OrbitalBlock(num_in=n_amp_block_in,
                                     n_hid = amp_hidden_size,
                                     # n_hid= amp_hidden_size if (n == self.N // 2 - 1) else [4],
                                     # n_hid=[1] if (n == self.N // 2 - 1 and not self.combined_amp_phase_blocks and self.mask_to_restricted_hilbert) else amp_hidden_size,
                                     # n_hid=[min(x, num_possible_layer_inputs(n)) for x in amp_hidden_size],
                                     num_out=n_amp_block_out,
                                     hidden_activation=amp_hidden_activation,
                                     bias=amp_bias,
                                     batch_norm=self.amp_batch_norm,
                                     batch_norm_momentum=self.batch_norm_momentum,
                                     out_activation=out_activation).to(self.device)
            self.amp_layers.append(amp_i)

            if make_with_phase and not self.combined_amp_phase_blocks:
                if n < num_lut:
                    # Use an explicit look-up-table (lut) for this qubit.
                    phase_i = OrbitalLUT(num_in=n_in,
                                         dim_vals_in=dim_vals_in,
                                         num_out=self._n_out_phase,
                                         out_activation=out_activation)
                else:
                    phase_i = OrbitalBlock(num_in=n_in,
                                           n_hid=phase_hidden_size,
                                           # n_hid=[min(x, num_possible_layer_inputs(n)) for x in phase_hidden_size],
                                           num_out=self._n_out_phase,
                                           hidden_activation=phase_hidden_activation,
                                           bias=phase_bias,
                                           batch_norm=self.phase_batch_norm,
                                           batch_norm_momentum=self.batch_norm_momentum,
                                           out_activation=out_activation).to(self.device)
                self.phase_layers.append(phase_i)

        self.amp_layers = nn.ModuleList(self.amp_layers)
        self.phase_layers = nn.ModuleList(self.phase_layers)

        if amp_activation is not None:
            self.amplitude_activation = amp_activation()
        else:
            self.amplitude_activation = amp_activation

        if phase_activation is not None:
            self.phase_activation = phase_activation()
        else:
            self.phase_activation = phase_activation

        self.predict()
        self.combine_amp_phase(True)

        self._idx_shell_basis_vec = torch.LongTensor([2 ** n for n in range(self.N // 2)])
        self._idx_spin_basis_vec = torch.LongTensor([2 ** n for n in range(self.N)])

        # def weights_init(m):
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight, gain=1e-2)
        #         # if m.bias is not None:
        #         #     if len(m.bias) <= 10:
        #         #         torch.nn.init.constant_(m.bias, 0.5*math.log(0.5))
        #
        # self.apply(weights_init)

        print("model prepared.")

    def sample_by_density(self, mode=True):
        self.sample_by_density = mode

    def sample(self, mode=True):
        self.sampling = mode

    def predict(self):
        self.sample(False)

    def combine_amp_phase(self, mode=True):
        self.combined_amp_phase = mode

    def clear_cache(self):
        pass

    def modules(self):
        '''
        Hack for torchsso: See https://github.com/cybertronai/pytorch-sso/blob/master/torchsso/optim/secondorder.py.
        '''
        modules = []
        for module in super(ComplexAutoregressiveMachine1D_OrbitalNade, self).modules():
            if len(list(module.children())) > 0:
                continue
            if type(module) in [nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
                modules.append(module)
        return modules

    @torch.no_grad()
    def _state2idx(self, s):
        return ((s > 0).long() * self._idx_shell_basis_vec[:s.shape[-1]].to(s)).sum(-1)

    @torch.no_grad()
    def _order_spins(self, s1, s2):
        idx1 = self._state2idx(s1)
        idx2 = self._state2idx(s2)
        order = 2 * (idx1 < idx2).long() - 1  # 1 if idx1 < idx2, -1 otherwise
        order[idx1 == idx2] = 0  # 0 if idx1 == idx2
        return order

    @torch.no_grad()
    def __get_restricted_hilbert_mask(self, x_alpha, x_beta, i):
        # If we are using a restricted Hilbert space, there will be certain inputs where
        # one or both of the output bits can only take one value to preserve the physicality
        # of the state.  Here we define masks for these cases.
        if not self.use_restricted_hilbert:
            return None
        else:

            if i >= max(self._min_n_set, 1):
                if self._alpha_beta_restricted:
                    mask = torch.zeros(len(x_alpha), 4)

                    # Recall amp ordering of [|0,0>, |1,0>, |0,1>, |1,1>]
                    is_alpha_down_idxs = torch.LongTensor([0, 2])
                    is_alpha_up_idxs = torch.LongTensor([1, 3])
                    is_beta_down_idxs = torch.LongTensor([0, 1])
                    is_beta_up_idxs = torch.LongTensor([2, 3])
                    all_idxs = torch.LongTensor([0, 1, 2, 3])

                    for (_n_alpha_up, _n_alpha_down, _n_beta_up, _n_beta_down) in zip(
                            self.n_alpha_up, self.n_alpha_down, self.n_beta_up, self.n_beta_down
                        ):
                        n_alpha_up, n_beta_up = (x_alpha > 0).sum(1), (x_beta > 0).sum(1)
                        n_alpha_down, n_beta_down = x_alpha.shape[-1] - n_alpha_up, x_beta.shape[-1] - n_beta_up

                        if len(self.n_alpha_up)==1:
                            set_alpha_down_idxs = torch.where(n_alpha_up >= _n_alpha_up)[0]
                            set_alpha_up_idxs = torch.where(n_alpha_down >= _n_alpha_down)[0]
                            set_beta_down_idxs = torch.where(n_beta_up >= _n_beta_up)[0]
                            set_beta_up_idxs = torch.where(n_beta_down >= _n_beta_down)[0]

                            _mask = torch.ones(len(x_alpha), 4)
                            for set_idx, is_idx in zip(
                                    [set_alpha_down_idxs, set_alpha_up_idxs, set_beta_down_idxs, set_beta_up_idxs],
                                    [is_alpha_up_idxs, is_alpha_down_idxs, is_beta_up_idxs, is_beta_down_idxs]):
                                _mask[set_idx.repeat_interleave(len(is_idx)),
                                      is_idx.repeat(len(set_idx))] = 0

                        else:
                            set_alpha_down_idxs = torch.where(n_alpha_up >= _n_alpha_up)[0]
                            set_alpha_up_idxs = torch.where(n_alpha_down >= _n_alpha_down)[0]
                            set_beta_down_idxs = torch.where(n_beta_up >= _n_beta_up)[0]
                            set_beta_up_idxs = torch.where(n_beta_down >= _n_beta_down)[0]

                            already_invalid_mask = (n_alpha_up > _n_alpha_up) | (n_alpha_down > _n_alpha_down) | (n_beta_up > _n_beta_up) | (n_beta_down > _n_beta_down)
                            already_invalid_idxs = torch.where(already_invalid_mask)[0]

                            _mask = torch.ones(len(x_alpha), 4)
                            for set_idx, is_idx in zip(
                                    [set_alpha_down_idxs, set_alpha_up_idxs, set_beta_down_idxs, set_beta_up_idxs, already_invalid_idxs],
                                    [is_alpha_up_idxs, is_alpha_down_idxs, is_beta_up_idxs, is_beta_down_idxs, all_idxs]):
                                _mask[set_idx.repeat_interleave(len(is_idx)),
                                      is_idx.repeat(len(set_idx))] = 0

                        mask+=_mask

                    mask.clamp_(max=1)

                else:
                    mask = torch.ones(len(x_alpha), 4)

                    n_up = (x_alpha > 0).sum(1) + (x_beta > 0).sum(1)
                    n_down = (x_alpha.shape[-1] + x_beta.shape[-1]) - n_up

                    set_down_mask = torch.where(n_up >= self.n_tot_up)[0]
                    set_up_mask = torch.where(n_down >= self.n_tot_down)[0]
                    not_both_up_mask = torch.where(n_up == (self.n_tot_up - 1))[0]
                    not_both_both_mask = torch.where(n_down == (self.n_tot_down - 1))[0]

                    # Recall amp ordering of [|0,0>, |1,0>, |0,1>, |1,1>]
                    has_up_idxs = torch.LongTensor([1, 2, 3])
                    has_down_idxs = torch.LongTensor([0, 1, 2])
                    both_down_idxs = torch.LongTensor([0])
                    both_up_idxs = torch.LongTensor([3])

                    for set_idx, is_idx in zip([set_down_mask, set_up_mask, not_both_up_mask, not_both_both_mask],
                                               [has_up_idxs, has_down_idxs, both_up_idxs, both_down_idxs]):
                        mask[set_idx.repeat_interleave(len(is_idx)),
                             is_idx.repeat(len(set_idx))] = 0

            else:
                mask = torch.ones(len(x_alpha), 4)

            return mask

    def __get_x_ins(self, x, i):
        batch_size = x.shape[0]

        x_ins, x_order = [], None

        for use_spin_sym in [self.use_amp_spin_sym, self.use_phase_spin_sym]:

            if i == 0:
                x_order = torch.zeros(batch_size)  # Order of zeros <--> all configured orbitals are the same.
                x_in = torch.zeros(batch_size, 1)
                x1, x2 = torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)
            else:
                x1, x2 = x[:, :2*i:2].clone(), x[:, 1:2*i:2].clone()

                # If using spin-exchange sym, we want to map the inputs such that the input
                # is invarient under swapping alpha and beta electrons.
                if use_spin_sym:
                    x_order = self._order_spins(x1, x2)
                    if self.input_encoding is InputEncoding.INTEGER:
                        # |0,0>, |0,1>, |1,0>, |1,1> --> -1,0,0,1 (i.e. spin order invarient mapping).
                        x_in = (x1 + x2) - 1
                    else:  # self.input_encoding is InputEncoding.BINARY
                        # if x_order = 1/0 (i.e. idx(x1) <= idx(x2)) --> [x1, x2]
                        # elif x_order = -1 (i.e. idx(x1) > idx(x2)) --> [x2, x1]
                        _mask12, _mask21 = (x_order >= 0), (x_order < 0)
                        x_in = torch.zeros(batch_size, 2 * i).to(x1.device)
                        x_in[_mask12] = torch.cat([x1[_mask12], x2[_mask12]], -1)
                        x_in[_mask21] = torch.cat([x2[_mask21], x1[_mask21]], -1)

                else:
                    # x_order = torch.ones(batch_size)
                    if self.input_encoding is InputEncoding.INTEGER:
                        # |0,0>, |0,1>, |1,0>, |1,1> --> 0,1,2,3.
                        x_in = 2 * x1 + x2
                    else:  # self.input_encoding is InputEncoding.BINARY
                        x_in = torch.cat([x1, x2], -1)

            x_ins.append(x_in.to(self.device))

        # x_order = 0 --> input was [x2,x1] x1!=x2 --> should swap outputs for |01> and |10>.
        # x_order = 1 --> input was [x1,x2] with x1==x2  --> outputs for |01> and |10> should be equal.
        # x_order = 2 --> input was [x1,x2] with x1!=x2  --> outputs for |01> and |10> left unchanged.
        # print("x", x_order)
        if x_order is not None:
            x_order += 1
            x_order = x_order.long()

        return x_ins, x1, x2, x_order

    def __get_conditional_output(self, x_ins, i, ret_phase=True, out_device=None):
        if out_device is None:
            out_device = self.out_device
        if self.combined_amp_phase_blocks:
            if self.aggregate_phase or i == (self.N // 2 - 1):
                amp_i, phase_i = self.amp_layers[i](x_ins[0])
            else:
                amp_i = self.amp_layers[i](x_ins[0]).to(out_device)
                phase_i = torch.zeros(len(amp_i), self._n_out_phase)
        else:
            amp_i = self.amp_layers[i](x_ins[0])
            if ret_phase and (self.aggregate_phase or i == (self.N // 2 - 1)):
                if self.aggregate_phase:
                    phase_i = self.phase_layers[i](x_ins[1]).to(out_device)
                else:
                    phase_i = self.phase_layers[0](x_ins[1]).to(out_device)
            else:
                phase_i = torch.zeros(len(amp_i), self._n_out_phase)

        if ret_phase:
            return amp_i.to(out_device), phase_i.to(out_device)
        else:
            return amp_i.to(out_device)

    def __apply_symmetries(self, amp_i, phase_i, x_order):
        if self.use_amp_spin_sym:
            # Recall :
            #   x_order = 0 --> input was [x2,x1] x1!=x2 --> should swap outputs for |01> and |10>.
            #   x_order = 1 --> input was [x1,x2] with x1==x2  --> outputs for |01> and |10> should be equal.
            #   x_order = 2 --> input was [x1,x2] with x1!=x2  --> outputs for |01> and |10> left unchanged.

            # 5 outputs (deltas for |01>s):
            # print(x_order)
            idx2sort = torch.LongTensor([[0, 3, 4, 2], [0, 1, 1, 2], [0, 4, 3, 2]])
            amp_i = ( amp_i[:, [0, 1, 1, 2]] + amp_i.gather(1, idx2sort[x_order]) ) / 2
        else:
            amp_i =  amp_i[:, [0, 1, 2, 3]]

        if self.use_phase_spin_sym:
            # Shouldn't ever be configured.
            phase_i = phase_i[:, [0, 1, 1, 2]]

        return amp_i, phase_i

    # @torch.no_grad()
    def __apply_phase_shifts(self, phase, x_alpha, x_beta, x_tot_order=None):
        if self.use_phase_spin_sym:
            if x_tot_order is None:
                x_tot_order = self._order_spins(x_alpha, x_beta)

            # Insert symettry of spin-exchanged pairs.
            phase_shift_mask = (x_tot_order > 0)

            N_01 = ( (x_alpha[phase_shift_mask] <= 0) & (x_beta[phase_shift_mask] > 0) ).sum(1)
            if phase.dim() > 1:
                N_01.unsqueeze_(-1)
            phase[phase_shift_mask.to(phase.device)] += (math.pi * (N_01.fmod(2))).to(phase.device)

        return phase

    def __apply_activations(self, amp_i, phase_i, i, amp_mask=None, masking=None):
        if masking is None:
            masking = self.masking
        if ( masking is NadeMasking.NONE
            or (masking is NadeMasking.PARTIAL and i==(self.N // 2 - 1))):
            amp_mask = None
        if self.amplitude_activation is not None:
            # if (amp_mask is not None):
            #     amp_mask[amp_mask.sum(-1)==0] = 1
            amp_i = self.amplitude_activation(amp_i, amp_mask)
            # NEW
            if (amp_mask is not None) and (len(self.n_alpha_up) > 1):
                amp_i[amp_mask.sum(-1)==0] = float('-inf')
        if self.phase_activation is not None:
            if self.aggregate_phase:
                phase_i = self.phase_activation(phase_i, amp_mask)
            else:
                phase_i = self.phase_activation(phase_i)
        return amp_i, phase_i

    def _forward_sample(self, batch_size, ret_output=True, masking=None, max_batch_size=None):
        '''Generate 'batch_size' states from the underlying distribution.'''
        states = torch.zeros(1, 2, requires_grad=False)
        probs = torch.FloatTensor([1]).to(self.device)
        counts = torch.LongTensor([batch_size])

        blockidx2spin = torch.FloatTensor([[-1, -1], [1, -1], [-1, 1], [1, 1]])

        if ret_output:
            amps, phases = None, None

        # self.masking_reg = 0
        # use_reg = False

        for i in range(self.N // 2):
            # 1. Get inputs
            x_ins, x1, x2, x_order = self.__get_x_ins(states, i)
            # batch_size_i = len(x_ins[0])

            # 2. Calculate the outputs of the i-th block.
            if ret_output:
                amp_i, phase_i = self.__get_conditional_output(x_ins, i, ret_phase=True)
            else:
                amp_i = self.__get_conditional_output(x_ins, i, ret_phase=False)
                phase_i = torch.zeros_like(amp_i)

            _amp_mask = self.__get_restricted_hilbert_mask(x1, x2, i)

            amp_i, phase_i = self.__apply_symmetries(amp_i, phase_i, x_order)
            amp_i, phase_i = self.__apply_activations(amp_i, phase_i, i, _amp_mask, masking)

            # amp_i[_amp_mask.sum(-1, keep_dim=True)==0] = math.log(1/8)

            # 3. Sample the next states.
            #   1) Convert log_amplitudes to probabilites.
            #   2) Sample one label of the next qudit for each occurance (count) of each unique state.
            #   3) Update the states, counts and probabilites accordingly.
            #   4) Update the amplitudes and phases if we are returning the wavefunction as well.
            with torch.no_grad():

                if self.amplitude_encoding is AmplitudeEncoding.LOG_AMP:
                    probs_i = amp_i.detach().exp().pow(2)
                    # probs_i[_amp_mask.sum(-1, keep_dim=True) == 0] = 1/4
                    # probs_i = amp_samp_i.detach().exp().pow(2)
                else:
                    raise NotImplementedError()

                next_probs = probs.unsqueeze(1) * probs_i.to(probs)

                # Work around for https://github.com/numpy/numpy/issues/8317
                probs_i_np = probs_i.cpu().numpy().astype('float64')
                probs_i_np /= np.sum(probs_i_np, -1, keepdims=True)

                # if i == self.N // 2 - 1:
                #     print("probs_i (un-masked)", probs_i_np)
                    # amp_samp_i, _ = self.__apply_activations(amp_i_raw, phase_i_raw, i, _amp_mask)
                    # probs_i_tmp = amp_samp_i.detach().exp().pow(2)
                    # probs_i_tmp /= np.sum(probs_i_tmp, -1, keepdims=True)
                    # print("probs_i (masked)", probs_i_tmp)

                new_sample_counts = torch.LongTensor( multinomial_arr(counts, probs_i_np) )

                # Throw away unphysical samples.
                new_sample_counts *= _amp_mask.to(new_sample_counts)

                new_sample_mask = (new_sample_counts > 0)
                num_new_samples_per_state = new_sample_mask.sum(1)
                new_sample_next_idxs = torch.where(new_sample_counts > 0)[1]

                if i == 0:
                    states = blockidx2spin[new_sample_next_idxs]
                else:
                    states = torch.cat(
                        [states.repeat_interleave(num_new_samples_per_state, 0), blockidx2spin[new_sample_next_idxs]],
                        1)
                counts = new_sample_counts[new_sample_mask]
                probs = next_probs[new_sample_mask]

                if max_batch_size is not None:
                    if len(states) > max_batch_size:
                        raise MaxBatchSizeExceededError

            if ret_output:
                if i == 0:
                    amps = amp_i[new_sample_mask.to(amp_i.device)]
                    phases = phase_i[new_sample_mask.to(phase_i.device)]
                else:
                    amps = amps.repeat_interleave(num_new_samples_per_state.to(amps.device)).to(amp_i.device) + \
                           amp_i[new_sample_mask.to(amp_i.device)]

                    phases = phases.repeat_interleave(num_new_samples_per_state.to(phases.device)).to(phase_i.device) + \
                                 phase_i[new_sample_mask.to(phase_i.device)]

        phases = self.__apply_phase_shifts(phases, states[:, ::2], states[:, 1::2])

        ret = [states.to(self.out_device), counts, probs]

        if ret_output:
            if self.combined_amp_phase:
                output = torch.stack([amps, phases], -1).to(self.out_device)
            else:
                output = (amps.to(self.out_device), phases.to(self.out_device))
            ret.append(output)

        return ret

    def _forward_predict(self, x, masking=None):
        # if masking is None:
        #     masking = self.masking
        if x.dim() < 2:
            x.unsqueeze_(0)

        num_spins = x.shape[-1]

        x_alpha, x_beta = x[:, ::2].clone(), x[:, 1::2].clone()

        outputs = []
        for i in range(num_spins // 2):
            x_ins, x1, x2, x_order = self.__get_x_ins(x, i)
            # if _amp_mask is not None:
            amp_i, phase_i = self.__get_conditional_output(x_ins, i, ret_phase=True)
            amp_i, phase_i = self.__apply_symmetries(amp_i, phase_i, x_order)

            if masking is not NadeMasking.NONE:
                amp_i, phase_i = self.__apply_activations(amp_i, phase_i, i, self.__get_restricted_hilbert_mask(x1, x2, i), masking)
            else:
                amp_i, phase_i = self.__apply_activations(amp_i, phase_i, i, None, masking)

            if i==(num_spins // 2 - 1):
                phase_i = self.__apply_phase_shifts(phase_i, x_alpha, x_beta)

            if self.combined_amp_phase:
                out_i = torch.stack([amp_i, phase_i], -1)
            else:
                out_i = amp_i, phase_i

            outputs.append(out_i.to(self.out_device))

        return torch.stack(outputs, 1)

    def forward(self, x, *args, **kwargs):
        if self.sampling:
            return self._forward_sample(x, *args, **kwargs)
        else:
            x = x.float().to(self.device)
            return self._forward_predict(x, *args, **kwargs).to(self.out_device)
