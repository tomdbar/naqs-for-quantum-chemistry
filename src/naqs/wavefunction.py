import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.naqs.network.activations import SoftmaxLogProbAmps
from src.naqs.network.base import AmplitudeEncoding, InputEncoding, NadeMasking
from src.naqs.network.nade import ComplexAutoregressiveMachine1D_OrbitalNade
from src.utils.hilbert import Encoding
from src.utils.system import mk_dir

import math


class _NAQSComplex_Base(ABC):
    _cplx_dtype = np.complex64

    def __init__(self,
                 model,
                 hilbert,
                 qubit_ordering=-1,  # 1: default, 0: random, -1:reverse, list:custom)

                 device=None,
                 out_device="cpu",

                 *args,
                 **kwargs
                 ):

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            # I trust that you know what you are doing!
            self.device = device
        if out_device is None:
            self.out_device = self.device
        else:
            # I still trust that you know what you are doing!
            self.out_device = out_device

        self.hilbert = hilbert
        self._N_model = self.hilbert.N - self.hilbert.N_occ
        self._N_fixed = self.hilbert.N_occ
        self.encoding = hilbert.encoding

        if self.encoding == Encoding.BINARY:
            self.qubit_vals = torch.FloatTensor([0, 1])
        elif self.encoding == Encoding.SIGNED:
            self.qubit_vals = torch.FloatTensor([-1, 1])
        else:
            raise ValueError("{} is not a recognised encoding.".format(self.encoding))

        if qubit_ordering == 1:
            # order qubits as passed.
            self.permute_qubits = False
            self.qubit2model_permutation = self.model2qubit_permutation = np.arange(self.hilbert.N)
        else:
            self.permute_qubits = True
            if qubit_ordering == -1:
                # reverse qubits
                self.qubit2model_permutation = np.concatenate([np.arange(self._N_fixed),
                                                               np.arange(self.hilbert.N - 1, self._N_fixed - 1, -1)])
            elif qubit_ordering == 0:
                # random ordering
                self.qubit2model_permutation = np.concatenate([np.arange(self._N_fixed),
                                                               self._N_fixed + np.random.permutation(self._N_model)])
            else:
                # assume a custom ordering has been passed
                qubit_ordering = np.array(qubit_ordering)
                assert len(qubit_ordering) == self.hilbert.N, f"Custom qubit ordering must have length {self.hilbert.N}"
                assert len(
                    set(qubit_ordering)) == self.hilbert.N, f"Custom qubit ordering have each qubit exactly once."
                if not (qubit_ordering.argsort()[:self._N_fixed] == np.arange(self._N_fixed)).all():
                    qubit_ordering = np.concatenate([np.arange(self._N_fixed),
                                                     np.delete(qubit_ordering,
                                                               qubit_ordering.argsort()[:self._N_fixed])])
                    print(f"Custom qubit ordering modifed so that the {self._N_fixed} fixed electrons are first.")

                self.qubit2model_permutation = np.array(qubit_ordering)
            self.model2qubit_permutation = np.argsort(self.qubit2model_permutation)

        self.model = model(*args, **kwargs)

        self.train_model()
        self.predict_model()

    def train_model(self):
        self.model.train()

    def eval_model(self):
        self.model.eval()

    def sample_model(self):
        self.model.sample()

    def predict_model(self):
        self.model.predict()

    def _qubit2idx(self, vals):
        if self.encoding == Encoding.BINARY:
            idxs = vals
        else:  # self.encoding==Encoding.SIGNED
            idxs = (vals + 1) / 2
        return idxs.type(torch.long)

    def _idx2qubit(self, idxs):
        if self.encoding == Encoding.BINARY:
            vals = idxs.type(torch.float64)
        else:  # self.encoding==Encoding.SIGNED
            vals = self.qubit_vals.gather(0, idxs.type(torch.long))
        return vals

    def _state2model(self, s):
        if not self.model.sampling:
            s = s[..., self.qubit2model_permutation]
            if self._N_fixed > 0:
                s = s[..., self._N_fixed:]
        return s.float()

    def _model2state(self, log_psi, s):
        if not self.model.sampling:
            if self._N_fixed > 0:
                # Set prob. amp. of fixed states being un-occupied --> 1.
                # Set prob. amp. of fixed states being un-occupied --> 0.
                try:
                    log_psi = torch.cat([self.__fixed_log_psi, log_psi], dim=2)
                except:
                    self.__fixed_log_psi = torch.zeros(len(s), 2, self._N_fixed, 2).to(log_psi)
                    self.__fixed_log_psi[:, 0, :, 0] = -10
                    log_psi = torch.cat([self.__fixed_log_psi, log_psi], dim=2)
            log_psi = log_psi[:, self.model2qubit_permutation, ...]
        return log_psi

    def _evaluate_model(self, s):
        '''
        Evaluates the model to give conditional distributions.

        The model output is expected to be of the form: [batch, qubit, qubit_value, cond_prob].

        Note: This is the function we will overwrite to implement more complex symettries and
        restrictions on the permitted model.
        '''
        return self._model2state(self.model(self._state2model(s)), s)

    def _evaluate_log_psi(self, s, gather_state=True):
        '''
        Returns model output in the form:
            gather_state == False : [batch, qubit, qubit_value, cond_prob]
            gather_state == True  : [batch, qubit, cond_prob]
        '''
        model_output = self._evaluate_model(s)

        if gather_state:
            model_output = model_output.gather(-2,
                                               self._qubit2idx(s).view(model_output.shape[0], -1, 1, 1).repeat(1, 1, 1,
                                                                                                               2))

        if self.model.amplitude_encoding is AmplitudeEncoding.AMP:
            # [batch, ..., [amp, phase]] --> [batch, ..., [log_amp, phase]]
            amp, phase = torch.split(model_output,1,-1)
            model_output = torch.stack([amp.log(), phase], -1)
        return model_output

    def log_psi(self, s, ret_complex=False, combine_conditionals=True):
        log_psi = self._evaluate_log_psi(s, gather_state=True)

        if combine_conditionals:
            log_psi = log_psi.sum(axis=1)
            # log_psi[..., 1] = log_psi[..., 1].clamp_(0, math.pi)
            # amp, phase = torch.chunk(log_psi, 2, -1)
            # log_psi = torch.stack([amp, math.pi * F.hardtanh(phase, 0, 1)], -1)
            # log_psi[..., 1] = F.hardtanh(log_psi[..., 1], 0, math.pi)
        log_psi = log_psi.squeeze()

        if ret_complex:
            log_psi = log_psi.numpy()
            log_psi = log_psi[..., 0] + 1j * log_psi[..., 1]
            log_psi = log_psi.astype(self._cplx_dtype)

        return log_psi

    def psi(self, s, ret_complex=False, combine_conditionals=True):
        log_psi = self.log_psi(s, ret_complex=False, combine_conditionals=combine_conditionals)

        psi = torch.stack([log_psi[..., 0].exp() * log_psi[..., 1].cos(),
                           log_psi[..., 0].exp() * log_psi[..., 1].sin()],
                          -1)

        if ret_complex:
            psi = psi.data.numpy()
            psi = psi[..., 0] + 1j * psi[..., 1]
            psi = psi.astype(self._cplx_dtype)

        return psi

    def amplitude(self, s, combine_conditionals=True):
        log_amps = self.log_psi(s, combine_conditionals=False)[..., 0]

        if combine_conditionals:
            log_amps = log_amps.sum(axis=-1).squeeze()
        if log_amps.dim() == 0:
            log_amps.unsqueeze_(0)

        return log_amps.exp()

    def phase(self, s, combine_conditionals=True):
        phases = self.log_psi(s, combine_conditionals=False)[..., 1]

        if combine_conditionals:
            phases = phases.sum(axis=-1).squeeze()

        return phases

    @abstractmethod
    def sample(self, num_samples=1, ret_probs=False):
        raise NotImplementedError()

    def parameters(self, group_idx=None):
        if group_idx is not None:
            raise NotImplementedError()
        return self.model.parameters()

    def count_parameters(self, print_verbose=True):
        if print_verbose:
            print("---modules : parameters---\n")
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            if print_verbose:
                print(f"{name} : {param}")
            total_params += param
        print(f"\n--> Total trainable params: {total_params}")
        return total_params

    def save(self, fname, quiet):
        dir = os.path.dirname(fname)
        if dir != '':
            mk_dir(dir, quiet)
        checkpoint = {
            'model:state_dict': self.model.state_dict(),
            'wavefunction:permute_qubits': self.permute_qubits,
            'wavefunction:qubit2model_permutation': self.qubit2model_permutation,
            'wavefunction:model2qubit_permutation': self.model2qubit_permutation
        }
        if os.path.splitext(fname)[-1] != '.pth':
            fname += '.pth'
        torch.save(checkpoint, fname)
        if not quiet: print("Saved NAQSComplex wavefunction model to {}.".format(fname))
        return fname

    def load(self, fname):
        checkpoint = torch.load(fname, map_location=self.device)
        self.model.load_state_dict(checkpoint['model:state_dict'])
        self.permute_qubits = checkpoint['wavefunction:permute_qubits']
        self.qubit2model_permutation = checkpoint['wavefunction:qubit2model_permutation']
        self.model2qubit_permutation = checkpoint['wavefunction:model2qubit_permutation']
        print("Loaded NAQSComplex wavefunction from {}.".format(fname))

    @torch.no_grad()
    def save_psi(self, fname="psi", subspace_args={}, normalise=True):
        basis, basis_idxs = self.hilbert.get_subspace(**subspace_args, ret_idxs=True)

        amps = self.amplitude(basis)
        if normalise:
            amps /= amps.pow(2).sum().pow(0.5)
        phases = self.phase(basis)

        sort_args = torch.argsort(amps, descending=True)

        basis, basis_idxs = basis[sort_args], basis_idxs[sort_args]
        psi = torch.stack([ amps[sort_args], phases[sort_args] ]).transpose(0, 1)

        np.savetxt(f"{fname}.txt", psi.detach().numpy(), fmt='%5e')
        np.savetxt(f"{fname}_basis.txt", basis.clamp(min=0), fmt='%i')
        np.savetxt(f"{fname}_basis_idxs.txt", basis_idxs, fmt='%i')

        print("Saved psi (as amplitude/phase), basis and basis_idxs to : \n\t{}\n\t{}\n\t{}".format(
            fname,
            f"{fname}_basis",
            f"{fname}_basis_idxs"
        ))

class NAQSComplex_NADE_orbitals(_NAQSComplex_Base):

    def __init__(self,
                 hilbert,
                 N_up=None,
                 N_alpha=None,
                 N_beta=None,

                 qubit_ordering=-1,  # 1: default, 0: random, -1:reverse, list:custom

                 num_lut=0,

                 input_encoding=InputEncoding.BINARY,

                 n_electrons=None,
                 n_alpha_electrons=None,
                 n_beta_electrons=None,
                 # mask_to_restricted_hilbert=True,
                 masking = NadeMasking.PARTIAL,

                 amp_hidden_size=[],
                 amp_hidden_activation=nn.ReLU,
                 amp_bias=True,

                 phase_hidden_size=[],
                 phase_hidden_activation=nn.ReLU,
                 phase_bias=True,

                 combined_amp_phase_blocks=False,

                 use_amp_spin_sym=True,
                 use_phase_spin_sym=True,
                 aggregate_phase=True,

                 amp_batch_norm=False,
                 phase_batch_norm=False,
                 batch_norm_momentum=1,

                 amp_activation=SoftmaxLogProbAmps,
                 phase_activation=None,
                 ):

        super().__init__(
            hilbert=hilbert,
            qubit_ordering=qubit_ordering,

            model=ComplexAutoregressiveMachine1D_OrbitalNade,

            num_qubits = hilbert.N - hilbert.N_occ,

            num_lut = num_lut,

            input_encoding=input_encoding,

            n_electrons=n_electrons,
            n_alpha_electrons=n_alpha_electrons,
            n_beta_electrons=n_beta_electrons,
            # mask_to_restricted_hilbert=mask_to_restricted_hilbert,
            masking=masking,

            amp_hidden_size=amp_hidden_size,
            amp_hidden_activation=amp_hidden_activation,
            amp_bias=amp_bias,

            phase_hidden_size=phase_hidden_size,
            phase_hidden_activation=phase_hidden_activation,
            phase_bias=phase_bias,

            amp_activation=amp_activation,
            phase_activation=phase_activation,

            combined_amp_phase_blocks=combined_amp_phase_blocks,

            use_amp_spin_sym=use_amp_spin_sym,
            use_phase_spin_sym=use_phase_spin_sym,
            aggregate_phase=aggregate_phase,

            amp_batch_norm=amp_batch_norm,
            phase_batch_norm=phase_batch_norm,
            batch_norm_momentum=batch_norm_momentum
        )
        if qubit_ordering==1:
            self.state2model_permutation_shell = torch.arange(self._N_model//2)
        elif qubit_ordering==-1:
            self.qubit2model_permutation = torch.stack([torch.arange(self._N_model-2,-1,-2), torch.arange(self._N_model-1,-1,-2)],1).reshape(-1)
            self.qubit2model_permutation = torch.stack(
                [torch.arange(self._N_model - 2, -1, -2), torch.arange(self._N_model - 1, -1, -2)], 1).reshape(-1)
            self.model2qubit_permutation = np.argsort(self.qubit2model_permutation)
            print("self.qubit2model_permutation (overwritten) :", self.qubit2model_permutation)
            print("self.model2qubit_permutation (overwritten) :", self.model2qubit_permutation)
            self.state2model_permutation_shell = torch.arange(self._N_model//2-1, -1, -1)
        else:
            self.state2model_permutation_shell = self.qubit2model_permutation[1::2]//2
        self.model2state_permutation_shell = np.argsort(self.state2model_permutation_shell)
        print("self.state2model_permutation_shell :", self.state2model_permutation_shell)
        print("self.model2state_permutation_shell :", self.model2state_permutation_shell)

        if ((N_up is None) and (N_alpha is None) and (N_beta is None)):
            self.use_hilbert_restrictions, self._hilbert_restrictions = False, None
        else:
            restricted_subspace = self.hilbert.get_subspace(N_up, N_alpha, N_beta)
            if len(restricted_subspace) == 0:
                raise ValueError("Invalid restrictions on subspace.")
            self.use_hilbert_restrictions = True
            if (N_alpha is None) and (N_beta is None):
                self._hilbert_restrictions = [N_up]
            else:
                self._hilbert_restrictions = [N_alpha, N_beta]

    def _evaluate_log_psi(self, s, gather_state=True):
        '''
        Returns model output in the form:
            gather_state == False : [batch, qubit, qubit_value, cond_prob]
            gather_state == True  : [batch, qubit, cond_prob]
        '''
        model_output = self._evaluate_model(s)

        if gather_state:
            model_output = model_output.gather(-2,
                                               self.state2shell(s).view(model_output.shape[0], -1, 1, 1).repeat(1, 1, 1, 2))
            # print(f"\nGathered output\n {model_output}")

        if self.model.amplitude_encoding is AmplitudeEncoding.AMP:
            # [batch, ..., [amp, phase]] --> [batch, ..., [log_amp, phase]]
            amp, phase = torch.split(model_output,1,-1)
            model_output = torch.stack([amp.log(), phase], -1)
        return model_output

    def parameters(self, group_idx=None):
        if group_idx is None or group_idx==-1:
            return self.model.parameters()

        elif group_idx==0:
            # Return non-lut params
            params = [p for p in self.model.amp_layers[self.model.num_lut:].parameters()]
            params += [p for p in self.model.phase_layers[self.model.num_lut:].parameters()]
            if self.model.amplitude_activation is not None:
                params += [p for p in self.model.amplitude_activation.parameters()]
            if self.model.phase_activation is not None:
                params += [p for p in self.model.phase_activation.parameters()]
            return params

        elif group_idx==1:
            # Return lut params
            params = [p for p in self.model.amp_layers[:self.model.num_lut].parameters()]
            params += [p for p in self.model.phase_layers[:self.model.num_lut].parameters()]
            return params

        else:
            raise NotImplementedError()

    def conditional_parameters(self, cond_idx=None):
        if cond_idx is None:
            return self.model.parameters()
        else:
            params = [p for p in self.model.amp_layers[cond_idx].parameters()]
            if cond_idx >= self.model.num_lut and self.model.amplitude_activation is not None:
                    params += [p for p in self.model.amplitude_activation.parameters()]

            if (self.model.aggregate_phase) or (cond_idx == self._N_model // 2 - 1):
                params += [p for p in self.model.phase_layers[cond_idx].parameters()]
                if cond_idx >= self.model.num_lut and self.model.phase_activation is not None:
                    params += [p for p in self.model.phase_activation.parameters()]

            return params

    # def parameters(self, group_idx=None):
    #     if group_idx is None or group_idx==-1:
    #         return self.model.parameters()
    #     else:
    #         params = [p for p in self.model.amp_layers[group_idx].parameters()]
    #         params += [p for p in self.model.phase_layers[group_idx].parameters()]
    #         # if self.model.amplitude_activation is not None:
    #         #     params += [p for p in self.model.amplitude_activation.parameters()]
    #         # if self.model.phase_activation is not None:
    #         #     params += [p for p in self.model.phase_activation.parameters()]
    #         return params

    def state2shell(self, s):
        shp = s.shape
        return (s.view(shp[0], shp[-1]//2, 2).clamp_min(0) * torch.LongTensor([1, 2])).sum(-1)

    def _evaluate_model(self, s):
        '''
        Evaluates the model to give conditional distributions.

        The model output is expected to be of the form: [batch, qubit, qubit_value, cond_prob].

        Note: This is the function we will overwrite to implement more complex symettries and
        restrictions on the permitted model.
        '''
        self.model.predict()
        model_output = self.model(self._state2model(s))
        return model_output[:, self.model2state_permutation_shell, ...]

    def truncated_log_psi(self, k_trunc):
        states, log_psi = self.model.top_k(k_trunc)
        states = states[:, self.model2qubit_permutation]
        return states, log_psi

    def sample(self, num_samples=1, ret_probs=True, ret_log_psi=True, ret_norm_reg=False, eval_mode=False,
               *args, **kwargs):
        '''Sample from the probability distribution given by the wavefunction.
        '''
        self.model.sample()
        if eval_mode:
            model_was_training = self.model.training
            self.model.eval()
        else:
            model_was_eval = not self.model.training
            self.model.train()

        if not ret_norm_reg:
            states, counts, probs, log_psi = self.model(num_samples, *args, **kwargs)
        else:
            states, counts, probs, log_psi, norm_reg = self.model(num_samples, ret_norm_reg=True)
        states = self.hilbert.to_state_tensor( states[:, self.model2qubit_permutation] )

        output = [states, counts]
        if ret_probs:
            output.append(probs)
        if ret_log_psi:
            output.append(log_psi)
        if ret_norm_reg:
            output.append(norm_reg)

        if eval_mode:
            if model_was_training:
                self.model.train()
        else:
            if model_was_eval:
                self.model.eval()

        return output

# class NAQSComplex_MLP(_NAQSComplex_Base):
#     _cplx_dtype = np.complex64
#
#     def __init__(self,
#                  hilbert,
#                  qubit_ordering=-1,  # 1: default, 0: random, -1:reverse, list:custom
#
#                  num_layers=3,
#                  n_hidden=32,
#                  connect_qubits=[],
#                  share_weights=False,
#                  bias=True,
#                  in_activation=None,
#                  hid_activation=nn.ReLU,
#                  amp_activation=None,
#                  phase_activation=None,
#                  aggregate_phase=True,
#                  init_weight_scale=0.01,
#                  use_prob_log=False,
#                  round_phases=True,
#                  round_phases_scaling_factor=100,
#                  device=None,
#                  out_device="cpu",
#                  max_batch_size=-1,
#                  use_checkpointing=False,
#                  batch_norm=False,
#                  batch_norm_momentum=1,
#                  weight_norm=False
#                  ):
#
#         super().__init__(
#             hilbert=hilbert,
#             qubit_ordering=qubit_ordering,
#
#             model=ComplexAutoregressiveMachine1D_MLP,
#
#             N=hilbert.N - hilbert.N_occ,
#             in_channels=1,
#             hid_channels=n_hidden,
#             out_channels=2,
#             num_layers=num_layers,
#             connect_qubits=connect_qubits,
#             share_weights=share_weights,
#             in_activation=in_activation,
#             hid_activation=hid_activation,
#             amp_activation=amp_activation,
#             phase_activation=phase_activation,
#             aggregate_phase=aggregate_phase,
#             init_weight_scale=init_weight_scale,
#             use_prob_log=use_prob_log,
#             bias=bias,
#             max_batch_size=max_batch_size,
#             use_checkpointing=use_checkpointing,
#             batch_norm=batch_norm,
#             batch_norm_momentum=batch_norm_momentum,
#             weight_norm=weight_norm
#         )
#
#     @torch.no_grad()
#     def sample(self, num_samples=1, ret_probs=False):
#         self.model.sample()
#         model_was_training = self.model.training
#         self.model.eval()
#
#         def next_conditional_prob(s):
#             if s.dim() < 3:
#                 # take [batch, spins] --> [batch, 1, spins] as there is one input channel.
#                 s = s.unsqueeze(1)
#             log_psi = self._normalise_in_log_space(self.model(s).to("cpu"))
#             # If we are sampling, we are always looking for the probabilty that states are in zero.
#             log_psi = log_psi[:, 0, ...]
#             probs = log_psi[..., 0].exp().pow(2)
#             return probs.squeeze()
#
#         samps = torch.zeros((num_samples, 1, self.hilbert.N))
#
#         if ret_probs:
#             probs = torch.ones(num_samples)
#
#         for i in range(self.hilbert.N):
#             if i == 0:
#                 last_samps = torch.zeros((num_samples, 1))
#             else:
#                 last_samps = samps[..., i - 1].clone()
#             probs_next = next_conditional_prob(last_samps)
#             next_samp_one = (probs_next < torch.rand(num_samples))
#
#             vals = self._idx2qubit(next_samp_one.type(torch.long))
#             samps[..., i] = vals.unsqueeze(-1)
#
#             if ret_probs:
#                 probs_next[next_samp_one.squeeze()] = 1 - probs_next[next_samp_one.squeeze()]
#                 probs *= probs_next
#
#         if self.permute_qubits:
#             samps = samps[..., self.model2qubit_permutation]
#
#         self.model.predict()
#         if model_was_training:
#             self.model.train()
#
#         # (batch, channel, qubit) --> (batch, qubit)
#         samps.squeeze_(1)
#
#         if ret_probs:
#             return samps, probs
#         else:
#             return samps
#
#
# class NAQSComplex_RNN(_NAQSComplex_Base):
#
#     def __init__(self,
#                  hilbert,
#                  qubit_ordering=-1,  # 1: default, 0: random, -1:reverse, list:custom
#
#                  num_qubit_vals=2,
#                  rnn_cell=nn.GRUCell,
#              rnn_input_size=1,
#                  rnn_hidden_size=16,
#                  rnn_bias=True,
#                  rnn_num_layers=1,
#
#                  mlp_hidden_size=[64],
#                  mlp_hidden_activation=nn.ReLU,
#                  mlp_bias=True,
#
#                  amp_activation=SoftmaxLogProbAmps,
#                  phase_activation=ScaledSoftSign
#                  ):
#
#         super().__init__(
#             hilbert=hilbert,
#             qubit_ordering=qubit_ordering,
#
#             model=ComplexAutoregressiveMachine1D_RNN,
#
#             num_qubit_vals=num_qubit_vals,
#             rnn_cell=rnn_cell,
#             rnn_input_size=rnn_input_size,
#             rnn_hidden_size=rnn_hidden_size,
#             rnn_bias=rnn_bias,
#             rnn_num_layers=rnn_num_layers,
#
#             mlp_hidden_size=mlp_hidden_size,
#             mlp_hidden_activation=mlp_hidden_activation,
#             mlp_bias=mlp_bias,
#
#             amp_activation=amp_activation,
#             phase_activation=phase_activation
#         )
#
#     @torch.no_grad()
#     def sample(self, num_samples=1, ret_probs=False):
#         '''Sample from the probability distribution given by the wavefunction.
#
#         This sampling method makes no restriction on the allowed states
#         to sample (i.e. any state from the Hilbert state could be sampled).
#         '''
#         self.model.sample()
#         model_was_training = self.model.training
#         self.model.eval()
#
#         samples = torch.zeros((num_samples, self.hilbert.N))
#
#         if ret_probs:
#             probs = torch.ones(num_samples)
#
#         last_vals = torch.zeros((num_samples, 1))
#         for i in range(self.hilbert.N):
#             q_probs = self.model(last_vals)[..., 0].exp().pow(2)
#             q_idx = torch.multinomial(q_probs, 1).squeeze()
#             q_vals = self._idx2qubit(q_idx)
#             samples[..., i] = q_vals
#
#             if ret_probs:
#                 probs *= q_probs[torch.arange(len(q_probs)), q_idx]
#
#         if self.permute_qubits:
#             samples = samples[..., self.model2qubit_permutation]
#
#         self.model.predict()
#         if model_was_training:
#             self.model.train()
#
#         # (batch, channel, qubit) --> (batch, qubit)
#         samples.squeeze_(1)
#
#         if ret_probs:
#             return samples, probs
#         else:
#             return samples
#
#
# class NAQSComplex_MPS(_NAQSComplex_Base):
#
#     def __init__(self,
#                  hilbert,
#                  N_up=None,
#                  N_alpha=None,
#                  N_beta=None,
#
#                  qubit_ordering=-1,  # 1: default, 0: random, -1:reverse, list:custom
#
#                  num_qubit_vals=2,
#                  # rnn_cell=nn.GRUCell,
#                  rnn=nn.GRU,
#
#                  rnn_shared=False,
#                  mlp_shared=False,
#
#                  rnn_input_size=1,
#                  rnn_hidden_size=16,
#                  rnn_bias=True,
#                  rnn_num_layers=1,
#
#                  mlp_hidden_size=[64],
#                  mlp_hidden_activation=nn.ReLU,
#                  mlp_bias=True,
#
#                  post_rnn_batch_norm=False,
#                  post_mlp_batch_norm=False,
#                  batch_norm_momentum=1,
#
#                  amp_activation=SoftmaxLogProbAmps,
#                  phase_activation=ScaledSoftSign
#                  ):
#
#         super().__init__(
#             hilbert=hilbert,
#             qubit_ordering=qubit_ordering,
#
#             model=ComplexAutoregressiveMachine1D_MPS,
#
#             num_qubits = hilbert.N - hilbert.N_occ,
#             num_qubit_vals=num_qubit_vals,
#
#             rnn_shared=rnn_shared,
#             mlp_shared=mlp_shared,
#
#             rnn=rnn,
#             rnn_input_size=rnn_input_size,
#             rnn_hidden_size=rnn_hidden_size,
#             rnn_bias=rnn_bias,
#             rnn_num_layers=rnn_num_layers,
#
#             mlp_hidden_size=mlp_hidden_size,
#             mlp_hidden_activation=mlp_hidden_activation,
#             mlp_bias=mlp_bias,
#
#             post_rnn_batch_norm=post_rnn_batch_norm,
#             post_mlp_batch_norm=post_mlp_batch_norm,
#             batch_norm_momentum=batch_norm_momentum,
#
#             amp_activation=amp_activation,
#             phase_activation=phase_activation
#         )
#
#         if ((N_up is None) and (N_alpha is None) and (N_beta is None)):
#             self.use_hilbert_restrictions, self._hilbert_restrictions = False, None
#         else:
#             restricted_subspace = self.hilbert.get_subspace(N_up, N_alpha, N_beta)
#             if len(restricted_subspace) == 0:
#                 raise ValueError("Invalid restrictions on subspace.")
#             self.use_hilbert_restrictions = True
#             if (N_alpha is None) and (N_beta is None):
#                 self._hilbert_restrictions = [N_up]
#             else:
#                 self._hilbert_restrictions = [N_alpha, N_beta]
#
#     def _evaluate_model(self, s):
#         '''
#         Evaluates the model to give conditional distributions.
#
#         The model output is expected to be of the form: [batch, qubit, qubit_value, cond_prob].
#
#         Note: This is the function we will overwrite to implement more complex symettries and
#         restrictions on the permitted model.
#         '''
#         if not self.use_hilbert_restrictions:
#           return super()._evaluate_model(s)
#
#         else:
#             model_was_sampling = self.model.sampling
#             model_was_combined_amp_phase = self.model.combined_amp_phase
#
#             # Shift all inputs by one, to ensure AR property.
#             s_in = s.clone()
#             s = self._state2model(s)
#             s = s.roll(1, -1)
#             s[..., 0] *= 0
#
#             self.model.sample() # <-- Important to set this after _state2model!
#             self.model.combine_amp_phase(False)
#
#             model_outputs = []
#
#             # Calculate the value we will default conditional amplitudes/probabilites to
#             # if we need to deterministically set a qubit.
#             if self.model.amplitude_encoding is (AmplitudeEncoding.AMP or AmplitudeEncoding.PROB):
#                 def set_qubits(cond_amps, up_mask, down_mask):
#                     m = torch.ones_like(cond_amps, requires_grad=False)
#                     m[down_mask, 0] = 1 / cond_amps.detach()[down_mask, 0] # Set amp/prob psi_i(0) = 1
#                     m[down_mask, 1] = 0 # Set amp/prob psi_i(1) = 0
#                     m[up_mask, 0] = 0  # Set amp/prob psi_i(0) = 0
#                     m[up_mask, 1] = 1 / cond_amps.detach()[up_mask, 1]  # Set amp/prob psi_i(1) = 1
#                     return cond_amps * m
#
#             else: # --> AmplitudeEncoding.LOG_AMP or AmplitudeEncoding.LOG_PROB
#                 if self.model.amplitude_encoding is AmplitudeEncoding.LOG_PROB:
#                     log0 = torch.FloatTensor([1e-10]).log()
#                     log1 = 0.5 * (1 - log0.exp()).log()
#                 else: # --> AmplitudeEncoding.LOG_AMP
#                     log0 = torch.FloatTensor([1e-3]).log()
#                     log1 = 0.5*(1-(2*log0).exp()).log()
#                 def set_qubits(cond_amps, up_mask, down_mask):
#                     m = torch.ones_like(cond_amps, requires_grad=False).to(cond_amps)
#                     m[down_mask, 0] = log1 / cond_amps.detach()[down_mask, 0] # Set amp/prob psi_i(0) = 1
#                     m[down_mask, 1] = log0 / cond_amps.detach()[down_mask, 1] # Set amp/prob psi_i(1) = 0
#                     m[up_mask, 0] = log0 / cond_amps.detach()[up_mask, 0] # Set amp/prob psi_i(0) = 1
#                     m[up_mask, 1] = log1 / cond_amps.detach()[up_mask, 1] # Set amp/prob psi_i(1) = 1
#                     return cond_amps * m
#
#             # We only have a single restriction (i.e. total magnetisation is fixed).
#             if len(self._hilbert_restrictions)==1:
#                 num_up, num_down = torch.zeros(s.shape[0], requires_grad=False), torch.zeros(s.shape[0], requires_grad=False)
#                 max_up, max_down = self._hilbert_restrictions[0], self._N_model - self._hilbert_restrictions[0]
#
#                 # s = self._state2model(s)
#                 for i, s_i in enumerate(s.T):
#                     # print("i")
#                     if i > 0:
#                         num_up += (s_i > 0)
#                         num_down += (s_i <= 0)
#                     up_mask, down_mask = (num_down >= max_down), (num_up >= max_up)
#                     cond_amps_i, cond_phase_i = self.model(s_i)
#
#                     # We only have to start checking validity once the smallest number
#                     # of electrons set to 'up' have possibly been sampled.
#                     # if i >= min(max_up, max_down):
#                     cond_amps_i = set_qubits(cond_amps_i, up_mask, down_mask)
#                     model_out_i = torch.stack([cond_amps_i, cond_phase_i], -1)
#                     model_outputs.append(model_out_i)
#
#             # We have two restrictions (i.e. total magnetisation in alpha/beta shells.).
#             else:
#                 num_up_alpha, num_down_alpha, num_up_beta, num_down_beta = [torch.zeros(s.shape[0]) for _ in range(4)]
#                 max_up_alpha, max_down_alpha = self._hilbert_restrictions[0], math.floor(self._N_model/2) - self._hilbert_restrictions[0]
#                 max_up_beta, max_down_beta = self._hilbert_restrictions[1], math.ceil(self._N_model/2) - self._hilbert_restrictions[1]
#
#                 is_beta = (self.qubit2model_permutation % 2)
#
#                 for i, s_i in enumerate(s.T):
#                     if i > 0:
#                         if is_beta[i-1]:
#                             num_up_beta += (s_i > 0)
#                             num_down_beta += (s_i <= 0)
#                         else:
#                             num_up_alpha += (s_i > 0)
#                             num_down_alpha += (s_i <= 0)
#
#                     if is_beta[i]:
#                         up_mask, down_mask = (num_down_beta >= max_down_beta), (num_up_beta >= max_up_beta)
#                     else:
#                         up_mask, down_mask = (num_down_alpha >= max_down_alpha), (num_up_alpha >= max_up_alpha)
#
#                     cond_amps_i, cond_phase_i = self.model(s_i)
#                     # We only have to start checking validity once the smallest number
#                     # of electrons set to 'up' have possibly been sampled.
#                     # if i >= min(max_up, max_down):
#                     cond_amps_i = set_qubits(cond_amps_i, up_mask, down_mask)
#                     model_out_i = torch.stack([cond_amps_i, cond_phase_i], -1)
#                     model_outputs.append(model_out_i)
#
#             model_output = torch.stack(model_outputs, 1)
#
#             if not model_was_sampling:
#                 self.model.predict()
#             if model_was_combined_amp_phase:
#                 self.model.combine_amp_phase(True)
#
#             return self._model2state(model_output, s_in)
#
#     @torch.no_grad()
#     def sample(self, num_samples=1, ret_probs=False):
#         '''Sample from the probability distribution given by the wavefunction.
#         '''
#         self.model.sample()
#         model_was_training = self.model.training
#         self.model.eval()
#
#         samples = torch.zeros((num_samples, self.hilbert.N))
#
#         if self.use_hilbert_restrictions:
#             q_probs_up = torch.FloatTensor([0, 1], device=self.model.out_device)
#             q_probs_down = torch.FloatTensor([1, 0], device=self.model.out_device)
#             if len(self._hilbert_restrictions)==1:
#                 num_up, num_down = torch.zeros(num_samples), torch.zeros(num_samples)
#                 max_up, max_down = self._hilbert_restrictions[0], self._N_model - self._hilbert_restrictions[0]
#             else:
#                 num_up_alpha, num_down_alpha, num_up_beta, num_down_beta = [torch.zeros(num_samples) for _ in range(4)]
#                 max_up_alpha, max_down_alpha = self._hilbert_restrictions[0], math.floor(self._N_model / 2) - self._hilbert_restrictions[0]
#                 max_up_beta, max_down_beta = self._hilbert_restrictions[1], math.ceil(self._N_model / 2) - self._hilbert_restrictions[1]
#                 is_beta = (self.qubit2model_permutation % 2)
#
#         if ret_probs:
#             probs = torch.ones(num_samples)
#
#         last_vals = torch.zeros(num_samples)
#         for i in range(self.hilbert.N):
#             q_probs = self.model(last_vals)[..., 0].exp().pow(2)
#
#             if self.use_hilbert_restrictions:
#                 if len(self._hilbert_restrictions) == 1:
#                     up_mask, down_mask = (num_down >= max_down), (num_up >= max_up)
#                 elif is_beta[i]:
#                     up_mask, down_mask = (num_down_beta >= max_down_beta), (num_up_beta >= max_up_beta)
#                 else:
#                     up_mask, down_mask = (num_down_alpha >= max_down_alpha), (num_up_alpha >= max_up_alpha)
#                 q_probs[up_mask] = q_probs_up
#                 q_probs[down_mask] = q_probs_down
#
#             q_idx = torch.multinomial(q_probs, 1).squeeze()
#             q_vals = self._idx2qubit(q_idx)
#             samples[..., i] = q_vals
#             last_vals = q_vals
#
#             if self.use_hilbert_restrictions:
#                 if len(self._hilbert_restrictions) == 1:
#                     num_up += (q_vals > 0)
#                     num_down += (q_vals <= 0)
#                 elif is_beta[i]:
#                     num_up_beta += (q_vals > 0)
#                     num_down_beta += (q_vals <= 0)
#                 else:
#                     num_up_alpha += (q_vals > 0)
#                     num_down_alpha += (q_vals <= 0)
#
#             if ret_probs:
#                 probs *= q_probs[torch.arange(len(q_probs)), q_idx]
#
#         if self.permute_qubits:
#             samples = samples[..., self.model2qubit_permutation]
#
#         self.model.predict()
#         if model_was_training:
#             self.model.train()
#
#         # (batch, channel, qubit) --> (batch, qubit)
#         samples.squeeze_(1)
#
#         if ret_probs:
#             return samples, probs
#         else:
#             return samples
#
#
# class NAQSComplex_NADE(_NAQSComplex_Base):
#
#     _temp = 0.1
#
#     def __init__(self,
#                  hilbert,
#                  N_up=None,
#                  N_alpha=None,
#                  N_beta=None,
#
#                  qubit_ordering=-1,  # 1: default, 0: random, -1:reverse, list:custom
#
#                  num_qubit_vals=2,
#
#                  mlp_hidden_size=[],
#                  mlp_hidden_activation=nn.ReLU,
#                  mlp_bias=True,
#
#                  post_mlp_batch_norm=False,
#                  batch_norm_momentum=1,
#
#                  affine_readout=False,
#                  amp_activation=SoftmaxLogProbAmps,
#                  phase_activation=None
#                  ):
#
#         super().__init__(
#             hilbert=hilbert,
#             qubit_ordering=qubit_ordering,
#
#             model=ComplexAutoregressiveMachine1D_NADE,
#
#             num_qubits = hilbert.N - hilbert.N_occ,
#             num_qubit_vals=num_qubit_vals,
#
#             mlp_hidden_size=mlp_hidden_size,
#             mlp_hidden_activation=mlp_hidden_activation,
#             mlp_bias=mlp_bias,
#
#             post_mlp_batch_norm=post_mlp_batch_norm,
#             batch_norm_momentum=batch_norm_momentum,
#
#             affine_readout=affine_readout,
#             amp_activation=amp_activation,
#             phase_activation=phase_activation
#         )
#
#         if ((N_up is None) and (N_alpha is None) and (N_beta is None)):
#             self.use_hilbert_restrictions, self._hilbert_restrictions = False, None
#         else:
#             restricted_subspace = self.hilbert.get_subspace(N_up, N_alpha, N_beta)
#             if len(restricted_subspace) == 0:
#                 raise ValueError("Invalid restrictions on subspace.")
#             self.use_hilbert_restrictions = True
#             if (N_alpha is None) and (N_beta is None):
#                 self._hilbert_restrictions = [N_up]
#             else:
#                 self._hilbert_restrictions = [N_alpha, N_beta]
#
#     def _evaluate_model(self, s):
#         '''
#         Evaluates the model to give conditional distributions.
#
#         The model output is expected to be of the form: [batch, qubit, qubit_value, cond_prob].
#
#         Note: This is the function we will overwrite to implement more complex symettries and
#         restrictions on the permitted model.
#         '''
#         if not self.use_hilbert_restrictions:
#           return super()._evaluate_model(s)
#
#         else:
#             model_was_sampling = self.model.sampling
#             model_was_combined_amp_phase = self.model.combined_amp_phase
#
#             # Shift all inputs by one, to ensure AR property.
#             s_in = s.clone()
#             s = self._state2model(s)
#             s = s.roll(1, -1)
#             s[..., 0] *= 0
#
#             self.model.sample() # <-- Important to set this after _state2model!
#             self.model.combine_amp_phase(False)
#
#             model_outputs = []
#
#             # Calculate the value we will default conditional amplitudes/probabilites to
#             # if we need to deterministically set a qubit.
#             if self.model.amplitude_encoding is (AmplitudeEncoding.AMP or AmplitudeEncoding.PROB):
#                 def set_qubits(cond_amps, up_mask, down_mask):
#                     m = torch.zeros_like(cond_amps, requires_grad=False)
#                     m[down_mask, 0] = 1 - cond_amps.detach()[down_mask, 0] # Set amp/prob psi_i(0) = 1
#                     m[down_mask, 1] = 0 - cond_amps.detach()[down_mask, 1] # Set amp/prob psi_i(1) = 0
#                     m[up_mask, 0] = 0 - cond_amps.detach()[up_mask, 0]  # Set amp/prob psi_i(0) = 0
#                     m[up_mask, 1] = 1 - cond_amps.detach()[up_mask, 1]  # Set amp/prob psi_i(1) = 1
#                     return cond_amps * m
#
#             else: # --> AmplitudeEncoding.LOG_AMP or AmplitudeEncoding.LOG_PROB
#                 if self.model.amplitude_encoding is AmplitudeEncoding.LOG_PROB:
#                     log0 = torch.FloatTensor([1e-10]).log()
#                     log1 = 0.5 * (1 - log0.exp()).log()
#                 else: # --> AmplitudeEncoding.LOG_AMP
#                     log0 = self._temp * torch.FloatTensor([1e-3]).log()
#                     log1 = 0.5*(1-(2*log0).exp()).log()
#                 def set_qubits(cond_amps, up_mask, down_mask):
#                     m = torch.zeros_like(cond_amps, requires_grad=False).to(cond_amps)
#                     m[down_mask, 0] = log1 - cond_amps.detach()[down_mask, 0] # Set amp/prob psi_i(0) = 1
#                     m[down_mask, 1] = log0 - cond_amps.detach()[down_mask, 1] # Set amp/prob psi_i(1) = 0
#                     m[up_mask, 0] = log0 - cond_amps.detach()[up_mask, 0] # Set amp/prob psi_i(0) = 1
#                     m[up_mask, 1] = log1 - cond_amps.detach()[up_mask, 1] # Set amp/prob psi_i(1) = 1
#                     return cond_amps + m
#
#
#             # We only have a single restriction (i.e. total magnetisation is fixed).
#             if len(self._hilbert_restrictions)==1:
#                 num_up, num_down = torch.zeros(s.shape[0], requires_grad=False), torch.zeros(s.shape[0], requires_grad=False)
#                 max_up, max_down = self._hilbert_restrictions[0], self._N_model - self._hilbert_restrictions[0]
#
#                 # s = self._state2model(s)
#                 for i, s_i in enumerate(s.T):
#                     # print("i")
#                     if i > 0:
#                         num_up += (s_i > 0)
#                         num_down += (s_i <= 0)
#                     up_mask, down_mask = (num_down >= max_down), (num_up >= max_up)
#                     cond_amps_i, cond_phase_i = self.model(s_i)
#
#                     # We only have to start checking validity once the smallest number
#                     # of electrons set to 'up' have possibly been sampled.
#                     if i >= min(max_up, max_down):
#                         cond_amps_i = set_qubits(cond_amps_i, up_mask, down_mask)
#                     model_out_i = torch.stack([cond_amps_i, cond_phase_i], -1)
#                     model_outputs.append(model_out_i)
#
#             # We have two restrictions (i.e. total magnetisation in alpha/beta shells.).
#             else:
#                 num_up_alpha, num_down_alpha, num_up_beta, num_down_beta = [torch.zeros(s.shape[0]) for _ in range(4)]
#                 max_up_alpha, max_down_alpha = self._hilbert_restrictions[0], math.floor(self._N_model/2) - self._hilbert_restrictions[0]
#                 max_up_beta, max_down_beta = self._hilbert_restrictions[1], math.ceil(self._N_model/2) - self._hilbert_restrictions[1]
#
#                 is_beta = (self.qubit2model_permutation % 2)
#
#                 for i, s_i in enumerate(s.T):
#                     if i > 0:
#                         if is_beta[i-1]:
#                             num_up_beta += (s_i > 0)
#                             num_down_beta += (s_i <= 0)
#                         else:
#                             num_up_alpha += (s_i > 0)
#                             num_down_alpha += (s_i <= 0)
#
#                     if is_beta[i]:
#                         up_mask, down_mask = (num_down_beta >= max_down_beta), (num_up_beta >= max_up_beta)
#                     else:
#                         up_mask, down_mask = (num_down_alpha >= max_down_alpha), (num_up_alpha >= max_up_alpha)
#
#                     cond_amps_i, cond_phase_i = self.model(s_i)
#                     # We only have to start checking validity once the smallest number
#                     # of electrons set to 'up' have possibly been sampled.
#                     # if i >= min(max_up, max_down):
#                     cond_amps_i = set_qubits(cond_amps_i, up_mask, down_mask)
#                     model_out_i = torch.stack([cond_amps_i, cond_phase_i], -1)
#                     model_outputs.append(model_out_i)
#
#             model_output = torch.stack(model_outputs, 1)
#
#             if not model_was_sampling:
#                 self.model.predict()
#             if model_was_combined_amp_phase:
#                 self.model.combine_amp_phase(True)
#
#             return self._model2state(model_output, s_in)
#
#     @torch.no_grad()
#     def sample(self, num_samples=1, ret_probs=False):
#         '''Sample from the probability distribution given by the wavefunction.
#         '''
#         self.model.sample()
#         model_was_training = self.model.training
#         self.model.eval()
#
#         samples = torch.zeros((num_samples, self.hilbert.N))
#
#         if self.use_hilbert_restrictions:
#             q_probs_up = torch.FloatTensor([0, 1], device=self.model.out_device)
#             q_probs_down = torch.FloatTensor([1, 0], device=self.model.out_device)
#             if len(self._hilbert_restrictions)==1:
#                 num_up, num_down = torch.zeros(num_samples), torch.zeros(num_samples)
#                 max_up, max_down = self._hilbert_restrictions[0], self._N_model - self._hilbert_restrictions[0]
#             else:
#                 num_up_alpha, num_down_alpha, num_up_beta, num_down_beta = [torch.zeros(num_samples) for _ in range(4)]
#                 max_up_alpha, max_down_alpha = self._hilbert_restrictions[0], math.floor(self._N_model / 2) - self._hilbert_restrictions[0]
#                 max_up_beta, max_down_beta = self._hilbert_restrictions[1], math.ceil(self._N_model / 2) - self._hilbert_restrictions[1]
#                 is_beta = (self.qubit2model_permutation % 2)
#
#         if ret_probs:
#             probs = torch.ones(num_samples)
#
#         last_vals = torch.zeros(num_samples)
#         for i in range(self.hilbert.N):
#             q_probs = self.model(last_vals)[..., 0].exp().pow(2)
#
#             if self.use_hilbert_restrictions:
#                 if len(self._hilbert_restrictions) == 1:
#                     up_mask, down_mask = (num_down >= max_down), (num_up >= max_up)
#                 elif is_beta[i]:
#                     up_mask, down_mask = (num_down_beta >= max_down_beta), (num_up_beta >= max_up_beta)
#                 else:
#                     up_mask, down_mask = (num_down_alpha >= max_down_alpha), (num_up_alpha >= max_up_alpha)
#                 q_probs[up_mask] = q_probs_up
#                 q_probs[down_mask] = q_probs_down
#
#             q_idx = torch.multinomial(q_probs, 1).squeeze()
#             q_vals = self._idx2qubit(q_idx)
#             samples[..., i] = q_vals
#             last_vals = q_vals
#
#             if self.use_hilbert_restrictions:
#                 if len(self._hilbert_restrictions) == 1:
#                     num_up += (q_vals > 0)
#                     num_down += (q_vals <= 0)
#                 elif is_beta[i]:
#                     num_up_beta += (q_vals > 0)
#                     num_down_beta += (q_vals <= 0)
#                 else:
#                     num_up_alpha += (q_vals > 0)
#                     num_down_alpha += (q_vals <= 0)
#
#             if ret_probs:
#                 probs *= q_probs[torch.arange(len(q_probs)), q_idx]
#
#         if self.permute_qubits:
#             samples = samples[..., self.model2qubit_permutation]
#
#         self.model.predict()
#         if model_was_training:
#             self.model.train()
#
#         # (batch, channel, qubit) --> (batch, qubit)
#         samples.squeeze_(1)
#
#         if ret_probs:
#             return samples, probs
#         else:
#             return samples
#
# class NAQSComplex_NADE_sym(_NAQSComplex_Base):
#
#     def __init__(self,
#                  hilbert,
#                  N_up=None,
#                  N_alpha=None,
#                  N_beta=None,
#
#                  qubit_ordering=-1,  # 1: default, 0: random, -1:reverse, list:custom
#
#                  num_qubit_vals=2,
#
#                  mlp_hidden_size=[],
#                  readout_hidden_size=[],
#                  mlp_hidden_activation=nn.ReLU,
#                  mlp_bias=True,
#
#                  batch_norm=False,
#                  batch_norm_momentum=1,
#
#                  amp_activation=SoftmaxLogProbAmps,
#                  phase_activation=None
#                  ):
#
#         super().__init__(
#             hilbert=hilbert,
#             qubit_ordering=qubit_ordering,
#
#             model=ComplexAutoregressiveMachine1D_NADE_sym,
#
#             num_qubits = hilbert.N - hilbert.N_occ,
#             num_qubit_vals=num_qubit_vals,
#
#             mlp_hidden_size=mlp_hidden_size,
#             readout_hidden_size=readout_hidden_size,
#             mlp_hidden_activation=mlp_hidden_activation,
#             mlp_bias=mlp_bias,
#
#             batch_norm=batch_norm,
#             batch_norm_momentum=batch_norm_momentum,
#
#             amp_activation=amp_activation,
#             phase_activation=phase_activation
#         )
#
#         if ((N_up is None) and (N_alpha is None) and (N_beta is None)):
#             self.use_hilbert_restrictions, self._hilbert_restrictions = False, None
#         else:
#             restricted_subspace = self.hilbert.get_subspace(N_up, N_alpha, N_beta)
#             if len(restricted_subspace) == 0:
#                 raise ValueError("Invalid restrictions on subspace.")
#             self.use_hilbert_restrictions = True
#             if (N_alpha is None) and (N_beta is None):
#                 self._hilbert_restrictions = [N_up]
#             else:
#                 self._hilbert_restrictions = [N_alpha, N_beta]
#
#     def _evaluate_model(self, s):
#         '''
#         Evaluates the model to give conditional distributions.
#
#         The model output is expected to be of the form: [batch, qubit, qubit_value, cond_prob].
#
#         Note: This is the function we will overwrite to implement more complex symettries and
#         restrictions on the permitted model.
#         '''
#         if not self.use_hilbert_restrictions:
#           return super()._evaluate_model(s)
#
#         else:
#             model_was_sampling = self.model.sampling
#             model_was_combined_amp_phase = self.model.combined_amp_phase
#
#             # Shift all inputs by one, to ensure AR property.
#             s_in = s.clone()
#             s = self._state2model(s)
#             s = s.roll(1, -1)
#             s[..., 0] *= 0
#
#             self.model.sample() # <-- Important to set this after _state2model!
#             self.model.combine_amp_phase(False)
#
#             model_outputs = []
#
#             # Calculate the value we will default conditional amplitudes/probabilites to
#             # if we need to deterministically set a qubit.
#             if self.model.amplitude_encoding is (AmplitudeEncoding.AMP or AmplitudeEncoding.PROB):
#                 def set_qubits(cond_amps, up_mask, down_mask):
#                     m = torch.zeros_like(cond_amps, requires_grad=False)
#                     m[down_mask, 0] = 1 - cond_amps.detach()[down_mask, 0] # Set amp/prob psi_i(0) = 1
#                     m[down_mask, 1] = 0 - cond_amps.detach()[down_mask, 1] # Set amp/prob psi_i(1) = 0
#                     m[up_mask, 0] = 0 - cond_amps.detach()[up_mask, 0]  # Set amp/prob psi_i(0) = 0
#                     m[up_mask, 1] = 1 - cond_amps.detach()[up_mask, 1]  # Set amp/prob psi_i(1) = 1
#                     return cond_amps * m
#
#             else: # --> AmplitudeEncoding.LOG_AMP or AmplitudeEncoding.LOG_PROB
#                 if self.model.amplitude_encoding is AmplitudeEncoding.LOG_PROB:
#                     log0 = torch.FloatTensor([1e-10]).log()
#                     log1 = 0.5 * (1 - log0.exp()).log()
#                 else: # --> AmplitudeEncoding.LOG_AMP
#                     log0 = self._temp * torch.FloatTensor([1e-3]).log()
#                     log1 = 0.5*(1-(2*log0).exp()).log()
#                 def set_qubits(cond_amps, up_mask, down_mask):
#                     m = torch.zeros_like(cond_amps, requires_grad=False).to(cond_amps)
#                     m[down_mask, 0] = log1 - cond_amps.detach()[down_mask, 0] # Set amp/prob psi_i(0) = 1
#                     m[down_mask, 1] = log0 - cond_amps.detach()[down_mask, 1] # Set amp/prob psi_i(1) = 0
#                     m[up_mask, 0] = log0 - cond_amps.detach()[up_mask, 0] # Set amp/prob psi_i(0) = 1
#                     m[up_mask, 1] = log1 - cond_amps.detach()[up_mask, 1] # Set amp/prob psi_i(1) = 1
#                     return cond_amps + m
#
#
#             # We only have a single restriction (i.e. total magnetisation is fixed).
#             if len(self._hilbert_restrictions)==1:
#                 num_up, num_down = torch.zeros(s.shape[0], requires_grad=False), torch.zeros(s.shape[0], requires_grad=False)
#                 max_up, max_down = self._hilbert_restrictions[0], self._N_model - self._hilbert_restrictions[0]
#
#                 # s = self._state2model(s)
#                 for i, s_i in enumerate(s.T):
#                     # print("i")
#                     if i > 0:
#                         num_up += (s_i > 0)
#                         num_down += (s_i <= 0)
#                     up_mask, down_mask = (num_down >= max_down), (num_up >= max_up)
#                     cond_amps_i, cond_phase_i = self.model(s_i)
#
#                     # We only have to start checking validity once the smallest number
#                     # of electrons set to 'up' have possibly been sampled.
#                     if i >= min(max_up, max_down):
#                         cond_amps_i = set_qubits(cond_amps_i, up_mask, down_mask)
#                     model_out_i = torch.stack([cond_amps_i, cond_phase_i], -1)
#                     model_outputs.append(model_out_i)
#
#             # We have two restrictions (i.e. total magnetisation in alpha/beta shells.).
#             else:
#                 num_up_alpha, num_down_alpha, num_up_beta, num_down_beta = [torch.zeros(s.shape[0]) for _ in range(4)]
#                 max_up_alpha, max_down_alpha = self._hilbert_restrictions[0], math.floor(self._N_model/2) - self._hilbert_restrictions[0]
#                 max_up_beta, max_down_beta = self._hilbert_restrictions[1], math.ceil(self._N_model/2) - self._hilbert_restrictions[1]
#
#                 is_beta = (self.qubit2model_permutation % 2)
#
#                 for i, s_i in enumerate(s.T):
#                     if i > 0:
#                         if is_beta[i-1]:
#                             num_up_beta += (s_i > 0)
#                             num_down_beta += (s_i <= 0)
#                         else:
#                             num_up_alpha += (s_i > 0)
#                             num_down_alpha += (s_i <= 0)
#
#                     if is_beta[i]:
#                         up_mask, down_mask = (num_down_beta >= max_down_beta), (num_up_beta >= max_up_beta)
#                     else:
#                         up_mask, down_mask = (num_down_alpha >= max_down_alpha), (num_up_alpha >= max_up_alpha)
#
#                     cond_amps_i, cond_phase_i = self.model(s_i)
#                     # We only have to start checking validity once the smallest number
#                     # of electrons set to 'up' have possibly been sampled.
#                     # if i >= min(max_up, max_down):
#                     cond_amps_i = set_qubits(cond_amps_i, up_mask, down_mask)
#                     model_out_i = torch.stack([cond_amps_i, cond_phase_i], -1)
#                     model_outputs.append(model_out_i)
#
#             model_output = torch.stack(model_outputs, 1)
#
#             if not model_was_sampling:
#                 self.model.predict()
#             if model_was_combined_amp_phase:
#                 self.model.combine_amp_phase(True)
#
#             return self._model2state(model_output, s_in)
#
#     @torch.no_grad()
#     def sample(self, num_samples=1, ret_probs=False):
#         '''Sample from the probability distribution given by the wavefunction.
#         '''
#         self.model.sample()
#         model_was_training = self.model.training
#         self.model.eval()
#
#         samples = torch.zeros((num_samples, self.hilbert.N))
#
#         if self.use_hilbert_restrictions:
#             q_probs_up = torch.FloatTensor([0, 1], device=self.model.out_device)
#             q_probs_down = torch.FloatTensor([1, 0], device=self.model.out_device)
#             if len(self._hilbert_restrictions)==1:
#                 num_up, num_down = torch.zeros(num_samples), torch.zeros(num_samples)
#                 max_up, max_down = self._hilbert_restrictions[0], self._N_model - self._hilbert_restrictions[0]
#             else:
#                 num_up_alpha, num_down_alpha, num_up_beta, num_down_beta = [torch.zeros(num_samples) for _ in range(4)]
#                 max_up_alpha, max_down_alpha = self._hilbert_restrictions[0], math.floor(self._N_model / 2) - self._hilbert_restrictions[0]
#                 max_up_beta, max_down_beta = self._hilbert_restrictions[1], math.ceil(self._N_model / 2) - self._hilbert_restrictions[1]
#                 is_beta = (self.qubit2model_permutation % 2)
#
#         if ret_probs:
#             probs = torch.ones(num_samples)
#
#         last_vals = torch.zeros(num_samples)
#         for i in range(self.hilbert.N):
#             q_probs = self.model(last_vals)[..., 0].exp().pow(2)
#
#             if self.use_hilbert_restrictions:
#                 if len(self._hilbert_restrictions) == 1:
#                     up_mask, down_mask = (num_down >= max_down), (num_up >= max_up)
#                 elif is_beta[i]:
#                     up_mask, down_mask = (num_down_beta >= max_down_beta), (num_up_beta >= max_up_beta)
#                 else:
#                     up_mask, down_mask = (num_down_alpha >= max_down_alpha), (num_up_alpha >= max_up_alpha)
#                 q_probs[up_mask] = q_probs_up
#                 q_probs[down_mask] = q_probs_down
#
#             q_idx = torch.multinomial(q_probs, 1).squeeze()
#             q_vals = self._idx2qubit(q_idx)
#             samples[..., i] = q_vals
#             last_vals = q_vals
#
#             if self.use_hilbert_restrictions:
#                 if len(self._hilbert_restrictions) == 1:
#                     num_up += (q_vals > 0)
#                     num_down += (q_vals <= 0)
#                 elif is_beta[i]:
#                     num_up_beta += (q_vals > 0)
#                     num_down_beta += (q_vals <= 0)
#                 else:
#                     num_up_alpha += (q_vals > 0)
#                     num_down_alpha += (q_vals <= 0)
#
#             if ret_probs:
#                 probs *= q_probs[torch.arange(len(q_probs)), q_idx]
#
#         if self.permute_qubits:
#             samples = samples[..., self.model2qubit_permutation]
#
#         self.model.predict()
#         if model_was_training:
#             self.model.train()
#
#         # (batch, channel, qubit) --> (batch, qubit)
#         samples.squeeze_(1)
#
#         if ret_probs:
#             return samples, probs
#         else:
#             return samples
#
# class NAQSComplex_NADE_shell(_NAQSComplex_Base):
#
#     def __init__(self,
#                  hilbert,
#                  N_up=None,
#                  N_alpha=None,
#                  N_beta=None,
#
#                  qubit_ordering=-1,  # 1: default, 0: random, -1:reverse, list:custom
#
#                  num_qubit_vals=2,
#
#                  num_lut=0,
#
#                  n_alpha_electrons=None,
#                  n_beta_electrons=None,
#
#                  amp_hidden_size=[],
#                  amp_hidden_activation=nn.ReLU,
#                  amp_bias=True,
#
#                  phase_hidden_size=[],
#                  phase_hidden_activation=nn.ReLU,
#                  phase_bias=True,
#
#                  use_amp_exchange_sym=True,
#                  aggregate_phase=True,
#
#                  amp_batch_norm=False,
#                  phase_batch_norm=False,
#                  batch_norm_momentum=1,
#
#                  use_unique_network_passes=True,
#                  cache_unique_network_passes=True,
#
#                  amp_activation=SoftmaxLogProbAmps,
#                  phase_activation=None,
#                  ):
#
#         super().__init__(
#             hilbert=hilbert,
#             qubit_ordering=qubit_ordering,
#
#             model=ComplexAutoregressiveMachine1D_NADE_shell,
#
#             num_qubits = hilbert.N - hilbert.N_occ,
#             num_qubit_vals=num_qubit_vals,
#
#             num_lut = num_lut,
#
#             n_alpha_electrons=n_alpha_electrons,
#             n_beta_electrons=n_beta_electrons,
#
#             amp_hidden_size=amp_hidden_size,
#             amp_hidden_activation=amp_hidden_activation,
#             amp_bias=amp_bias,
#
#             phase_hidden_size=phase_hidden_size,
#             phase_hidden_activation=phase_hidden_activation,
#             phase_bias=phase_bias,
#
#             amp_activation=amp_activation,
#             phase_activation=phase_activation,
#
#             use_amp_exchange_sym=use_amp_exchange_sym,
#             aggregate_phase=aggregate_phase,
#
#             amp_batch_norm=amp_batch_norm,
#             phase_batch_norm=phase_batch_norm,
#             batch_norm_momentum=batch_norm_momentum,
#
#             use_unique_network_passes=use_unique_network_passes,
#             cache_unique_network_passes=cache_unique_network_passes
#         )
#         if qubit_ordering==1:
#             self.state2model_permutation_shell = torch.arange(self._N_model//2)
#         elif qubit_ordering==-1:
#             self.qubit2model_permutation = torch.stack([torch.arange(self._N_model-2,-1,-2), torch.arange(self._N_model-1,-1,-2)],1).reshape(-1)
#             print("self.qubit2model_permutation (overwritten) :", self.qubit2model_permutation)
#             self.state2model_permutation_shell = torch.arange(self._N_model//2-1, -1, -1)
#         else:
#             self.state2model_permutation_shell = self.qubit2model_permutation[1::2]//2
#         self.model2state_permutation_shell = np.argsort(self.state2model_permutation_shell)
#         print("self.state2model_permutation_shell :", self.state2model_permutation_shell)
#         print("self.model2state_permutation_shell :", self.model2state_permutation_shell)
#
#         if ((N_up is None) and (N_alpha is None) and (N_beta is None)):
#             self.use_hilbert_restrictions, self._hilbert_restrictions = False, None
#         else:
#             restricted_subspace = self.hilbert.get_subspace(N_up, N_alpha, N_beta)
#             if len(restricted_subspace) == 0:
#                 raise ValueError("Invalid restrictions on subspace.")
#             self.use_hilbert_restrictions = True
#             if (N_alpha is None) and (N_beta is None):
#                 self._hilbert_restrictions = [N_up]
#             else:
#                 self._hilbert_restrictions = [N_alpha, N_beta]
#
#     def _evaluate_log_psi(self, s, gather_state=True):
#         '''
#         Returns model output in the form:
#             gather_state == False : [batch, qubit, qubit_value, cond_prob]
#             gather_state == True  : [batch, qubit, cond_prob]
#         '''
#         model_output = self._evaluate_model(s)
#
#         if gather_state:
#             model_output = model_output.gather(-2,
#                                                self.state2shell(s).view(model_output.shape[0], -1, 1, 1).repeat(1, 1, 1, 2))
#             # print(f"\nGathered output\n {model_output}")
#
#         if self.model.amplitude_encoding is AmplitudeEncoding.AMP:
#             # [batch, ..., [amp, phase]] --> [batch, ..., [log_amp, phase]]
#             amp, phase = torch.split(model_output,1,-1)
#             model_output = torch.stack([amp.log(), phase], -1)
#         return model_output
#
#     def parameters(self, group_idx=None):
#         if group_idx is None or group_idx==-1:
#             return self.model.parameters()
#
#         elif group_idx==0:
#             # Return non-lut params
#             params = [p for p in self.model.amp_layers[self.model.num_lut:].parameters()]
#             params += [p for p in self.model.phase_layers[self.model.num_lut:].parameters()]
#             if self.model.amplitude_activation is not None:
#                 params += [p for p in self.model.amplitude_activation.parameters()]
#             if self.model.phase_activation is not None:
#                 params += [p for p in self.model.phase_activation.parameters()]
#             return params
#
#         # elif group_idx==1:
#         #     # Return non-lut params
#         #     params = [p for p in self.model.amp_layers[self.model.num_lut:].parameters()]
#         #     params = [p for p in self.model.phase_layers[self.model.num_lut:].parameters()]
#         #     if self.model.amplitude_activation is not None:
#         #         params += [p for p in self.model.amplitude_activation.parameters()]
#         #     if self.model.phase_activation is not None:
#         #         params += [p for p in self.model.phase_activation.parameters()]
#         #     return params
#
#         elif group_idx==1:
#             # Return lut params
#             params = [p for p in self.model.amp_layers[:self.model.num_lut].parameters()]
#             params += [p for p in self.model.phase_layers[:self.model.num_lut].parameters()]
#             return params
#
#         else:
#             raise NotImplementedError()
#
#     # def parameters(self, group_idx=None):
#     #     if group_idx is None or group_idx==-1:
#     #         return self.model.parameters()
#     #     else:
#     #         params = [p for p in self.model.amp_layers[group_idx].parameters()]
#     #         params += [p for p in self.model.phase_layers[group_idx].parameters()]
#     #         # if self.model.amplitude_activation is not None:
#     #         #     params += [p for p in self.model.amplitude_activation.parameters()]
#     #         # if self.model.phase_activation is not None:
#     #         #     params += [p for p in self.model.phase_activation.parameters()]
#     #         return params
#
#     def state2shell(self, s):
#         shp = s.shape
#         return (s.view(shp[0], shp[-1]//2, 2).clamp_min(0) * torch.LongTensor([1, 2])).sum(-1)
#
#     def _evaluate_model(self, s):
#         '''
#         Evaluates the model to give conditional distributions.
#
#         The model output is expected to be of the form: [batch, qubit, qubit_value, cond_prob].
#
#         Note: This is the function we will overwrite to implement more complex symettries and
#         restrictions on the permitted model.
#         '''
#         self.model.predict()
#         model_output = self.model(self._state2model(s))
#         return model_output[:, self.model2state_permutation_shell, ...]
#
#     def truncated_log_psi(self, k_trunc):
#         states, log_psi = self.model.top_k(k_trunc)
#         states = states[:, self.model2qubit_permutation]
#         return states, log_psi
#
#     def sample(self, num_samples=1, ret_probs=True, ret_log_psi=True, ret_norm_reg=False, eval_mode=False):
#         '''Sample from the probability distribution given by the wavefunction.
#         '''
#         self.model.sample()
#         if eval_mode:
#             model_was_training = self.model.training
#             self.model.eval()
#         else:
#             model_was_eval = not self.model.training
#             self.model.train()
#
#         if not ret_norm_reg:
#             states, counts, probs, log_psi = self.model(num_samples)
#         else:
#             states, counts, probs, log_psi, norm_reg = self.model(num_samples, ret_norm_reg=True)
#         states = self.hilbert.to_state_tensor( states[:, self.model2qubit_permutation] )
#
#         output = [states, counts]
#         if ret_probs:
#             output.append(probs)
#         if ret_log_psi:
#             output.append(log_psi)
#         if ret_norm_reg:
#             output.append(norm_reg)
#
#         if eval_mode:
#             if model_was_training:
#                 self.model.train()
#         else:
#             if model_was_eval:
#                 self.model.eval()
#
#         return output
#
