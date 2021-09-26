import os
import math
import numbers
import time
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
import pandas as pd

import torch
from torch.nn import functional as F

from abc import ABC, abstractmethod
from bisect import bisect
from collections import Counter
from collections import deque

try:
    import torchsso
    TORCH_SSO_FOUND = True
except:
    TORCH_SSO_FOUND = False

from src.naqs.network.nade import MaxBatchSizeExceededError
import src.naqs.network.torch_utils as torch_utils
import src.utils.complex as cplx
from src.utils.sparse_math import sparse_dense_mv
from src.utils.system import mk_dir
from src.optimizer.hamiltonian import PauliHamiltonian
from src.optimizer.utils import LogKey, KFACOptimizer

def to_sparse_vector(data, idxs, size=None):
    if size is None:
        size = len(idxs)
    vec = csr_matrix((data, (idxs, [0]*len(idxs))), shape=(size,1))
    return vec

def median_(val, freq):
    ord = np.argsort(val)
    cdf = np.cumsum(freq[ord])
    return val[ord][np.searchsorted(cdf, cdf[-1] // 2)]

class OptimizerBase(ABC):
    '''
    Base class for optimizing the energy a wavefunction anzatz.
    '''

    def __init__(self,
                 wavefunction,
                 qubit_hamiltonian,
                 pre_compute_H=True,

                 n_electrons=None,
                 n_alpha_electrons=None,
                 n_beta_electrons=None,
                 n_fixed_electrons=None,
                 n_excitations_max=None,

                 reweight_samples_by_psi=False,
                 normalise_psi=False,

                 normalize_grads=False,
                 grad_clip_factor=3,
                 grad_clip_memory_length=50,
                 optimizer=torch.optim.Adam,
                 optimizer_args={'lr': 1e-3},
                 scheduler=None,
                 scheduler_args=None,

                 save_loc='./',

                 pauli_hamiltonian_fname=None,
                 overwrite_pauli_hamiltonian=False,
                 pauli_hamiltonian_dtype=np.float32,

                 verbose=False):

        self.wavefunction = wavefunction
        self.hilbert = self.wavefunction.hilbert
        self.qubit_hamiltonian = qubit_hamiltonian

        self.device = "cpu"

        self.reweight_samples_by_psi = reweight_samples_by_psi

        self.normalise_psi = normalise_psi
        self.n_electrons = n_electrons
        self.n_alpha_electrons = n_alpha_electrons
        self.n_beta_electrons = n_beta_electrons
        self.n_fixed_electrons = n_fixed_electrons
        self.n_excitations_max = n_excitations_max

        self.subspace_args = {"N_up": self.n_electrons,
                              "N_alpha": self.n_alpha_electrons,
                              "N_beta": self.n_beta_electrons,
                              "N_occ": self.n_fixed_electrons,
                              "N_exc_max": self.n_excitations_max}

        self.optimizer_callable = optimizer
        self.optimizer_args = optimizer_args

        self.scheduler_callable = scheduler
        self.scheduler_args = scheduler_args

        self.grad_clip_factor = grad_clip_factor
        self.grad_clip_memory_length = grad_clip_memory_length

        self.normalize_grads = normalize_grads
        self.save_loc = save_loc
        self.verbose = verbose

        self.pauli_hamiltonian_fname = pauli_hamiltonian_fname
        self.overwrite_pauli_hamiltonian = overwrite_pauli_hamiltonian

        restricted_idxs = self.hilbert.get_subspace(ret_states=False,
                                                    ret_idxs=True,
                                                    **self.subspace_args)

        self.pauli_hamiltonian = PauliHamiltonian.get(self.hilbert,
                                                      qubit_hamiltonian,
                                                      hamiltonian_fname=self.pauli_hamiltonian_fname,
                                                      restricted_idxs=restricted_idxs,
                                                      verbose=self.verbose,
                                                      n_excitations_max=self.n_excitations_max,
                                                      dtype=pauli_hamiltonian_dtype)


        if pre_compute_H and not self.pauli_hamiltonian.is_frozen():
            print("Pre-computing Hamiltonian.")
            # for idxs in torch.split(restricted_idxs, 1000000):
            #     self.pauli_hamiltonian.update_H(idxs, check_unseen=False, assume_unique=True)
            self.pauli_hamiltonian.update_H(restricted_idxs, check_unseen=False, assume_unique=True)
            self.pauli_hamiltonian.freeze_H()
            if self.overwrite_pauli_hamiltonian:
                self.pauli_hamiltonian.save_H(self.pauli_hamiltonian_fname)

        self.sampled_idxs = Counter()
        self.reset_log()
        self.reset_optimizer()

    def reset_log(self):
        '''Reset the logging tools.

        Resets the log to a dictionary of empty lists.  Resets the total number of steps and run time.
        '''
        self.log = {LogKey.E: [], LogKey.E_LOC: [], LogKey.E_LOC_VAR: [], LogKey.N_UNIQUE_SAMP: [], LogKey.TIME: []}
        self.last_samples = []
        self.n_steps = 0
        self.n_epochs = 0
        self.run_time = 0

    def reset_optimizer(self, cond_idx=None):
        '''Reset the optimization tools.

        Reset the optimizer and the tracked list of gradient norms used for gradient clipping.
        '''
        print("Resetting optimizer", end="...")
        if TORCH_SSO_FOUND:
            opt_list = [KFACOptimizer, torchsso.optim.SecondOrderOptimizer]
        else:
            opt_list = [KFACOptimizer]

        if self.optimizer_callable in opt_list:
            self.optimizer = self.optimizer_callable(self.wavefunction.model, **self.optimizer_args)

        else:
            if type(self.optimizer_args) is not dict:
                args = []
                for idx, args_i in enumerate(self.optimizer_args):
                    args_i['params'] = self.wavefunction.parameters(idx)
                    args.append(args_i)
                self.optimizer = self.optimizer_callable(args)
            else:
                # self.optimizer = self.optimizer_callable(self.wavefunction.parameters(), **self.optimizer_args)
                print(f"subnetwork {cond_idx}", end="...")
                self.optimizer = self.optimizer_callable(self.wavefunction.conditional_parameters(cond_idx), **self.optimizer_args)

        print("done.")

        if self.scheduler_callable is not None:
            print("Resetting scheduler", end="...")
            self.scheduler = self.scheduler_callable(self.optimizer, **self.scheduler_args)
            print("done.")
        else:
            self.scheduler = None

        self.__grad_norms = [deque([], self.grad_clip_memory_length) for _ in range(len(self.optimizer.param_groups))]

    @torch.no_grad()
    def calculate_energy(self, normalise_psi=None):
        '''Calculate the 'true' energy of the current wavefunction using the entire physically valid
        Hilbert space.

        Note that this might be very slow/intractable for large systems.

        normalise_psi : Whether the distribution over the physically valid Hilbert space provided
                        by the wavefunction should be renormalised (it may have total probabilty < 1).
        '''
        states, states_idx = self.hilbert.get_subspace(ret_states=True,
                                                       ret_idxs=True,
                                                       use_restricted_idxs=False,
                                                       **self.subspace_args)

        self.pauli_hamiltonian.update_H(states_idx, check_unseen=True, assume_unique=True)

        # Here, we are computing all physically valid couplings, so we might as well
        # freeze the Hamiltonian once its been updated, as no new couplings we add
        # will be relevant.
        self.pauli_hamiltonian.freeze_H()

        psi = self.wavefunction.psi(states, ret_complex=True)
        if normalise_psi:
            psi /= np.sum(np.abs(psi) ** 2) ** 0.5

        energy = psi.conj().dot( sparse_dense_mv(self.pauli_hamiltonian.get_restricted_H(), psi) )

        return energy.real

    @torch.no_grad()
    def calculate_local_energy(self, states_idx, psi=None, set_unsampled_states_to_zero=True, ret_complex=False):
        '''Calculate the local energy for each state.

        The local energy is given by:
                E_loc(|s>) = ( 1/|Psi(|s>) ) * sum_{s'} |Psi(|s'>) <s|H|s'>
        (or E_loc(|s>) = ( 1/|Psi^*(|s>) ) * sum_{s'} |Psi^*(|s'>) <s'|H|s> for conj.)

        We have a choice to make: namely if we pass a subset of states {|s>}, which couple
        to a larger subset of states {|s'>}, do we compute psi for all |s'> even if they are
        not in the origional sample, or do we treat un-sampled amplitudes as zero?

        states_idx : The states for which we want to calculate the local energies.
        psi : The complex amplitudes of states_idx (if None these will be computed on-demand).
        set_unsampled_states_to_zero : Whether to assume all states not in states_idx have zero
                                       amplitude.  If false, and un-sampled but still coupled
                                       states will be computed.
        ret_complex : Return as compelx numpy array (False) or complex torch array (True default).
        '''
        if psi is None:
            psi = self.wavefunction.psi(self.hilbert.idx2state(states_idx, use_restricted_idxs=False), ret_complex=True)
        else:
            psi = psi.detach()
            if cplx.is_complex(psi):
                psi = cplx.torch_to_numpy(psi)

        self.pauli_hamiltonian.update_H(states_idx, check_unseen=True, assume_unique=True)

        if set_unsampled_states_to_zero:
            local_energy = (sparse_dense_mv(self.pauli_hamiltonian.get_H(states_idx), psi) / psi).conj()

        else:
            raise NotImplementedError()
            # Note we are not using cython functions here as they are not optimised yer.
            # coupled_state_idxs = self.pauli_hamiltonian.get_coupled_state_idxs(states_idx, ret_unqiue=True)
            # coupled_state_idxs = np.sort(coupled_state_idxs)
            # coupled_psi = self.wavefunction.psi(self.hilbert.idx2state(coupled_state_idxs), ret_complex=True)
            #
            # coupled_psi_sparse = to_sparse_vector(coupled_psi, coupled_state_idxs, H.shape[0])
            # local_energy = np.squeeze(H.dot(coupled_psi_sparse)[states_idx].toarray()) / psi

        if not ret_complex:
            local_energy = cplx.np_to_torch(local_energy)

        return local_energy

    @abstractmethod
    def solve_H(self, states_idx=None):
        raise NotImplementedError()

    @abstractmethod
    def pre_train(self):
        raise NotImplementedError()

    def _SGD_step(self, states, states_idx,
                  log_psi=None, sample_weights=None, log_psi_eval=None, regularisation_loss=None,
                  n_samps=None, e_loc_clip_factor=None):
        '''Take a step of gradient descent for samples states.

           (0. Downstream sparse calculations are faster if states/states_idxs are sorted by ascending
               state_idx, so sort first if needed.)
            1. Compute log amplitudes of states (with network gradients): log_Psi(|s>).
            2. Compute the local energies for each state (no network gradients requried): E_loc(|s>).
            3. Compute the expectation << O >> = 2*<< log_Psi(|s>) E_loc(|s>) >>_{|s>~|Psi(|s>)|^2}.
            4. Backpropagate << O >> --> << (dlog_Psi(|s>) / dtheta) E_loc(|s>) >>, i.e. the variational gradients.
            5. Take step of gradient descent.
            6. From the local energies of each state, calculate the overall energy estimation and their varience.

        states : The sampled states.
        states_idx : The sampled state_idxs.
        log_psi : The log amplitudes of the states.
        '''
        # if not assume_sorted:
        #     sort_args = np.argsort(states_idx)
        #     states = states[sort_args]
        #     states_idx = states_idx[sort_args]
        #     log_psi = log_psi[sort_args]
        if self.verbose:
            print("Entering _SGD_step(...)")
            t = time.time()

        self.sampled_idxs.update(self.hilbert.to_idx_array(states_idx).squeeze())

        # 1. Compute log amplitudes of states (with network gradients): log_Psi(|s>).
        if log_psi is None:
            log_psi = self.wavefunction.log_psi(states)
            if self.verbose:
                print(f"log_psi : {time.time()-t:.4f}s")
                t = time.time()

        # 2.Compute the local energies for each state (no network gradients requried): E_loc(|s>).
        e_loc = self.calculate_local_energy(states_idx.squeeze(), psi=cplx.exp(log_psi.detach()))
        if self.verbose:
            print(f"e_loc : {time.time()-t:.4f}s")
            t = time.time()

        # 3. Compute the expectation << O >> = 2*<< log_Psi(|s>) E_loc(|s>) >>_{|s>~|Psi(|s>)|^2}.
        if sample_weights is None:
            if self.reweight_samples_by_psi:
                sample_weights = log_psi.detach()[..., 0].exp().pow(2)
            else:
                raise NotImplementedError("Re-weighting by the number of samples is not yet implemented.")

            if self.normalise_psi:
                sample_weights /= sample_weights.sum()

        if sample_weights.dim() < 2:
            sample_weights = sample_weights.unsqueeze(-1)

        e_loc_corr = e_loc - (sample_weights * e_loc).sum(axis=0).detach()
        exp_op = 2 * cplx.real(sample_weights * cplx.scalar_mult(log_psi, e_loc_corr)).sum(axis=0)
        # exp_op -= 2 * cplx.real(cplx.scalar_mult(
        #                         (sample_weights * e_loc_corr).sum(axis=0),
        #                         (sample_weights * log_psi).sum(axis=0)))

        if self.verbose:
            print(f"<<grad>> : {time.time()-t:.4f}s")
            t = time.time()

        # 4. Backpropagate << O >> --> << (dlog_Psi(|s>) / dtheta) E_loc(|s>) >>, i.e. the variational gradients.
        self.optimizer.zero_grad()
        if self.normalize_grads:
            exp_op = exp_op / (exp_op.detach()).abs()
        if regularisation_loss is not None:
            # print(exp_op)
            # print(regularisation_loss)
            exp_op = exp_op + regularisation_loss
        exp_op.backward()
        del exp_op # <-- served it's purpose, so free up the memory.
        self._clip_grads()

        if self.verbose:
            print(f"backprop: {time.time()-t:.4f}s")
            t = time.time()

        # 5. Take step of gradient descent.
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.verbose:
            print(f"step : {time.time()-t:.4f}s")
            t = time.time()

        if self.verbose:
            print(f"<<E>>, var(<<E>>) : {time.time() - t:.4f}s")

        with torch.no_grad():
            # 6. From the local energies of each state, calculate the overall energy estimation and their varience.
            if log_psi_eval is not None:
                e_loc = self.calculate_local_energy(states_idx.squeeze(), psi=cplx.exp(log_psi_eval.detach()))

            sample_weights /= sample_weights.sum()

            local_energy = cplx.real(sample_weights * e_loc).sum()
            local_energy_variance = ((cplx.real(e_loc) - local_energy).pow(2) * sample_weights.squeeze()).sum()

        return local_energy.item(), local_energy_variance.item()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    def _clip_grads(self):
        if self.grad_clip_factor is not None:
            for grad_norms, group in zip(self.__grad_norms, self.optimizer.param_groups):
                max_norm = self.grad_clip_factor * np.mean(grad_norms).item() if len(grad_norms) > 0 else 1e3
                # Will be fixed to work with grads on different devices in 1.5.1:
                #     norm = torch.nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type=2)
                # Until then, use my custom clipper.
                norm = torch_utils.clip_grad_norm_(group['params'], max_norm, norm_type=2)
                try:
                    norm = norm.item()
                except:
                    pass
                grad_norms.append(min(max_norm, norm))

    def __format_checkpoint_fname(self, fname):
        '''Formats a checkpoint file location as an absolute path with a '.pth' file extension.

        If a relative path is passed, this returns the absolute path relative to self.save_loc.
        '''
        if os.path.splitext(fname)[-1] != '.pth':
            fname += '.pth'
        if not os.path.isabs(fname):
            fname = os.path.join(self.save_loc, fname)
        return fname


    def save(self, fname="energy_optimizer", quiet=False):
        '''Save the current optimizer and all information required to load and restart optimisation.

        The energy optimisation needs to save the following attributes:
            - The network optimizer.
            - The log.
            - The number of steps taken / number of epochs / total running time.
            
        Additionally, the wavefunction itself must be saved, but this is handled internally in the
        object.
        '''
        fname = self.__format_checkpoint_fname(fname)
        if not quiet:
            print(f"Saving checkpoint {fname}.", end="...")

        dir = os.path.dirname(fname)
        if dir != '':
            mk_dir(dir, quiet)

        wavefunction_fname = os.path.splitext(fname)[0] + '_naqs'
        wavefunction_fname = self.wavefunction.save(wavefunction_fname, quiet)

        checkpoint = {
            'optimizer:state_dict': self.optimizer.state_dict(),
            'run_time': self.run_time,
            'n_steps': self.n_steps,
            'n_epochs': self.n_epochs,
            'log': self.log,
            'sampled_idxs': self.sampled_idxs,
            'wavefunction:fname':wavefunction_fname,
            'hamiltonian_fname':self.pauli_hamiltonian_fname
        }

        torch.save(checkpoint, fname)
        if not quiet:
            print("done.")

        if self.overwrite_pauli_hamiltonian:
            self.pauli_hamiltonian.save(self.pauli_hamiltonian_fname)

    def load(self, fname="energy_optimizer", quiet=False):
        '''Load a saved optimizer checkpoint.
        
        The energy optimisation needs to load the following attributes:
            - The network optimizer.
            - The log.
            - The number of steps taken / number of epochs / total running time.
            
        Additionally, the wavefunction itself must be loaded, but this is handled internally in the
        object.  Here, we will try to load the wavefunction from the specified file path, but will
        not raise an exception if it is not found.  Instead we raise a warning and assume the user
        will locate and load the wavefunction manually.
        '''
        fname = self.__format_checkpoint_fname(fname)
        if not quiet:
            print("Loading checkpoint {}.".format(fname), end="...")
        try:
            checkpoint = torch.load(fname, map_location=self.device)
        except:
            checkpoint = torch.jit.load(fname, map_location=self.device)

        try:
            self.wavefunction.load(checkpoint['wavefunction:fname'])
        except:
            print(f"\twavefunction not found (expected at {checkpoint['wavefunction:fname']})")

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer:state_dict'])
        except:
            print("\tOptimizer could not be loaded.")
        self.log = checkpoint['log']
        
        self.n_steps = checkpoint['n_steps']
        self.n_epochs = checkpoint['n_epochs']
        self.run_time = checkpoint['run_time']
        self.sampled_idxs = checkpoint['sampled_idxs']

        if not quiet:
            print("done.")

    def save_psi(self, fname, subspace_args={}, quiet=False):
        '''Save the wavefunction amplitudes to file.'''
        fname = os.path.join(self.save_loc, fname)

        if not quiet:
            print("Saving psi to {}.".format(fname), end="...")

        dir = os.path.dirname(fname)
        if dir != '':
            mk_dir(dir, quiet)

        if subspace_args == {}:
            subspace_args = {
                "N_up": self.n_electrons,
                "N_alpha": self.n_alpha_electrons,
                "N_beta": self.n_beta_electrons,
                "N_occ": self.n_fixed_electrons}

        self.wavefunction.save_psi(fname, subspace_args)

        print("done.")

    def save_log(self, fname="log", quiet=False):
        '''Save the optimizer's log to file.
        '''
        fname = os.path.join(self.save_loc, fname)
        dir = os.path.dirname(fname)
        if dir != '':
            mk_dir(dir, quiet)

        path, ext = os.path.splitext(fname)
        if ext != ".pkl":
            fname = path + ".pkl"

        ITERS = "Iteration"
        df = None
        for key, value in self.log.items():
            df_key = pd.DataFrame(value, columns=[ITERS, key])
            if df is not None:
                df = pd.merge(df, df_key, how="outer", on=ITERS)
            else:
                df = df_key

        if df is not None:
            df = df.sort_values(ITERS).reset_index(drop=True)

            df.to_pickle(fname)
            print("Log saved to", fname)
        else:
            print("Log is empty.")

class ExactSamplingOptimizer(OptimizerBase):
    '''
    Optimize a wavefunction ansatz using exhaustive sampling of the physically valid Hilbert space.
    '''

    def __init__(self,
                 num_batches_per_epoch=None,
                 **kwargs):
        kwargs['reweight_samples_by_psi'] = True
        super().__init__(**kwargs)
        self.num_batches_per_epoch = num_batches_per_epoch
        if self.num_batches_per_epoch is None:
            self.__batch_size = len(self.hilbert.get_subspace(ret_states=False, ret_idxs=True, **self.subspace_args))
        else:
            subspace_size = len(self.hilbert.get_subspace(ret_states=False, ret_idxs=True, **self.subspace_args))
            self.__batch_size = math.ceil(subspace_size / self.num_batches_per_epoch)

    def solve_H(self, states_idx=None):
        ss, ss_idxs = self.wavefunction.hilbert.get_subspace(ret_idxs=True, **self.subspace_args)
        H = self.pauli_hamiltonian.get_H(ss_idxs)
        eig_val, eig_vec = sp.sparse.linalg.eigs(H, k=1, which='SR', maxiter=1e9)
        eig_vec = eig_vec * np.sign(eig_vec[0])

        return eig_val[0], eig_vec[0]

    def pre_train(self, n_epochs, target_amps=None, use_equal_unset_amps=False,
                  optimizer=torch.optim.Adam, optimizer_args={'lr': 5e-3},
                  output_freq=50):
        '''Pre-train the NAQS wavefunction towards a given set of target amplitudes.

        This can be useful to, for example, train the system towards the Hartree-Fock state before
        commencing with energy-based optimization.
        '''
        ss, ss_idxs = self.wavefunction.hilbert.get_subspace(ret_idxs=True, **self.subspace_args)
        optimizer = optimizer(self.wavefunction.parameters(), **optimizer_args)

        target = torch.zeros_like(ss_idxs, dtype=torch.float)
        if target_amps is None:
            target[0] = 1
        else:
            target[:len(target_amps)] = torch.FloatTensor(target_amps)
            target[len(target_amps):] = np.sum(target_amps) / (len(ss_idxs) - len(target_amps))

        if not use_equal_unset_amps:
            len_target = len(target_amps) if target_amps is not None else 1
        else:
            len_target = len(ss_idxs)

        train_mask = torch.zeros_like(ss_idxs)
        train_mask[:len_target] = 1

        dataset = torch.utils.data.TensorDataset(ss, target, train_mask.bool())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.__batch_size, shuffle=self.num_batches_per_epoch is not None)

        t0 = time.time()
        print("Pre-training NAQS amplitudes...")
        for i in range(1, n_epochs + 1):
            for states, amps, mask in dataloader:
                optimizer.zero_grad()
                if mask.sum() > 0:
                    psi = self.wavefunction.amplitude(states)
                    if self.normalise_psi:
                        psi = psi / psi.detach().pow(2).sum().pow(0.5)
                    loss = F.binary_cross_entropy(psi[mask],
                                                  amps[mask])
                    loss.backward()
                    optimizer.step()

            if ((i) % output_freq == 0) or (i == 1):
                time_per_epoch = (time.time() - t0) / output_freq
                print(f"\t Epoch {i} : loss = {loss.item():.5e}, |psi|^2 = {psi.norm().detach().item():.3f}, epoch time={time_per_epoch:.2f}s")
                t0 = time.time()
        optimizer.zero_grad()
        print("done.")

    def pre_flatten(self, n_epochs,
                  optimizer=torch.optim.Adam, optimizer_args={'lr': 5e-3},
                  output_freq=50):
        '''Pre-train the NAQS wavefunction towards a given set of target amplitudes.

        This can be useful to, for example, train the system towards the Hartree-Fock state before
        commencing with energy-based optimization.
        '''
        ss, ss_idxs = self.wavefunction.hilbert.get_subspace(ret_idxs=True, **self.subspace_args)
        optimizer = optimizer(self.wavefunction.parameters(), **optimizer_args)

        target = torch.ones_like(ss_idxs, dtype=torch.float) / math.sqrt(len(ss))

        dataset = torch.utils.data.TensorDataset(ss, target)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.__batch_size, shuffle=self.num_batches_per_epoch is not None)

        t0 = time.time()
        print("Pre-training NAQS amplitudes...")
        for i in range(1, n_epochs + 1):
            for states, amps in dataloader:
                optimizer.zero_grad()
                psi = self.wavefunction.amplitude(states)
                if self.normalise_psi:
                    psi = psi / psi.detach().pow(2).sum().pow(0.5)
                loss = F.binary_cross_entropy(psi, target)
                loss.backward()
                optimizer.step()

            if ((i) % output_freq == 0) or (i == 1):
                time_per_epoch = (time.time() - t0) / output_freq
                print(f"\t Epoch {i} : loss = {loss.item():.5e}, |psi|^2 = {psi.norm().detach().item():.3f}, epoch time={time_per_epoch:.2f}s")
                t0 = time.time()
        optimizer.zero_grad()
        print("done.")

    def run(self, n_epochs, save_freq=None, save_final=False, reset_log=False, reset_optimizer=False, output_freq=50):
        '''Optimise the wavefunction.

        Runs the optimisation process for n_epochs epoch's.  The number of SGD steps per epoch
        is determined by self.num_batches_per_epoch.

        n_epochs : The number of epoch's to run for.
        save_freq : How often (in epoch's) to save the wavefunction.  None means don't save.
        save_final : Whether to save the final wavefunction.
        reset_log : Whether to reset the logging.
        reset_optimizer : Whether to reset the parameter optimizer.
        output_freq : How often (in epoch's) to print statistics on training.
        '''
        # self.wavefunction.train_model()

        if reset_log:
            self.reset_log()
        if reset_optimizer:
            self.reset_optimizer()

        __run_time_at_last_log = self.run_time
        __num_steps_at_last_log = self.n_steps

        states, states_idx = self.hilbert.get_subspace(ret_states=True, ret_idxs=True, **self.subspace_args)

        # Quickly print some information about the number of states and number of mini-batches.
        n_str = f"{len(states_idx)}" if len(states_idx) < 1e6 else f"{len(states_idx):.3e}"
        samp_str = f"Each epoch processes {n_str} samples"
        if (self.num_batches_per_epoch is not None) and (self.num_batches_per_epoch > 1):
            samp_str += f" in {self.num_batches_per_epoch} mini-batches"
        print(f"Training NAQS energy. {samp_str}.\n")

        dataset = torch.utils.data.TensorDataset(states, states_idx)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.__batch_size,
                                                 shuffle=self.num_batches_per_epoch is not None,
                                                 pin_memory=True,
                                                 # num_workers=16
                                                 )
        self.wavefunction.model.clear_cache()

        for i in range(n_epochs):
            for s, s_idx in dataloader:
                t_start_step = time.time()
                local_energy, local_energy_variance = self._SGD_step(s, s_idx)

                self.n_steps += 1
                self.run_time += (time.time() - t_start_step)

                self.log[LogKey.E_LOC].append((self.n_steps, local_energy))
                self.log[LogKey.E_LOC_VAR].append((self.n_steps, local_energy_variance))
                self.log[LogKey.N_UNIQUE_SAMP].append((self.n_steps, len(s_idx)))
                self.log[LogKey.TIME].append((self.n_steps, self.run_time))

            self.n_epochs += 1

            if ((self.n_epochs) % output_freq == 0) or (self.n_epochs == 1):
                energy = self.calculate_energy(normalise_psi=True)
                self.log[LogKey.E].append((self.n_steps, energy))

                time_per_epoch = (self.run_time - __run_time_at_last_log) / output_freq
                __run_time_at_last_log = self.run_time

                steps_in_epochs = self.n_steps - __num_steps_at_last_log
                __num_steps_at_last_log = self.n_steps

                print(f"Epoch {self.n_epochs} ({steps_in_epochs} SGD steps) : "
                      + f"<E>={energy:.5f}, "
                      + f"<E_loc>={local_energy:.5f}, "
                      + f"var(<E_loc>)={local_energy_variance:.5f}, "
                      + f"epoch time={time_per_epoch:.2f}s, total time={self.run_time:.1f}s",
                      end="\t\t\t\n")

            if save_freq is not None:
                if self.n_epochs % save_freq == 0:
                    print("\t saving progress...", end="")
                    self.save(os.path.join(self.save_loc, f"opt_{self.n_steps}steps"), quiet=True)
                    print("done.")
        if save_final:
            self.save(quiet=False)

class PartialSamplingOptimizer(OptimizerBase):
    '''
    Optimize a wavefunction ansatz using partial sampling of the Hilbert space.
    '''

    def __init__(self,
                 n_samples,
                 n_samples_max = 1e9,
                 n_unq_samples_min = 1000,
                 n_unq_samples_max = 1e6,
                 # n_samples_steps=None,
                 log_exact_energy=True,
                 **kwargs):
        kwargs['reweight_samples_by_psi'] = False
        super().__init__(**kwargs)
        self.log_exact_energy = log_exact_energy
        self.n_samples = n_samples
        self.n_samples_max = int(n_samples_max)
        self.n_unq_samples_min = int(n_unq_samples_min)
        self.n_unq_samples_max = int(n_unq_samples_max)
        # self.n_samples_steps = n_samples_steps

    def get_n_samples(self):
        return self.n_samples
    #     if isinstance(self.n_samples, numbers.Number):
    #         n_samples = self.n_samples
    #     else:
    #         n_samples = self.n_samples[bisect(self.n_samples_steps, self.n_steps)]
    #     self.exact_sampling = (self.n_samples <= 0)
    #     return n_samples

    def solve_H(self, n_samps=None, ret_n_samps=True):
        if n_samps is None:
            n_samps = self.get_n_samples()
        states, counts, probs, log_psi = self.wavefunction.sample(n_samps)

        N_DIAG_MAX = 10000

        n_unq = len(states)
        if n_unq > N_DIAG_MAX:
            print(f"Limiting number of sampled states from {n_unq} is to most likely {N_DIAG_MAX}.")
            sort_args = np.argsort(counts)
            states = states[ sort_args[-N_DIAG_MAX:] ]

        print(f"Diagonalising sampled Hilbert subspace of size: {n_unq}x{n_unq}", end="...")

        H = self.pauli_hamiltonian.get_H(self.hilbert.state2idx(states, use_restricted_idxs=False).squeeze())
        eig_val, eig_vec = sp.sparse.linalg.eigs(H, k=1, which='SR', maxiter=1e9)
        eig_vec = eig_vec * np.sign(eig_vec[0])

        print("done.")

        if ret_n_samps:
            return eig_val[0].real, eig_vec[0], n_unq
        else:
            return eig_val[0].real, eig_vec[0]

    def pre_train(self, n_epochs, target_amps=None, use_equal_unset_amps=False,
                  optimizer=torch.optim.Adam, optimizer_args={'lr': 5e-3},
                  output_freq=50):
        '''Pre-train the NAQS wavefunction towards a given set of target amplitudes.

        This can be useful to, for example, train the system towards the Hartree-Fock state before
        commencing with energy-based optimization.
        '''
        ss, ss_idxs = self.wavefunction.hilbert.get_subspace(ret_idxs=True, **self.subspace_args)
        optimizer = optimizer(self.wavefunction.parameters(), **optimizer_args)

        if target_amps is None:
            target = torch.FloatTensor([1])
            target_states = ss[:1]
        else:
            target = torch.FloatTensor(target_amps)
            target_states = ss[:len(target_amps)]

        if use_equal_unset_amps:
            amp_other = torch.FloatTensor([(np.sum(t**2 for t in target) / (len(ss_idxs) - len(target_amps)))**0.5]*self.n_samples)
            target = torch.cat([target, amp_other])

        t0 = time.time()
        print("Pre-training NAQS amplitudes...")
        for i in range(1, n_epochs + 1):
            optimizer.zero_grad()
            # psi = self.wavefunction.amplitude(s)
            # loss = F.binary_cross_entropy(psi, target)

            log_psi = self.wavefunction.log_psi(target_states.clone())

            if use_equal_unset_amps:
                if log_psi.dim() == 1:
                    log_psi.unsqueeze_(0)
                states, counts, probs, sample_log_psi = self.wavefunction.sample(self.n_samples)
                log_psi = torch.cat([log_psi, sample_log_psi], 0)

            psi = log_psi[...,0].exp()
            # if self.normalise_psi:
            #     psi /= psi.pow(2).sum().pow(0.5)
            loss = F.binary_cross_entropy(psi, target)

            loss.backward()
            optimizer.step()

            if ((i) % output_freq == 0) or (i == 1):
                time_per_epoch = (time.time() - t0) / output_freq
                print(f"\t Epoch {i} : loss = {loss.item():.5e}, |psi|^2 = {psi.norm().detach().item():.3f}, epoch time={time_per_epoch:.2f}s")
                t0 = time.time()
        optimizer.zero_grad()
        print("done.")

    def pre_flatten(self, n_epochs, n_samps=1e6, flatten_phase=False, norm_reg_weight=0,
                    use_sampling=True, max_batch_size=-1,
                    optimizer=torch.optim.Adam, optimizer_args={'lr': 5e-3},
                    output_freq=50):
        '''Pre-train the NAQS wavefunction towards a given set of target amplitudes.

        This can be useful to, for example, train the system towards the Hartree-Fock state before
        commencing with energy-based optimization.
        '''
        print("Pre-flattening NAQS amplitudes", end="...")

        ss, ss_idxs = self.wavefunction.hilbert.get_subspace(ret_idxs=True, **self.subspace_args)
        optimizer = optimizer(self.wavefunction.parameters(), **optimizer_args)

        log_amp_target = math.log(1 / math.sqrt(len(ss)))
        if flatten_phase:
            log_amp_target = [log_amp_target, 0]

        if not use_sampling:
            states, states_idxs = self.hilbert.get_subspace(ret_idxs=True, **self.subspace_args)
            if max_batch_size < 0:
                max_batch_size = len(states)
            n_batches = (len(states)-1)//max_batch_size + 1
            print(f"using {n_batches} batch(es) of size of at most {max_batch_size}.")

            def run_epoch():
                for j, idx_batch in enumerate(torch.randperm(len(states)).chunk(n_batches)):
                    optimizer.zero_grad()
                    log_psi = self.wavefunction.log_psi(states[idx_batch])
                    if not flatten_phase:
                        log_psi = log_psi[..., 0]
                    loss = F.mse_loss(log_psi.view(-1),
                                      torch.FloatTensor([log_amp_target] * len(idx_batch)).to(log_psi).view(-1))
                    loss.backward()
                    optimizer.step()
                return loss, log_psi

        else:
            def run_epoch():
                optimizer.zero_grad()
                states, counts, probs, log_psi = self.wavefunction.sample(n_samps)
                if not flatten_phase:
                    log_psi = log_psi[..., 0]
                loss = F.mse_loss(log_psi.view(-1),
                                  torch.FloatTensor([log_amp_target] * len(counts)).to(log_psi).view(-1))
                loss.backward()
                optimizer.step()

        t0 = time.time()
        for i in range(1, n_epochs + 1):

            loss, log_psi = run_epoch()

            if ((i) % output_freq == 0) or (i == 1):
                time_per_epoch = (time.time() - t0)
                if i!=1:
                    time_per_epoch /= output_freq
                print(f"\t Epoch {i} : loss = {loss.item():.5e}, |psi|^2 = {log_psi[...,0].exp().norm().detach().item():.3f}, epoch time={time_per_epoch:.2f}s")
                t0 = time.time()
        optimizer.zero_grad()
        print("done.")

    def run(self, n_epochs, save_freq=None, save_final=False, reset_log=False, reset_optimizer=False, output_freq=50):
        '''Optimise the wavefunction.

        Runs the optimisation process for n_epochs epoch's.  The number of SGD steps per epoch
        is determined by self.num_batches_per_epoch.

        n_epochs : The number of epoch's to run for.
        save_freq : How often (in epoch's) to save the wavefunction.  None means don't save.
        save_final : Whether to save the final wavefunction.
        reset_log : Whether to reset the logging.
        reset_optimizer : Whether to reset the parameter optimizer.
        output_freq : How often (in epoch's) to print statistics on training.
        '''
        # self.wavefunction.train_model()

        if reset_log:
            self.reset_log()
        if reset_optimizer:
            self.reset_optimizer()

        __run_time_at_last_log = self.run_time
        __num_steps_at_last_log = self.n_steps

        # Quickly print some information about the number of states and number of mini-batches.
        if self.reweight_samples_by_psi:
            s_info = "(normalised) wavefunction amplitude squared"
        else:
            s_info = "frequency"
        print(f"Training NAQS energy.  Samples will be weighted by their {s_info}.")
        self.wavefunction.model.clear_cache()

        if self.n_steps == 0:
            self.save(os.path.join(self.save_loc, f"opt_{self.n_steps}steps"), quiet=False)

        def get_samples(last_action=0):
            action = 0  # -1 decrease samples, 0 return samples, +1 increase samples

            try:
                states, counts, probs, log_psi = self.wavefunction.sample(self.n_samples,
                                                                          max_batch_size=self.n_unq_samples_max)
                n_unq, sampling_completed = len(states), True
            except MaxBatchSizeExceededError:
                print("MaxBatchSizeExceededError")
                n_unq, sampling_completed = self.n_unq_samples_max + 1, False
                action = -1

            # Only check the number of unique samples if we can still increase/decrease the number of samples used.
            if ((self.n_samples != self.n_unq_samples_min) and (self.n_samples != self.n_samples_max)) or not sampling_completed:

                # If we want more unique samples, increase the sample size.
                if n_unq < self.n_unq_samples_min and last_action >= 0:
                    action = 1
                    self.n_samples = int( min(self.n_samples * 10, self.n_samples_max) )
                    print(f"\t...{n_unq} unique samples generated --> increasing batch size to {self.n_samples/1e6:.1f}M at epoch {self.n_epochs}.")

                # If we want more fewer samples, decrease the sample size.
                elif n_unq > self.n_unq_samples_max and last_action <= 0:
                    action = -1
                    self.n_samples = int( max(self.n_samples / 10, self.n_unq_samples_min) )
                    print(f"\t...{n_unq} unique samples generated --> decreasing batch size to {self.n_samples/1e6:.1f}M at epoch {self.n_epochs}.")

            if action != 0:
                if sampling_completed:
                    # If sampling was completed, let's clear as much memory as we can before trying again.
                    log_psi = log_psi.detach()
                    del states, counts, probs, log_psi
                    torch.cuda.empty_cache()
                return get_samples(action)
            else:
                return states, counts, probs, log_psi

        # n_train_per_subnet = 100

        for i in range(n_epochs):
            t_start_step = time.time()
            # if (i % n_train_per_subnet)==0:
            #     self.reset_optimizer(cond_idx=(i // n_train_per_subnet) % (self.wavefunction._N_model // 2))

            # if i % 50 == 0:
            #     self.verbose = True
            # else:
            #     self.verbose = False

            states, counts, probs, log_psi = get_samples()

            if self.verbose:
                print(f"generate samples : {time.time()-t_start_step:.4f}s")
            if self.reweight_samples_by_psi:
                weights = log_psi.detach()[...,0].exp().pow(2)
                weights /= weights.sum()
            else:
                weights = counts.float() / counts.sum().float()
                # weights = counts.float() / self.n_samples

            local_energy, local_energy_variance = self._SGD_step(states, self.hilbert.state2idx(states, use_restricted_idxs=False),
                                                                 log_psi,
                                                                 sample_weights=weights,
                                                                 n_samps=self.n_samples * weights)


            self.n_steps += 1
            self.run_time += (time.time() - t_start_step)

            self.log[LogKey.E_LOC].append((self.n_steps, local_energy))
            self.log[LogKey.E_LOC_VAR].append((self.n_steps, local_energy_variance))
            self.log[LogKey.N_UNIQUE_SAMP].append((self.n_steps, len(weights)))
            self.log[LogKey.TIME].append((self.n_steps, self.run_time))

            self.n_epochs += 1

            if ((self.n_epochs) % output_freq == 0) or (self.n_epochs == 1):
                if self.log_exact_energy:
                    energy = self.calculate_energy(normalise_psi=True)
                    energy_str = f"{energy:.5f}"
                else:
                    energy = None
                    energy_str = "N/A"
                self.log[LogKey.E].append((self.n_steps, energy))

                local_energies = [x[1] for x in self.log[LogKey.E_LOC][-min(output_freq,self.n_epochs):]]
                local_energy = np.mean(local_energies)
                local_energy_err = np.std(local_energies)

                time_per_epoch = (self.run_time - __run_time_at_last_log) / output_freq
                __run_time_at_last_log = self.run_time

                steps_in_epochs = self.n_steps - __num_steps_at_last_log
                __num_steps_at_last_log = self.n_steps

                n_samps = counts.sum().item()

                if n_samps < 1e3:
                    samp_str = f"{n_samps:.1f}"
                elif n_samps < 1e6:
                    samp_str = f"{(n_samps / 1e3):.1f}k"
                elif n_samps < 1e9:
                    samp_str = f"{(n_samps / 1e6):.1f}M"
                else:
                    samp_str = f"{(n_samps / 1e9):.1f}B"

                print(f"Epoch {self.n_epochs} ({steps_in_epochs} SGD steps with {samp_str} samples ({len(weights)} unq.) : "
                      + f"<E>={energy_str}, "
                      + f"<E_loc>={local_energy:.5f} +\- {local_energy_err:.5f}, "
                      + f"var(<E_loc>)={local_energy_variance:.5f}, "
                      + f"epoch time={time_per_epoch:.2f}s, total time={self.run_time:.1f}s",
                      end="\t\t\t\n")

            if save_freq is not None:
                if self.n_epochs % save_freq == 0:
                    print("\t saving progress...", end="")
                    self.save(os.path.join(self.save_loc, f"opt_{self.n_steps}steps"), quiet=True)
                    print("done.")

        if save_final:
            self.save(quiet=False)

class DensitySamplingOptimizer(OptimizerBase):
    '''
    Optimize a wavefunction ansatz using partial sampling of the Hilbert space.
    '''

    def __init__(self,
                 dP,
                 dP_steps=None,
                 log_exact_energy=True,
                 **kwargs):
        kwargs['reweight_samples_by_psi'] = True
        super().__init__(**kwargs)
        self.log_exact_energy = log_exact_energy
        self.dP = dP
        self.dP_steps = dP_steps

    def get_dP(self):
        if isinstance(self.dP, numbers.Number):
            n_samples = self.dP
        else:
            n_samples = self.dP[bisect(self.dP_steps, self.n_steps)]
        self.exact_sampling = (self.dP <= 0)
        return n_samples

    def pre_train(self, n_epochs, dP, target_amps=None, use_equal_unset_amps=False,
                  optimizer=torch.optim.Adam, optimizer_args={'lr': 5e-3},
                  output_freq=50):
        '''Pre-train the NAQS wavefunction towards a given set of target amplitudes.

        This can be useful to, for example, train the system towards the Hartree-Fock state before
        commencing with energy-based optimization.
        '''
        raise NotImplementedError()

    def pre_flatten(self, n_epochs, dP=None, norm_reg_weight=0,
                  optimizer=torch.optim.Adam, optimizer_args={'lr': 5e-3},
                  output_freq=50):
        '''Pre-train the NAQS wavefunction towards a given set of target amplitudes.

        This can be useful to, for example, train the system towards the Hartree-Fock state before
        commencing with energy-based optimization.
        '''
        if dP is None:
            dP = self.get_dP()

        ss, ss_idxs = self.wavefunction.hilbert.get_subspace(ret_idxs=True, **self.subspace_args)
        optimizer = optimizer(self.wavefunction.parameters(), **optimizer_args)

        amp_target = 1 / math.sqrt(len(ss))

        t0 = time.time()
        print("Pre-flattening NAQS amplitudes...")
        for i in range(1, n_epochs + 1):
            optimizer.zero_grad()

            states, counts, probs, log_psi, norm_reg = self.wavefunction.sample(dP)
            psi = log_psi[...,0].exp()
            loss = F.binary_cross_entropy(psi, torch.FloatTensor([amp_target]*len(counts)).to(psi))

            loss.backward()
            optimizer.step()

            if ((i) % output_freq == 0) or (i == 1):
                time_per_epoch = (time.time() - t0) / output_freq
                print(f"\t Epoch {i} : loss = {loss.item():.5e}, |psi|^2 = {psi.norm().detach().item():.3f}, epoch time={time_per_epoch:.2f}s")
                t0 = time.time()
        optimizer.zero_grad()
        print("done.")

    def run(self, n_epochs,  save_freq=None, save_final=False, reset_log=False, reset_optimizer=False, output_freq=50):
        '''Optimise the wavefunction.

        Runs the optimisation process for n_epochs epoch's.  The number of SGD steps per epoch
        is determined by self.num_batches_per_epoch.

        n_epochs : The number of epoch's to run for.
        save_freq : How often (in epoch's) to save the wavefunction.  None means don't save.
        save_final : Whether to save the final wavefunction.
        reset_log : Whether to reset the logging.
        reset_optimizer : Whether to reset the parameter optimizer.
        output_freq : How often (in epoch's) to print statistics on training.
        '''
        # self.wavefunction.train_model()

        if reset_log:
            self.reset_log()
        if reset_optimizer:
            self.reset_optimizer()

        __run_time_at_last_log = self.run_time
        __num_steps_at_last_log = self.n_steps

        # Quickly print some information about the number of states and number of mini-batches.
        print(f"Training NAQS energy.  Samples will be weighted by their (normalised) wavefunction amplitude squared.")
        self.wavefunction.model.clear_cache()

        for i in range(n_epochs):
            t_start_step = time.time()
            dP = self.get_dP_samples()
            states, probs, log_psi = self.wavefunction.sample_dP(dP)
            if self.verbose:
                print(f"generate samples : {time.time()-t_start_step:.4f}s")
            weights = log_psi.detach()[...,0].exp().pow(2)
            weights /= weights.sum()
            # reg_loss = 1e-3 * self.wavefunction.model.masking_loss
            local_energy, local_energy_variance = self._SGD_step(states, self.hilbert.state2idx(states),
                                                                 log_psi,
                                                                 sample_weights=weights)
                                                                 # regularisation_loss=reg_loss)


            self.n_steps += 1
            self.run_time += (time.time() - t_start_step)

            self.log[LogKey.E_LOC].append((self.n_steps, local_energy))
            self.log[LogKey.E_LOC_VAR].append((self.n_steps, local_energy_variance))
            self.log[LogKey.TIME].append((self.n_steps, self.run_time))

            self.n_epochs += 1

            if ((self.n_epochs) % output_freq == 0) or (self.n_epochs == 1):
                if self.log_exact_energy:
                    energy = self.calculate_energy(normalise_psi=True)
                else:
                    energy = None
                self.log[LogKey.E].append((self.n_steps, energy))

                time_per_epoch = (self.run_time - __run_time_at_last_log) / output_freq
                __run_time_at_last_log = self.run_time

                steps_in_epochs = self.n_steps - __num_steps_at_last_log
                __num_steps_at_last_log = self.n_steps

                print(f"Epoch {self.n_epochs} ({steps_in_epochs} SGD steps with dP={dP:.3e} ({len(weights)} unq. states) : "
                      + f"<E>={energy:.5f}, "
                      + f"<E_loc>={local_energy:.5f}, "
                      + f"var(<E_loc>)={local_energy_variance:.5f}, "
                      + f"epoch time={time_per_epoch:.2f}s, total time={self.run_time:.1f}s",
                      end="\t\t\t\n")

            if save_freq is not None:
                if self.n_epochs % save_freq == 0:
                    print("\t saving progress...", end="")
                    self.save(os.path.join(self.save_loc, f"opt_{self.n_steps}steps"), quiet=True)
                    print("done.")

        if save_final:
            self.save(quiet=False)