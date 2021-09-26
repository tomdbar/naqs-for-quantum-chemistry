import os

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, csr_matrix, load_npz, save_npz

from src.utils.hilbert import Encoding
from src.utils.system import mk_dir

import pickle

from src.utils.hamiltonian_math import get_Hij_cy, popcount_parity

import time

from abc import ABC, abstractmethod

def numberOfSetBits(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24

def bitCountParity(i):
    # res = np.zeros_like(i)
    # for _ in range(i[0].itemsize):
    #     res ^= i & 1
    #     i >>= 1
    # return 1 - 2 * res
    return 1 - 2*(np.fmod(numberOfSetBits(i),2))

class PauliHamiltonian():

    @staticmethod
    def _format_fnames(fname, fname_H=None):
        f, ext = os.path.splitext(fname)
        if f[-5:] != "_info":
            fbase, fname = f, f"{f}_info"
        else:
            fbase, fname = f[:-5], f

        fname_H = f"{fbase}.npz"
        fname = f"{fname}.npz"

        return fname, fname_H

    @staticmethod
    def get(hilbert, qubit_hamiltonian, hamiltonian_fname=None, restricted_idxs=None, n_excitations_max=None, verbose=False, dtype=np.float32):
        pauli_hamiltonian = None
        if hamiltonian_fname is not None:
            hamiltonian_fname, hamiltonian_fname_npz = PauliHamiltonian._format_fnames(hamiltonian_fname)
            if os.path.exists(hamiltonian_fname):
                pauli_hamiltonian = _PauliHamiltonianDynamic(hilbert, qubit_hamiltonian, restricted_idxs, n_excitations_max, verbose, dtype)
                pauli_hamiltonian.load(hamiltonian_fname)
            elif os.path.exists(hamiltonian_fname_npz):
                pauli_hamiltonian = _PauliHamiltonianFrozen(hilbert, qubit_hamiltonian, hamiltonian_fname_npz, restricted_idxs, n_excitations_max, verbose, dtype)

        if pauli_hamiltonian is None:
            pauli_hamiltonian = _PauliHamiltonianDynamic(hilbert, qubit_hamiltonian, restricted_idxs, n_excitations_max, verbose, dtype)

        return pauli_hamiltonian

        # else:
        #     return _PauliHamiltonianFrozen(hilbert, qubit_hamiltonian, hamiltonian_fname, restricted_idxs, n_excitations_max, verbose, dtype)
        # if hamiltonian_fname is None:
        #     return _PauliHamiltonianDynamic(hilbert, qubit_hamiltonian, restricted_idxs, n_excitations_max, verbose, dtype)
        # else:
        #     return _PauliHamiltonianFrozen(hilbert, qubit_hamiltonian, hamiltonian_fname, restricted_idxs, n_excitations_max, verbose, dtype)


class __PauliHamiltonianBase(ABC):

    def __init__(self, hilbert, qubit_hamiltonian,
                 restricted_idxs=None, n_excitations_max=None,
                 verbose=False,
                 dtype=np.float32):
        self.hilbert = hilbert
        assert self.hilbert.encoding == Encoding.SIGNED, "PauliCouplings requires Encoding.SIGNED."
        self.qubit_hamiltonian = qubit_hamiltonian
        self.restricted_idxs = self.hilbert.full2restricted_idx(restricted_idxs)
        self.n_excitations_max = n_excitations_max
        self.dtype = dtype
        self.verbose = verbose

        d = self.hilbert.size
        self.H = csr_matrix(([], ([], [])), shape=(d, d), dtype=self.dtype)
        # self.cached_couplings = np.zeros((d, d), dtype=bool)
        self._cached_idxs = np.array([], dtype=hilbert.get_idx_dtype("np"))

        self._frozen_H = False
        self._restricted_H = None

    def __get_new_H_subspace(self, idxs):
        return self.H[idxs[:,np.newaxis],idxs]

    def get_H(self, idxs=None):
        # t = time.time()
        if idxs is not None:
            idxs = self.hilbert.full2restricted_idx(idxs)
            try:
                is_full_sample = (len(idxs) == len(self.restricted_idxs))
            except:
                is_full_sample = False
            if is_full_sample:
                H = self.get_restricted_H()
            else:
                H = self.__get_new_H_subspace(idxs)
        else:
            H = self.H
        # print(f"\tupdate self.get_H(), {time.time() - t:.4f}")
        return H

    def get_restricted_H(self):
        if self._frozen_H and (self._restricted_H is not None):
            H = self._restricted_H
        else:
            H = self.__get_new_H_subspace(self.restricted_idxs)
            if self._frozen_H:
                self._restricted_H = H
        return H

    def get_coupled_state_idxs(self, state_idxs, return_unique=False):
        '''Return the coupled indexes (columns) corresponding to each state (row),
        from the sparse matrix, H.

        return_unique : If True, returns a flat list of unqiue state_idxs.  If False,
                        a list of lists of coupled states for each input is returned.
        '''
        coupled_idxs = [self.H.indices[self.H.indptr[idx]:self.H.indptr[idx + 1]] for idx in state_idxs]
        if return_unique:
            coupled_idxs = np.unique(np.concatenate(coupled_idxs))
        return coupled_idxs

    def freeze_H(self):
        '''Freeze further updates to H.

        This is useful if we know we have updated H with all possible couplings.
        By freezing H at this point, we can simply return with no-action if more
        update steps are called, rather than re-checking the cache every time.
        '''
        self._frozen_H = True

    def is_frozen(self):
        return self._frozen_H

    def load_H(self, fname):
        if os.path.splitext(fname)[-1] != ".npz":
            fname += ".npz"
        print(f"Loading sparse Hamiltonian matrix from '{fname}'", end="...")
        self.H = load_npz(fname)
        print("done.")
        if self.H.dtype is not np.dtype(self.dtype):
            print(f"\tLoaded dtype ({self.H.dtype}) does not match expected dtype ({self.dtype})." +
                  f"\tCasting loaded matix to {self.dtype}")
        print(f"\tMatrix has size {self.H.shape} and {self.H.nnz} non-zero elements.")

    def save_H(self, fname):
        if os.path.splitext(fname)[-1] != ".npz":
            fname += ".npz"

        dir = os.path.dirname(fname)
        if dir != '':
            mk_dir(dir, False)

        print(f"Saving sparse Hamiltonian matrix to '{fname}'", end="...")
        d = self.H.data
        d[np.abs(self.H.data) < 1e-12] = 0
        self.H.data = d
        self.H.eliminate_zeros()
        save_npz(fname, self.H)
        print(f"done.")

    def load(self, fname, fname_H=None):
        fname, fname_H = PauliHamiltonian._format_fnames(fname, fname_H)

        print(f"Loading PauliHamiltonian from '{fname}'", end="...")
        with np.load(fname, allow_pickle=True) as f_in:
            info = f_in['info']
        print("done.")

        self._cached_idxs = info[0]
        self._frozen_H = info[1]

        self.load_H(fname_H)

    def save(self, fname, fname_H=None):
        fname, fname_H = PauliHamiltonian._format_fnames(fname, fname_H)

        dir = os.path.dirname(fname)
        if dir != '':
            mk_dir(dir, False)

        info = np.array([self._cached_idxs, self._frozen_H, fname_H])

        print(f"Saving PauliHamiltonian to '{fname}'", end="...")
        np.savez(fname, info=info)
        self.save_H(fname_H)
        print("done.")

    @abstractmethod
    def unfreeze_H(self):
        '''Allow further updates to H.

        This is useful if you want to allow updates to H haveing previously frozen
        it!
        '''
        raise NotImplementedError()

    @abstractmethod
    def update_H(self, idx):
        '''Update the Hamiltonian with couplings for given states.

        Computes the couplings for state_idx, adds them to the sparse Hamiltonian
        matrix and returns the updated Hamiltonian.
        '''
        raise NotImplementedError()

class _PauliHamiltonianDynamic(__PauliHamiltonianBase):
    '''Calculates and formats the coupling information.

    Takes in a qubit hamiltonian from openfermion, pre-computes as much as possible,
    then calculates the coupling Hamiltonian on-demand, whilst (trying to!) minimise
    repeated computations.

    Pre-computes for following for each coupling term (Pauli string):
        parities_idx : bit-strings corresponding to the relative parity of coupled states.
        YZ_sites_idx : bit-strings corresponding to the positions of pauli X/Y operators.
        couplings : the coupling pre-factor for the bit string.
        bin2couplingsign : lut to convert the bit-string resulting from:
                            states bit string AND YZ_sites_idx
                           to the sign of the resulting coupling.

    Stores the Hamiltonian as a sparse matrix, self.H, that is filled in 'on-demand'.
    The sparse matrix spance the full Hilbert space (2^N x 2^N), as memory isn't an
    issue for this (most empty) matrix.  As such, the idx of states in the Hilbert
    space also index the rows/columns of the Hamiltonian matrix.
    '''

    _MAX_RAM = 50e9  # Process matrices larger than _MAX_RAM bytes this in chunks of this size.

    def __init__(self, hilbert, qubit_hamiltonian,
                 restricted_idxs=None, n_excitations_max=None, verbose=False, dtype=np.float32):
        super().__init__(hilbert, qubit_hamiltonian, restricted_idxs, n_excitations_max, verbose, dtype)

        print("\tpre-processing Hamiltonian terms and coupling lookups", end="...\n\t")
        self.XY_sites_idx, self.YZ_sites_idx, self.couplings = self.__calc_coupling_info()

        self._unique_XY_sites_idx, self._unique2all_XY_sites_idx = np.unique(self.XY_sites_idx, return_inverse=True)
        self._unique_YZ_sites_idx, self._unique2all_YZ_sites_idx = np.unique(self.YZ_sites_idx, return_inverse=True)

        self._unique_XY_sites_idx = self.hilbert.to_idx_array(self._unique_XY_sites_idx)
        self._unique2all_XY_sites_idx = self.hilbert.to_idx_array(self._unique2all_XY_sites_idx)

        # self._unique2all_XY_sites_idx = self._unique2all_XY_sites_idx.astype(np.int32)
        # self._unique2all_YZ_sites_idx = self._unique2all_YZ_sites_idx.astype(np.int32)

        self.dtype = dtype

        print(f"Pauli Hamiltonian has K={len(self.couplings)} terms:", end="\n\t\t--> ")
        print(f"there are {len(self._unique_XY_sites_idx)} unique XY bit-strings (1 if v_k,n = {{X,Y}}, 0 if v_k,n = {{I,Z}})),",
              end="\n\t\t--> ")
        print(f"there are {len(self._unique_YZ_sites_idx)} unique YZ bit-strings (1 if v_k,n = {{Y,Z}}, 0 if v_k,n = {{I,X}})).\n")

    def unfreeze_H(self):
        '''Allow further updates to H.

        This is useful if you want to allow updates to H having previously frozen it!
        '''
        self._frozen_H = False
        self._restricted_H = None

    def update_H(self, state_idx, check_unseen=True, assume_unique=False):
        '''Update the Hamiltonian with couplings for given states.

        Computes the couplings for state_idx, adds them to the sparse Hamiltonian
        matrix and returns the updated Hamiltonian.

        check_unseen : Whether to check if the passed states have already been
                       cached.  If False, any previous result is overwritten.
                       If True, already computed states are ignored.

        are_unique : Whether the passed states are unique.  If true, this can
                     allows for faster checking of the cache.
        '''
        if not self._frozen_H:
            # Only check, compute and update if H is not frozen!

            # First we make sure we have the state_idxs as a numpy array of unique values.
            state_i_idx = self.hilbert.to_idx_array(state_idx)
            if self.verbose:
                t=time.time()
                print(f"Sampled {len(state_i_idx)} unique states (dtype={state_i_idx.dtype}):", end="\n\t--> ")
            if check_unseen:
                state_i_idx = np.setdiff1d(state_i_idx, self._cached_idxs, assume_unique=assume_unique)
                if self.verbose:
                    print(f"gives {len(state_i_idx)} un-cached states ({time.time()-t:.4f}s).", end="\n\t--> ")
                if len(state_i_idx) == 0:
                    # If nothing new is asked for, return immediately.
                    return self.get_H()

            P_k_bits_by_unique_YZ_sites = np.bitwise_and(state_i_idx[:, None], self._unique_YZ_sites_idx[None, :])
            if self.verbose:
                print(f"P_k_bits_by_unique_YZ_sites has shape {P_k_bits_by_unique_YZ_sites.shape} ({P_k_bits_by_unique_YZ_sites.dtype}) when considering unique YZ bit strings) ({time.time()-t:.4f}s).",
                      end="\n\t--> ")
            P_k_by_unique_YZ_sites = popcount_parity(P_k_bits_by_unique_YZ_sites)
            if self.verbose:
                print(f"P_k_by_unique_YZ_sites has shape {P_k_by_unique_YZ_sites.shape} when considering unique YZ bit strings) ({time.time()-t:.4f}s).",
                      end="\n\t--> ")

            M = len(state_i_idx)
            Kxy = len(self._unique_XY_sites_idx)

            j_idx_full = np.bitwise_xor(state_i_idx[:, None], self._unique_XY_sites_idx[None, :]).ravel()
            # j_idx_full = np.bitwise_xor(state_i_idx[None, :], self._unique_XY_sites_idx[:, None]).ravel()
            if self.verbose:
                print(f"Calculated coupled states (i.e. j_idx_full : {j_idx_full.shape}, {j_idx_full.dtype}) ({time.time()-t:.4f}s).",
                      end="\n\t--> ")

            # ij_idxs = self.hilbert.full2restricted_idx(np.concatenate([state_i_idx, j_idx_full]))
            # i_idx, j_idx = ij_idxs[:len(state_i_idx)], ij_idxs[len(state_i_idx):]
            i_idx = self.hilbert.full2restricted_idx(state_i_idx)
            j_idx = self.hilbert.full2restricted_idx(j_idx_full)

            if self.verbose:
                print(f"Converted states from full --> restricted idxs ({time.time()-t:.4f}s).",
                      end="\n\t--> ")

            physical_coupling_mask = np.where(j_idx >= 0)[0]

            if self.verbose:
                print(f"Calculated physical_coupling_mask (M={M}, Kxy={Kxy}, len(H_ij) - len(physical_coupling_mask)={M*Kxy - len(physical_coupling_mask)})({time.time()-t:.4f}s).",
                      end="\n\t--> ")

            # Finally, all that remains is to calculate the couplings.
            H_ij = get_Hij_cy(state_i_idx, self._unique_XY_sites_idx, self._unique2all_XY_sites_idx,
                              P_k_by_unique_YZ_sites, self._unique2all_YZ_sites_idx,
                              self.couplings.squeeze())

            if self.verbose:
                print(f"Calculated couplings (i.e. H_ij's) ({H_ij.shape}) ({time.time()-t:.4f}s).",
                      end="\n\t--> ")

            H_ij = H_ij[physical_coupling_mask]
            i_idx, j_idx = i_idx[physical_coupling_mask // Kxy], j_idx[physical_coupling_mask]

            if self.verbose:
                print(f"Apply mask to Hamiltonian data ({time.time()-t:.4f}s).",
                      end="\n\t--> ")

            H_new = csr_matrix((H_ij, (i_idx, j_idx)),
                               shape=(self.hilbert.size, self.hilbert.size))

            if self.verbose:
                print(f"Created new sparse Hamiltonian with {H_new.nnz} non-zero elements ({time.time()-t:.4f}s).",
                      end="\n\t--> ")

            self._cached_idxs = np.concatenate((self._cached_idxs, state_i_idx))

            # H_new.data[np.abs(H_new.data) < 1e-9] = 0
            # H_new.eliminate_zeros()

            nnz_old = self.H.nnz
            self.H = self.H + H_new

            if self.verbose:
                print("Updating H:", end="\n\t\t--> ")
                print(f"from {nnz_old} non-zero elements", end=" ")
                print(f"to {self.H.nnz} non-zero elements. ({time.time()-t:.4f}s)")

        return self.H


    def __calc_coupling_info(self):
        '''Pre-compute the internal storage of the set qubit hamiltonian.

        This pre-computes parities_idx, YZ_sites_idx and couplings.
        '''
        # n_qubits = self.qubit_hamiltonian.many_body_order()
        n_qubits = self.hilbert.N
        couplings, XY_sites, YZ_sites = [], [], []

        # Iterate over all terms in the qubit Hamiltonian.
        for coupling_idx, (term, coupling) in enumerate(self.qubit_hamiltonian.terms.items()):
            valid_term, num_exc = True, 0
            XY_site = torch.zeros(n_qubits, dtype=torch.int8)
            YZ_site = torch.zeros(n_qubits, dtype=torch.int8)
            num_Y = 0
            for qubit_idx, pauli_idx in term:
                if pauli_idx in ['X', 'Y']:
                    XY_site[qubit_idx] = 1
                    if pauli_idx == 'Y':
                        num_Y += 1
                        YZ_site[qubit_idx] = 1
                    if qubit_idx < self.hilbert.N_occ:
                        valid_term = False  # Can't flip a fixed qubit!
                        break
                    elif self.n_excitations_max is not None:
                        num_exc += 1
                        if num_exc > self.n_excitations_max:
                            valid_term = False  # Not allowed this many excitations!
                            break
                elif pauli_idx == 'Z':
                    YZ_site[qubit_idx] = 1

            if valid_term:
                # XY_site_idx = (self.hilbert._idx_basis_vec * XY_site).sum()
                # XY_sites.append(XY_site_idx)
                #
                # YZ_site_idx = (self.hilbert._idx_basis_vec * YZ_site).sum()
                # YZ_sites.append(YZ_site_idx)

                # couplings.append(1j ** (num_Y) * coupling)

                XY_sites.append(XY_site)
                YZ_sites.append(YZ_site)
                couplings.append( (1j ** (num_Y)).real * coupling)

        # XY_sites = self.hilbert.to_idx_array(torch.stack(XY_sites))
        # YZ_sites = self.hilbert.to_idx_array(torch.stack(YZ_sites))
        # couplings = np.expand_dims(np.stack(couplings), 1)

        XY_sites = self.hilbert.to_idx_array( (self.hilbert._idx_basis_vec.unsqueeze(0) * torch.stack(XY_sites)).sum(1) )
        YZ_sites = self.hilbert.to_idx_array( (self.hilbert._idx_basis_vec.unsqueeze(0) * torch.stack(YZ_sites)).sum(1) )
        couplings = np.expand_dims(np.stack(couplings), 1).astype(self.dtype)

        # if couplings.imag.sum() == 0:
        #     # If there are no imaginary valued couplings (there aren't...)
        #     couplings = couplings.real.astype(self.dtype)

        return XY_sites, YZ_sites, couplings

class _PauliHamiltonianFrozen(__PauliHamiltonianBase):

    def __init__(self, hilbert, qubit_hamiltonian,
                 hamiltonian_fname,
                 restricted_idxs=None, n_excitations_max=None, verbose=False, dtype=np.float32):
        super().__init__(hilbert, qubit_hamiltonian, restricted_idxs, n_excitations_max, verbose, dtype)
        self.hamiltonian_fname = hamiltonian_fname
        self.load_H(self.hamiltonian_fname)
        self.freeze_H()

    def unfreeze_H(self, *args, **kwargs):
        '''Allow further updates to H.

        This is useful if you want to allow updates to H haveing previously frozen
        it!
        '''
        raise NotImplementedError("Can't unfreeze the PauliHamiltonian.  Presumably this Hamiltonian was loaded from a file" +
                                  "and so was created without any of the dynamic back-end for calculating new couplings.")

    def update_H(self, *args, **kwargs):
        '''Update the Hamiltonian with couplings for given states.

        Computes the couplings for state_idx, adds them to the sparse Hamiltonian
        matrix and returns the updated Hamiltonian.
        '''
        if not self._frozen_H:
            raise NotImplementedError("Can't update the PauliHamiltonian.  Presumably this Hamiltonian was loaded from a file" +
                                      "and so was created without any of the dynamic back-end for calculating new couplings.")