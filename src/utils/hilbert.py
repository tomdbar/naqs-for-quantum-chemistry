import numpy as np
import torch
import torch.nn.functional as F
from enum import Enum
import numpy as np
import math

from scipy.special import comb
from itertools import permutations, combinations, product

from collections import defaultdict
from operator import itemgetter

from src.utils.hilbert_math import make_basis_idxs_cy

from abc import ABC, abstractmethod

from collections.abc import Iterable

import time


class Encoding(Enum):
    BINARY = 0
    SIGNED = 1


class Hilbert():

    @staticmethod
    def get(N, N_alpha=None, N_beta=None, *args, **kwargs):
        if (N_alpha is None) and (N_beta is None):
            return _HilbertFull(N, *args, **kwargs)
        elif isinstance(N_alpha, Iterable) or isinstance(N_beta, Iterable):
            return _HilbertPartiallyRestricted(N, N_alpha, N_beta, *args, **kwargs)
        else:
            return _HilbertRestricted(N, N_alpha, N_beta, *args, **kwargs)


class _HilbertBase(ABC):

    @abstractmethod
    def get_subspace(self):
        raise NotImplementedError()

    @abstractmethod
    def get_basis(self):
        raise NotImplementedError()

    @abstractmethod
    def state2idx(self):
        raise NotImplementedError()

    @abstractmethod
    def idx2state(self):
        raise NotImplementedError()

    @abstractmethod
    def restricted2full_idx(self):
        raise NotImplementedError()

    @abstractmethod
    def full2restricted_idx(self):
        raise NotImplementedError()

    def to_state_tensor(self, state):
        if torch.is_tensor(state):
            return state.to(self._state_torch_dtype)
        else:
            return torch.tensor(state, dtype=self._state_torch_dtype)

    def to_idx_tensor(self, idx):
        if torch.is_tensor(idx):
            return idx.to(self._idx_torch_dtype)
        return torch.tensor(idx, dtype=self._idx_torch_dtype)

    def to_state_array(self, state):
        if isinstance(state, np.ndarray):
            return state.astype(self._state_np_dtype)
        elif torch.is_tensor(state):
            return state.numpy().astype(self._state_np_dtype)
        else:
            return np.array(state, dtype=self._state_np_dtype)

    def to_idx_array(self, idx):
        if isinstance(idx, np.ndarray):
            return idx.astype(self._idx_np_dtype)
        elif torch.is_tensor(idx):
            return idx.numpy().astype(self._idx_np_dtype)
        else:
            return np.array(idx, dtype=self._idx_np_dtype)

    def to_state(self, state):
        if torch.is_tensor(state):
            return self.to_state_tensor(state)
        elif isinstance(state, np.ndarray):
            return self.to_state_array(state)
        else:
            raise ValueError("state must be PyTorch tensor or numpy array.")

    def to_idx(self, idx):
        if torch.is_tensor(idx):
            return self.to_idx_tensor(idx)
        elif isinstance(idx, np.ndarray):
            return self.to_idx_array(idx)
        else:
            raise ValueError("idx must be PyTorch tensor or numpy array.")

    def get_state_dtype(self, type="torch"):
        type = type.lower()
        if type == "torch":
            return self._state_torch_dtype
        elif type == "np" or type == "numpy":
            return self._state_np_dtype
        else:
            raise ValueError("type must be 'torch' or 'numpy'.")

    def get_idx_dtype(self, type="torch"):
        type = type.lower()
        if type == "torch":
            return self._idx_torch_dtype
        elif type == "np" or type == "numpy":
            return self._idx_np_dtype
        else:
            raise ValueError("type must be 'torch' or 'numpy'.")


class _HilbertFull(_HilbertBase):
    _MAKE_BASIS_THRESHOLD = 30

    def __init__(self, N, N_occ=None, encoding=Encoding.BINARY, make_basis=None, verbose=False):
        self.N = N
        self.size = 2**N
        self.verbose = verbose

        if self.verbose:
            print("preparing _HilbertFull...")

        self._state_torch_dtype, self._state_np_dtype = torch.int8, np.int8

        # if self.N < 8:
        #     self._idx_torch_dtype, self._idx_np_dtype = torch.int8, np.int8
        if self.N < 16:
            self._idx_torch_dtype, self._idx_np_dtype = torch.int16, np.int16
        elif self.N < 30:
            # elif self.N < 26:
            self._idx_torch_dtype, self._idx_np_dtype = torch.int32, np.int32
        else:
            self._idx_torch_dtype, self._idx_np_dtype = torch.int64, np.int64

        # self._idx_torch_dtype, self._idx_np_dtype = torch.int32, np.int32

        if self.verbose: print(f"\tPreparing basis information", end="...")

        self.N_occ = 0 if N_occ is None else N_occ
        #         self._idx_basis_vec = torch.FloatTensor([2 ** n for n in range(N - 1, -1, -1)])
        self._idx_basis_vec = self.to_idx_tensor([2 ** n for n in range(N)])
        if self.N_occ > 0:
            self._idx_pad = self._idx_basis_vec[:self.N_occ].sum().item()
            self._idx_stride = 2 ** N_occ
        else:
            self._idx_pad = 0

        if self.verbose: print("done.")

        if encoding not in [Encoding.BINARY, Encoding.SIGNED]:
            raise ValueError("{} is not a recognised encoding.".format(encoding))
        self.encoding = encoding

        if self.verbose: print(f"\tHilbert encoding: {Encoding.BINARY}")

        self.basis_states, self.basis_idxs = None, None
        if make_basis is None:
            make_basis = (N <= self._MAKE_BASIS_THRESHOLD)
        if make_basis:
            if N > self._MAKE_BASIS_THRESHOLD:
                raise RuntimeWarning(
                    f"Warning: setting make_basis=True for a Hilbert space with N={N} is likely to take a long time...")
            if self.verbose: print("\tMaking basis", end="...")
            self.basis_states, self.basis_idxs = self.get_basis(ret_states=True, ret_idxs=True)
        else:
            if self.verbose: print("\tMaking __basis_idxs", end="...")
            self.basis_idxs = self.to_idx_tensor(self.__pad_idxs(self.__make_basis_idxs()))
        if self.verbose: print("done.")

        self.subspaces = {}

    def __check_config(self, N_up, N_alpha, N_beta, N_occ, N_exc_max):
        if ((N_up is not None)
                and (N_alpha is not None)
                and (N_beta is not None)):
            assert N_up == N_alpha + N_beta, f"N_up ({N_up}) must be the sum of N_alpha ({N_alpha}) and N_beta ({N_beta})"

        elif ((N_alpha is not None) and (N_beta is not None)):
            N_up = N_alpha + N_beta

        elif ((N_up is not None) and (N_alpha is not None)):
            N_beta = N_up - N_alpha

        elif ((N_up is not None) and (N_beta is not None)):
            N_alpha = N_up - N_beta

        elif (N_alpha is not None):
            N_up = N_alpha
            N_beta = 0

        elif (N_beta is not None):
            N_up = N_beta
            N_alpha = 0

        elif (N_up is not None):
            assert N_up <= self.N, f"N_up ({N_up}) must be <= N ({self.N})"

        if (N_occ is not None):
            assert N_occ >= self.N_occ, f"Hilbert space if configured with self.N_occ={self.N_occ} < {N_occ}"
            if N_occ == self.N_occ:
                N_occ = None
            else:
                N_occ -= self.N_occ

        if (N_exc_max is not None):
            assert N_exc_max <= N_up, f"Maximum number of excitations (N_exc) can not exceed total number of 1's (N_up)."

        if (N_up is not None):
            assert self.N_occ <= N_up, f"self.N_occ ({self.N_occ}) must be <= N_up ({N_up})"
            N_up -= self.N_occ

        if (N_alpha is not None):
            N_alpha -= math.ceil(self.N_occ / 2)

        if (N_beta is not None):
            N_beta -= math.floor(self.N_occ / 2)

        return N_up, N_alpha, N_beta, N_occ, N_exc_max

    def get_subspace(self, N_up=None, N_alpha=None, N_beta=None, N_occ=None, N_exc_max=None,
                     ret_states=True, ret_idxs=False, use_restricted_idxs=False):
        N_up, N_alpha, N_beta, N_occ, N_exc_max = self.__check_config(N_up, N_alpha, N_beta, N_occ, N_exc_max)
        key = (N_up, N_alpha, N_beta, N_occ, N_exc_max)
        if key in self.subspaces:
            space_states, space_idxs = self.subspaces[key]
        else:
            space_idxs = self.__basis_idxs
            space_bool = space_idxs > 0

            mask = None
            if ((N_up is not None) and
                    (N_alpha is None) and
                    (N_beta is None)):

                mask = (space_bool.sum(1) == N_up)

            elif ((N_up is not None) and
                  (N_alpha is not None) and
                  (N_beta is not None)):

                mask_alpha = (space_bool[:, ::2].sum(1) == N_alpha)
                mask_beta = (space_bool[:, 1::2].sum(1) == N_beta)

                mask = (mask_alpha & mask_beta)

            if N_occ is not None:
                mask_occ = (space_bool[:, :N_occ] > 0).sum(1) == N_occ
                if mask is not None:
                    mask = (mask & mask_occ)
                else:
                    mask = mask_occ

            if N_exc_max is not None:
                mask_exc = (space_bool[:, N_up:] > 0).sum(1) <= N_exc_max
                if mask is not None:
                    mask = (mask & mask_exc)
                else:
                    mask = mask_exc

            if mask is not None:
                space_idxs = space_idxs[mask]
                space_bool = space_bool[mask]

            # space_states = torch.tensor(space_bool, dtype=torch.float32).squeeze()
            # space_idxs = torch.tensor(space_idxs, dtype=torch.long).sum(1)
            space_states = self.to_state_tensor(space_bool).squeeze()
            space_idxs = self.to_idx_tensor(space_idxs.sum(1))

            space_states, space_idxs = self.__pad_space(space_states, space_idxs)

            if self.encoding == Encoding.SIGNED:
                space_states = 2 * space_states - 1

            self.subspaces[key] = (space_states, space_idxs)

        if ret_states and ret_idxs:
            return space_states, space_idxs
        elif ret_states:
            return space_states
        elif ret_idxs:
            return space_idxs

    def __make_basis_idxs(self):
        N = self.N - self.N_occ
        # dim = np.arange(2 ** N, dtype=self._idx_np_dtype)
        # self.__basis_idxs = (dim[:, None] & (1 << np.arange(N, dtype=self._idx_np_dtype)))
        self.__basis_idxs = make_basis_idxs_cy(N)
        return self.to_idx_tensor(self.__basis_idxs).sum(1)
        # return torch.tensor(self.__basis_idxs, dtype=torch.long).sum(1)

    def get_basis(self, ret_states=True, ret_idxs=False, use_restricted_idxs=False):
        if self.basis_states is not None:
            basis_states = self.basis_states.clone()
            basis_idxs = self.basis_idxs.clone()
        else:
            basis_idxs = self.__make_basis_idxs()
            basis_bool = (self.__basis_idxs > 0)

            basis_states = self.to_state_tensor(basis_bool).squeeze()
            # basis_states = torch.tensor(basis_bool, dtype=torch.int8).squeeze()
            # basis_idxs = torch.tensor(self.__basis_idxs, dtype=torch.long).sum(1)

            basis_states, basis_idxs = self.__pad_space(basis_states, basis_idxs)

            if self.encoding == Encoding.SIGNED:
                basis_states = 2 * basis_states - 1

        if ret_states and ret_idxs:
            return self.to_state_tensor(basis_states), self.to_idx_tensor(basis_idxs)
        elif ret_states:
            return self.to_state_tensor(basis_states)
        elif ret_idxs:
            return self.to_idx_tensor(basis_idxs)

    def __pad_states(self, states):
        if self.N_occ > 0:
            states = F.pad(states, (self.N_occ, 0), value=1)
        return states

    def __pad_idxs(self, idxs):
        if self.N_occ > 0:
            idxs = (idxs << self.N_occ) + self._idx_pad
        return idxs

    def __pad_space(self, states, idxs):
        return self.__pad_states(states), self.__pad_idxs(idxs)

    def state2idx(self, state, use_restricted_idxs=False):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state)
        state_clamped = state.clamp_min(0)
        idxs = (state_clamped * self._idx_basis_vec).sum(dim=-1, keepdim=True)
        return self.to_idx_tensor(idxs)

    def idx2state(self, idx, use_restricted_idxs=False):
        if self.N_occ > 0:
            valid_idx = ((idx - self._idx_pad) % self._idx_stride == 0)
            if not valid_idx.all():
                raise Exception("The following idxs are not valid in the configured Hilbert space:\n", idx[~valid_idx])

            idx = (idx - 3) >> self.N_occ

        if not torch.is_tensor(idx):
            idx = torch.LongTensor(idx)

        if self.basis_states is None:
            state = torch.zeros(self.N, len(idx))
            for i in range(self.N - 1, -1, -1):
                state[i] = idx.fmod(2)
            state = state.transpose(0, 1)
            if self.encoding == Encoding.SIGNED:
                state = 2 * state - 1

        else:
            state = self.basis_states.index_select(0, idx.long())
        return self.to_state_tensor(state)

    def restricted2full_idx(self, idx):
        return idx

    def full2restricted_idx(self, idx):
        return idx


class _HilbertRestricted(_HilbertBase):

    def __init__(self, N, N_alpha=None, N_beta=None,
                 encoding=Encoding.BINARY, make_basis=True, verbose=False):
        self.N = N
        self.N_up = N_alpha + N_beta
        self.N_occ = 0
        self.N_alpha = N_alpha
        self.N_beta = N_beta

        self.__check_config(self.N_up, self.N_alpha, self.N_beta, None, None)

        self.size = int(comb(math.ceil(self.N / 2), N_alpha) * comb(math.floor(self.N / 2), N_beta))

        self.verbose = verbose

        if self.verbose:
            print("preparing _HilbertRestricted...")

        # min_bits = math.ceil(math.log(self.size, 2))
        self._state_torch_dtype, self._state_np_dtype = torch.int8, np.int8

        # if N < 8:
        #     self._idx_torch_dtype, self._idx_np_dtype = torch.int8, np.int8
        if N < 16:
            self._idx_torch_dtype, self._idx_np_dtype = torch.int16, np.int16
        elif N < 30:
            self._idx_torch_dtype, self._idx_np_dtype = torch.int32, np.int32
        else:
            self._idx_torch_dtype, self._idx_np_dtype = torch.int64, np.int64

        if encoding not in [Encoding.BINARY, Encoding.SIGNED]:
            raise ValueError("{} is not a recognised encoding.".format(encoding))
        self.encoding = encoding

        if self.verbose:
            print(f"\tHilbert encoding: {Encoding.BINARY}")

        if self.verbose:
            print(f"\tPreparing basis information", end="...")

        if not make_basis:
            raise NotImplementedError("_HilbertRestricted must have make_basis=True.")

        self._idx_basis_vec = self.to_idx_tensor([2 ** n for n in range(N)])

        self.basis_states, self.basis_idxs, self.restricted2full_basis_idxs = self.__prepare_basis()

        if self.N <= 30:
            # Faster look up, but requires more memory.
            self.use_full2restricted_lut_arr = True
            full2restricted_basis_idxs = -1 * np.ones(2 ** self.N)
            full2restricted_basis_idxs[self.restricted2full_basis_idxs] = np.arange(len(self.restricted2full_basis_idxs))
            self.full2restricted_basis_idxs = self.to_idx_tensor(full2restricted_basis_idxs)
        else:
            self.use_full2restricted_lut_arr = False
            self.full2restricted_basis_idxs = defaultdict(lambda: -1,
                                                          np.stack([self.restricted2full_basis_idxs,
                                                                    np.arange(len(self.restricted2full_basis_idxs))]).T)

        if self.verbose:
            print("done.")

        self.subspaces = {(None, None): (self.basis_states.clone(), self.basis_idxs.clone())}

    def __prepare_basis(self):

        alpha_set_bits = np.array(list(combinations(np.arange(0, self.N, step=2), self.N_alpha)))
        beta_set_bits = np.array(list(combinations(np.arange(1, self.N, step=2), self.N_beta)))

        # alphabeta_set_bits = np.array(list(product(alpha_set_bits, beta_set_bits))).reshape(-1, self.N_up)
        alphabeta_set_bits = np.array([np.concatenate(x) for x in product(alpha_set_bits, beta_set_bits)])

        restricted_basis = np.zeros((len(alphabeta_set_bits), self.N), dtype=self._state_np_dtype)

        restricted_basis[
            np.broadcast_to(np.arange(len(alphabeta_set_bits))[:, None], alphabeta_set_bits.shape),
            alphabeta_set_bits
        ] = 1

        alphabeta_set_bits = (2 ** alphabeta_set_bits).sum(-1)
        restricted_hilbert_idxs = np.arange(len(restricted_basis))

        if self.encoding == Encoding.SIGNED:
            restricted_basis = 2 * restricted_basis - 1

        return (self.to_state_tensor(restricted_basis),
                self.to_idx_tensor(restricted_hilbert_idxs),
                self.to_idx_tensor(alphabeta_set_bits))


    def __check_config(self, N_up, N_alpha, N_beta, N_occ, N_exc_max):
        if ((N_up is not None)
                and (N_alpha is not None)
                and (N_beta is not None)):
            assert N_up == N_alpha + N_beta, f"N_up ({N_up}) must be the sum of N_alpha ({N_alpha}) and N_beta ({N_beta})"

        elif ((N_alpha is not None) and (N_beta is not None)):
            N_up = N_alpha + N_beta

        elif ((N_up is not None) and (N_alpha is not None)):
            N_beta = N_up - N_alpha

        elif ((N_up is not None) and (N_beta is not None)):
            N_alpha = N_up - N_beta

        elif (N_alpha is not None):
            N_up = N_alpha
            N_beta = 0

        elif (N_beta is not None):
            N_up = N_beta
            N_alpha = 0

        elif (N_up is not None):
            assert N_up <= self.N, f"N_up ({N_up}) must be <= N ({self.N})"

        if (N_occ is not None):
            if N_occ == self.N_occ:
                N_occ = None
            else:
                N_occ -= self.N_occ

        if (N_exc_max is not None):
            assert N_exc_max <= N_up, f"Maximum number of excitations (N_exc) can not exceed total number of 1's (N_up)."

        return N_up, N_alpha, N_beta, N_occ, N_exc_max

    def get_subspace(self, N_up=None, N_alpha=None, N_beta=None,
                     N_occ=None, N_exc_max=None,
                     ret_states=True, ret_idxs=False, use_restricted_idxs=False):
        if N_up is None:
            N_up = self.N_up
        if N_alpha is None:
            N_alpha = self.N_alpha
        if N_beta is None:
            N_beta = self.N_beta
        N_up, N_alpha, N_beta, N_occ, N_exc_max = self.__check_config(N_up,
                                                                      N_alpha,
                                                                      N_beta,
                                                                      N_occ,
                                                                      N_exc_max)
        key = (N_occ, N_exc_max)
        if key in self.subspaces:
            space_states, space_idxs = self.subspaces[key]
        else:
            mask = None
            space_states, space_idxs = self.basis_states.clone(), self.basis_idxs.clone()
            if N_occ is not None:
                mask_occ = (space_states[:, :N_occ] > 0).sum(1) == N_occ
                if mask is not None:
                    mask = (mask & mask_occ)
                else:
                    mask = mask_occ

            if N_exc_max is not None:
                mask_exc = (space_states[:, self.N_up:] > 0).sum(1) <= N_exc_max
                if mask is not None:
                    mask = (mask & mask_exc)
                else:
                    mask = mask_exc

            if mask is not None:
                space_states = space_states[mask]
                space_states = space_states[mask]

            self.subspaces[key] = (space_states, space_idxs)

        if not use_restricted_idxs:
            space_idxs = self.restricted2full_basis_idxs[space_idxs.long()]

        if ret_states and ret_idxs:
            return space_states, space_idxs
        elif ret_states:
            return space_states
        elif ret_idxs:
            return space_idxs

    def get_basis(self, ret_states=True, ret_idxs=False, use_restricted_idxs=False):
        basis_states = self.basis_states.clone()
        basis_idxs = self.basis_idxs.clone()

        if not use_restricted_idxs:
            basis_idxs = self.restricted2full_basis_idxs[basis_idxs.long()]

        if ret_states and ret_idxs:
            return self.to_state_tensor(basis_states), self.to_idx_tensor(basis_idxs)
        elif ret_states:
            return self.to_state_tensor(basis_states)
        elif ret_idxs:
            return self.to_idx_tensor(basis_idxs)

    def state2idx(self, state, use_restricted_idxs=False):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state)
        state_clamped = state.clamp_min(0)
        idxs = self.to_idx_tensor((state_clamped * self._idx_basis_vec).sum(dim=-1, keepdim=True))
        if use_restricted_idxs:
            # idxs = self.full2restricted_basis_idxs[idxs]
            idxs = self.full2restricted_idx(idxs)
        return self.to_idx_tensor(idxs)

    def idx2state(self, idx, use_restricted_idxs=False):
        if not torch.is_tensor(idx):
            idx = torch.LongTensor(idx)

        if not use_restricted_idxs:
            # idx = self.full2restricted_basis_idxs[idx]
            idx = self.full2restricted_idx(idx)

        state = self.basis_states.index_select(0, idx.long())

        return self.to_state_tensor(state)

    def restricted2full_idx(self, idx):
        np_out = False
        if not torch.is_tensor(idx):
            idx = torch.LongTensor(idx)
            np_out = True
        idx = self.restricted2full_basis_idxs[idx.long()]
        if np_out:
            idx = self.to_idx_array(idx)
        else:
            idx = self.to_idx_tensor(idx)
        return idx

    def full2restricted_idx(self, idx):
        if self.use_full2restricted_lut_arr:
            return self.__full2restricted_idx_arr_lut(idx)
        else:
            return self.__full2restricted_idx_dic_lut(idx)

    def __full2restricted_idx_arr_lut(self, idx):
        np_out = False
        if not torch.is_tensor(idx):
            idx = torch.LongTensor(idx)
            np_out = True
        idx = self.full2restricted_basis_idxs[idx.long()]
        if np_out:
            idx = self.to_idx_array(idx)
        else:
            idx = self.to_idx_tensor(idx)
        return idx

    def __full2restricted_idx_dic_lut(self, idx):
        if torch.is_tensor(idx):
            idx = idx.numpy()
            np_out = False
        else:
            np_out = True
        if len(idx) > 1:
            idx = np.fromiter( itemgetter(*idx.squeeze())(self.full2restricted_basis_idxs), self._idx_np_dtype, count=len(idx) )
        else:
            idx = np.array([self.full2restricted_basis_idxs[idx[0]]])

        if np_out:
            idx = self.to_idx_array(idx)
        else:
            idx = self.to_idx_tensor(idx)
        return idx


class _HilbertPartiallyRestricted(_HilbertBase):

    def __init__(self, N, N_alpha, N_beta,
                 encoding=Encoding.BINARY, make_basis=True, verbose=False):
        self.N = N
        self.N_alpha = self.__to_arr(N_alpha)
        self.N_beta = self.__to_arr(N_beta)
        self.N_up = self.__to_arr(self.N_alpha + self.N_beta)
        self.N_occ = 0

        self.__check_config(self.N_up, self.N_alpha, self.N_beta, None, None)

        self.size = sum( int(comb(math.ceil(self.N / 2), N_alpha) * comb(math.floor(self.N / 2), N_beta))
                         for N_alpha, N_beta in zip(self.N_alpha, self.N_beta) )

        self.verbose = verbose

        if self.verbose:
            print("preparing _HilbertRestricted...")

        # min_bits = math.ceil(math.log(self.size, 2))
        self._state_torch_dtype, self._state_np_dtype = torch.int8, np.int8

        # if N < 8:
        #     self._idx_torch_dtype, self._idx_np_dtype = torch.int8, np.int8
        if N < 16:
            self._idx_torch_dtype, self._idx_np_dtype = torch.int16, np.int16
        elif N < 30:
            self._idx_torch_dtype, self._idx_np_dtype = torch.int32, np.int32
        else:
            self._idx_torch_dtype, self._idx_np_dtype = torch.int64, np.int64

        if encoding not in [Encoding.BINARY, Encoding.SIGNED]:
            raise ValueError("{} is not a recognised encoding.".format(encoding))
        self.encoding = encoding

        if self.verbose:
            print(f"\tHilbert encoding: {Encoding.BINARY}")

        if self.verbose:
            print(f"\tPreparing basis information", end="...")

        if not make_basis:
            raise NotImplementedError("_HilbertRestricted must have make_basis=True.")

        self._idx_basis_vec = self.to_idx_tensor([2 ** n for n in range(N)])

        self.basis_states, self.basis_idxs, self.restricted2full_basis_idxs = self.__prepare_basis()

        if self.N <= 30:
            # Faster look up, but requires more memory.
            self.use_full2restricted_lut_arr = True
            full2restricted_basis_idxs = -1 * np.ones(2 ** self.N)
            full2restricted_basis_idxs[self.restricted2full_basis_idxs] = np.arange(len(self.restricted2full_basis_idxs))
            self.full2restricted_basis_idxs = self.to_idx_tensor(full2restricted_basis_idxs)
        else:
            self.use_full2restricted_lut_arr = False
            self.full2restricted_basis_idxs = defaultdict(lambda: -1,
                                                          np.stack([self.restricted2full_basis_idxs,
                                                                    np.arange(len(self.restricted2full_basis_idxs))]).T)

        if self.verbose:
            print("done.")

        self.subspaces = {(None, None): (self.basis_states.clone(), self.basis_idxs.clone())}

    def __prepare_basis(self):

        num_states = 0
        restricted_basis, restricted_hilbert_idxs, alphabeta_set_bits = [], [], []
        for n_alpha, n_beta in zip(self.N_alpha, self.N_beta):
            alpha_set_bits = np.array(list(combinations(np.arange(0, self.N, step=2), n_alpha)))
            beta_set_bits = np.array(list(combinations(np.arange(1, self.N, step=2), n_beta)))

            # alphabeta_set_bits = np.array(list(product(alpha_set_bits, beta_set_bits))).reshape(-1, self.N_up)
            _alphabeta_set_bits = np.array([np.concatenate(x) for x in product(alpha_set_bits, beta_set_bits)])

            _restricted_basis = np.zeros((len(_alphabeta_set_bits), self.N), dtype=self._state_np_dtype)

            _restricted_basis[
                np.broadcast_to(np.arange(len(_alphabeta_set_bits))[:, None], _alphabeta_set_bits.shape),
                _alphabeta_set_bits
            ] = 1

            _alphabeta_set_bits = (2 ** _alphabeta_set_bits).sum(-1)
            _restricted_hilbert_idxs = num_states + np.arange(len(_restricted_basis))
            num_states += len(_restricted_basis)

            if self.encoding == Encoding.SIGNED:
                _restricted_basis = 2 * _restricted_basis - 1

            restricted_basis.append(_restricted_basis)
            restricted_hilbert_idxs.append(_restricted_hilbert_idxs)
            alphabeta_set_bits.append(_alphabeta_set_bits)

        restricted_basis = np.concatenate(restricted_basis)
        restricted_hilbert_idxs = np.concatenate(restricted_hilbert_idxs)
        alphabeta_set_bits = np.concatenate(alphabeta_set_bits)

        return (self.to_state_tensor(restricted_basis),
                self.to_idx_tensor(restricted_hilbert_idxs),
                self.to_idx_tensor(alphabeta_set_bits))

    def __to_arr(self, N):
        if N is None:
            return None
        if not isinstance(N, Iterable):
            N = [N]
        if not isinstance(N, np.ndarray):
            N = np.array(N)
        return N

    def __check_config(self, N_up, N_alpha, N_beta, N_occ, N_exc_max):
        N_up, N_alpha, N_beta, N_occ, N_exc_max = (self.__to_arr(N_up),
                                                   self.__to_arr(N_alpha),
                                                   self.__to_arr(N_beta),
                                                   self.__to_arr(N_occ),
                                                   self.__to_arr(N_exc_max))

        if (N_up is None) and (N_alpha is not None) and (N_beta is not None):
            N_up = N_alpha + N_beta

        assert all(N_up == N_alpha + N_beta), f"N_up ({N_up}) must be the sum of N_alpha ({N_alpha}) and N_beta ({N_beta})"

        if ((N_up is not None) and (N_alpha is not None)):
            N_beta = N_up - N_alpha

        elif ((N_up is not None) and (N_beta is not None)):
            N_alpha = N_up - N_beta

        elif (N_alpha is not None):
            N_up = N_alpha
            N_beta = 0

        elif (N_beta is not None):
            N_up = N_beta
            N_alpha = 0

        elif (N_up is not None):
            assert all(N_up <= self.N), f"N_up ({N_up}) must be <= N ({self.N})"

        if (N_occ is not None):
            if N_occ == self.N_occ:
                N_occ = None
            else:
                N_occ -= self.N_occ

        if (N_exc_max is not None):
            assert N_exc_max <= N_up, f"Maximum number of excitations (N_exc) can not exceed total number of 1's (N_up)."

        return N_up, N_alpha, N_beta, N_occ, N_exc_max

    def get_subspace(self, N_up=None, N_alpha=None, N_beta=None,
                     N_occ=None, N_exc_max=None,
                     ret_states=True, ret_idxs=False, use_restricted_idxs=False):
        if N_occ is not None or N_exc_max is not None:
            # legacy arguments not needed in the final experiments.
            raise NotImplementedError()

        key = (N_occ, N_exc_max)
        if key in self.subspaces:
            space_states, space_idxs = self.subspaces[key]
        else:
            space_states, space_idxs = self.basis_states.clone(), self.basis_idxs.clone()
            self.subspaces[key] = (space_states, space_idxs)

        if not use_restricted_idxs:
            space_idxs = self.restricted2full_basis_idxs[space_idxs.long()]

        if ret_states and ret_idxs:
            return space_states, space_idxs
        elif ret_states:
            return space_states
        elif ret_idxs:
            return space_idxs

    def get_basis(self, ret_states=True, ret_idxs=False, use_restricted_idxs=False):
        basis_states = self.basis_states.clone()
        basis_idxs = self.basis_idxs.clone()

        if not use_restricted_idxs:
            basis_idxs = self.restricted2full_basis_idxs[basis_idxs.long()]

        if ret_states and ret_idxs:
            return self.to_state_tensor(basis_states), self.to_idx_tensor(basis_idxs)
        elif ret_states:
            return self.to_state_tensor(basis_states)
        elif ret_idxs:
            return self.to_idx_tensor(basis_idxs)

    def state2idx(self, state, use_restricted_idxs=False):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state)
        state_clamped = state.clamp_min(0)
        idxs = self.to_idx_tensor((state_clamped * self._idx_basis_vec).sum(dim=-1, keepdim=True))
        if use_restricted_idxs:
            # idxs = self.full2restricted_basis_idxs[idxs]
            idxs = self.full2restricted_idx(idxs)
        return self.to_idx_tensor(idxs)

    def idx2state(self, idx, use_restricted_idxs=False):
        if not torch.is_tensor(idx):
            idx = torch.LongTensor(idx)

        if not use_restricted_idxs:
            # idx = self.full2restricted_basis_idxs[idx]
            idx = self.full2restricted_idx(idx)

        state = self.basis_states.index_select(0, idx.long())

        return self.to_state_tensor(state)

    def restricted2full_idx(self, idx):
        np_out = False
        if not torch.is_tensor(idx):
            idx = torch.LongTensor(idx)
            np_out = True
        idx = self.restricted2full_basis_idxs[idx.long()]
        if np_out:
            idx = self.to_idx_array(idx)
        else:
            idx = self.to_idx_tensor(idx)
        return idx

    def full2restricted_idx(self, idx):
        if self.use_full2restricted_lut_arr:
            return self.__full2restricted_idx_arr_lut(idx)
        else:
            return self.__full2restricted_idx_dic_lut(idx)

    def __full2restricted_idx_arr_lut(self, idx):
        np_out = False
        if not torch.is_tensor(idx):
            idx = torch.LongTensor(idx)
            np_out = True
        idx = self.full2restricted_basis_idxs[idx.long()]
        if np_out:
            idx = self.to_idx_array(idx)
        else:
            idx = self.to_idx_tensor(idx)
        return idx

    def __full2restricted_idx_dic_lut(self, idx):
        if torch.is_tensor(idx):
            idx = idx.numpy()
            np_out = False
        else:
            np_out = True
        if len(idx) > 1:
            idx = np.fromiter( itemgetter(*idx.squeeze())(self.full2restricted_basis_idxs), self._idx_np_dtype, count=len(idx) )
        else:
            idx = np.array([self.full2restricted_basis_idxs[idx[0]]])

        if np_out:
            idx = self.to_idx_array(idx)
        else:
            idx = self.to_idx_tensor(idx)
        return idx


class Unitaries():
    default_unitaries = {"X": np.matrix([[1, 1],
                                         [1, -1]]) / np.sqrt(2),
                         "Y": np.matrix([[1j, 1],
                                         [1, 1j]]) / np.sqrt(2),
                         "Z": np.matrix([[1, 0],
                                         [0, 1]]),
                         "-X": -1 * np.matrix([[1, 1],
                                               [1, -1]]) / np.sqrt(2),
                         "-Y": -1 * np.matrix([[1j, 1],
                                               [1, 1j]]) / np.sqrt(2),
                         "-Z": -1 * np.matrix([[1, 0],
                                               [0, 1]])
                         }

    def __init__(self, unitaries=None, qubit_encoding=Encoding.SIGNED):

        if unitaries is None:
            user_unitaries = {}
        else:
            user_unitaries = unitaries
        assert type(user_unitaries) == dict, "Must pass unitaries as a dictionary object."

        self.qubit_encoding = qubit_encoding

        # Add user defined unitary rotations to the "X","Y","Z" defaults.
        # Note that user definitions will overwrite the defaults if there is a clash.
        unitaries = Unitaries.default_unitaries.copy()
        unitaries.update(user_unitaries)

        qubit0 = np.array([1, 0])
        qubit1 = np.array([0, 1])

        self.rotated_qubits = {}

        for q_idx, q_vec in enumerate([qubit0, qubit1]):
            rots = {}
            for basis, U in unitaries.items():
                rots[basis] = np.concatenate(np.matmul(U, q_vec).tolist()).ravel()
            self.rotated_qubits[q_idx] = rots

    def __qubit2idx(self, vals, encoding):
        if encoding == Encoding.BINARY:
            idxs = vals
        else:  # self.encoding==Encoding.SIGNED
            idxs = (vals + 1) / 2
        return idxs

    def get_rotated_coeffs(self, q_state, q_basis, encoding=None):
        if encoding is None:
            encoding = self.qubit_encoding
        coeffs = [self.rotated_qubits[q_idx][q_basis] for q_idx in self.__qubit2idx(q_state.squeeze(-1), encoding)]
        return np.array(coeffs).transpose()

    def rotate_state(self, states, state_bases, encoding=None):
        states = np.array(states)
        if len(states.shape) < 2:
            # [state] --> [batch=1, state]
            states = np.expand_dims(states, 0)
        if len(states.shape) < 3:
            # [batch, state] --> [batch, state, out_states]
            states = np.expand_dims(states, -1)
        # states = [state]
        coeffs = np.array([[1]] * states.shape[0])

        if encoding is None:
            encoding = self.qubit_encoding
        q0, q1 = -1 if encoding is Encoding.SIGNED else 0, 1

        for qubit_idx, (s, b) in enumerate(zip(states.transpose(1, 0, 2), state_bases)):
            if b != "Z":
                c0, c1 = self.get_rotated_coeffs(s, b, encoding)

                new_states = []
                new_coeffs = []

                for st, cf in zip(states.transpose(2, 0, 1), coeffs.transpose()):
                    new_state0 = st.copy()
                    new_state1 = st.copy()

                    new_state0[..., qubit_idx] = q0
                    new_state1[..., qubit_idx] = q1

                    new_coeff0 = cf * c0
                    new_coeff1 = cf * c1

                    new_states += [new_state0, new_state1]
                    new_coeffs += [new_coeff0, new_coeff1]

                new_states = np.stack(new_states).transpose(1, 2, 0)
                new_coeffs = np.stack(new_coeffs).transpose()

                states = new_states
                coeffs = new_coeffs

        return states.transpose(0, 2, 1), coeffs


# Helper functions to go from little-endian (Alexei) <--> big-endian (Tom)
def reverseBits(num, bitSize):
    # convert number into binary representation
    # output will be like bin(10) = '0b10101'
    binary = bin(num)

    reverse = binary[-1:1:-1]
    reverse = reverse + (bitSize - len(reverse)) * '0'

    # converts reversed binary string into integer
    return int(reverse, 2)

def unpackbits(x, num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = 2 ** np.arange(num_bits).reshape([1, num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

def packbits(x, num_bits):
    bit_vals = 2 ** np.arange(num_bits)
    return np.sum(bit_vals * x, axis=-1)

def flipbits(x, num_bits):
    return packbits(unpackbits(x, num_bits)[:, ::-1], num_bits)