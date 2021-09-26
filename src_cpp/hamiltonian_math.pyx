import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t
cdef extern int __builtin_popcount(unsigned int) nogil
cdef extern int __builtin_popcountll(unsigned long long) nogil


#http://nicolas-hug.com/blog/cython_notes

# Build with : python src_cpp/setup.py build_ext --inplace --force

'''
get_Hij_cy(...)
'''

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.longdouble_t, ndim=1] __inner_int64_longdouble(long M, long Kxy, long K,
                            long [:] _unique2all_XY_sites_idx,
                            long [:,:] P_k_by_unique_YZ_sites,
                            long [:] _unique2all_YZ_sites_idx,
                            long double [:] couplings,
                            cnp.ndarray[cnp.longdouble_t, ndim=1] H_ij):
    cdef:
        Py_ssize_t idx_i, k
        long idx_i_base

    for idx_i in prange(M, nogil=True, schedule="static"): # Sampled state idx
        idx_i_base = idx_i * Kxy
        for k in range(K): # Unique coupled state idx (i.e. term idx)
            H_ij[idx_i_base + _unique2all_XY_sites_idx[k]] += P_k_by_unique_YZ_sites[idx_i, _unique2all_YZ_sites_idx[k]] * couplings[k]

    return H_ij

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.longdouble_t, ndim=1] __inner_int32_longdouble(int M, int Kxy, int K,
                            long [:] _unique2all_XY_sites_idx,
                            long [:,:] P_k_by_unique_YZ_sites,
                            long [:] _unique2all_YZ_sites_idx,
                            long double [:] couplings,
                            cnp.ndarray[cnp.longdouble_t, ndim=1] H_ij):
    cdef:
        Py_ssize_t idx_i, k
        long idx_i_base
        # cnp.ndarray[cnp.float32_t, ndim=1] H_ij = np.zeros(M * Kxy, dtype=np.float32)

    for idx_i in prange(M, nogil=True, schedule="static"): # Sampled state idx
        idx_i_base = idx_i * Kxy
        for k in range(K): # Unique coupled state idx (i.e. term idx)
            H_ij[idx_i_base + _unique2all_XY_sites_idx[k]] += P_k_by_unique_YZ_sites[idx_i, _unique2all_YZ_sites_idx[k]] * couplings[k]

    return H_ij

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.longdouble_t, ndim=1] __inner_int16_longdouble(short M, short Kxy, short K,
                            long [:] _unique2all_XY_sites_idx,
                            long [:,:] P_k_by_unique_YZ_sites,
                            long [:] _unique2all_YZ_sites_idx,
                            long double [:] couplings,
                            cnp.ndarray[cnp.longdouble_t, ndim=1] H_ij):
    cdef:
        Py_ssize_t idx_i, k
        long idx_i_base

    for idx_i in prange(M, nogil=True, schedule="static"): # Sampled state idx
        idx_i_base = idx_i * Kxy
        for k in range(K): # Unique coupled state idx (i.e. term idx)
            H_ij[idx_i_base + _unique2all_XY_sites_idx[k]] += P_k_by_unique_YZ_sites[idx_i, _unique2all_YZ_sites_idx[k]] * couplings[k]

    return H_ij

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
# cdef cnp.ndarray[cnp.float64_t, ndim=1] __inner_int64_double(long M, long Kxy, long K,
#                             long [:] _unique2all_XY_sites_idx,
#                             long [:,:] P_k_by_unique_YZ_sites,
#                             long [:] _unique2all_YZ_sites_idx,
#                             double [:] couplings,
#                             cnp.ndarray[cnp.float64_t, ndim=1] H_ij):
cdef cnp.ndarray[cnp.float64_t, ndim=1] __inner_int64_double(int64_t M, int64_t Kxy, int64_t K,
                            int64_t [:] _unique2all_XY_sites_idx,
                            int64_t [:,:] P_k_by_unique_YZ_sites,
                            int64_t [:] _unique2all_YZ_sites_idx,
                            double [:] couplings,
                            cnp.ndarray[cnp.float64_t, ndim=1] H_ij):
    cdef:
        Py_ssize_t idx_i, k
        long idx_i_base

    for idx_i in prange(M, nogil=True, schedule="static"): # Sampled state idx
        idx_i_base = idx_i * Kxy
        for k in range(K): # Unique coupled state idx (i.e. term idx)
            H_ij[idx_i_base + _unique2all_XY_sites_idx[k]] += P_k_by_unique_YZ_sites[idx_i, _unique2all_YZ_sites_idx[k]] * couplings[k]

    return H_ij

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.float64_t, ndim=1] __inner_int32_double(int M, int Kxy, int K,
                            long [:] _unique2all_XY_sites_idx,
                            long [:,:] P_k_by_unique_YZ_sites,
                            long [:] _unique2all_YZ_sites_idx,
                            double [:] couplings,
                            cnp.ndarray[cnp.float64_t, ndim=1] H_ij):
    cdef:
        Py_ssize_t idx_i, k
        long idx_i_base
        # cnp.ndarray[cnp.float32_t, ndim=1] H_ij = np.zeros(M * Kxy, dtype=np.float32)

    for idx_i in prange(M, nogil=True, schedule="static"): # Sampled state idx
        idx_i_base = idx_i * Kxy
        for k in range(K): # Unique coupled state idx (i.e. term idx)
            H_ij[idx_i_base + _unique2all_XY_sites_idx[k]] += P_k_by_unique_YZ_sites[idx_i, _unique2all_YZ_sites_idx[k]] * couplings[k]

    return H_ij

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.float64_t, ndim=1] __inner_int16_double(short M, short Kxy, short K,
                            long [:] _unique2all_XY_sites_idx,
                            long [:,:] P_k_by_unique_YZ_sites,
                            long [:] _unique2all_YZ_sites_idx,
                            double [:] couplings,
                            cnp.ndarray[cnp.float64_t, ndim=1] H_ij):
    cdef:
        Py_ssize_t idx_i, k
        long idx_i_base

    for idx_i in prange(M, nogil=True, schedule="static"): # Sampled state idx
        idx_i_base = idx_i * Kxy
        for k in range(K): # Unique coupled state idx (i.e. term idx)
            H_ij[idx_i_base + _unique2all_XY_sites_idx[k]] += P_k_by_unique_YZ_sites[idx_i, _unique2all_YZ_sites_idx[k]] * couplings[k]

    return H_ij

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.float32_t, ndim=1] __inner_int64_float(long M, long Kxy, long K,
                            long [:] _unique2all_XY_sites_idx,
                            long [:,:] P_k_by_unique_YZ_sites,
                            long [:] _unique2all_YZ_sites_idx,
                            float [:] couplings,
                            cnp.ndarray[cnp.float32_t, ndim=1] H_ij):
    cdef:
        Py_ssize_t idx_i, k
        long idx_i_base

    for idx_i in prange(M, nogil=True, schedule="static"): # Sampled state idx
        idx_i_base = idx_i * Kxy
        for k in range(K): # Unique coupled state idx (i.e. term idx)
            H_ij[idx_i_base + _unique2all_XY_sites_idx[k]] += P_k_by_unique_YZ_sites[idx_i, _unique2all_YZ_sites_idx[k]] * couplings[k]

    return H_ij

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.float32_t, ndim=1] __inner_int32_float(int M, int Kxy, int K,
                            long [:] _unique2all_XY_sites_idx,
                            long [:,:] P_k_by_unique_YZ_sites,
                            long [:] _unique2all_YZ_sites_idx,
                            float [:] couplings,
                            cnp.ndarray[cnp.float32_t, ndim=1] H_ij):
    cdef:
        Py_ssize_t idx_i, k
        long idx_i_base

    for idx_i in prange(M, nogil=True, schedule="static"): # Sampled state idx
        idx_i_base = idx_i * Kxy
        for k in range(K): # Unique coupled state idx (i.e. term idx)
            H_ij[idx_i_base + _unique2all_XY_sites_idx[k]] += P_k_by_unique_YZ_sites[idx_i, _unique2all_YZ_sites_idx[k]] * couplings[k]

    return H_ij

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.float32_t, ndim=1] __inner_int16_float(short M, short Kxy, short K,
                            long [:] _unique2all_XY_sites_idx,
                            long [:,:] P_k_by_unique_YZ_sites,
                            long [:] _unique2all_YZ_sites_idx,
                            float [:] couplings,
                            cnp.ndarray[cnp.float32_t, ndim=1] H_ij):
    cdef:
        Py_ssize_t idx_i, k
        long idx_i_base

    for idx_i in prange(M, nogil=True, schedule="static"): # Sampled state idx
        idx_i_base = idx_i * Kxy
        for k in range(K): # Unique coupled state idx (i.e. term idx)
            H_ij[idx_i_base + _unique2all_XY_sites_idx[k]] += P_k_by_unique_YZ_sites[idx_i, _unique2all_YZ_sites_idx[k]] * couplings[k]

    return H_ij

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
def get_Hij_cy(state_i_idx, _unique_XY_sites_idx, _unique2all_XY_sites_idx,
               P_k_by_unique_YZ_sites, _unique2all_YZ_sites_idx,
               couplings):
    
    state_i_idx = state_i_idx.astype(np.long)

    M = len(state_i_idx)
    Kxy = len(_unique_XY_sites_idx)
    K = len(_unique2all_XY_sites_idx)
    couplings = couplings.squeeze()

    H_ij = np.zeros(M * Kxy, dtype=couplings.dtype)

    if couplings.dtype == np.float32:
        if state_i_idx.dtype == np.int16:
            H_ij = __inner_int16_float(M, Kxy, K,
                                _unique2all_XY_sites_idx.astype(np.long),
                                 P_k_by_unique_YZ_sites.astype(np.long),
                                 _unique2all_YZ_sites_idx.astype(np.long),
                                 couplings,
                                 H_ij)

        elif state_i_idx.dtype == np.int32:
            H_ij = __inner_int32_float(M, Kxy, K,
                                _unique2all_XY_sites_idx.astype(np.long),
                                 P_k_by_unique_YZ_sites.astype(np.long),
                                 _unique2all_YZ_sites_idx.astype(np.long),
                                 couplings,
                                 H_ij)

        else:
            H_ij = __inner_int64_float(M, Kxy, K,
                                _unique2all_XY_sites_idx.astype(np.long),
                                 P_k_by_unique_YZ_sites.astype(np.long),
                                 _unique2all_YZ_sites_idx.astype(np.long),
                                 couplings,
                                 H_ij)

    elif couplings.dtype == np.float64:
        if state_i_idx.dtype == np.int16:
            H_ij = __inner_int16_double(M, Kxy, K,
                                _unique2all_XY_sites_idx.astype(np.long),
                                 P_k_by_unique_YZ_sites.astype(np.long),
                                 _unique2all_YZ_sites_idx.astype(np.long),
                                 couplings,
                                 H_ij)

        elif state_i_idx.dtype == np.int32:
            H_ij = __inner_int32_double(M, Kxy, K,
                                _unique2all_XY_sites_idx.astype(np.long),
                                 P_k_by_unique_YZ_sites.astype(np.long),
                                 _unique2all_YZ_sites_idx.astype(np.long),
                                 couplings,
                                 H_ij)

        else:
            H_ij = __inner_int64_double(M, Kxy, K,
                                _unique2all_XY_sites_idx.astype(np.long),
                                 P_k_by_unique_YZ_sites.astype(np.long),
                                 _unique2all_YZ_sites_idx.astype(np.long),
                                 couplings,
                                 H_ij)

    else:
        if state_i_idx.dtype == np.int16:
            H_ij = __inner_int16_longdouble(M, Kxy, K,
                                _unique2all_XY_sites_idx.astype(np.long),
                                 P_k_by_unique_YZ_sites.astype(np.long),
                                 _unique2all_YZ_sites_idx.astype(np.long),
                                 couplings,
                                 H_ij)

        elif state_i_idx.dtype == np.int32:
            H_ij = __inner_int32_longdouble(M, Kxy, K,
                                _unique2all_XY_sites_idx.astype(np.long),
                                 P_k_by_unique_YZ_sites.astype(np.long),
                                 _unique2all_YZ_sites_idx.astype(np.long),
                                 couplings,
                                 H_ij)

        else:
            H_ij = __inner_int64_longdouble(M, Kxy, K,
                                _unique2all_XY_sites_idx.astype(np.long),
                                 P_k_by_unique_YZ_sites.astype(np.long),
                                 _unique2all_YZ_sites_idx.astype(np.long),
                                 couplings,
                                 H_ij)

    return np.asarray(H_ij)


'''
popcount_parity(...)
'''

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cnp.ndarray[char, ndim=2] __popcount_parity_uint8_cy(uint8_t[:, :] arr):
    """
    Iterates over the elements of an 2D array and replaces them by their popcount
    Parameters:
    -----------
    arr: numpy_array, dtype=np.uint8, shape=(m, n)
       The array for which the popcounts should be computed.
    """
    cdef:
        Py_ssize_t i, j, x = arr.shape[0], y = arr.shape[1]
        cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
    for i in prange(x, nogil=True):
        for j in range(y):
            out[i,j] = 1 - 2*(__builtin_popcount(arr[i,j])%2)

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cnp.ndarray[char, ndim=2] __popcount_parity_uint16_cy(uint16_t[:, :] arr):
    """
    Iterates over the elements of an 2D array and replaces them by their popcount
    Parameters:
    -----------
    arr: numpy_array, dtype=np.uint16, shape=(m, n)
       The array for which the popcounts should be computed.
    """
    cdef:
        Py_ssize_t i, j, x = arr.shape[0], y = arr.shape[1]
        cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
    for i in prange(x, nogil=True):
        for j in range(y):
            out[i,j] = 1 - 2*(__builtin_popcount(arr[i,j])%2)

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cnp.ndarray[char, ndim=2] __popcount_parity_uint32_cy(uint32_t[:, :] arr):
    """
    Iterates over the elements of an 2D array and replaces them by their popcount
    Parameters:
    -----------
    arr: numpy_array, dtype=np.uint32, shape=(m, n)
       The array for which the popcounts should be computed.
    """
    cdef:
        Py_ssize_t i, j, x = arr.shape[0], y = arr.shape[1]
        cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
    for i in prange(x, nogil=True):
        for j in range(y):
            out[i,j] = 1 - 2*(__builtin_popcount(arr[i,j])%2)

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cnp.ndarray[char, ndim=2] __popcount_parity_uint64_cy(uint64_t[:, :] arr):
    """
    Iterates over the elements of an 2D array and replaces them by their popcount
    Parameters:
    -----------
    arr: numpy_array, dtype=np.uint64, shape=(m, n)
       The array for which the popcounts should be computed.
    """
    cdef:
        Py_ssize_t i, j, x = arr.shape[0], y = arr.shape[1]
        cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
    for i in prange(x, nogil=True):
        for j in range(y):
            out[i,j] = 1 - 2*(__builtin_popcountll(arr[i,j])%2)

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cnp.ndarray[char, ndim=2] __popcount_parity_int8_cy(int8_t[:, :] arr):
    """
    Iterates over the elements of an 2D array and replaces them by their popcount
    Parameters:
    -----------
    arr: numpy_array, dtype=np.int8, shape=(m, n)
       The array for which the popcounts should be computed.
    """
    cdef:
        Py_ssize_t i, j, x = arr.shape[0], y = arr.shape[1]
        cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
    for i in prange(x, nogil=True):
        for j in range(y):
            out[i,j] = 1 - 2*(__builtin_popcount(arr[i,j])%2)

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cnp.ndarray[char, ndim=2] __popcount_parity_int16_cy(int16_t[:, :] arr):
    """
    Iterates over the elements of an 2D array and replaces them by their popcount
    Parameters:
    -----------
    arr: numpy_array, dtype=np.int16, shape=(m, n)
       The array for which the popcounts should be computed.
    """
    cdef:
        Py_ssize_t i, j, x = arr.shape[0], y = arr.shape[1]
        cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
    for i in prange(x, nogil=True):
        for j in range(y):
            out[i,j] = 1 - 2*(__builtin_popcount(arr[i,j])%2)

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cnp.ndarray[char, ndim=2] __popcount_parity_int32_cy(int32_t[:, :] arr):
    """
    Iterates over the elements of an 2D array and replaces them by their popcount
    Parameters:
    -----------
    arr: numpy_array, dtype=np.int32, shape=(m, n)
       The array for which the popcounts should be computed.
    """
    cdef:
        Py_ssize_t i, j, x = arr.shape[0], y = arr.shape[1]
        cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
    for i in prange(x, nogil=True):
        for j in range(y):
            out[i,j] = 1 - 2*(__builtin_popcount(arr[i,j])%2)

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cnp.ndarray[char, ndim=2] __popcount_parity_int64_cy(int64_t[:, :] arr):
    """
    Iterates over the elements of an 2D array and replaces them by their popcount
    Parameters:
    -----------
    arr: numpy_array, dtype=np.int64, shape=(m, n)
       The array for which the popcounts should be computed.
    """
    cdef:
        Py_ssize_t i, j, x = arr.shape[0], y = arr.shape[1]
        cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
    for i in prange(x, nogil=True):
        for j in range(y):
            out[i,j] = 1 - 2*(__builtin_popcountll(arr[i,j])%2)

    return out

def popcount_parity(arr):
    """
    Computes the parity of popcount of each element of a numpy array.
    http://en.wikipedia.org/wiki/Hamming_weight
    Parameters:
    -----------
    arr: numpy_array
         The array of integers for which the popcounts should be computed.
    """
    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)

    if arr.dtype==np.int8:
        return __popcount_parity_int8_cy(arr)
    if arr.dtype==np.uint8:
        return __popcount_parity_uint8_cy(arr)
    if arr.dtype==np.int16:
        return __popcount_parity_int16_cy(arr)
    elif arr.dtype==np.uint16:
        return __popcount_parity_uint16_cy(arr)
    elif arr.dtype==np.int32:
        return __popcount_parity_int32_cy(arr)
    elif arr.dtype==np.uint32:
        return __popcount_parity_uint32_cy(arr)
    elif arr.dtype==np.int64:
        return __popcount_parity_int64_cy(arr)
    elif arr.dtype==np.uint64:
        return __popcount_parity_uint64_cy(arr)
    else:
        raise TypeError(f"Unsupported array dtype for popcount_parity(...): {arr.dtype}.")
