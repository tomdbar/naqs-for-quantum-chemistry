import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange

#http://nicolas-hug.com/blog/cython_notes

# Build with : python src_cpp/setup.py build_ext --inplace --force

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[int, ndim=2] __make_basis_idxs_cy(int N):
    cdef:
        Py_ssize_t i, j
        cnp.ndarray[int, ndim=1] d = 1 << np.arange(N, dtype=np.int32)
        cnp.ndarray[int, ndim=2] out = np.empty((2**N, N), dtype=np.int32)

    for i in prange(2**N, nogil=True):
        for j in range(N):
            out[i,j] = i & d[j]

    return np.asarray(out)

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[long, ndim=2] __make_basis_idxs_int64_cy(int N):
    cdef:
        Py_ssize_t i, j
        cnp.ndarray[int, ndim=1] d = 1 << np.arange(N, dtype=np.int64)
        cnp.ndarray[int, ndim=2] out = np.empty((2**N, N), dtype=np.int64)

    for i in prange(2**N, nogil=True):
        for j in range(N):
            out[i,j] = i & d[j]

    return np.asarray(out)

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
def make_basis_idxs_cy(N, dtype=np.int32):
    if dtype is np.int64:
        return __make_basis_idxs_int64_cy(N)
    else:
        return __make_basis_idxs_cy(N)


# '''
# popcount(...)
# '''
#
# from libc.stdint cimport uint32_t, uint16_t, uint64_t, int32_t, int16_t, int64_t
# cdef extern int __builtin_popcount(unsigned int) nogil
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef cnp.ndarray[char, ndim=2] __popcount_uint16_cy(uint16_t[:, :] arr):
#     cdef:
#         int i, j
#         int x = arr.shape[0], y = arr.shape[1]
#         cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
#     for i in prange(x, nogil=True):
#         for j in range(y):
#             out[i,j] = 1 - 2*(__builtin_popcount(arr[i,j])%2)
#
#     return out
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef cnp.ndarray[char, ndim=2] __popcount_uint32_cy(uint32_t[:, :] arr):
#     cdef:
#         int i, j
#         int x = arr.shape[0], y = arr.shape[1]
#         cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
#     for i in prange(x, nogil=True):
#         for j in range(y):
#             out[i,j] = 1 - 2*(__builtin_popcount(arr[i,j])%2)
#
#     return out
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef cnp.ndarray[char, ndim=2] __popcount_uint64_cy(uint64_t[:, :] arr):
#     cdef:
#         int i, j
#         int x = arr.shape[0], y = arr.shape[1]
#         cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
#     for i in prange(x, nogil=True):
#         for j in range(y):
#             out[i,j] = 1 - 2*(__builtin_popcount(arr[i,j])%2)
#
#     return out
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef cnp.ndarray[char, ndim=2] __popcount_int16_cy(int16_t[:, :] arr):
#     """
#     Iterates over the elements of an 2D array and replaces them by their popcount
#     Parameters:
#     -----------
#     arr: numpy_array, dtype=np.uint32, shape=(m, n)
#        The array for which the popcounts should be computed.
#     """
#     cdef:
#         int i, j
#         int x = arr.shape[0], y = arr.shape[1]
#         cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
#     for i in prange(x, nogil=True):
#         for j in range(y):
#             out[i,j] = 1 - 2*(__builtin_popcount(arr[i,j])%2)
#
#     return out
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef cnp.ndarray[char, ndim=2] __popcount_int32_cy(int32_t[:, :] arr):
#     """
#     Iterates over the elements of an 2D array and replaces them by their popcount
#     Parameters:
#     -----------
#     arr: numpy_array, dtype=np.uint32, shape=(m, n)
#        The array for which the popcounts should be computed.
#     """
#     cdef:
#         int i, j
#         int x = arr.shape[0], y = arr.shape[1]
#         cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
#     for i in prange(x, nogil=True):
#         for j in range(y):
#             out[i,j] = 1 - 2*(__builtin_popcount(arr[i,j])%2)
#
#     return out
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef cnp.ndarray[char, ndim=2] __popcount_int64_cy(int64_t[:, :] arr):
#     """
#     Iterates over the elements of an 2D array and replaces them by their popcount
#     Parameters:
#     -----------
#     arr: numpy_array, dtype=np.uint32, shape=(m, n)
#        The array for which the popcounts should be computed.
#     """
#     cdef:
#         int i, j
#         int x = arr.shape[0], y = arr.shape[1]
#         cnp.ndarray[char, ndim=2] out = np.zeros((x,y), dtype=np.int8)
#     for i in prange(x, nogil=True):
#         for j in range(y):
#             out[i,j] = 1 - 2*(__builtin_popcount(arr[i,j])%2)
#
#     return out
#
# def check_physicality(arr, N_up, N_alpha=None):
#     """
#     Computes the parity of popcount of each element of a numpy array.
#     http://en.wikipedia.org/wiki/Hamming_weight
#     Parameters:
#     -----------
#     arr: numpy_array
#          The array of integers for which the popcounts should be computed.
#     """
#     if len(arr.shape) == 1:
#         arr = arr.reshape(-1, 1)
#
#     __popcount_check_cy = __popcount_int16_cy(arr, N_up)
#
#
#
#     # if arr.dtype==np.int16:
#     #     return __popcount_int16_cy(arr)
#     # elif arr.dtype==np.uint16:
#     #     return __popcount_parity_uint16_cy(arr)
#     # elif arr.dtype==np.int32:
#     #     return __popcount_parity_int32_cy(arr)
#     # elif arr.dtype==np.uint32:
#     #     return __popcount_parity_uint32_cy(arr)
#     # elif arr.dtype==np.int64:
#     #     return __popcount_parity_int64_cy(arr)
#     # elif arr.dtype==np.uint64:
#     #     return __popcount_parity_uint64_cy(arr)
#     else:
#         raise TypeError(f"Unsupported array dtype for popcount_parity(...): {arr.dtype}.")
