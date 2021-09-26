import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange

import torch
import time

#http://nicolas-hug.com/blog/cython_notes

# Build with : python src_cpp/setup.py build_ext --inplace --force

def __type_mv(m, v):
    '''Type check/set a real matrix (m) and complex vector (v).

    We will either have:
        1. n_bit = 32 --> m is np.float32 and v is np.complex 64
        2. n_bit = 64 --> m is np.float64 and v is np.complex 128'''

    if m.dtype is np.dtype(np.float64):
        n_bit = 64
    elif m.dtype is np.dtype(np.float32):
        n_bit = 32
    else:
        raise Exception("m must have dtype of np.float32 or np.float64.")

    if not np.iscomplexobj(v):
        if n_bit == 32:
            v = v.astype(np.complex64)
        else:
            v = v.astype(np.complex128)
    else:
        if (v.dtype is np.dtype(np.complex128)) and n_bit == 32:
            n_bit = 64
            m = m.astype(np.float64)
        elif (v.dtype is np.dtype(np.complex64)) and n_bit == 64:
            v = v.astype(np.complex128)

    idx_type = m.indices.dtype

    return m, v, n_bit, idx_type

###############################################
# Sparse matrix, dense vector multiplication. #
###############################################

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
def sparse_dense_mv(m, v, par=None):
    t = time.time()
    if par is None:
        # Approximately where the setup overhead of parallelism balances with the speed increase,
        # in my experience, on my machines etc...
        par = (m.shape[0] > 500)

    m, v, n_bit, idx_type = __type_mv(m, v)

    out = np.zeros(m.shape[1], dtype=v.dtype)

    if n_bit == 64:
        if par:
            if idx_type == np.int32:
                out =  __sparse_dense_par_mv_64bitElem_32bitIdx(m.data, m.indices, m.indptr, v, m.shape[1], out)
            else:
                out =  __sparse_dense_par_mv_64bitElem_64bitIdx(m.data, m.indices, m.indptr, v, m.shape[1], out)
        else:
            if idx_type == np.int32:
                out =  __sparse_dense_mv_64bitElem_32bitIdx(m.data, m.indices, m.indptr, v, m.shape[1], out)
            else:
                out =  __sparse_dense_mv_64bitElem_64bitIdx(m.data, m.indices, m.indptr, v, m.shape[1], out)
    else:
        if par:
            if idx_type == np.int32:
                out =  __sparse_dense_par_mv_32bitElem_32bitIdx(m.data, m.indices, m.indptr, v, m.shape[1], out)
            else:
                out =  __sparse_dense_par_mv_32bitElem_64bitIdx(m.data, m.indices, m.indptr, v, m.shape[1], out)
        else:
            if idx_type == np.int32:
                out =  __sparse_dense_mv_32bitElem_32bitIdx(m.data, m.indices, m.indptr, v, m.shape[1], out)
            else:
                out =  __sparse_dense_mv_32bitElem_64bitIdx(m.data, m.indices, m.indptr, v, m.shape[1], out)

    return np.asarray(out)

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.complex128_t, ndim=1] __sparse_dense_par_mv_64bitElem_32bitIdx(double [:] data,
                                                                       int [:] indices,
                                                                       int [:] indptr,
                                                                       cnp.ndarray[cnp.complex128_t, ndim=1] v,
                                                                       Py_ssize_t dim,
                                                                       cnp.ndarray[cnp.complex128_t, ndim=1] out):
    cdef Py_ssize_t i, j

    with nogil:
        for j in prange(dim, schedule="dynamic"):
            for i in range(indptr[j+1]-indptr[j]):
                out[j] = out[j] + data[indptr[j]+i] * v[indices[indptr[j]+i]]

    return out

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.complex64_t, ndim=1] __sparse_dense_par_mv_32bitElem_32bitIdx(float [:] data,
                                                                      int [:] indices,
                                                                      int [:] indptr,
                                                                      cnp.ndarray[cnp.complex64_t, ndim=1] v,
                                                                      Py_ssize_t dim,
                                                                      cnp.ndarray[cnp.complex64_t, ndim=1] out):
    cdef Py_ssize_t i, j

    with nogil:
        for j in prange(dim, schedule="dynamic"):
            for i in range(indptr[j+1]-indptr[j]):
                out[j] = out[j] + data[indptr[j]+i] * v[indices[indptr[j]+i]]

    return out

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.complex128_t, ndim=1] __sparse_dense_mv_64bitElem_32bitIdx(double [:] data,
                                                                   int [:] indices,
                                                                   int [:] indptr,
                                                                   cnp.ndarray[cnp.complex128_t, ndim=1] v,
                                                                   Py_ssize_t dim,
                                                                   cnp.ndarray[cnp.complex128_t, ndim=1] out):
    '''Input is assumed to be in csr form.
        --> data[indptr[i]:indptr[i+1]] gives the matrix element values for i-th row.
        --> indices[indptr[i]:indptr[i+1]] gives the column values for i-th row.
    '''
    cdef Py_ssize_t i, j

    for i in range(dim):
        # For i-th row.
        for j in range(indptr[i+1]-indptr[i]):
            # For j=1st, 2nd, ... non-zero element in that row,
            #   column value is indices[indptr[i]+j]
            out[i] = out[i] + data[indptr[i]+j] * v[indices[indptr[i]+j]]

    return out

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.complex64_t, ndim=1] __sparse_dense_mv_32bitElem_32bitIdx(float [:] data,
                                                                  int [:] indices,
                                                                  int [:] indptr,
                                                                  cnp.ndarray[cnp.complex64_t, ndim=1] v,
                                                                  Py_ssize_t dim,
                                                                  cnp.ndarray[cnp.complex64_t, ndim=1] out):
    '''Input is assumed to be in csr form.
        --> data[indptr[i]:indptr[i+1]] gives the matrix element values for i-th row.
        --> indices[indptr[i]:indptr[i+1]] gives the column values for i-th row.
    '''
    cdef Py_ssize_t i, j

    for i in range(dim):
        # For i-th row.
        for j in range(indptr[i+1]-indptr[i]):
            # For j=1st, 2nd, ... non-zero element in that row,
            #   column value is indices[indptr[i]+j]
            out[i] = out[i] + data[indptr[i]+j] * v[indices[indptr[i]+j]]

    return out

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.complex128_t, ndim=1] __sparse_dense_par_mv_64bitElem_64bitIdx(double [:] data,
                                                                       long [:] indices,
                                                                       long [:] indptr,
                                                                       cnp.ndarray[cnp.complex128_t, ndim=1] v,
                                                                       Py_ssize_t dim,
                                                                       cnp.ndarray[cnp.complex128_t, ndim=1] out):
    cdef Py_ssize_t i, j

    with nogil:
        for j in prange(dim, schedule="dynamic"):
            for i in range(indptr[j+1]-indptr[j]):
                out[j] = out[j] + data[indptr[j]+i] * v[indices[indptr[j]+i]]

    return out

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.complex64_t, ndim=1] __sparse_dense_par_mv_32bitElem_64bitIdx(float [:] data,
                                                                      long [:] indices,
                                                                      long [:] indptr,
                                                                      cnp.ndarray[cnp.complex64_t, ndim=1] v,
                                                                      Py_ssize_t dim,
                                                                      cnp.ndarray[cnp.complex64_t, ndim=1] out):
    cdef Py_ssize_t i, j

    with nogil:
        for j in prange(dim, schedule="dynamic"):
            for i in range(indptr[j+1]-indptr[j]):
                out[j] = out[j] + data[indptr[j]+i] * v[indices[indptr[j]+i]]

    return out

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.complex128_t, ndim=1] __sparse_dense_mv_64bitElem_64bitIdx(double [:] data,
                                                                   long [:] indices,
                                                                   long [:] indptr,
                                                                   cnp.ndarray[cnp.complex128_t, ndim=1] v,
                                                                   Py_ssize_t dim,
                                                                   cnp.ndarray[cnp.complex128_t, ndim=1] out):
    '''Input is assumed to be in csr form.
        --> data[indptr[i]:indptr[i+1]] gives the matrix element values for i-th row.
        --> indices[indptr[i]:indptr[i+1]] gives the column values for i-th row.
    '''
    cdef Py_ssize_t i, j

    for i in range(dim):
        # For i-th row.
        for j in range(indptr[i+1]-indptr[i]):
            # For j=1st, 2nd, ... non-zero element in that row,
            #   column value is indices[indptr[i]+j]
            out[i] = out[i] + data[indptr[i]+j] * v[indices[indptr[i]+j]]

    return out

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.complex64_t, ndim=1] __sparse_dense_mv_32bitElem_64bitIdx(float [:] data,
                                                                  long [:] indices,
                                                                  long [:] indptr,
                                                                  cnp.ndarray[cnp.complex64_t, ndim=1] v,
                                                                  Py_ssize_t dim,
                                                                  cnp.ndarray[cnp.complex64_t, ndim=1] out):
    '''Input is assumed to be in csr form.
        --> data[indptr[i]:indptr[i+1]] gives the matrix element values for i-th row.
        --> indices[indptr[i]:indptr[i+1]] gives the column values for i-th row.
    '''
    cdef Py_ssize_t i, j

    for i in range(dim):
        # For i-th row.
        for j in range(indptr[i+1]-indptr[i]):
            # For j=1st, 2nd, ... non-zero element in that row,
            #   column value is indices[indptr[i]+j]
            out[i] = out[i] + data[indptr[i]+j] * v[indices[indptr[i]+j]]

    return out

################################################
# Sparse matrix, sparse vector multiplication. #
################################################

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
def sparse_sparse_mv(m, v, v_idxs, assume_sorted=False):
    t = time.time()

    m, v, n_bit, idx_type = __type_mv(m, v)
    v_idxs = v_idxs.astype(idx_type)

    out = np.zeros(v_idxs.shape[0], dtype=v.dtype)

    if not assume_sorted:
        sort_args = np.argsort(v_idxs)
        v_idxs = v_idxs[sort_args]
        v = v[sort_args]

        unsort_args = np.zeros_like(sort_args)
        unsort_args[sort_args] = np.arange(len(sort_args))

    if n_bit == 64:
        if idx_type == np.int32:
            out =  __sparse_sparse_par_mv_64bitElem_32bitIdx(m.data, m.indices, m.indptr, v, v_idxs, len(v), out)
        else:
            out =  __sparse_sparse_par_mv_64bitElem_64bitIdx(m.data, m.indices, m.indptr, v, v_idxs, len(v), out)
    else:
        if idx_type == np.int32:
            out =  __sparse_sparse_par_mv_32bitElem_32bitIdx(m.data, m.indices, m.indptr, v, v_idxs, len(v), out)
        else:
            out =  __sparse_sparse_par_mv_32bitElem_64bitIdx(m.data, m.indices, m.indptr, v, v_idxs, len(v), out)

    if not assume_sorted:
        return np.asarray(out)[unsort_args]
    else:
        return np.asarray(out)


@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.complex64_t, ndim=1] __sparse_sparse_par_mv_32bitElem_32bitIdx(float [:] data,
                                                                      int [:] indices,
                                                                      int [:] indptr,
                                                                      cnp.ndarray[cnp.complex64_t, ndim=1] v,
                                                                      int [:] v_idxs,
                                                                      Py_ssize_t dim,
                                                                      cnp.ndarray[cnp.complex64_t, ndim=1] out):
    cdef Py_ssize_t i_m, i_m_max, i_v, j, k

    with nogil:
        for k in prange(dim, schedule="static"):
            j = v_idxs[k]

            # i_m, i_m_max, i_v = indptr[j], indptr[j+1], 0
            i_m, i_m_max, i_v = 0, indptr[j+1]-indptr[j], 0

            while (i_m < i_m_max) and (i_v < dim):
                if indices[indptr[j]+i_m] < v_idxs[i_v]:
                    i_m = i_m + 1
                elif  v_idxs[i_v] < indices[indptr[j]+i_m]:
                    i_v = i_v + 1
                else:
                    out[k] = out[k] + data[indptr[j]+i_m] * v[i_v]
                    i_m = i_m + 1
                    i_v = i_v + 1

    return out

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.complex128_t, ndim=1] __sparse_sparse_par_mv_64bitElem_64bitIdx(double [:] data,
                                                                       long [:] indices,
                                                                       long [:] indptr,
                                                                       cnp.ndarray[cnp.complex128_t, ndim=1] v,
                                                                       long [:] v_idxs,
                                                                       Py_ssize_t dim,
                                                                       cnp.ndarray[cnp.complex128_t, ndim=1] out):
    cdef Py_ssize_t i_m, i_m_max, i_v, j, k

    with nogil:
        for k in prange(dim, schedule="static"):
            j = v_idxs[k]

            # i_m, i_m_max, i_v = indptr[j], indptr[j+1], 0
            i_m, i_m_max, i_v = 0, indptr[j+1]-indptr[j], 0

            while (i_m < i_m_max) and (i_v < dim):
                if indices[indptr[j]+i_m] < v_idxs[i_v]:
                    i_m = i_m + 1
                elif  v_idxs[i_v] < indices[indptr[j]+i_m]:
                    i_v = i_v + 1
                else:
                    out[k] = out[k] + data[indptr[j]+i_m] * v[i_v]
                    i_m = i_m + 1
                    i_v = i_v + 1

    return out

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.complex64_t, ndim=1] __sparse_sparse_par_mv_32bitElem_64bitIdx(float [:] data,
                                                                      long [:] indices,
                                                                      long [:] indptr,
                                                                      cnp.ndarray[cnp.complex64_t, ndim=1] v,
                                                                      long [:] v_idxs,
                                                                      Py_ssize_t dim,
                                                                      cnp.ndarray[cnp.complex64_t, ndim=1] out):
    cdef Py_ssize_t i_m, i_m_max, i_v, j, k

    with nogil:
        for k in prange(dim, schedule="static"):
            j = v_idxs[k]

            # i_m, i_m_max, i_v = indptr[j], indptr[j+1], 0
            i_m, i_m_max, i_v = 0, indptr[j+1]-indptr[j], 0

            while (i_m < i_m_max) and (i_v < dim):
                if indices[indptr[j]+i_m] < v_idxs[i_v]:
                    i_m = i_m + 1
                elif  v_idxs[i_v] < indices[indptr[j]+i_m]:
                    i_v = i_v + 1
                else:
                    out[k] = out[k] + data[indptr[j]+i_m] * v[i_v]
                    i_m = i_m + 1
                    i_v = i_v + 1

    return out

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef cnp.ndarray[cnp.complex64_t, ndim=1] __sparse_sparse_par_mv_64bitElem_32bitIdx(double [:] data,
                                                                      int [:] indices,
                                                                      int [:] indptr,
                                                                      cnp.ndarray[cnp.complex128_t, ndim=1] v,
                                                                      int [:] v_idxs,
                                                                      Py_ssize_t dim,
                                                                      cnp.ndarray[cnp.complex128_t, ndim=1] out):
    cdef Py_ssize_t i_m, i_m_max, i_v, j, k

    with nogil:
        for k in prange(dim, schedule="static"):
            j = v_idxs[k]

            # i_m, i_m_max, i_v = indptr[j], indptr[j+1], 0
            i_m, i_m_max, i_v = 0, indptr[j+1]-indptr[j], 0

            while (i_m < i_m_max) and (i_v < dim):
                if indices[indptr[j]+i_m] < v_idxs[i_v]:
                    i_m = i_m + 1
                elif  v_idxs[i_v] < indices[indptr[j]+i_m]:
                    i_v = i_v + 1
                else:
                    out[k] = out[k] + data[indptr[j]+i_m] * v[i_v]
                    i_m = i_m + 1
                    i_v = i_v + 1

    return out


#####################################################
# Sparse matrix, dense < vector | matrix | vector > #
#####################################################

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
def sparse_dense_exp_op(m, v, par=None):

    m, v, n_bit = __type_mv(m, v)

    if n_bit==32:
        return __sparse_dense_exp_op_32bit(m.data, m.indices, m.indptr, v, m.shape[1])
    else:
        return __sparse_dense_exp_op_64bit(m.data, m.indices, m.indptr, v, m.shape[1])

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef complex __sparse_dense_exp_op_64bit(double [:] data,
                                         int [:] indices,
                                         int [:] indptr,
                                         cnp.ndarray[cnp.complex128_t, ndim=1] v,
                                         Py_ssize_t dim):
    cdef:
        Py_ssize_t i, j
        complex out_i
        complex out = 0

    with nogil:
        for i in prange(dim, schedule='dynamic'):
            # For i-th row.
            out_i = 0
            for j in range(indptr[j+1]-indptr[j]):
                # For j=1st, 2nd, ... non-zero element in that row,
                #   column value is indices[indptr[i]+j]
                out_i = out_i + data[indptr[i]+j] * v[indices[indptr[i]+j]]

            out = out + v[i].conjugate()*out_i

    return out

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cdef complex __sparse_dense_exp_op_32bit(float [:] data,
                                         int [:] indices,
                                         int [:] indptr,
                                         cnp.ndarray[cnp.complex64_t, ndim=1] v,
                                         Py_ssize_t dim):
    cdef:
        Py_ssize_t i, j
        complex out_i
        complex out = 0

    with nogil:
        for i in prange(dim, schedule='dynamic'):
            # For i-th row.
            out_i = 0
            for j in range(indptr[j+1]-indptr[j]):
                # For j=1st, 2nd, ... non-zero element in that row,
                #   column value is indices[indptr[i]+j]
                out_i = out_i + data[indptr[i]+j] * v[indices[indptr[i]+j]]

            out = out + v[i].conjugate()*out_i

    return out