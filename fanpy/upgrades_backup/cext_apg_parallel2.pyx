import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython
from libc.stdlib cimport malloc, free


# DEF npairs = 100
# DEF npairs_squared = 10000
# DEF nparams = 1000000

# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# cdef long[npairs] col_inds_apg(long[:] orbpairs, int nspin) nogil:
#     cdef long col_inds[npairs]
#     cdef Py_ssize_t i, j, k
#     for k in range(len(orbpairs) // 2):
#         i = orbpairs[2 * k]
#         j = orbpairs[2 * k + 1]
#         col_inds[k] = nspin * i - i * (i + 1) // 2 + (j - i - 1)
#     return col_inds

# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# cdef long[:] col_inds_apig(long[:] orbpairs):
#     cdef long col_inds[npairs]
#     cdef Py_ssize_t i, j, k
#     for k in range(len(orbpairs) // 2):
#         col_inds[k] = orbpairs[2 * k]
#     return col_inds

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double ryser(double[:, :] submatrix, long[:] orbpairs, int nspin) nogil:
    cdef int n = submatrix.shape[0]

    # cdef long col_inds[npairs]
    cdef long *col_inds = <long *> malloc(sizeof(long) * n)
    cdef int i, j, k
    for k in range(n):
        i = orbpairs[2 * k]
        j = orbpairs[2 * k + 1]
        col_inds[k] = nspin * i - i * (i + 1) // 2 + (j - i - 1)

    cdef double rowsum, rowsumprod
    cdef double perm = 0.0
    cdef unsigned long exp = 1 << n
    cdef unsigned long x
    cdef unsigned long y, z, z_matrix
    for x in range(exp):
        rowsumprod = 1.0
        for y in range(n):
            rowsum = 0.0
            for z in range(n):
                z_matrix = col_inds[z]
                if x & (1 << z) != 0:
                    rowsum += submatrix[y, z_matrix]
            rowsumprod *= rowsum
        perm += rowsumprod * bitparity(x)
    if n % 2 == 1:
        perm *= -1

    free(col_inds)
    return perm

# FIXME: nogil results in bad update of deriv
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void ryser_deriv(
    double[:, :] submatrix,
    long[:] orbpairs,
    long sign,
    int nspin,
    double* deriv,
) nogil:
    cdef int n = submatrix.shape[0]
    cdef int num_orbpairs = submatrix.shape[1]

    # cdef long col_inds[npairs]
    cdef long *col_inds = <long *> malloc(sizeof(long) * n)
    cdef int i, j, k
    for k in range(n):
        i = orbpairs[2 * k]
        j = orbpairs[2 * k + 1]
        col_inds[k] = nspin * i - i * (i + 1) // 2 + (j - i - 1)

    cdef double rowsum
    # cdef double perm = 0.0
    cdef unsigned long exp = 1 << n
    cdef unsigned long x
    cdef unsigned int y, z, y_matrix, z_matrix

    # cdef double rowsum_vec[npairs]
    cdef double *rowsum_vec = <double *> malloc(sizeof(double) * n)
    # cdef double rowsumprod[npairs_squared]
    cdef double *rowsumprod = <double *> malloc(sizeof(double) * n ** 2)
    cdef double temp = 1.0
    # cdef double rowsumprod_vec[npairs_squared]
    cdef double *rowsumprod_vec = <double *> malloc(sizeof(double) * n ** 2)
    # cdef double perm_vec[npairs_squared]
    cdef double *perm_vec = <double *> malloc(sizeof(double) * n ** 2)
    cdef int p, q

    for p in range(n):
        for q in range(n):
            perm_vec[p * n + q] = 0.0
    for x in range(exp):
        # rowsumprod = 1.0
        for p in range(n):
            for q in range(n):
                rowsumprod[p * n + q] = 1.0
                rowsumprod_vec[p * n + q] = 1.0
        for y in range(n):
            rowsum = 0.0
            for q in range(n):
                rowsum_vec[q] = 0.0
            for z in range(n):
                z_matrix = col_inds[z]
                # if x includes z
                if x & (1 << z) != 0:
                    rowsum += submatrix[y, z_matrix]
                    rowsum_vec[z] = submatrix[y, z_matrix]
            # rowsumprod *= rowsum
            for q in range(n):
                # fixe: can't handle zeros
                # rowsumprod[q] *= rowsum - rowsum_vec[q]
                rowsumprod_vec[y * n + q] = rowsum - rowsum_vec[q]

        # create array such that each element (i, j) is a product of elements in column j except
        # for element at row i
        for q in range(n):
            temp = 1.0
            for p in range(n):
                rowsumprod[p * n + q] *= temp
                temp *= rowsumprod_vec[p * n + q]
            temp = 1.0
            for p in range(n - 1, -1, -1):
                rowsumprod[p * n + q] *= temp
                temp *= rowsumprod_vec[p * n + q]
        # perm += rowsumprod * bitparity(x)
        for p in range(n):
            for q in range(n):
                # if x excludes q
                # if x & (1 << q) == 0 and rowsumprod_vec[p * n + q] != 0:
                if x & (1 << q) == 0:
                    perm_vec[p * n + q] += rowsumprod[p * n + q] * (-bitparity(x))
    if n % 2 == 1:
        # perm *= -1
        for p in range(n):
            for q in range(n):
                perm_vec[p * n + q] *= -1

    for p in range(n):
        for q in range(n):
            # deriv[p * num_orbpairs + col_inds[q]] += sign * perm_vec[p * n + q]
            deriv[p * n + q] += sign * perm_vec[p * n + q]

    free(col_inds)
    free(rowsum_vec)
    free(rowsumprod)
    free(rowsumprod_vec)
    free(perm_vec)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef long countbits(unsigned long n) nogil:
    cdef unsigned long q = n
    q = (q & <unsigned long>0x5555555555555555) + ((q & <unsigned long>0xAAAAAAAAAAAAAAAA) >> 1)
    q = (q & <unsigned long>0x3333333333333333) + ((q & <unsigned long>0xCCCCCCCCCCCCCCCC) >> 2)
    q = (q & <unsigned long>0x0F0F0F0F0F0F0F0F) + ((q & <unsigned long>0xF0F0F0F0F0F0F0F0) >> 4)
    q = (q & <unsigned long>0x00FF00FF00FF00FF) + ((q & <unsigned long>0xFF00FF00FF00FF00) >> 8)
    q = (q & <unsigned long>0x0000FFFF0000FFFF) + ((q & <unsigned long>0xFFFF0000FFFF0000) >> 16)
    # This last & isn't strictly necessary.
    q = (q & <unsigned long>0x00000000FFFFFFFF) + ((q & <unsigned long>0xFFFFFFFF00000000) >> 32)
    return q;

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef long bitparity (unsigned long n) nogil:
    return 1 - (countbits(n) & 1)*2

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _olp_internal(list orbpairs_sign, double[:, :] params, long nspin):
    cdef long[:, :] orbpairs_sign_view = np.array(orbpairs_sign)
    cdef int nelec = 2 * params.shape[0]

    cdef double output = 0.0
    cdef Py_ssize_t i
    for i in prange(orbpairs_sign_view.shape[0], nogil=True, schedule='static'):
    # for i in range(orbpairs_sign_view.shape[0]):
        output += (
            orbpairs_sign_view[i, nelec] *
            # ryser(params, col_inds_apg(orbpairs_sign_view[i, :nelec], nspin))
            ryser(params, orbpairs_sign_view[i, :nelec], nspin)
        )
    return output


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _olp_deriv_internal(list orbpairs_sign, double[:, :] params, long nspin):
    cdef long[:, :] orbpairs_sign_view = np.array(orbpairs_sign)
    cdef int npair = params.shape[0]
    cdef int nelec = 2 * params.shape[0]
    cdef int nparams = params.size
    cdef int num_orbpairs = params.shape[1]

    deriv = np.zeros(params.size)
    cdef double[:] deriv_view = deriv
    # deriv = <double*>malloc(sizeof(double)*nparams)

    cdef int i, j, k
    cdef Py_ssize_t p, q
    cdef int col_ind
    for i in prange(orbpairs_sign_view.shape[0], nogil=True, schedule='static'):
        deriv_local = <double*>malloc(sizeof(double)*npair**2)
        for p in range(npair ** 2):
            deriv_local[p] = 0
        ryser_deriv(
            params,
            orbpairs_sign_view[i, :nelec],
            orbpairs_sign_view[i, nelec],
            nspin,
            deriv_local
        )
        with gil:
            for p in range(npair):
                for q in range(npair):
                    j = orbpairs_sign_view[i, 2 * q]
                    k = orbpairs_sign_view[i, 2 * q + 1]
                    col_ind = nspin * j - j * (j + 1) // 2 + (k - j - 1)
                    deriv_view[p * num_orbpairs + col_ind] += deriv_local[p * npair + q]
        free(deriv_local)
    return deriv
