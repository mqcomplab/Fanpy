import numpy as np
cimport numpy as np

cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef long[:] get_col_inds(long[:, ::1] orbpairs, int nspin):
    cdef num_orbpairs = orbpairs.shape[0]
    cdef int i, j, k
    cdef long[:] output = np.zeros(num_orbpairs, dtype=int)
    for k in range(num_orbpairs):
        i = orbpairs[k, 0]
        j = orbpairs[k, 0]
        # col_ind = (iK - i(i+1)/2) + (j - i)
        output[k] = nspin * i - i * (i + 1) // 2 + (j - i - 1)
    return output
