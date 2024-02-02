import numpy as np
cimport numpy as np

cimport cython
from permanent.permanent import permanent
from cpython cimport bool


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray[np.int32_t, ndim=1] get_col_inds(list orbpairs, int nspin):
    cdef Py_ssize_t num_orbpairs = (len(orbpairs) - 1) / 2
    cdef Py_ssize_t i, j, k

    output = np.zeros(num_orbpairs, dtype=np.int32)
    cdef int[:] output_view = output
    # output = []

    for k in range(num_orbpairs):
        i = orbpairs[2 * k]
        j = orbpairs[2 * k + 1]
        # col_ind = (iK - i(i+1)/2) + (j - i)
        output_view[k] = nspin * i - i * (i + 1) // 2 + (j - i - 1)
        # output.append(nspin * i - i * (i + 1) // 2 + (j - i - 1))
    return output


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _olp_internal(
    list orbpair_generator,
    np.ndarray[double, ndim=2] params,
    dict dict_orbpair_ind,
    # int nspin,
):
    cdef tuple orbpairs_sign, orbpairs
    cdef int sign, i, n
    cdef (int, int) orbp
    cdef np.ndarray[np.int32_t, ndim=1] col_inds = np.zeros(
        (len(orbpair_generator[0]) - 1) / 2, dtype=np.int32
    )
    val = 0
    for orbpairs_sign in orbpair_generator:
        n = len(orbpairs_sign)
        orbpairs = orbpairs_sign[:n-1]
        sign = orbpairs_sign[n-1]
        if len(orbpairs) == 0:
            continue

        for i in range((n - 1) / 2):
            orbp = (orbpairs[2 * i], orbpairs[2 * i + 1])
            col_inds[i] = dict_orbpair_ind[orbp]
        # col_inds = np.array([dict_orbpair_ind[orbp] for orbp in orbpairs], dtype=int)
        # FIXME: converting all orbpairs is slow for some reason
        # col_inds = get_col_inds(orbpairs, nspin)
        val += sign * permanent(params[:, col_inds])

    return val


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _olp_deriv_internal(
    list orbpair_generator,
    np.ndarray[double, ndim=2] params,
    dict dict_orbpair_ind,
):
    cdef tuple orbpairs_sign, orbpairs
    cdef int num_orbpairs, sign, i, n, j, deriv
    cdef (int, int) orbp
    cdef np.ndarray[np.int32_t, ndim=1] col_inds = np.zeros(
        (len(orbpair_generator[0]) - 1) / 2, dtype=np.int32
    )

    cdef np.ndarray[np.int32_t, ndim=1] row_inds = np.arange(params.shape[0], dtype=np.int32)

    cdef int col_removed, row_removed
    cdef np.ndarray[np.int32_t, ndim=1] row_inds_trunc, col_inds_trunc

    num_orbpairs = params.shape[1]

    vals = np.zeros(params.size)
    cdef double[:] vals_view = vals
    for orbpairs_sign in orbpair_generator:
        n = len(orbpairs_sign) - 1
        orbpairs = orbpairs_sign[:n]
        sign = orbpairs_sign[n]

        if n == 0:
            continue
        elif n == 1:
            vals_view[col_inds[0]] += sign
            continue

        for i in range(n / 2):
            orbp = (orbpairs[2 * i], orbpairs[2 * i + 1])
            col_inds[i] = dict_orbpair_ind[orbp]

        for col_removed in col_inds:
            col_inds_trunc = col_inds[col_inds != col_removed]
            for row_removed in row_inds:
                deriv = row_removed * num_orbpairs + col_removed
                # vals_view[deriv] += sign * self.compute_permanent(col_inds, deriv=deriv)

                # cut out rows and columns that corresponds to the element with which the permanent
                # is derivatized
                row_inds_trunc = row_inds[row_inds != row_removed]
                vals_view[deriv] += sign * permanent(
                    params[row_inds_trunc[:, None], col_inds_trunc[None, :]]
                )

    return vals


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _olp_deriv_internal_ap1rog(
    np.ndarray[double, ndim=2] params,
    np.ndarray[long, ndim=1] col_inds,
    np.ndarray[long, ndim=1] row_inds,
):
    vals = np.zeros(params.size)
    cdef double[:] vals_view = vals
    cdef int num_cols = params.shape[1]
    cdef int col_removed, row_removed, deriv
    cdef np.ndarray[long, ndim=1] row_inds_trunc, col_inds_trunc

    for col_removed in col_inds:
        col_inds_trunc = col_inds[col_inds != col_removed]
        for row_removed in row_inds:
            deriv = row_removed * num_cols + col_removed
            # vals_view[deriv] += sign * self.compute_permanent(col_inds, deriv=deriv)

            # cut out rows and columns that corresponds to the element with which the permanent
            # is derivatized
            row_inds_trunc = row_inds[row_inds != row_removed]
            if row_inds_trunc.size == col_inds_trunc.size == 0:
                vals_view[deriv] += 1
            else:
                vals_view[deriv] += permanent(
                    params[row_inds_trunc[:, None], col_inds_trunc[None, :]]
                )

    return vals


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef list generate_complete_pmatch(tuple indices, int sign=1):
    cdef Py_ssize_t n, n_inner, i
    cdef list output, subsets_pair
    cdef tuple scheme_inner_sign, scheme
    cdef int inner_sign

    n = len(indices)
    if n % 2 == 1 or n < 2:
        return []
    elif n == 2:
        return [(indices[0], indices[1], sign)]
    else:
        output = []
        # smaller subset (all pairs without the last two indices)
        subsets_pairs = generate_complete_pmatch(indices[:n-2], sign=sign)
        for scheme_inner_sign in subsets_pairs:
            n_inner = len(scheme_inner_sign)
            scheme = scheme_inner_sign[:n_inner-1]
            inner_sign = scheme_inner_sign[n_inner-1]
            # add in the last two indices
            output += [scheme + indices[n-2:] + (inner_sign,)]
            # starting from the last
            for i in reversed(range(n // 2 - 1)):
                # replace ith pair in the scheme with last pair
                output += [
                    scheme[: 2 * i]
                    + (scheme[2 * i], indices[n-2], scheme[2 * i + 1], indices[n-1])
                    + scheme[2 * (i + 1) :]
                    + (-inner_sign,)
                ]
                output += [
                    scheme[: 2 * i]
                    + (scheme[2 * i], indices[n-1], scheme[2 * i + 1], indices[n-2])
                    + scheme[2 * (i + 1) :]
                    + (inner_sign,)
                ]
        return output
# [[k for j in i[0] for k in j] + [i[-1]] for i in generate_complete_pmatch(range(2*i))]


# def generate_general_pmatch(
#     np.ndarray[long, ndim=1] indices,
#     connectivity_matrix
# ):
#     cdef Py_ssize_t n, j, sign, n_inner, inner_sign
#     cdef long ind_one, ind_two
#     # cdef np.uint8_t[:] mask_bool
#     cdef np.ndarray[long, ndim=1] mask_ind
#     cdef list output, scheme_inner_sign, scheme
#     n = indices.size
#     if n == 2:
#         return [[indices[0], indices[1], 1]]
#     elif n > 2:
#         output = []
#         ind_one = indices[0]
#         for j in np.where(connectivity_matrix[0, 1:])[0]:
#             if j % 2 == 0:
#                 sign = 1
#             else:
#                 sign = -1
#             j += 1
#             ind_two = indices[j]
#             # filter out indices that are not used
#             mask_bool = np.logical_not(np.isin(indices, [ind_one, ind_two])).astype(np.uint8)
#             mask_ind = np.where(mask_bool)[0]
#             mask_ind = mask_ind[mask_ind > 0]

#             for scheme_inner_sign in generate_general_pmatch(
#                 indices[mask_ind], connectivity_matrix[mask_ind[:, None], mask_ind[None, :]]
#             ):
#                 n_inner = len(scheme_inner_sign)
#                 scheme = scheme_inner_sign[:n_inner-1]
#                 inner_sign = scheme_inner_sign[n_inner-1]
#                 output += [[ind_one, ind_two] + scheme + [sign * inner_sign]]
#         return output


from functools import lru_cache
@lru_cache(maxsize=1000)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def generate_general_pmatch(tuple indices, tuple connectivity_flat):
    """
    Parameters
    ----------
    indices : tuple
        Indices that are occupied.
    connectivity_flat tuple
        Components of the upper triangular matrix (offset 1) of the connectivity graph.
    """
    # indices = tuple(indices)
    #connectivity_matrix_full = np.ones((indices.size, indices.size), dtype=bool)
    #connectivity_matrix_full[np.triu_indices(indices.size, k=1)] = connectivity_matrix
    #connectivity_matrix_full[np.tril_indices(indices.size, k=-1)] = connectivity_matrix
    #connectivity_matrix = connectivity_matrix_full
    cdef int n = len(indices)
    if n == 2:
        return [(indices[0], indices[1], 1)]

    cdef list output = []
    cdef int j, sign, ind_one, ind_two, k, n_inner, inner_sign
    cdef bint is_connected
    cdef tuple new_indices, temp, scheme_inner_sign, scheme
    ind_one = indices[0]
    for j, is_connected in enumerate(connectivity_flat[:n-1]):
        if not is_connected:
            continue
        sign = (-1) ** j
        j += 1
        ind_two = indices[j]
        # filter out indices that are not used
        new_indices = indices[1:j] + indices[j+1:]
        #mask_bool = ~np.isin(indices, [ind_one, ind_two])
        #mask_ind = np.where(mask_bool)[0]
        #mask_ind = mask_ind[mask_ind > 0]
        j -= 1
        temp = connectivity_flat[n - 1 : n - 1 + j - 1]
        for k in range(2, j + 1):
            temp += connectivity_flat[
                (2 * n - 1 - (k - 1)) * (k - 1) // 2 + j - k + 2: (2 * n - 1 - k) * k // 2 + j - k
            ]
        temp += connectivity_flat[(2 * n - 1 - j) * j // 2 + 1:]
        # temp = connectivity_matrix[mask_ind[:, None], mask_ind[None, :]]
        for scheme_inner_sign in generate_general_pmatch(
                new_indices, temp
            # tuple(indices[mask_ind]),
            # tuple(temp[np.triu_indices(temp.shape[0], k=1)])
        ):
            n_inner = len(scheme_inner_sign)
            scheme = scheme_inner_sign[:n_inner-1]
            inner_sign = scheme_inner_sign[n_inner-1]
            output.append((ind_one, ind_two) + scheme + (sign * inner_sign,))
    return output
