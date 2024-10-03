import numpy as np
cimport numpy as np

cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def sign_excite_one(np.int64_t[:] occ_indices, np.int64_t[:] vir_indices):
    cdef Py_ssize_t num_occ = occ_indices.size

    bin_sizes = [0] * (num_occ + 1)
    # assume occ_indices is ordered
    # assume vir_indices is ordered
    cdef Py_ssize_t a, counter
    counter = 0
    for a in vir_indices:
        while counter < num_occ and a > occ_indices[counter]:
            counter += 1
        bin_sizes[counter] += 1

    cdef Py_ssize_t i, j, num_jumps
    output = []
    for i in range(num_occ):
        # i is the position in the occ_indices
        for j in range(num_occ + 1):
            # j is the position of the spaces beween occ_indices
            # 0 is the space before index 0
            # 1 is the space between indices 0 and 1,
            # n is the space after n-1
            num_jumps = i + j
            if j > i:
                num_jumps -= 1
            if num_jumps % 2 == 0:
                output += [1] * bin_sizes[j]
            else:
                output += [-1] * bin_sizes[j]
    return output


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def sign_excite_two(np.int64_t[:] occ_indices, np.int64_t[:] vir_indices):
    cdef int num_occ = len(occ_indices)

    bin_sizes = [0] * (num_occ + 1)
    # cdef long[:] bin_sizes = np.zeros(num_occ + 1, dtype=int)
    # assume occ_indices is ordered
    # assume vir_indices is ordered
    cdef int a
    cdef int counter = 0
    for a in vir_indices:
        while counter < num_occ and a > occ_indices[counter]:
            counter += 1
        bin_sizes[counter] += 1

    output = []
    cdef int i1, i2, j1, j2, num_jumps, sign_j1, len_j1
    for i1 in range(num_occ):
        for i2 in range(i1 + 1, num_occ):
            # i1 and i2 are the positions in the occ_indices
            for j1 in range(num_occ + 1):
                # j is the position of the spaces beween occ_indices
                # 0 is the space before index 0
                # 1 is the space between indices 0 and 1,
                # n is the space after n-1

                if bin_sizes[j1] == 0:
                    continue

                # when j1 == j2
                num_jumps = i1 + i2 - 1 + 2 * j1
                # sign resulting from creations where j1 == j2
                if num_jumps % 2 == 0:
                    sign_j1 = 1
                else:
                    sign_j1 = -1

                signs_j2 = []
                for j2 in range(j1 + 1, num_occ + 1):
                    if bin_sizes[j2] == 0:
                        continue

                    num_jumps = i1 + i2 - 1 + j2 + j1
                    num_jumps -= sum([j2 > i1, j2 > i2, j1 > i1, j1 > i2])

                    if num_jumps % 2 == 0:
                        sign = 1
                    else:
                        sign = -1

                    signs_j2 += [sign] * bin_sizes[j2]

                for len_j1 in reversed(range(bin_sizes[j1])):
                    output += [sign_j1] * len_j1 + signs_j2

    return output


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def sign_excite_two_ab(
        np.int64_t[:] occ_alpha,
        np.int64_t[:] occ_beta,
        np.int64_t[:] vir_alpha,
        np.int64_t[:] vir_beta,
):
    cdef int num_occ_alpha = len(occ_alpha)
    cdef int num_occ_beta = len(occ_beta)

    bin_sizes_alpha = [0] * (num_occ_alpha + 1)
    bin_sizes_beta = [0] * (num_occ_beta + 1)
    # cdef long[:] bin_sizes_alpha = np.zeros(num_occ_alpha + 1, dtype=int)
    # cdef long[:] bin_sizes_beta = np.zeros(num_occ_beta + 1, dtype=int)
    # assume occ_alpha, vir_alpha, occ_beta, vir_beta are ordered

    cdef int counter = 0
    cdef int a
    for a in vir_alpha:
        while counter < num_occ_alpha and a > occ_alpha[counter]:
            counter += 1
        bin_sizes_alpha[counter] += 1

    counter = 0
    for a in vir_beta:
        while counter < num_occ_beta and a > occ_beta[counter]:
            counter += 1
        bin_sizes_beta[counter] += 1

    cdef int i_a, i_b, j_a, j_b, num_jumps
    output = []
    for i_a in range(num_occ_alpha):
        # i_a is the position in the occ_alpha_indices
        for i_b in range(num_occ_beta):
            # i_b is the position in the occ_beta_indices
            for j_a in range(num_occ_alpha + 1):
                # j_a is the position of the spaces beween occ_alpha_indices
                # 0 is the space before index 0
                # 1 is the space between indices 0 and 1,
                # n is the space after n-1
                # if not bins_alpha[j_a]:
                if not bin_sizes_alpha[j_a]:
                    continue
                output_j_b = []
                for j_b in range(num_occ_beta + 1):
                    # j_b is the position of the spaces beween occ_beta_indices
                    # 0 is the space before index 0
                    # 1 is the space between indices 0 and 1,
                    # n is the space after n-1
                    # if not bins_beta[j_b]:
                    if not bin_sizes_beta[j_b]:
                        continue
                    num_jumps = i_a + i_b + j_b + j_a
                    if j_a > i_a:
                        num_jumps -= 1
                    if j_b > i_b:
                        num_jumps -= 1
                    if num_jumps % 2 == 0:
                        output_j_b += [1] * bin_sizes_beta[j_b]
                    else:
                        output_j_b += [-1] * bin_sizes_beta[j_b]
                output += output_j_b * bin_sizes_alpha[j_a]
    return output
