import numpy as np
cimport numpy as np

cimport cython
from libc.math cimport log2

# NOTE: Limited to 61 orbitals (b/c signed 64 bit integer only allos 61 bits)
# NOTE: need to have more than two virtual/occupied alpha/beta orbitals
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double _integrate_sd_sd_zero(
        double[:, ::1] one_int,
        double[:, ::1] two_int_ijij,
        double[:, ::1] two_int_ijji,
        int[1000] shared_alpha,
        int[1000] shared_beta,
        int counter_shared_alpha,
        int counter_shared_beta,
):
    """Return integrals of the given Slater determinant with itself.

    Parameters
    ----------
    shared_indices : np.ndarray
        Integer indices of the orbitals that are occupied in the Slater determinant.

    Returns
    -------
    integrals : 3-tuple of float
        Integrals of the given Slater determinant with itself.
        The one-electron (first element), coulomb (second element), and exchange (third element)
        integrals of the given Slater determinant with itself.

    """
    cdef double one_electron = 0
    cdef double coulomb = 0
    cdef double exchange = 0

    cdef int counter = 0
    cdef int i, j

    for i in shared_alpha[:counter_shared_alpha]:
        one_electron += one_int[i, i]
        for j in shared_alpha[:counter]:
            coulomb += two_int_ijij[i, j]
            exchange -= two_int_ijji[i, j]
        for j in shared_beta[:counter_shared_beta]:
            coulomb += two_int_ijij[i, j]
        counter += 1

    counter = 0
    for i in shared_beta[:counter_shared_beta]:
        one_electron += one_int[i, i]
        for j in shared_beta[:counter]:
            coulomb += two_int_ijij[i, j]
            exchange -= two_int_ijji[i, j]
        counter += 1

    return one_electron + coulomb + exchange


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double _integrate_sd_sd_one(
        double[:, ::1] one_int,
        double[:, :, :, ::1] two_int,
        int[1000] diff_sd1,
        int[1000] diff_sd2,
        int[1000] shared_alpha,
        int[1000] shared_beta,
        int counter_shared_alpha,
        int counter_shared_beta,
):
    """Return integrals of the given Slater determinant with its first order excitation.

    Parameters
    ----------
    diff_sd1 : int
        Index of the orbital that is occupied in the first Slater determinant and not occupied
        in the second.
    diff_sd2 : int
        Index of the orbital that is occupied in the second Slater determinant and not occupied
        in the first.
    shared_alpha : np.ndarray
        Integer indices of the alpha orbitals that are shared between the first and second
        Slater determinants.
    shared_beta : np.ndarray
        Integer indices of the beta orbitals that are shared between the first and second Slater
        determinants.

    Returns
    -------
    integrals : 3-tuple of float
        The one-electron (first element), coulomb (second element), and exchange (third element)
        integrals of the given Slater determinant with its first order excitations.

    """
    # pylint:disable=C0103
    cdef double one_electron = 0
    cdef double coulomb = 0
    cdef double exchange = 0

    cdef Py_ssize_t nspatial = one_int.shape[0]

    # get spatial indices
    cdef int spatial_a = diff_sd1[0]
    cdef int spatial_b = diff_sd2[0]
    cdef bint a_is_alpha = True
    cdef bint b_is_alpha = True
    if spatial_a >= nspatial:
        a_is_alpha = False
        spatial_a -= nspatial
    if spatial_b >= nspatial:
        b_is_alpha = False
        spatial_b -= nspatial

    if a_is_alpha != b_is_alpha:
        return 0.0

    cdef int i

    one_electron += one_int[spatial_a, spatial_b]
    for i in shared_alpha[:counter_shared_alpha]:
        coulomb += two_int[spatial_a, i, spatial_b, i]
        if a_is_alpha:
            exchange -= two_int[spatial_a, i, i, spatial_b]
    for i in shared_beta[:counter_shared_beta]:
        coulomb += two_int[spatial_a, i, spatial_b, i]
        if not a_is_alpha:
            exchange -= two_int[spatial_a, i, i, spatial_b]

    return one_electron + coulomb + exchange

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double _integrate_sd_sd_two(double[:, :, :, ::1] two_int, int[1000] diff_sd1, int[1000] diff_sd2):
    """Return integrals of the given Slater determinant with its second order excitation.

    Parameters
    ----------
    diff_sd1 : 2-tuple of int
        Indices of the orbitals that are occupied in the first Slater determinant and not
        occupied in the second.
    diff_sd2 : 2-tuple of int
        Indices of the orbitals that are occupied in the second Slater determinant and not
        occupied in the first.

    Returns
    -------
    integrals : 3-tuple of float
        The one-electron (first element), coulomb (second element), and exchange (third element)
        integrals of the given Slater determinant with itself.

    """
    # pylint:disable=C0103
    cdef double one_electron = 0
    cdef double coulomb = 0
    cdef double exchange = 0

    cdef Py_ssize_t nspatial = two_int.shape[0]

    # get spatial indices
    cdef int spatial_a = diff_sd1[0]
    cdef int spatial_b = diff_sd1[1]
    cdef int spatial_c = diff_sd2[0]
    cdef int spatial_d = diff_sd2[1]
    cdef bint a_is_alpha = True
    cdef bint b_is_alpha = True
    cdef bint c_is_alpha = True
    cdef bint d_is_alpha = True
    if spatial_a >= nspatial:
        a_is_alpha = False
        spatial_a -= nspatial
    if spatial_b >= nspatial:
        b_is_alpha = False
        spatial_b -= nspatial
    if spatial_c >= nspatial:
        c_is_alpha = False
        spatial_c -= nspatial
    if spatial_d >= nspatial:
        d_is_alpha = False
        spatial_d -= nspatial

    if b_is_alpha == d_is_alpha and a_is_alpha == c_is_alpha:
        coulomb += two_int[spatial_a, spatial_b, spatial_c, spatial_d]
    if b_is_alpha == c_is_alpha and a_is_alpha == d_is_alpha:
        exchange -= two_int[spatial_a, spatial_b, spatial_d, spatial_c]

    return one_electron + coulomb + exchange


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double _integrate_sd_sd_deriv_zero(
    double[:, ::1] one_int,
    double[:, :, :, ::1]two_int,
    int[1000] shared_alpha,
    int[1000] shared_beta,
    int counter_shared_alpha,
    int counter_shared_beta,
    int x,
    int y,
):
    """Return the derivative of the integrals of the given Slater determinant with itself.

    Parameters
    ----------
    x : int
        Row of the antihermitian matrix (of the given spin) at which the integral will be
        derivatized.
    y : int
        Column of the antihermitian matrix (of the given spin) at which the integral will be
        derivatized.
    shared_alpha : np.ndarray
        Integer indices of the alpha orbitals that are occupied by the Slater determinant.
        Dtype must be int.
    shared_beta : np.ndarray
        Integer indices of the beta orbitals that are occupied by the Slater determinant.
        Dtype must be int.

    Returns
    -------
    integrals : 3-tuple of float
        The derivatives (with respect to the given parameter) of the one-electron (first
        element), coulomb (second element), and exchange (third element) integrals of the given
        Slater determinant with itself.

    """
    # pylint:disable=R0912,R0915
    cdef double one_electron = 0
    cdef double coulomb = 0
    cdef double exchange = 0

    cdef int i, j

    for j in shared_alpha[:counter_shared_alpha]:
        if j == x:
            one_electron -= 2 * one_int[x, y]
            for i in shared_beta[:counter_shared_beta]:
                coulomb -= 2 * two_int[x, i, y, i]
            for i in shared_alpha[:counter_shared_alpha]:
                if i == x:
                    continue
                coulomb -= 2 * two_int[x, i, y, i]
                exchange += 2 * two_int[x, i, i, y]
            break

    for j in shared_beta[:counter_shared_beta]:
        if j == x:
            one_electron -= 2 * one_int[x, y]
            for i in shared_alpha[:counter_shared_alpha]:
                coulomb -= 2 * two_int[x, i, y, i]
            for i in shared_beta[:counter_shared_beta]:
                if i == x:
                    continue
                coulomb -= 2 * two_int[x, i, y, i]
                exchange += 2 * two_int[x, i, i, y]
            break

    for j in shared_alpha[:counter_shared_alpha]:
        if j == y:
            one_electron += 2 * one_int[x, y]
            for i in shared_beta[:counter_shared_beta]:
                coulomb += 2 * two_int[x, i, y, i]
            for i in shared_alpha[:counter_shared_alpha]:
                if i == y:
                    continue
                coulomb += 2 * two_int[x, i, y, i]
                exchange -= 2 * two_int[x, i, i, y]
            break

    for j in shared_beta[:counter_shared_beta]:
        if j == y:
            one_electron += 2 * one_int[x, y]
            for i in shared_alpha[:counter_shared_alpha]:
                coulomb += 2 * two_int[x, i, y, i]
            for i in shared_beta[:counter_shared_beta]:
                if i == y:
                    continue
                coulomb += 2 * two_int[x, i, y, i]
                exchange -= 2 * two_int[x, i, i, y]
            break

    return one_electron + coulomb + exchange


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double _integrate_sd_sd_deriv_one(
        double[:, ::1] one_int,
        double[:, :, :, ::1] two_int,
        int[1000] diff_sd1,
        int[1000] diff_sd2,
        int[1000] shared_alpha,
        int[1000] shared_beta,
        int counter_shared_alpha,
        int counter_shared_beta,
        int x,
        int y,
):
    """Return derivative of integrals of given Slater determinant with its first excitation.

    Parameters
    ----------
    diff_sd1 : 1-tuple of int
        Index of the orbital that is occupied in the first Slater determinant and not occupied
        in the second.
    diff_sd2 : 1-tuple of int
        Index of the orbital that is occupied in the second Slater determinant and not occupied
        in the first.
    x : int
        Row of the antihermitian matrix at which the integral will be derivatized.
    y : int
        Column of the antihermitian matrix at which the integral will be derivatized.
    shared_alpha : np.ndarray
        Integer indices of the alpha orbitals that are shared between the first and second
        Slater determinant.
        Dtype must be int.
    shared_beta : np.ndarray
        Integer indices of the beta orbitals that are shared between the first and second Slater
        Dtype must be int.

    Returns
    -------
    integrals : 3-tuple of float
        The derivatives (with respect to the given parameter) of the one-electron (first
        element), coulomb (second element), and exchange (third element) integrals of the given
        Slater determinant with its first order excitation.

    """
    # pylint:disable=C0103,R0912,R0915
    cdef double one_electron = 0
    cdef double coulomb = 0
    cdef double exchange = 0

    cdef Py_ssize_t nspatial = one_int.shape[0]

    # get spatial indices
    cdef int spatial_a = diff_sd1[0]
    cdef int spatial_b = diff_sd2[0]
    cdef bint a_is_alpha = True
    cdef bint b_is_alpha = True
    if spatial_a >= nspatial:
        a_is_alpha = False
        spatial_a -= nspatial
    if spatial_b >= nspatial:
        b_is_alpha = False
        spatial_b -= nspatial

    cdef int i, j

    # selected (spin orbital) x = a
    if x == spatial_a and a_is_alpha == b_is_alpha:
        one_electron -= one_int[y, spatial_b]
        # spin of a, b = alpha
        if b_is_alpha:
            for i in shared_beta[:counter_shared_beta]:
                if (not a_is_alpha and i == spatial_a) or (not b_is_alpha and i == spatial_b):
                    continue
                coulomb -= two_int[y, i, spatial_b, i]
            for i in shared_alpha[:counter_shared_alpha]:
                if (a_is_alpha and i == spatial_a) or (b_is_alpha and i == spatial_b):
                    continue
                coulomb -= two_int[y, i, spatial_b, i]
                exchange += two_int[y, i, i, spatial_b]
        # spin of a, b = beta
        else:
            for i in shared_alpha[:counter_shared_alpha]:
                if (a_is_alpha and i == spatial_a) or (b_is_alpha and i == spatial_b):
                    continue
                coulomb -= two_int[y, i, spatial_b, i]

            for i in shared_beta[:counter_shared_beta]:
                if (not a_is_alpha and i == spatial_a) or (not b_is_alpha and i == spatial_b):
                    continue
                coulomb -= two_int[y, i, spatial_b, i]
                exchange += two_int[y, i, i, spatial_b]
    # selected (spin orbital) x = b
    elif x == spatial_b and a_is_alpha == b_is_alpha:
        one_electron -= one_int[spatial_a, y]
        # spin of a, b = alpha
        if a_is_alpha:
            for i in shared_beta[:counter_shared_beta]:
                if (not a_is_alpha and i == spatial_a) or (not b_is_alpha and i == spatial_b):
                    continue
                coulomb -= two_int[spatial_a, i, y, i]
            for i in shared_alpha[:counter_shared_alpha]:
                if (a_is_alpha and i == spatial_a) or (b_is_alpha and i == spatial_b):
                    continue
                coulomb -= two_int[spatial_a, i, y, i]
                exchange += two_int[spatial_a, i, i, y]
        # spin of a, b = beta
        else:
            for i in shared_alpha[:counter_shared_alpha]:
                if (a_is_alpha and i == spatial_a) or (b_is_alpha and i == spatial_b):
                    continue
                coulomb -= two_int[i, spatial_a, i, y]
            for i in shared_beta[:counter_shared_beta]:
                if (not a_is_alpha and i == spatial_a) or (not b_is_alpha and i == spatial_b):
                    continue
                coulomb -= two_int[spatial_a, i, y, i]
                exchange += two_int[spatial_a, i, i, y]
    # non selected (spin orbital) x, spin of a, b = 0
    if a_is_alpha and b_is_alpha:
        for j in shared_alpha[:counter_shared_alpha]:
            if j == x:
                coulomb -= two_int[x, spatial_a, y, spatial_b]
                coulomb -= two_int[x, spatial_b, y, spatial_a]
                exchange += two_int[x, spatial_b, spatial_a, y]
                exchange += two_int[x, spatial_a, spatial_b, y]
        for j in shared_beta[:counter_shared_beta]:
            if j == x:
                coulomb -= two_int[spatial_a, x, spatial_b, y]
                coulomb -= two_int[spatial_b, x, spatial_a, y]
    # non selected (spin orbital) x, spin of a, b = 1
    elif not a_is_alpha and not b_is_alpha:
        for j in shared_beta[:counter_shared_beta]:
            if j == x:
                coulomb -= two_int[x, spatial_a, y, spatial_b]
                coulomb -= two_int[x, spatial_b, y, spatial_a]
                exchange += two_int[x, spatial_b, spatial_a, y]
                exchange += two_int[x, spatial_a, spatial_b, y]
        for j in shared_alpha[:counter_shared_alpha]:
            if j == x:
                coulomb -= two_int[x, spatial_a, y, spatial_b]
                coulomb -= two_int[x, spatial_b, y, spatial_a]

    # selected (spin orbital) y = a
    if y == spatial_a and a_is_alpha == b_is_alpha:
        one_electron += one_int[x, spatial_b]
        # spin of a, b = alpha
        if b_is_alpha:
            for i in shared_beta[:counter_shared_beta]:
                if (not a_is_alpha and i == spatial_a) or (not b_is_alpha and i == spatial_b):
                    continue
                coulomb += two_int[x, i, spatial_b, i]
            for i in shared_alpha[:counter_shared_alpha]:
                if (a_is_alpha and i == spatial_a) or (b_is_alpha and i == spatial_b):
                    continue
                coulomb += two_int[x, i, spatial_b, i]
                exchange -= two_int[x, i, i, spatial_b]
        # spin of a, b = beta
        else:
            for i in shared_alpha[:counter_shared_alpha]:
                if (a_is_alpha and i == spatial_a) or (b_is_alpha and i == spatial_b):
                    continue
                coulomb += two_int[x, i, spatial_b, i]
            for i in shared_beta[:counter_shared_beta]:
                if (not a_is_alpha and i == spatial_a) or (not b_is_alpha and i == spatial_b):
                    continue
                coulomb += two_int[x, i, spatial_b, i]
                exchange -= two_int[x, i, i, spatial_b]
    # selected (spin orbital) x = b
    elif y == spatial_b and a_is_alpha == b_is_alpha:
        one_electron += one_int[spatial_a, x]
        # spin of a, b = alpha
        if a_is_alpha:
            for i in shared_beta[:counter_shared_beta]:
                if (not a_is_alpha and i == spatial_a) or (not b_is_alpha and i == spatial_b):
                    continue
                coulomb += two_int[spatial_a, i, x, i]
            for i in shared_alpha[:counter_shared_alpha]:
                if (a_is_alpha and i == spatial_a) or (b_is_alpha and i == spatial_b):
                    continue
                coulomb += two_int[spatial_a, i, x, i]
                exchange -= two_int[spatial_a, i, i, x]
        # spin of a, b = beta
        else:
            for i in shared_alpha[:counter_shared_alpha]:
                if (a_is_alpha and i == spatial_a) or (b_is_alpha and i == spatial_b):
                    continue
                coulomb += two_int[spatial_a, i, x, i]
            for i in shared_beta[:counter_shared_beta]:
                if (not a_is_alpha and i == spatial_a) or (not b_is_alpha and i == spatial_b):
                    continue
                coulomb += two_int[spatial_a, i, x, i]
                exchange -= two_int[spatial_a, i, i, x]
    # non selected (spin orbital) x, spin of a, b = 0
    if a_is_alpha and b_is_alpha:
        for j in shared_alpha[:counter_shared_alpha]:
            if j == y:
                coulomb += two_int[x, spatial_a, y, spatial_b]
                coulomb += two_int[x, spatial_b, y, spatial_a]
                exchange -= two_int[x, spatial_a, spatial_b, y]
                exchange -= two_int[x, spatial_b, spatial_a, y]
        for j in shared_beta[:counter_shared_beta]:
            if j == y:
                coulomb += two_int[spatial_a, x, spatial_b, y]
                coulomb += two_int[spatial_b, x, spatial_a, y]
    # non selected (spin orbital) x, spin of a, b = 1
    if not a_is_alpha and not b_is_alpha:
        for j in shared_beta[:counter_shared_beta]:
            if j == y:
                coulomb += two_int[x, spatial_a, y, spatial_b]
                coulomb += two_int[x, spatial_b, y, spatial_a]
                exchange -= two_int[x, spatial_a, spatial_b, y]
                exchange -= two_int[x, spatial_b, spatial_a, y]
        for j in shared_alpha[:counter_shared_alpha]:
            if j == y:
                coulomb += two_int[x, spatial_a, y, spatial_b]
                coulomb += two_int[x, spatial_b, y, spatial_a]

    return one_electron + coulomb + exchange


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double _integrate_sd_sd_deriv_two(
        double[:, :, :, ::1] two_int,
        int[1000] diff_sd1,
        int[1000] diff_sd2,
        int x,
        int y
):
    """Return derivative of integrals of given Slater determinant with its second excitation.

    Parameters
    ----------
    diff_sd1 : 2-tuple of int
        Indices of the orbitals that are occupied in the first Slater determinant and not
        occupied in the second.
    diff_sd2 : 2-tuple of int
        Indices of the orbitals that are occupied in the second Slater determinant and not
        occupied in the first.
    x : int
        Row of the antihermitian matrix at which the integral will be derivatized.
    y : int
        Column of the antihermitian matrix at which the integral will be derivatized.

    Returns
    -------
    integrals : 3-tuple of float
        The derivatives (with respect to the given parameter) of the one-electron (first
        element), coulomb (second element), and exchange (third element) integrals of the given
        Slater determinant with its first order excitation.

    """
    # pylint: disable=C0103,R0912,R0915
    cdef double one_electron = 0
    cdef double coulomb = 0
    cdef double exchange = 0

    cdef Py_ssize_t nspatial = two_int.shape[0]

    # get spatial indices
    cdef int spatial_a = diff_sd1[0]
    cdef int spatial_b = diff_sd1[1]
    cdef int spatial_c = diff_sd2[0]
    cdef int spatial_d = diff_sd2[1]
    cdef bint a_is_alpha = True
    cdef bint b_is_alpha = True
    cdef bint c_is_alpha = True
    cdef bint d_is_alpha = True
    if spatial_a >= nspatial:
        a_is_alpha = False
        spatial_a -= nspatial
    if spatial_b >= nspatial:
        b_is_alpha = False
        spatial_b -= nspatial
    if spatial_c >= nspatial:
        c_is_alpha = False
        spatial_c -= nspatial
    if spatial_d >= nspatial:
        d_is_alpha = False
        spatial_d -= nspatial

    if x == spatial_a:
        if a_is_alpha and b_is_alpha and c_is_alpha and d_is_alpha:
            coulomb -= two_int[y, spatial_b, spatial_c, spatial_d]
            exchange += two_int[y, spatial_b, spatial_d, spatial_c]
        elif a_is_alpha and not b_is_alpha and c_is_alpha and not d_is_alpha:
            coulomb -= two_int[y, spatial_b, spatial_c, spatial_d]
        # elif not a_is_alpha and b_is_alpha and not c_is_alpha and d_is_alpha:
        #     coulomb -= two_int[spatial_b, y, spatial_d, spatial_c]
        # elif a_is_alpha and not b_is_alpha and not c_is_alpha and d_is_alpha:
        #     exchange += two_int[y, spatial_b, spatial_d, spatial_c]
        # elif not a_is_alpha and b_is_alpha and c_is_alpha and not d_is_alpha:
        #     exchange += two_int[spatial_b, y, spatial_c, spatial_d]
        elif not a_is_alpha and not b_is_alpha and not c_is_alpha and not d_is_alpha:
            coulomb -= two_int[y, spatial_b, spatial_c, spatial_d]
            exchange += two_int[y, spatial_b, spatial_d, spatial_c]
    if x == spatial_b:
        if a_is_alpha and b_is_alpha and c_is_alpha and d_is_alpha:
            exchange += two_int[y, spatial_a, spatial_c, spatial_d]
            coulomb -= two_int[y, spatial_a, spatial_d, spatial_c]
        # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
        # returns orbitals in increasing order (which means second index cannot be alpha if
        # the first is beta)
        # NOTE: b will not be alpha if a is beta (spin of x = spin of b)
        # elif not a_is_alpha and b_is_alpha and c_is_alpha and not d_is_alpha:
        #     exchange += two_int[y, spatial_a, spatial_c, spatial_d]
        # NOTE: d will not be alpha if c is beta
        # elif a_is_alpha and not b_is_alpha and not c_is_alpha and d_is_alpha:
        #     exchange += two_int[spatial_a, y, spatial_d, spatial_c]
        # elif not a_is_alpha and b_is_alpha and not c_is_alpha and d_is_alpha:
        #     coulomb -= two_int[y, spatial_a, spatial_d, spatial_c]
        elif a_is_alpha and not b_is_alpha and c_is_alpha and not d_is_alpha:
            coulomb -= two_int[spatial_a, y, spatial_c, spatial_d]
        elif not a_is_alpha and not b_is_alpha and not c_is_alpha and not d_is_alpha:
            exchange += two_int[y, spatial_a, spatial_c, spatial_d]
            coulomb -= two_int[y, spatial_a, spatial_d, spatial_c]
    if x == spatial_c:
        if a_is_alpha and b_is_alpha and c_is_alpha and d_is_alpha:
            coulomb -= two_int[spatial_a, spatial_b, y, spatial_d]
            exchange += two_int[spatial_b, spatial_a, y, spatial_d]
        elif a_is_alpha and not b_is_alpha and c_is_alpha and not d_is_alpha:
            coulomb -= two_int[spatial_a, spatial_b, y, spatial_d]
        # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
        # returns orbitals in increasing order (which means second index cannot be alpha if
        # the first is beta)
        # elif not a_is_alpha and b_is_alpha and not c_is_alpha and d_is_alpha:
        #     coulomb -= two_int[spatial_b, spatial_a, spatial_d, y]
        # elif not a_is_alpha and b_is_alpha and c_is_alpha and not d_is_alpha:
        #     exchange += two_int[spatial_b, spatial_a, y, spatial_d]
        # NOTE: b will not be alpha if a is beta
        # elif a_is_alpha and not b_is_alpha and not c_is_alpha and d_is_alpha:
        #     exchange += two_int[spatial_a, spatial_b, spatial_d, y]
        elif not a_is_alpha and not b_is_alpha and not c_is_alpha and not d_is_alpha:
            coulomb -= two_int[spatial_a, spatial_b, y, spatial_d]
            exchange += two_int[spatial_b, spatial_a, y, spatial_d]
    if x == spatial_d:
        if a_is_alpha and b_is_alpha and c_is_alpha and d_is_alpha:
            exchange += two_int[spatial_a, spatial_b, y, spatial_c]
            coulomb -= two_int[spatial_b, spatial_a, y, spatial_c]
        # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
        # returns orbitals in increasing order (which means second index cannot be alpha if
        # the first is beta)
        # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
        # elif a_is_alpha and not b_is_alpha and not c_is_alpha and d_is_alpha:
        #     exchange += two_int[spatial_a, spatial_b, y, spatial_c]
        # NOTE: b will not be alpha if a is beta
        # elif not a_is_alpha and b_is_alpha and c_is_alpha and not d_is_alpha:
        #     exchange += two_int[spatial_b, spatial_a, spatial_c, y]
        # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
        # elif not a_is_alpha and b_is_alpha and not c_is_alpha and d_is_alpha:
        #     coulomb -= two_int[spatial_b, spatial_a, y, spatial_c]
        elif a_is_alpha and not b_is_alpha and c_is_alpha and not d_is_alpha:
            coulomb -= two_int[spatial_a, spatial_b, spatial_c, y]
        elif not a_is_alpha and not b_is_alpha and not c_is_alpha and not d_is_alpha:
            exchange += two_int[spatial_a, spatial_b, y, spatial_c]
            coulomb -= two_int[spatial_b, spatial_a, y, spatial_c]

    if y == spatial_a:
        if a_is_alpha and b_is_alpha and c_is_alpha and d_is_alpha:
            coulomb += two_int[x, spatial_b, spatial_c, spatial_d]
            exchange -= two_int[x, spatial_b, spatial_d, spatial_c]
        elif a_is_alpha and not b_is_alpha and c_is_alpha and not d_is_alpha:
            coulomb += two_int[x, spatial_b, spatial_c, spatial_d]
        # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
        # returns orbitals in increasing order (which means second index cannot be alpha if
        # the first is beta)
        # NOTE: d will not be alpha if c is beta
        # elif not a_is_alpha and b_is_alpha and not c_is_alpha and d_is_alpha:
        #     coulomb += two_int[spatial_b, x, spatial_d, spatial_c]
        # elif a_is_alpha and not b_is_alpha and not c_is_alpha and d_is_alpha:
        #     exchange -= two_int[x, spatial_b, spatial_d, spatial_c]
        # NOTE: b will not be alpha if a is beta (spin of x = spin of a)
        # elif not a_is_alpha and b_is_alpha and c_is_alpha and not d_is_alpha:
        #     exchange -= two_int[spatial_b, x, spatial_c, spatial_d]
        elif not a_is_alpha and not b_is_alpha and not c_is_alpha and not d_is_alpha:
            coulomb += two_int[x, spatial_b, spatial_c, spatial_d]
            exchange -= two_int[x, spatial_b, spatial_d, spatial_c]
    if y == spatial_b:
        if a_is_alpha and b_is_alpha and c_is_alpha and d_is_alpha:
            exchange -= two_int[x, spatial_a, spatial_c, spatial_d]
            coulomb += two_int[x, spatial_a, spatial_d, spatial_c]
        # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
        # returns orbitals in increasing order (which means second index cannot be alpha if
        # the first is beta)
        # NOTE: b will not be alpha if a is beta (spin of x = spin of b)
        # elif not a_is_alpha and b_is_alpha and c_is_alpha and not d_is_alpha:
        #     exchange -= two_int[x, spatial_a, spatial_c, spatial_d]
        # NOTE: d will not be alpha if c is beta
        # elif a_is_alpha and not b_is_alpha and not c_is_alpha and d_is_alpha:
        #     exchange -= two_int[spatial_a, x, spatial_d, spatial_c]
        # NOTE: b will not be alpha if a is beta (spin of x = spin of b)
        # elif not a_is_alpha and b_is_alpha and not c_is_alpha and d_is_alpha:
        #     coulomb += two_int[x, spatial_a, spatial_d, spatial_c]
        elif a_is_alpha and not b_is_alpha and c_is_alpha and not d_is_alpha:
            coulomb += two_int[spatial_a, x, spatial_c, spatial_d]
        elif not a_is_alpha and not b_is_alpha and not c_is_alpha and not d_is_alpha:
            exchange -= two_int[x, spatial_a, spatial_c, spatial_d]
            coulomb += two_int[x, spatial_a, spatial_d, spatial_c]
    if y == spatial_c:
        if a_is_alpha and b_is_alpha and c_is_alpha and d_is_alpha:
            coulomb += two_int[spatial_a, spatial_b, x, spatial_d]
            exchange -= two_int[spatial_b, spatial_a, x, spatial_d]
        elif a_is_alpha and not b_is_alpha and c_is_alpha and not d_is_alpha:
            coulomb += two_int[spatial_a, spatial_b, x, spatial_d]
        # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
        # returns orbitals in increasing order (which means second index cannot be alpha if
        # the first is beta)
        # NOTE: d will not be alpha if c is beta (spin of x = spin of c)
        # elif not a_is_alpha and b_is_alpha and not c_is_alpha and d_is_alpha:
        #     coulomb += two_int[spatial_b, spatial_a, spatial_d, x]
        # NOTE: a will not be alpha if b is beta
        # elif not a_is_alpha and b_is_alpha and c_is_alpha and not d_is_alpha:
        #     exchange -= two_int[spatial_b, spatial_a, x, spatial_d]
        # NOTE: d will not be alpha if c is beta (spin of x = spin of c)
        # elif a_is_alpha and not b_is_alpha and not c_is_alpha and d_is_alpha:
        #     exchange -= two_int[spatial_a, spatial_b, spatial_d, x]
        elif not a_is_alpha and not b_is_alpha and not c_is_alpha and not d_is_alpha:
            coulomb += two_int[spatial_a, spatial_b, x, spatial_d]
            exchange -= two_int[spatial_b, spatial_a, x, spatial_d]
    if y == spatial_d:
        if a_is_alpha and b_is_alpha and c_is_alpha and d_is_alpha:
            exchange -= two_int[spatial_a, spatial_b, x, spatial_c]
            coulomb += two_int[spatial_b, spatial_a, x, spatial_c]
        # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
        # returns orbitals in increasing order (which means second index cannot be alpha if
        # the first is beta)
        # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
        # elif a_is_alpha and not b_is_alpha and not c_is_alpha and d_is_alpha:
        #     exchange -= two_int[spatial_a, spatial_b, x, spatial_c]
        # NOTE: b will not be alpha if a is beta
        # elif not a_is_alpha and b_is_alpha and c_is_alpha and not d_is_alpha:
        #     exchange -= two_int[spatial_b, spatial_a, spatial_c, x]
        # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
        # elif not a_is_alpha and b_is_alpha and not c_is_alpha and d_is_alpha:
        #     coulomb += two_int[spatial_b, spatial_a, x, spatial_c]
        elif a_is_alpha and not b_is_alpha and c_is_alpha and not d_is_alpha:
            coulomb += two_int[spatial_a, spatial_b, spatial_c, x]
        elif not a_is_alpha and not b_is_alpha and not c_is_alpha and not d_is_alpha:
            exchange -= two_int[spatial_a, spatial_b, x, spatial_c]
            coulomb += two_int[spatial_b, spatial_a, x, spatial_c]

    return one_electron + coulomb + exchange


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double integrate_sd_sd(
    double[:, ::1] one_int,
    double[:, ::1] two_int_ijij,
    double[:, ::1] two_int_ijji,
    double[:, :, :, ::1] two_int,
    long sd1,
    long sd2,
):
    r"""Integrate the Hamiltonian with against two Slater determinants.

    .. math::

        H_{\mathbf{m}\mathbf{n}} &=
        \left< \mathbf{m} \middle| \hat{H} \middle| \mathbf{n} \right>\\
        &= \sum_{ij}
            h_{ij} \left< \mathbf{m} \middle| a^\dagger_i a_j \middle| \mathbf{n} \right>
        + \sum_{i<j, k<l} g_{ijkl}
        \left< \mathbf{m} \middle| a^\dagger_i a^\dagger_j a_l a_k \middle| \mathbf{n} \right>\\

    In the first summation involving :math:`h_{ij}`, only the terms where :math:`\mathbf{m}` and
    :math:`\mathbf{n}` are different by at most single excitation will contribute to the
    integral. In the second summation involving :math:`g_{ijkl}`, only the terms where
    :math:`\mathbf{m}` and :math:`\mathbf{n}` are different by at most double excitation will
    contribute to the integral.

    Parameters
    ----------
    sd1 : int
        Slater Determinant against which the Hamiltonian is integrated.
    sd2 : int
        Slater Determinant against which the Hamiltonian is integrated.
    sign : {1, -1, None}
        Sign change resulting from cancelling out the orbitals shared between the two Slater
        determinants.
        Computes the sign if none is provided.
        Make sure that the provided sign is correct. It will not be checked to see if its
        correct.
    deriv : {int, None}
        Index of the Hamiltonian parameter against which the integral is derivatized.
        Default is no derivatization.

    Returns
    -------
    one_electron : float
        One-electron energy.
    coulomb : float
        Coulomb energy.
    exchange : float
        Exchange energy.

    Raises
    ------
    ValueError
        If `sign` is not `1`, `-1` or `None`.

    """
    cdef Py_ssize_t nspatial = two_int.shape[0]

    # FIXME: assumed number of spatial orbitals less than 1000
    # cdef long[:] shared_alpha = np.zeros((1000,), dtype=int)
    # cdef int counter_shared_alpha = 0
    # cdef long[:] shared_beta = np.zeros((1000,), dtype=int)
    # cdef int counter_shared_beta = 0
    # cdef long[:] diff_sd1 = np.zeros((1000,), dtype=int)
    # cdef int counter_diff_sd1 = 0
    # cdef long[:] diff_sd2 = np.zeros((1000,), dtype=int)
    # cdef int counter_diff_sd2 = 0
    cdef int[1000] shared_alpha
    cdef int counter_shared_alpha = 0
    cdef int[1000] shared_beta
    cdef int counter_shared_beta = 0
    cdef int[1000] diff_sd1
    cdef int counter_diff_sd1 = 0
    cdef int[1000] diff_sd2
    cdef int counter_diff_sd2 = 0

    cdef bint is_occ_sd1
    cdef bint is_occ_sd2
    cdef int num_transpositions = 0

    cdef int i
    cdef int start_ind_sd1 = <int>log2(sd1 & -sd1)
    cdef int end_ind_sd1 = <int>log2(sd1)
    cdef int start_ind_sd2 = <int>log2(sd2 & -sd2)
    cdef int end_ind_sd2 = <int>log2(sd2)
    for i in range(min([start_ind_sd1, start_ind_sd2]), max([end_ind_sd1, end_ind_sd2]) + 1):
        is_occ_sd1 = False
        if sd1 & 2**i:
            is_occ_sd1 = True

        is_occ_sd2 = False
        if sd2 & 2**i:
            is_occ_sd2 = True

        if not is_occ_sd1 and not is_occ_sd2:
            continue

        if is_occ_sd1 and is_occ_sd2:
            if i < nspatial:
                shared_alpha[counter_shared_alpha] = i
                counter_shared_alpha += 1
            else:
                shared_beta[counter_shared_beta] = i - nspatial
                counter_shared_beta += 1
        elif is_occ_sd1 and not is_occ_sd2:
            diff_sd1[counter_diff_sd1] = i
            num_transpositions += counter_shared_alpha
            if i >= nspatial:
                num_transpositions += counter_shared_beta
            counter_diff_sd1 += 1

        elif is_occ_sd2 and not is_occ_sd1:
            diff_sd2[counter_diff_sd2] = i
            num_transpositions += counter_shared_alpha
            if i >= nspatial:
                num_transpositions += counter_shared_beta
            num_transpositions += counter_diff_sd2
            counter_diff_sd2 += 1
    # NOTE: the creators for the left side (diff_sd2) are reverse ordered (largest to smallest from
    # left to right). This means that it needs to be reshuffled.
    num_transpositions += counter_diff_sd2 * (counter_diff_sd2 - 1) / 2

    # if two Slater determinants do not have the same number of electrons
    if counter_diff_sd1 != counter_diff_sd2:
        return 0.0
    if counter_diff_sd1 > 2:
        return 0.0

    # shared_alpha = shared_alpha[:counter_shared_alpha]
    # shared_beta = shared_beta[:counter_shared_beta]
    # diff_sd1 = diff_sd1[:counter_diff_sd1]
    # diff_sd2 = diff_sd2[:counter_diff_sd2]

    cdef int sign = (-1) ** num_transpositions
    # cdef int j
    # cdef int sign_check = 1
    # cdef int sign_check_counter = 0
    # # remove orbitals from sd1
    # for i in diff_sd1[:counter_diff_sd1]:
    #     # find out alpha orbitals that are jumped over
    #     if i < nspatial:
    #         for j in shared_alpha[:counter_shared_alpha]:
    #             if i < j:
    #                 break
    #             sign_check *= -1
    #     # find out beta orbitals that are jumped over
    #     else:
    #         sign_check *= (-1) ** counter_shared_alpha
    #         for j in shared_beta[:counter_shared_beta]:
    #             j += nspatial
    #             if i <= j:
    #                 break
    #             sign_check *= -1
    #     # account for removed orbitals
    #     sign_check *= (-1) ** sign_check_counter
    #     sign_check_counter += 1
    # # add orbitals to sd with removed electrons
    # for i in diff_sd2[counter_diff_sd2 - 1::-1]:
    #     # number of alpha orbitals jumped over
    #     if i < nspatial:
    #         for j in shared_alpha[:counter_shared_alpha]:
    #             if i < j:
    #                 break
    #             sign_check *= -1
    #     # number of beta orbitals jumped over
    #     else:
    #         sign_check *= (-1) ** counter_shared_alpha
    #         for j in shared_beta[:counter_shared_beta]:
    #             j += nspatial
    #             if i <= j:
    #                 break
    #             sign_check *= -1
    # assert sign_check == sign

    cdef double integral
    # two sd's are the same
    if counter_diff_sd1 == 0 and counter_shared_alpha + counter_shared_beta > 0:
        integral = _integrate_sd_sd_zero(
            one_int,
            two_int_ijij,
            two_int_ijji,
            shared_alpha,
            shared_beta,
            counter_shared_alpha,
            counter_shared_beta,
        )
        # two sd's are different by single excitation
    elif counter_diff_sd1 == 1:
        integral = _integrate_sd_sd_one(
            one_int,
            two_int,
            diff_sd1,
            diff_sd2,
            shared_alpha,
            shared_beta,
            counter_shared_alpha,
            counter_shared_beta,
        )
    # two sd's are different by double excitation
    else:
        integral = _integrate_sd_sd_two(two_int, diff_sd1, diff_sd2)

    return sign * integral


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate zero division check
cdef double _integrate_sd_sd_deriv(
        double[:, ::1] one_int,
        double[:, :, :, ::1] two_int,
        long sd1,
        long sd2,
        int deriv
):
    r"""Return derivative of the CI matrix element with respect to the antihermitian elements.

    Parameters
    ----------
    sd1 : int
        Slater Determinant against which the Hamiltonian is integrated.
    sd2 : int
        Slater Determinant against which the Hamiltonian is integrated.
    deriv : int
        Index of the Hamiltonian parameter against which the integral is derivatized.

    Returns
    -------
    one_electron : float
        One-electron energy derivatized with respect to the given index.
    coulomb : float
        Coulomb energy derivatized with respect to the given index.
    exchange : float
        Exchange energy derivatized with respect to the given index.

    Raises
    ------
    ValueError
        If the given `deriv` is not an integer greater than or equal to 0 and less than the
        number of parameters.

    Notes
    -----
    Integrals are not assumed to be real. The performance benefit (at the moment) for assuming
    real orbitals is not much.

    """
    cdef Py_ssize_t nspatial = two_int.shape[0]

    # FIXME: assumed number of spatial orbitals less than 1000
    # FIXME: duplicated code
    cdef int[1000] shared_alpha
    cdef int counter_shared_alpha = 0
    cdef int[1000] shared_beta
    cdef int counter_shared_beta = 0
    cdef int[1000] diff_sd1
    cdef int counter_diff_sd1 = 0
    cdef int[1000] diff_sd2
    cdef int counter_diff_sd2 = 0

    cdef bint is_occ_sd1
    cdef bint is_occ_sd2
    cdef int num_transpositions = 0

    cdef int i
    cdef int start_ind_sd1 = <int>log2(sd1 & -sd1)
    cdef int end_ind_sd1 = <int>log2(sd1)
    cdef int start_ind_sd2 = <int>log2(sd2 & -sd2)
    cdef int end_ind_sd2 = <int>log2(sd2)
    for i in range(min([start_ind_sd1, start_ind_sd2]), max([end_ind_sd1, end_ind_sd2]) + 1):
        is_occ_sd1 = False
        if sd1 & 2**i:
            is_occ_sd1 = True

        is_occ_sd2 = False
        if sd2 & 2**i:
            is_occ_sd2 = True

        if not is_occ_sd1 and not is_occ_sd2:
            continue

        if is_occ_sd1 and is_occ_sd2:
            if i < nspatial:
                shared_alpha[counter_shared_alpha] = i
                counter_shared_alpha += 1
            else:
                shared_beta[counter_shared_beta] = i - nspatial
                counter_shared_beta += 1
        elif is_occ_sd1 and not is_occ_sd2:
            diff_sd1[counter_diff_sd1] = i
            num_transpositions += counter_shared_alpha
            if i >= nspatial:
                num_transpositions += counter_shared_beta
            counter_diff_sd1 += 1

        elif is_occ_sd2 and not is_occ_sd1:
            diff_sd2[counter_diff_sd2] = i
            num_transpositions += counter_shared_alpha
            if i >= nspatial:
                num_transpositions += counter_shared_beta
            num_transpositions += counter_diff_sd2
            counter_diff_sd2 += 1
    # NOTE: the creators for the left side (diff_sd2) are reverse ordered (largest to smallest from
    # left to right). This means that it needs to be reshuffled.
    num_transpositions += counter_diff_sd2 * (counter_diff_sd2 - 1) / 2

    # if two Slater determinants do not have the same number of electrons
    if counter_diff_sd1 != counter_diff_sd2:
        return 0.0
    if counter_diff_sd1 > 2:
        return 0.0

    cdef int sign = (-1) ** num_transpositions

    # turn deriv into indices of the matrix, (x, y), where x < y
    # ind = i
    cdef int n = nspatial
    cdef int k
    cdef int x
    for k in range(n + 1):
        x = k
        if deriv - n * (x + 1) + (x + 1) * (x + 2) / 2 < 0:
            break
    # ind_flat = j
    cdef int y = (deriv + (x + 1) * (x + 2) / 2) % n

    cdef double integral

    # two sd's are the same
    if counter_diff_sd1 == 0:
        integral = _integrate_sd_sd_deriv_zero(
            one_int,
            two_int,
            shared_alpha,
            shared_beta,
            counter_shared_alpha,
            counter_shared_beta,
            x,
            y,
        )
    # two sd's are different by single excitation
    elif counter_diff_sd1 == 1:
        integral = _integrate_sd_sd_deriv_one(
            one_int,
            two_int,
            diff_sd1,
            diff_sd2,
            shared_alpha,
            shared_beta,
            counter_shared_alpha,
            counter_shared_beta,
            x,
            y,
        )
    # two sd's are different by double excitation
    else:
        integral = _integrate_sd_sd_deriv_two(
            two_int, diff_sd1, diff_sd2, x, y
        )

    return sign * integral


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double integrate_wfn_sd(
        double[:, ::1] one_int,
        double[:, ::1] two_int_ijij,
        double[:, ::1] two_int_ijji,
        double[:, :, :, ::1] two_int,
        wfn,
        long sd,
        wfn_deriv=None,
        # ham_deriv=None,
):
    r"""Integrate the Hamiltonian with against a wavefunction and Slater determinant.

    .. math::

        \left< \Phi \middle| \hat{H} \middle| \Psi \right>
        = \sum_{\mathbf{m} \in S_\Phi}
            f(\mathbf{m}) \left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>

    where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
    :math:`\Phi` is the Slater determinant. The :math:`S_{\Phi}` is the set of Slater
    determinants for which :math:`\left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>` is
    not zero, which are the :math:`\Phi` and its first and second order excitations for a
    chemical Hamiltonian.

    Parameters
    ----------
    wfn : Wavefunction
        Wavefunction against which the Hamiltonian is integrated.
        Needs to have the following in `__dict__`: `get_overlap`.
    sd : int
        Slater Determinant against which the Hamiltonian is integrated.
    wfn_deriv : {int, None}
        Index of the wavefunction parameter against which the integral is derivatized.
        Default is no derivatization.
    ham_deriv : {int, None}
        Index of the Hamiltonian parameter against which the integral is derivatized.
        Default is no derivatization.

    Returns
    -------
    one_electron : float
        One-electron energy.
    coulomb : float
        Coulomb energy.
    exchange : float
        Exchange energy.

    Raises
    ------
    ValueError
        If integral is derivatized to both wavefunction and Hamiltonian parameters.

    """
    # pylint: disable=C0103
    # if wfn_deriv is not None and ham_deriv is not None:
    #     raise ValueError(
    #         "Integral can be derivatized with respect to at most one out of the "
    #         "wavefunction and Hamiltonian parameters."
    #     )

    cdef int nspatial = one_int.shape[0]

    cdef int[1000] occ_indices
    cdef int counter_occ = 0
    cdef int[1000] vir_indices
    cdef int counter_vir = 0

    cdef int i
    for i in range(2 * nspatial):
        if sd & 2**i:
            occ_indices[counter_occ] = i
            counter_occ += 1
        else:
            vir_indices[counter_vir] = i
            counter_vir += 1

    cdef double integral = 0

    integral += (
        integrate_sd_sd(one_int, two_int_ijij, two_int_ijji, two_int, sd, sd) * wfn.get_overlap(sd, deriv=wfn_deriv)
    )

    cdef int counter_i = 0
    cdef int counter_a = 0
    cdef int a, j, b
    cdef long sd_m
    for i in occ_indices[: counter_occ]:
        counter_a = 0
        for a in vir_indices[: counter_vir]:
            sd_m = sd ^ (<long> 1 << i) | (<long> 1 << a)

            integral += (
                integrate_sd_sd(one_int, two_int_ijij, two_int_ijji, two_int, sd, sd_m)
                * wfn.get_overlap(sd_m, deriv=wfn_deriv)
            )
            for j in occ_indices[counter_i + 1 : counter_occ]:
                for b in vir_indices[counter_a + 1 : counter_vir]:
                    sd_m = sd ^ (<long> 1 << i) ^ (<long> 1 << j)
                    sd_m |= (<long> 1 << a) | (<long> 1 << b)

                    integral += (
                        integrate_sd_sd(one_int, two_int_ijij, two_int_ijji, two_int, sd, sd_m)
                        * wfn.get_overlap(sd_m, deriv=wfn_deriv)
                    )
            counter_a += 1
        counter_i += 1

    return integral


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double integrate_wfn_sd_deriv(
        double[:, ::1] one_int,
        double[:, :, :, ::1] two_int,
        wfn,
        long sd,
        int ham_deriv,
):
    # pylint: disable=C0103

    cdef int nspatial = one_int.shape[0]

    cdef int[1000] occ_indices
    cdef int counter_occ = 0
    cdef int[1000] vir_indices
    cdef int counter_vir = 0

    cdef int i
    for i in range(2 * nspatial):
        if sd & 2**i:
            occ_indices[counter_occ] = i
            counter_occ += 1
        else:
            vir_indices[counter_vir] = i
            counter_vir += 1

    cdef double integral = 0

    integral += (
        _integrate_sd_sd_deriv(one_int, two_int, sd, sd, deriv=ham_deriv) * wfn.get_overlap(sd)
    )

    cdef int counter_i = 0
    cdef int counter_a = 0
    cdef int a, j, b
    cdef long sd_m
    for i in occ_indices[: counter_occ]:
        counter_a = 0
        for a in vir_indices[: counter_vir]:
            sd_m = sd ^ (<long> 1 << i) | (<long> 1 << a)

            integral += (
                _integrate_sd_sd_deriv(one_int, two_int, sd, sd_m, deriv=ham_deriv)
                * wfn.get_overlap(sd_m)
            )
            for j in occ_indices[counter_i + 1 : counter_occ]:
                for b in vir_indices[counter_a + 1 : counter_vir]:
                    sd_m = sd ^ (<long> 1 << i) ^ (<long> 1 << j)
                    sd_m |= (<long> 1 << a) | (<long> 1 << b)

                    integral += (
                        _integrate_sd_sd_deriv(one_int, two_int, sd, sd_m, deriv=ham_deriv)
                        * wfn.get_overlap(sd_m)
                    )
            counter_a += 1
        counter_i += 1

    return integral


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef long factorial(int n):
    cdef int i
    cdef long output
    output = 1
    for i in range(1, n+1):
        output *= i
    return output


# FIXME: this will probably cause problems if factorial is too large
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate zero division check
cdef int combinations(int k, int n):
    return factorial(k) / factorial(k - n) / factorial(n)


# NOTE: slower than integrate_wfn_sd
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef integrate_sd_wfn(
    ham,
        long sd,
        wfn,
        wfn_deriv=None
):
    r"""Integrate the Hamiltonian with against a Slater determinant and a wavefunction.

    .. math::

        \left< \Phi \middle| \hat{H} \middle| \Psi \right>
        = \sum_{\mathbf{m} \in S_\Phi}
            f(\mathbf{m}) \left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>

    where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
    :math:`\Phi` is the Slater determinant. The :math:`S_{\Phi}` is the set of Slater
    determinants for which :math:`\left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>` is
    not zero, which are the :math:`\Phi` and its first and second order excitations for a
    chemical Hamiltonian.

    Parameters
    ----------
    sd : int
        Slater Determinant against which the Hamiltonian is integrated.
    wfn : Wavefunction
        Wavefunction against which the Hamiltonian is integrated.
        Needs to have the following in `__dict__`: `get_overlap`.
    wfn_deriv : {int, None}
        Index of the wavefunction parameter against which the integral is derivatized.
        Default is no derivatization.

    Returns
    -------
    integrals : np.ndarray(3,)
        Integrals of the given Slater determinant and the wavefunction.
        First element corresponds to the one-electron energy, second to the coulomb energy, and
        third to the exchange energy.

    """
    # pylint: disable=C0103
    cdef int nspatial = ham.nspin / 2

    cdef int[1000] occ_alpha
    cdef int counter_occ_alpha = 0
    cdef int[1000] occ_beta
    cdef int counter_occ_beta = 0
    cdef int[1000] vir_alpha
    cdef int counter_vir_alpha = 0
    cdef int[1000] vir_beta
    cdef int counter_vir_beta = 0

    cdef int i
    for i in range(nspatial):
        if sd & (<long> 1 << i):
            occ_alpha[counter_occ_alpha] = i
            counter_occ_alpha += 1
        else:
            vir_alpha[counter_vir_alpha] = i
            counter_vir_alpha += 1
    for i in range(nspatial, 2 * nspatial):
        if sd & (<long> 1 << i):
            occ_beta[counter_occ_beta] = i
            counter_occ_beta += 1
        else:
            vir_beta[counter_vir_beta] = i
            counter_vir_beta += 1

    cdef double overlaps_zero
    cdef double[:] overlaps_one_alpha = np.zeros(counter_occ_alpha * counter_vir_alpha)
    cdef double[:] overlaps_one_beta = np.zeros(counter_occ_beta * counter_vir_beta)
    cdef double[:] overlaps_two_aa = np.zeros(
        combinations(counter_occ_alpha, 2) * combinations(counter_vir_alpha, 2)
    )
    cdef double[:] overlaps_two_ab = np.zeros(
        counter_occ_alpha * counter_occ_beta * counter_vir_alpha * counter_vir_beta
    )
    cdef double[:] overlaps_two_bb = np.zeros(
        combinations(counter_occ_beta, 2) * combinations(counter_vir_beta, 2)
    )

    overlaps_zero = wfn.get_overlap(sd, deriv=wfn_deriv)

    cdef int j, a, b
    cdef long sd_m
    cdef int counter_1 = 0
    cdef int counter_2 = 0
    cdef int counter_3 = 0
    cdef int counter_i = 0
    cdef int counter_a
    for i in occ_alpha[:counter_occ_alpha]:
        counter_a = 0
        for a in vir_alpha[:counter_vir_alpha]:
            sd_m = sd ^ (<long> 1 << i)
            sd_m |= (<long> 1 << a)
            overlaps_one_alpha[counter_1] = wfn.get_overlap(sd_m, deriv=wfn_deriv)
            counter_1 += 1
            for j in occ_alpha[counter_i + 1 : counter_occ_alpha]:
                for b in vir_alpha[counter_a + 1 : counter_vir_alpha]:
                    sd_m = sd ^ (<long> 1 << i) ^ (<long> 1 << j)
                    sd_m |= (<long> 1 << b) | (<long> 1 << a)
                    overlaps_two_aa[counter_2] = wfn.get_overlap(sd_m, deriv=wfn_deriv)
                    counter_2 += 1
            counter_a += 1
        counter_i += 1
        for j in occ_beta[:counter_occ_beta]:
            for a in vir_alpha[:counter_vir_alpha]:
                for b in vir_beta[:counter_vir_beta]:
                    sd_m = sd ^ (<long> 1 << i) ^ (<long> 1 << j)
                    sd_m |= (<long> 1 << b) | (<long> 1 << a)
                    overlaps_two_ab[counter_3] = wfn.get_overlap(sd_m, deriv=wfn_deriv)
                    counter_3 += 1

    counter_1 = 0
    counter_2 = 0
    counter_i = 0
    for i in occ_beta[:counter_occ_beta]:
        counter_a = 0
        for a in vir_beta[:counter_vir_beta]:
            sd_m = sd ^ (<long> 1 << i)
            sd_m |= (<long> 1 << a)
            overlaps_one_beta[counter_1] = wfn.get_overlap(sd_m, deriv=wfn_deriv)
            counter_1 += 1
            for j in occ_beta[counter_i + 1 : counter_occ_beta]:
                for b in vir_beta[counter_a + 1 : counter_vir_beta]:
                    sd_m = sd ^ (<long> 1 << i) ^ (<long> 1 << j)
                    sd_m |= (<long> 1 << b) | (<long> 1 << a)
                    overlaps_two_bb[counter_2] = wfn.get_overlap(sd_m, deriv=wfn_deriv)
                    counter_2 += 1
            counter_a +=1
        counter_i += 1

    # FIXME: is it possible to not convert it into a numpy array?
    occ_alpha_np = np.array([i for i in occ_alpha[:counter_occ_alpha]], dtype=int)
    occ_beta_np = np.array([i for i in occ_beta[:counter_occ_beta]], dtype=int)
    vir_alpha_np = np.array([i for i in vir_alpha[:counter_vir_alpha]], dtype=int)
    vir_beta_np = np.array([i for i in vir_beta[:counter_vir_beta]], dtype=int)
    occ_beta_np = np.subtract(occ_beta_np, nspatial)
    vir_beta_np = np.subtract(vir_beta_np, nspatial)

    output = np.zeros(3)
    output += np.sum(ham._integrate_sd_sds_zero(occ_alpha_np, occ_beta_np) * overlaps_zero, axis=1)

    output += np.sum(
        ham._integrate_sd_sds_one_alpha(occ_alpha_np, occ_beta_np, vir_alpha_np)
        * overlaps_one_alpha,
        axis=1
    )
    output += np.sum(
        ham._integrate_sd_sds_one_beta(occ_alpha_np, occ_beta_np, vir_beta_np)
        * overlaps_one_beta,
        axis=1
    )
    if counter_occ_alpha > 1 and counter_vir_alpha > 1:
        output[1:] += np.sum(
            ham._integrate_sd_sds_two_aa(occ_alpha_np, occ_beta_np, vir_alpha_np)
            * overlaps_two_aa,
            axis=1
        )
    if counter_occ_alpha > 0 and counter_occ_beta > 0 and counter_vir_alpha > 0 and counter_vir_beta > 0:
        output[1] += np.sum(
            ham._integrate_sd_sds_two_ab(
                occ_alpha_np, occ_beta_np, vir_alpha_np, vir_beta_np
            ) * overlaps_two_ab
        )
    if counter_occ_beta > 1 and counter_vir_beta > 1:
        output[1:] += np.sum(
             ham._integrate_sd_sds_two_bb(occ_alpha_np, occ_beta_np, vir_beta_np)
            * overlaps_two_bb, axis=1
        )
    # FIXME: move to _integrate methods
    return output


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef integrate_sd_wfn_deriv(ham, long sd, wfn):
    r"""Integrate the Hamiltonian with against a Slater determinant and a wavefunction.

    .. math::

        \left< \Phi \middle| \hat{H} \middle| \Psi \right>
        = \sum_{\mathbf{m} \in S_\Phi}
            f(\mathbf{m}) \left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>

    where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
    :math:`\Phi` is the Slater determinant. The :math:`S_{\Phi}` is the set of Slater
    determinants for which :math:`\left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>` is
    chemical Hamiltonian.

    Parameters
    ----------
    sd : int
        Slater Determinant against which the Hamiltonian is integrated.
    wfn : Wavefunction
        Wavefunction against which the Hamiltonian is integrated.
    ham_derivs : np.ndarray(N_derivs)
        Indices of the Hamiltonian parameter against which the integrals are derivatized.

    Returns
    -------
    integrals : np.ndarray(3, N_params)
        Integrals of the given Slater determinant and the wavefunction.
        First element corresponds to the one-electron energy, second to the coulomb energy, and
        third to the exchange energy.

    Raises
    ------
    TypeError
        If ham_derivs is not a one-dimensional numpy array of integers.
    ValueError
        If ham_derivs has any indices than is less than 0 or greater than or equal to nparams.

    Notes
    -----
    Providing only some of the Hamiltonian parameter indices will not make the code any faster.
    The integrals are derivatized with respect to all of Hamiltonian parameters and the
    appropriate derivatives are selected afterwards.

    """
    # pylint: disable=C0103
    cdef int nspatial = ham.nspin / 2

    cdef int[1000] occ_alpha
    cdef int counter_occ_alpha = 0
    cdef int[1000] occ_beta
    cdef int counter_occ_beta = 0
    cdef int[1000] vir_alpha
    cdef int counter_vir_alpha = 0
    cdef int[1000] vir_beta
    cdef int counter_vir_beta = 0

    cdef int i
    for i in range(nspatial):
        if sd & (<long> 1 << i):
            occ_alpha[counter_occ_alpha] = i
            counter_occ_alpha += 1
        else:
            vir_alpha[counter_vir_alpha] = i
            counter_vir_alpha += 1
    for i in range(nspatial, 2 * nspatial):
        if sd & (<long> 1 << i):
            occ_beta[counter_occ_beta] = i
            counter_occ_beta += 1
        else:
            vir_beta[counter_vir_beta] = i
            counter_vir_beta += 1

    cdef double overlaps_zero
    cdef double[:] overlaps_one_alpha = np.zeros(counter_occ_alpha * counter_vir_alpha)
    cdef double[:] overlaps_one_beta = np.zeros(counter_occ_beta * counter_vir_beta)
    cdef double[:] overlaps_two_aa = np.zeros(
        combinations(counter_occ_alpha, 2) * combinations(counter_vir_alpha, 2)
    )
    cdef double[:] overlaps_two_ab = np.zeros(
        counter_occ_alpha * counter_occ_beta * counter_vir_alpha * counter_vir_beta
    )
    cdef double[:] overlaps_two_bb = np.zeros(
        combinations(counter_occ_beta, 2) * combinations(counter_vir_beta, 2)
    )

    overlaps_zero = wfn.get_overlap(sd)

    cdef int j, a, b
    cdef long sd_m
    cdef int counter_1 = 0
    cdef int counter_2 = 0
    cdef int counter_3 = 0
    cdef int counter_i = 0
    cdef int counter_a
    for i in occ_alpha[:counter_occ_alpha]:
        counter_a = 0
        for a in vir_alpha[:counter_vir_alpha]:
            sd_m = sd ^ (<long> 1 << i)
            sd_m |= (<long> 1 << a)
            overlaps_one_alpha[counter_1] = wfn.get_overlap(sd_m)
            counter_1 += 1
            for j in occ_alpha[counter_i + 1 : counter_occ_alpha]:
                for b in vir_alpha[counter_a + 1 : counter_vir_alpha]:
                    sd_m = sd ^ (<long> 1 << i) ^ (<long> 1 << j)
                    sd_m |= (<long> 1 << b) | (<long> 1 << a)
                    overlaps_two_aa[counter_2] = wfn.get_overlap(sd_m)
                    counter_2 += 1
            counter_a += 1
        counter_i += 1
        for j in occ_beta[:counter_occ_beta]:
            for a in vir_alpha[:counter_vir_alpha]:
                for b in vir_beta[:counter_vir_beta]:
                    sd_m = sd ^ (<long> 1 << i) ^ (<long> 1 << j)
                    sd_m |= (<long> 1 << b) | (<long> 1 << a)
                    overlaps_two_ab[counter_3] = wfn.get_overlap(sd_m)
                    counter_3 += 1

    counter_1 = 0
    counter_2 = 0
    counter_i = 0
    for i in occ_beta[:counter_occ_beta]:
        counter_a = 0
        for a in vir_beta[:counter_vir_beta]:
            sd_m = sd ^ (<long> 1 << i)
            sd_m |= (<long> 1 << a)
            overlaps_one_beta[counter_1] = wfn.get_overlap(sd_m)
            counter_1 += 1
            for j in occ_beta[counter_i + 1 : counter_occ_beta]:
                for b in vir_beta[counter_a + 1 : counter_vir_beta]:
                    sd_m = sd ^ (<long> 1 << i) ^ (<long> 1 << j)
                    sd_m |= (<long> 1 << b) | (<long> 1 << a)
                    overlaps_two_bb[counter_2] = wfn.get_overlap(sd_m)
                    counter_2 += 1
            counter_a +=1
        counter_i += 1

    occ_alpha_np = np.array([i for i in occ_alpha[:counter_occ_alpha]])
    occ_beta_np = np.array([i for i in occ_beta[:counter_occ_beta]])
    vir_alpha_np = np.array([i for i in vir_alpha[:counter_vir_alpha]])
    vir_beta_np = np.array([i for i in vir_beta[:counter_vir_beta]])
    occ_beta_np = np.subtract(occ_beta_np, nspatial)
    vir_beta_np = np.subtract(vir_beta_np, nspatial)

    output = np.zeros((3, ham.nparams))

    if counter_occ_alpha > 0 and counter_occ_beta > 0 and counter_vir_alpha > 0:
        output[:] += np.squeeze(
            ham._integrate_sd_sds_deriv_zero_alpha(occ_alpha_np, occ_beta_np, vir_alpha_np) * overlaps_zero,
            axis=2,
        )
        output[:] += np.sum(
            ham._integrate_sd_sds_deriv_one_aa(occ_alpha_np, occ_beta_np, vir_alpha_np)
            * overlaps_one_alpha,
            axis=2,
        )
        output[1] += np.sum(
            ham._integrate_sd_sds_deriv_one_ba(occ_alpha_np, occ_beta_np, vir_alpha_np)
            * overlaps_one_alpha[0],
            axis=1,
        )
    if counter_occ_alpha > 0 and counter_occ_beta > 0 and counter_vir_beta > 0:
        output[:] += np.squeeze(
            ham._integrate_sd_sds_deriv_zero_beta(occ_alpha_np, occ_beta_np, vir_beta_np) * overlaps_zero,
            axis=2,
        )
        output[1] += np.sum(
            ham._integrate_sd_sds_deriv_one_ab(occ_alpha_np, occ_beta_np, vir_beta_np)
            * overlaps_one_beta[0],
            axis=1,
        )
        output[:] += np.sum(
            ham._integrate_sd_sds_deriv_one_bb(occ_alpha_np, occ_beta_np, vir_beta_np) * overlaps_one_beta,
            axis=2,
        )

    if counter_occ_alpha > 1 and counter_vir_alpha > 1:
        output[1:] += np.sum(
            ham._integrate_sd_sds_deriv_two_aaa(occ_alpha_np, occ_beta_np, vir_alpha_np)
            * overlaps_two_aa,
            axis=2,
        )
    if (
        counter_occ_alpha > 0
            and counter_occ_beta > 0
            and counter_vir_alpha > 0
            and counter_vir_beta > 0
    ):
        output[1] += np.sum(
            ham._integrate_sd_sds_deriv_two_aab(occ_alpha_np, occ_beta_np, vir_alpha_np, vir_beta_np)
            * overlaps_two_ab,
            axis=1,
        )
        output[1] += np.sum(
            ham._integrate_sd_sds_deriv_two_bab(occ_alpha_np, occ_beta_np, vir_alpha_np, vir_beta_np)
            * overlaps_two_ab,
            axis=1,
        )
    if counter_occ_beta > 1 and counter_vir_beta > 1:
        output[1:] += np.sum(
            ham._integrate_sd_sds_deriv_two_bbb(occ_alpha_np, occ_beta_np, vir_beta_np)
            * overlaps_two_bb,
            axis=2,
        )

    # FIXME: move to _integrate methods
    return output


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef get_energy_one_proj(wfn, ham, long[:] pspace):
    r"""Return the energy of the Schrodinger equation with respect to a reference wavefunction.

    .. math::

        E \approx \frac{\left< \Phi_{ref} \middle| \hat{H} \middle| \Psi \right>}
                        {\left< \Phi_{ref} \middle| \Psi \right>}

    where :math:`\Phi_{ref}` is some reference wavefunction. Let

    .. math::

        \left| \Phi_{ref} \right> = \sum_{\mathbf{m} \in S}
                                    g(\mathbf{m}) \left| \mathbf{m} \right>

    Then,

    .. math::

        \left< \Phi_{ref} \middle| \hat{H} \middle| \Psi \right>
        = \sum_{\mathbf{m} \in S}
            g^*(\mathbf{m}) \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>

    and

    .. math::

        \left< \Phi_{ref} \middle| \Psi \right> =
        \sum_{\mathbf{m} \in S} g^*(\mathbf{m}) \left< \mathbf{m} \middle| \Psi \right>

    Ideally, we want to use the actual wavefunction as the reference, but, without further
    simplifications, :math:`\Psi` uses too many Slater determinants to be computationally
    tractible. Then, we can truncate the Slater determinants as a subset, :math:`S`, such that
    the most significant Slater determinants are included, while the energy can be tractibly
    computed. This is equivalent to inserting a projection operator on one side of the integral

    .. math::

        \left< \Psi \right| \sum_{\mathbf{m} \in S}
        \left| \mathbf{m} \middle> \middle< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>
        = \sum_{\mathbf{m} \in S}
            f^*(\mathbf{m}) \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>

    Parameters
    ----------
    pspace : np.ndarray(int)
        List of Slater determinants that truncates the wavefunction.
    deriv : {int, None}
        Index of the selected parameters with respect to which the energy is derivatized.

    Returns
    -------
    energy : float
        Energy of the wavefunction with the given Hamiltonian.

    Raises
    ------
    TypeError
        If `refwfn` is not a CIWavefunction, int, or list/tuple of int.

    """
    cdef double[:, ::1] one_int = ham.one_int
    # FIXME: move to Hamiltonian class
    cdef double[:, ::1] two_int_ijij = np.ascontiguousarray(ham._cached_two_int_ijij)
    cdef double[:, ::1] two_int_ijji = np.ascontiguousarray(ham._cached_two_int_ijji)
    cdef double[:, :, :, ::1] two_int = ham.two_int

    cdef double overlap
    cdef double integral
    cdef double norm = 0
    cdef double energy = 0

    cdef int pspace_size = pspace.size
    cdef int i

    cdef double[:] overlaps = np.zeros(pspace.size)
    cdef double[:, :] integrals_sd_wfn = np.zeros((pspace.size, 3))
    for i in range(pspace_size):
        overlap = wfn.get_overlap(pspace[i])
        integral = np.sum(integrate_sd_wfn(ham, pspace[i], wfn))
        overlaps[i] = overlap
        for j in range(3):
            integrals_sd_wfn[i, j] = integrate_sd_wfn(ham, pspace[i], wfn)[j]

        energy += overlap * integral
        norm += overlap ** 2
    energy /= norm
    return energy


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef get_energy_one_proj_deriv(wfn, ham, long[:] pspace):
    cdef double[:, ::1] one_int = ham.one_int
    # FIXME: move to Hamiltonian class
    cdef double[:, ::1] two_int_ijij = np.ascontiguousarray(ham._cached_two_int_ijij)
    cdef double[:, ::1] two_int_ijji = np.ascontiguousarray(ham._cached_two_int_ijji)
    cdef double[:, :, :, ::1] two_int = ham.two_int

    cdef double overlap
    cdef double integral
    cdef double norm = 0
    cdef double energy = 0

    cdef int pspace_size = pspace.size
    cdef int wfn_nparams = wfn.nparams
    cdef int ham_nparams = ham.nparams

    cdef int i, j
    cdef double d_overlap
    cdef double d_integral_wfn
    cdef double[:] d_integral_ham = np.zeros(ham_nparams)
    cdef double[:] d_norm = np.zeros(wfn_nparams)
    cdef double[:] d_energy = np.zeros(wfn_nparams + ham_nparams)
    for i in range(pspace_size):
        overlap = wfn.get_overlap(pspace[i])
        integral = np.sum(integrate_sd_wfn(ham, pspace[i], wfn))

        energy += overlap * integral
        norm += overlap ** 2

        for j in range(wfn_nparams):
            d_overlap = wfn.get_overlap(pspace[i], j)
            d_integral_wfn = np.sum(integrate_sd_wfn(ham, pspace[i], wfn, wfn_deriv=j), axis=0)
            d_norm[j] += 2 * overlap * d_overlap
            d_energy[j] += d_overlap * integral
            d_energy[j] += overlap * d_integral_wfn

        d_integral_ham = np.sum(integrate_sd_wfn_deriv(ham, pspace[i], wfn), axis=0)
        for j in range(ham_nparams):
            d_energy[wfn_nparams + j] += overlap * d_integral_ham[j]
    energy /= norm
    for j in range(wfn_nparams):
        d_energy[j] /= norm
        d_energy[j] -= d_norm[j] * energy / norm
    for j in range(ham_nparams):
        d_energy[wfn_nparams + j] /= norm

    return energy, np.array(d_energy)


# NOTE: energy is computed,
# NOTE: refwfn is always list of slater determinants
# NOTE: only normalization constraint
# NOTE: hardcoded weights
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef system_nonlinear_objective(wfn, ham, long[:] ref_sds, long[:] pspace):
    cdef double[:, ::1] one_int = ham.one_int
    # FIXME: move to Hamiltonian class
    cdef double[:, ::1] two_int_ijij = np.ascontiguousarray(ham._cached_two_int_ijij)
    cdef double[:, ::1] two_int_ijji = np.ascontiguousarray(ham._cached_two_int_ijji)
    cdef double[:, :, :, ::1] two_int = ham.two_int

    cdef double overlap
    cdef double integral
    cdef double norm = 0
    cdef double energy = 0

    cdef int ref_sds_size = ref_sds.size
    cdef int i
    for i in range(ref_sds_size):
        overlap = wfn.get_overlap(ref_sds[i])
        integral = np.sum(integrate_sd_wfn(ham, ref_sds[i], wfn))

        energy += overlap * integral
        norm += overlap ** 2

    energy /= norm

    cdef int pspace_size = pspace.size
    # objective
    cdef double[::1] obj = np.zeros(pspace_size + 1)
    # <SD|H|Psi> - E<SD|Psi> == 0
    for i in range(pspace_size):
        overlap = wfn.get_overlap(pspace[i])
        integral = np.sum(integrate_sd_wfn(ham, pspace[i], wfn))
        obj[i] = integral - energy * overlap
    # constraints
    obj[pspace_size] = (norm - 1) * pspace_size

    return obj

    # ref_coeffs = np.array([wfn.get_overlap(i) for i in ref_sds])
    # norm = np.sum(ref_coeffs * np.array([wfn.get_overlap(i) for i in ref_sds]))
    # energy = np.sum(ref_coeffs * np.sum(np.array([integrate_sd_wfn(ham, i, wfn) for i in ref_sds]), axis=1)) / norm

    # obj = np.zeros(pspace_size + 1)
    # obj[: pspace_size] = np.sum(np.array(
    #     [integrate_sd_wfn(ham, i, wfn) for i in pspace]
    # ), axis=1) - energy * np.array([wfn.get_overlap(i) for i in pspace])
    # obj[pspace_size] = (norm - 1) * pspace_size

    # return obj


# NOTE: energy is computed,
# NOTE: refwfn is always list of slater determinants
# NOTE: only normalization constraint
# NOTE: hardcoded weights
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef system_nonlinear_jacobian(wfn, ham, long[:] ref_sds, long[:] pspace):
    cdef double[:, ::1] one_int = ham.one_int
    # FIXME: move to Hamiltonian class
    cdef double[:, ::1] two_int_ijij = np.ascontiguousarray(ham._cached_two_int_ijij)
    cdef double[:, ::1] two_int_ijji = np.ascontiguousarray(ham._cached_two_int_ijji)
    cdef double[:, :, :, ::1] two_int = ham.two_int

    cdef double overlap
    cdef double integral
    cdef double norm = 0
    cdef double energy = 0

    cdef int ref_sds_size = ref_sds.size
    cdef int pspace_size = pspace.size
    cdef int wfn_nparams = wfn.nparams
    cdef int ham_nparams = ham.nparams

    cdef int i, j
    cdef double d_overlap
    cdef double d_integral_wfn
    cdef double[:] d_integral_ham = np.zeros(ham_nparams)
    cdef double[:] d_norm = np.zeros(wfn_nparams)
    cdef double[:] d_energy = np.zeros(wfn_nparams + ham_nparams)
    for i in range(ref_sds_size):
        overlap = wfn.get_overlap(ref_sds[i])
        integral = np.sum(integrate_sd_wfn(ham, ref_sds[i], wfn))

        energy += overlap * integral
        norm += overlap ** 2

        for j in range(wfn_nparams):
            d_overlap = wfn.get_overlap(ref_sds[i], j)
            d_integral_wfn = np.sum(integrate_sd_wfn(ham, ref_sds[i], wfn, wfn_deriv=j), axis=0)
            d_norm[j] += 2 * overlap * d_overlap
            d_energy[j] += d_overlap * integral
            d_energy[j] += overlap * d_integral_wfn

        d_integral_ham = np.sum(integrate_sd_wfn_deriv(ham, ref_sds[i], wfn), axis=0)
        for j in range(ham_nparams):
            d_energy[wfn_nparams + j] += overlap * d_integral_ham[j]
    energy /= norm
    for j in range(wfn_nparams):
        d_energy[j] /= norm
        d_energy[j] -= d_norm[j] * energy / norm
    for j in range(ham_nparams):
        d_energy[wfn_nparams + j] /= norm

    # jacobian
    cdef double[:, ::1] jac = np.zeros((pspace_size + 1, wfn_nparams + ham_nparams))
    for i in range(pspace_size):
        overlap = wfn.get_overlap(pspace[i])
        integral = np.sum(integrate_sd_wfn(ham, pspace[i], wfn))

        for j in range(wfn_nparams):
            d_integral_wfn = np.sum(integrate_sd_wfn(ham, pspace[i], wfn, wfn_deriv=j), axis=0)
            d_overlap =  wfn.get_overlap(pspace[i], j)
            jac[i, j] = d_integral_wfn - energy * d_overlap - d_energy[j] * overlap

        d_integral_ham = np.sum(integrate_sd_wfn_deriv(ham, pspace[i], wfn), axis=0)
        for j in range(ham_nparams):
            jac[i, wfn_nparams + j] = d_integral_ham[j] - d_energy[wfn_nparams + j] * overlap

    # Add constraints
    for j in range(wfn_nparams):
        jac[pspace_size, j] = d_norm[j] * pspace_size


    return jac
