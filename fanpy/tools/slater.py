r"""Collection of functions used to construct and manipulate Slater determinants.

Slater determinants are represented with bitstrings that describe their occupation. The `0` would
correspond to an unoccupied orbital and the `1` would correspond to the occupied orbital. To ensure
that the lower order excitations are smaller in value, the orbital indices are counted from the
right to the left. For example, ` 00110011` will have the occupied orbitals with indices 0, 1, 4,
and 5.

It will be **assumed** that the creators within a Slater determinant are ordered from smallest to
largest from left to right. i.e. :math:`\ket{1, 3, 5} = a^\dagger_1 a^\dagger_3 a^\dagger_5 \ket{}`

For most of the time, the orbitals are spin orbitals, and their spin is designated by splitting the
orbitals into two blocks. If there are :math:`K` spatial orbitals, then the first block of :math:`K`
spin orbitals are the alpha orbitals, and the second block of :math:`K` spin orbitals are the beta
orbitals. The spin orbitals can be equivalently described with alternating alpha and beta spin
orbitals, but in the current module, the Slater determinant will be assumed to be organized in the
"block" format.

Though Python integers (in the binary format) can be used as a representation of the occupation
vector, the `int` object is used by default.

All references/changes to a Slater determinant should be made using this module, such that if we
decide to change the format of the Slater determinant, only this module needs to be changed.

Functions
---------
is_sd_compatible(sd) : bool
    Check if given Slater determinant is compatible.
occ(sd, i) : bool
    Check if a given Slater determinant has orbital `i` occupied.
occ_indices(sd) : tuple of ints
    Return indices of the occupied orbitals.
vir_indices(sd, norbs) : tuple of ints
    Return the indices of all of the virtual orbitals.
total_occ(sd) : int
    Return the total number of occupied orbitals in a Slater determinant.
is_alpha(i, nspatial) : bool
    Check if index `i` is an alpha spin orbital.
spatial_index(i, nspatial) : int
    Return the spatial orbital index that corresponds to the spin orbital `i`.
annihilate(sd, *indices) : {int, None}
    Annihilate the occupied orbital `i` from a Slater determinant.
create(sd, *indices) : {int, None}
    Create electrons in the orbitals that correspond to the given indices.
excite(sd, *indices) : {int, None}
    Excite electrons from occupied orbitals to virtual orbitals.
ground(nocc, norbs) : int
    Create a ground state Slater determinant.
shared_sd(sd1, sd2) : int
    Return indices of orbitals shared between two Slater determinants.
shared_orbs(sd1, sd2) : tuple of ints
    Return indices of orbitals shared between two Slater determinants
diff_orbs(sd1, sd2) : 2-tuple of tuple of ints
    Return indices of the orbitals that are not shared between two Slater determinants.
combine_spin(alpha_bits, beta_bits, nspatial) : int
    Construct a Slater determinant from the occupation of alpha and beta spin-orbitals.
split_spin(block_sd, nspatial) : 2-tuple of ints
    Split a Slater determinant into the alpha and beta parts.
interleave_index(i, nspatial) : int
    Convert orbital index in block-sd notation to that of interleaved-sd notation.
deinterleave_index(i, nspatial) : int
    Convert an orbital index in interleaved-sd notation to that of block-sd notation.
interleave(block_sd, nspatial) : int
    Convert block-sd to the interleaved-sd form.
deinterleave(shuffled_sd, nspatial) : int
    Turn sd from shuffled form to the block form.
get_spin(sd, nspatial) : float
    Return the spin of the given Slater determinant.
get_seniority(sd, nspatial) : int
    Return the seniority of the given Slater determinant.
sign_perm(jumbled_set, ordered_set=None, is_decreasing=True) : int
    Return the signature of the permutation that sorts a set of annihilators to increasing order.
sign_swap(sd, pos_current, pos_future) : int
    Return the signature of moving a creation operator to a specific position.

"""
# pylint: disable=C0103,C0302
import itertools as it

import numpy as np

import scipy.special


def is_sd_compatible(sd):
    """Check if given Slater determinant is compatible.

    Parameters
    ----------
    sd : int
        Some representation of a Slater determinant.

    Returns
    -------
    bool

    """
    return isinstance(sd, (int, np.integer))


def occ(sd, i):
    """Check if a given Slater determinant has orbital `i` occupied.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
    i : int
        Index of an orbital.

    Returns
    -------
    is_occ : bool
        True if orbital is occupied.
        False if orbital is not occupied.

    """
    return sd is not None and (sd >> i) % 2


def occ_indices(sd):
    """Return indices of the occupied orbitals.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.

    Returns
    -------
    occ_indices : np.array
        Orbitals occupied in the given Slater determinant.

    """
    if sd is None:
        return ()
    return np.array([i for i, occ in enumerate(bin(sd)[:1:-1]) if int(occ)], dtype=int)


def vir_indices(sd, norbs):
    """Return the indices of all of the virtual orbitals.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
    norbs : int
        Total number of orbitals.

    Returns
    -------
    vir_indices : np.ndarray
        Orbitals not occupied in the given Slater determinant.

    """
    # FIXME: there will almost always be more virtuals than occupieds (it will be faster to exclude
    #        occupied from total number of orbitals)
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if sd is None or norbs <= 0:
        return ()
    return np.array([i for i in range(norbs) if not sd & (1 << i)], dtype=int)


def total_occ(sd):
    """Return the total number of occupied orbitals in a Slater determinant.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.

    Returns
    -------
    total _occ : int
        Number of occupied orbitals.

    """
    if sd is None:  # pylint: disable=R1705
        return 0
    else:
        return bin(sd).count("1")


def is_alpha(i, nspatial):
    """Check if index `i` is an alpha spin orbital.

    Parameters
    ----------
    i : int
        Index of the spin orbital in the Slater determinant.
    nspatial : int
        Number of spatial orbitals.

    Returns
    -------
    is_alpha : bool
        True if alpha orbital.
        False if beta orbital.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.
        If `i < 0`.
        If `i > 2*nspatial`.

    """
    if __debug__ and (nspatial <= 0 or i < 0 or i > 2 * nspatial):
        raise ValueError("If `nspatial <= 0` or `i < 0` or `i > 2*nspatial`.")
    return i < nspatial


def spatial_index(i, nspatial):
    """Return the spatial orbital index that corresponds to the spin orbital `i`.

    Parameters
    ----------
    i : int
        Spin orbital index in the Slater determinant.
    nspatial : int
        Number of spatial orbitals.

    Returns
    -------
    ind_spatial : int
        Spatial orbital index that corresponds to the spin orbital `i`.

    Raises
    ------
    ValueError
        If `nspatial <= 0` or `i < 0` or `i > 2*nspatial`.

    """
    if __debug__ and (nspatial <= 0 or i < 0 or i > 2 * nspatial):
        raise ValueError("If `nspatial <= 0` or `i < 0` or `i > 2*nspatial`.")

    if is_alpha(i, nspatial):  # pylint: disable=R1705
        return i
    else:
        return i - nspatial


def spin_index(i, nspatial, spin="beta"):
    """Return the spin orbital index that corresponds to the spatial orbital `i`.

    Parameters
    ----------
    i : int
        Spatial orbital index in the Slater determinant.
    nspatial : int
        Number of spatial orbitals.
    spin : {'alpha', 'beta'}
        Spin of the spin orbital.

    Returns
    -------
    ind_spatial : int
        Spatial orbital index that corresponds to the spin orbital `i`.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.
        If `spin` is not 'alpha' or 'beta'.
        If `i` is less than 0 or greater than or euqal to the number of spatial orbitals.

    """
    if __debug__:
        if nspatial <= 0:
            raise ValueError("Number of spatial orbitals must be greater than zero.")
        if not 0 <= i < nspatial:
            raise ValueError(
                "Spatial orbital index must be greater than or equal to 0 and less than "
                "the number of spatial orbitals."
            )
        if spin not in ["alpha", "beta"]:
            raise ValueError("Spin of the orbital must be either alpha or beta.")

    if spin == "alpha":  # pylint: disable=R1705
        return i
    else:
        return i + nspatial


def annihilate(sd, *indices):
    """Annihilate the occupied orbital `i` from a Slater determinant.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
    indices : int
        The indices of the orbitals to annihilate.

    Returns
    -------
    sd : {int, None}
        Integer that describes the occupation of a Slater determinant as a bitstring.
        If an electron is annihilated in a virtual orbital, `None` is returned.

    """
    for i in indices:
        # if orbital is not occupied
        if not occ(sd, i):
            return None
        # if orbital is not occupied
        sd ^= 1 << i
    return sd


def create(sd, *indices):
    """Create electrons in the orbitals that correspond to the given indices.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
    indices : int
        The indices of the orbitals to create.

    Returns
    -------
    sd : {int, None}
        Integer that describes the occupation of a Slater determinant as a bitstring.
        If an electron is created in an occupied orbital, `None` is returned.

    """
    for i in indices:
        if not occ(sd, i) and sd is not None:
            sd |= 1 << i
        else:
            return None
    return sd


def excite(sd, *indices):
    """Excite electrons from occupied orbitals to virtual orbitals.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
    indices : int
        The indices of the orbitals to annihilate and create.
        The first half contain indices of orbitals to annihilate.
        The second half contain indices of orbitals to create.

    Returns
    -------
    excited_sd : {int, None}
        Integer that describes the occupation of a Slater determinant as a bitstring.
        If an occupied orbital is unoccupied or a virtual orbital is occupied, `None` is returned.

    Raises
    ------
    ValueError
        If the length of indices is not even (cannot evenly split up the indices in two).

    """
    if __debug__ and (len(indices) % 2) != 0:
        raise ValueError("Unqual number of creators and annihilators")
    sd = annihilate(sd, *indices[: len(indices) // 2])
    sd = create(sd, *indices[len(indices) // 2 :])
    return sd


def ground(nocc, norbs):
    """Create a ground state Slater determinant.

    If the number of electrons is odd, then the last electron is put into an alpha orbital.

    Parameters
    ----------
    nocc : int
        Number of occupied spin-orbitals.
    norbs : int
        Total number of spin-orbitals.

    Returns
    -------
    ground_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.

    Raises
    ------
    ValueError
        If `nocc > norbs`.
        If `norbs` is odd.

    Notes
    -----
    Assumes that the spin-orbitals are ordered by energy from lowest to greatest. The occupation is
    assumed to have the alpha block frist, then the beta block.

    """
    if __debug__:
        if nocc > norbs:
            raise ValueError(
                "Number of occupied spin-orbitals must be less than the total number of"
                " spin-orbitals"
            )
        if norbs % 2 != 0:
            raise ValueError("Total number of spin-orbitals must be even")
    return int("0b" + "1" * (nocc // 2 + nocc % 2), 2) + int(
        "0b" + "1" * (nocc // 2) + "0" * (norbs // 2), 2
    )


def shared_sd(sd1, sd2):
    """Return indices of orbitals shared between two Slater determinants.

    Parameters
    ----------
    sd1 : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
    sd2 : int
        Integer that describes the occupation of a Slater determinant as a bitstring.

    Returns
    -------
    shared_sd : int
        Orbitals shared by the two Slater determinants

    """
    return sd1 & sd2


def shared_orbs(sd1, sd2):
    """Return indices of orbitals shared between two Slater determinants.

    Parameters
    ----------
    sd1 : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
    sd2 : int
        Integer that describes the occupation of a Slater determinant as a bitstring.

    Returns
    -------
    shared_orbs : np.ndarray
        Orbitals shared by the two Slater determinants

    """
    return occ_indices(sd1 & sd2)


def diff_orbs(sd1, sd2):
    """Return indices of the orbitals that are not shared between two Slater determinants.

    Parameters
    ----------
    sd1 : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
    sd2 : int
        Integer that describes the occupation of a Slater determinant as a bitstring.

    Returns
    -------
    diff_orbs : 2-tuple of np.ndarray
        First tuple are the occupied orbital indices of `sd1` that are not occupied in `sd2`.
        Second tuple are the occupied orbital indices of `sd2` that are not occupied in `sd1`.

    """
    sd_diff = sd1 ^ sd2
    sd1_diff = sd_diff & sd1
    sd2_diff = sd_diff & sd2
    return (occ_indices(sd1_diff), occ_indices(sd2_diff))


def combine_spin(alpha_bits, beta_bits, nspatial):
    """Construct a Slater determinant from the occupation of alpha and beta spin-orbitals.

    Parameters
    ----------
    alpha_bits : int
        Integer that describes the occupation of alpha spin orbitals as a bitstring.
    beta_bits : int
        Integer that describes the occupation of beta spin orbitals as a bitstring.
    nspatial : int
        Total number of spatial orbitals.

    Returns
    -------
    block_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
        Indices less than `nspatial` correspond to the alpha spin orbitals.
        Indices greater than or equal to `nspatial` correspond to the beta spin orbitals.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.

    Notes
    -----
    Erratic behaviour if the total number of spatial orbitals is less than the the actual number
    (i.e. if there are any occupied orbitals with indices greater than nspatial).

    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if __debug__ and nspatial <= 0:
        raise ValueError("Number of spatial orbitals must be greater than 0.")
    return alpha_bits | (beta_bits << nspatial)


def split_spin(block_sd, nspatial):
    """Split a Slater determinant into the alpha and beta parts.

    Parameters
    ----------
    block_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
        Indices less than `nspatial` correspond to the alpha spin orbitals.
        Indices greater than or equal to `nspatial` correspond to the beta spin orbitals.
    nspatial : int
        Total number of spatial orbitals.

    Returns
    -------
    alpha_bits : int
        Integer that describes the occupation of alpha spin-orbitals as a bitstring.
    beta_bits : int
        Integer that describes the occupation of beta spin-orbitals as a bitstring.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.

    Notes
    -----
    Erratic behaviour if the total number of spatial orbitals is less than the the actual number
    (i.e. if there are any occupied orbitals with indices greater than nspatial).

    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if __debug__ and nspatial <= 0:
        raise ValueError("Number of spatial orbitals must be greater than 0")
    alpha_bits = block_sd & int("0b" + "1" * nspatial, 2)
    beta_bits = block_sd >> nspatial
    return (alpha_bits, beta_bits)


def interleave_index(i, nspatial):
    """Convert orbital index in block-sd notation to that of interleaved-sd notation.

    Parameters
    ----------
    i : int
        Index of orbital in block-sd notation.
    nspatial : int
        Number of spatial orbitals.

    Returns
    -------
    interleave_index : int
        Index of the same orbital in interleaved notation.

    Raises
    ------
    ValueError
        If the index is less than zero.
        If the index is greater than or equal to the number of spin orbitals.

    """
    if __debug__:
        if i < 0:
            raise ValueError("Index must be greater than or equal to zero.")
        if i >= 2 * nspatial:
            raise ValueError("Index must be less than the number of spin orbitals.")

    if i < nspatial:  # pylint: disable=R1705
        return 2 * i
    else:
        return 2 * (i - nspatial) + 1


def deinterleave_index(i, nspatial):
    """Convert an orbital index in interleaved-sd notation to that of block-sd notation.

    Parameters
    ----------
    i : int
        Index of orbital in interleaved-sd notation.
    nspatial : int
        Number of spatial orbitals.

    Returns
    -------
    block_index : int
        Index of the same orbital in block notation.

    Raises
    ------
    ValueError
        If the index is less than zero.
        If the index is greater than or equal to the number of spin orbitals.

    """
    if __debug__:
        if i < 0:
            raise ValueError("Index must be greater than or equal to zero.")
        if i >= 2 * nspatial:
            raise ValueError("Index must be less than the number of spin orbitals.")

    if i % 2 == 0:  # pylint: disable=R1705
        return i // 2
    else:
        return i // 2 + nspatial


def interleave(block_sd, nspatial):
    """Convert block-sd to the interleaved-sd form.

    Block form: alpha1, alpha2, ..., beta1, beta2, ...

    Shuffled form: alpha1, beta1, alpha2, beta2, ...

    Parameters
    ----------
    block_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
        Indices less than nspatial correspond to the alpha spin orbitals.
        Indices greater than or equal to nspatial correspond to the beta spin orbitals.
    nspatial : int
        Total number of spatial orbitals.

    Returns
    -------
    shuffled_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
        Odd indices correspond to the alpha spin orbitals.
        Even indices correspond to the beta spin orbitals.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.

    Notes
    -----
    Erratic behaviour if the total number of spatial orbitals is less than the the actual number
    (i.e. if there are any occupied orbitals with indices greater than nspatial).

    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if __debug__ and nspatial <= 0:
        raise ValueError("Number of spatial orbitals must be greater than 0")

    # OPTION 1
    # shuffled_sd = gmpy2.mpz(0)
    # for i in range(nspatial):
    #     if gmpy2.bit_test(block_sd, i):
    #         shuffled_sd |= 1 << 2 * i
    #     if gmpy2.bit_test(block_sd, i + nspatial):
    #         shuffled_sd |= 1 << 2 * i + 1

    # OPTION 2
    sd_bit = bin(block_sd)[2:]
    sd_bit = "0" * (nspatial * 2 - len(sd_bit)) + sd_bit
    alpha_bit, beta_bit = sd_bit[nspatial:], sd_bit[:nspatial]
    shuffled_bit = "0b" + "".join("".join(i) for i in zip(beta_bit, alpha_bit))
    shuffled_sd = int(shuffled_bit, 2)
    return shuffled_sd

    # OPTION 3
    # shuffled_sd = gmpy2.mpz(0)
    # i = gmpy2.bit_scan1(block_sd, 0)
    # while i is not None:
    #     shuffled_sd = create(shuffled_sd, interleave_index(i, nspatial))
    # return shuffled_sd


def deinterleave(shuffled_sd, nspatial):
    """Turn sd from shuffled form to the block form.

    Block form: alpha1, alpha2, ..., beta1, beta2, ...

    Shuffled form: alpha1, beta1, alpha2, beta2, ...

    Parameters
    ----------
    shuffled_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
        Odd indices correspond to the alpha spin orbitals.
        Even indices correspond to the beta spin orbitals.
    nspatial : int
        Total number of spatial orbitals.

    Returns
    -------
    block_sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
        Indices less than nspatial correspond to the alpha spin orbitals.
        Indices greater than or equal to nspatial correspond to the beta spin orbitals.

    Raises
    ------
    ValueError
        If `nspatial <= 0`.

    Notes
    -----
    Erratic behaviour if the total number of spatial orbitals is less than the the actual number
    (i.e. if there are any occupied orbitals with indices greater than nspatial).

    """
    # FIXME: no check for the total number of orbitals (can be less than actual number)
    if __debug__ and nspatial <= 0:
        raise ValueError("Number of spatial orbitals must be greater than 0")

    # OPTION 1
    # block_sd = gmpy2.mpz(0)
    # for i in range(nspatial):
    #     if gmpy2.bit_test(shuffled_sd, 2 * i):
    #         block_sd |= 1 << i
    #     if gmpy2.bit_test(shuffled_sd, 2 * i + 1):
    #         block_sd |= 1 << i + nspatial

    # OPTION 2
    sd_bit = bin(shuffled_sd)[2:]
    alpha_bit, beta_bit = sd_bit[-1::-2][::-1], sd_bit[-2::-2][::-1]
    alpha_bit = "0" * (nspatial - len(alpha_bit)) + alpha_bit
    beta_bit = "0" * (nspatial - len(beta_bit)) + beta_bit
    block_bit = "0b" + beta_bit + alpha_bit
    block_sd = int(block_bit, 2)
    return block_sd

    # OPTION 3
    # shuffled_sd = gmpy2.mpz(0)
    # i = gmpy2.bit_scan1(block_sd, 0)
    # while i is not None:
    #     shuffled_sd = create(shuffled_sd, deinterleave_index(i, nspatial))
    # return shuffled_sd


def get_spin(sd, nspatial):
    """Return the spin of the given Slater determinant.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
    nspatial : int
        Total number of spatial orbitals.

    Returns
    -------
    spin : float
        Spin of the given Slater determinant.

    """
    alpha_bits, beta_bits = split_spin(sd, nspatial)
    return (0.5) * (total_occ(alpha_bits) - total_occ(beta_bits))


def get_seniority(sd, nspatial):
    """Return the seniority of the given Slater determinant.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
    nspatial : int
        Total number of spatial orbitals.

    Returns
    -------
    seniority : int
        Seniority of the given Slater determinant.

    """
    alpha_bits, beta_bits = split_spin(sd, nspatial)
    return total_occ(alpha_bits ^ beta_bits)


def sign_perm(jumbled_set, ordered_set=None, is_decreasing=True):
    """Return the signature of the permutation that sorts a set of annihilators to increasing order.

    Parameters
    ----------
    jumbled_set : {tuple, list}
        Set of indices of the annihilators.
    ordered_set : {tuple, list}
        Set of indices ordered in strictly increasing order.
        If not provided, then the given indices are ordered.
    is_decreasing : bool
        If True, then the number of transpositions required to get strictly decreasing list is
        returned. Note that the `ordered_set` must still be given in an increasing order.
        Default is False.

    Returns
    -------
    sign : {1, -1}
        Signature of the permutation.

    Raises
    ------
    ValueError
        If `ordered_set` is not strictly increasing.

    """
    if is_decreasing:
        jumbled_set = jumbled_set[::-1]

    # get ordered set
    if ordered_set is None:
        ordered_set = sorted(jumbled_set)
    elif __debug__ and not all(i < j for i, j in zip(ordered_set, ordered_set[1:])):
        raise ValueError("ordered_set must be strictly increasing.")

    sign = 1
    # for each ordered number
    for i in ordered_set:
        for j in jumbled_set:
            # count the number of numbers that are greater than it
            if j > i:
                sign *= -1
            # skip over numbers to the right
            elif j == i:
                break
    return sign


def sign_swap(sd, pos_current, pos_future):
    """Return the signature of moving a creation operator to a specific position.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
    pos_current : int
        Index of the orbital that needs to be moved.
    pos_future : int
        Index to which the orbital is moved.

    Returns
    -------
    sign : {1, -1}
        Signature of the permutation.

    Raises
    ------
    ValueError
        If Slater determinant is None.
        If position is not a positive integer.
        If current orbital position is not occupied.

    """
    if __debug__:
        if sd is None:
            raise ValueError("Bad Slater determinant is given.")
        if not (isinstance(pos_current, (int, np.integer)) and pos_current >= 0):
            raise ValueError("The current orbital position must be a positive integer.")
        if not (isinstance(pos_future, (int, np.integer)) and pos_future >= 0):
            raise ValueError("The future orbital position must be a positive integer.")
        if not occ(sd, pos_current):
            raise ValueError("Given orbital is not occupied in the given Slater determinant.")

    # sd = gmpy2.mpz(sd)
    if pos_current < pos_future:
        # remove everything before pos_current (including pos_current)
        # remove everything after pos_future (excluding pos_future)
        num_trans = bin(sd)[:1:-1][pos_current + 1 : pos_future + 1].count("1")
    else:
        # remove everything after pos_current (including pos_current)
        # remove everything before pos_future (excluding pos_future)
        num_trans = bin(sd)[:1:-1][pos_future:pos_current].count("1")

    if num_trans % 2 == 0:  # pylint: disable=R1705
        return 1
    else:
        return -1


def sign_excite(sd, annihilators, creators):
    r"""Return the signature of applying annihilators then creators to the Slater determinant.

    .. math::

        a^\dagger_{j_N} \dots a^\dagger_{j_1} a_{i_M} \dots a_{i_1} \left| \Phi \right>
        = sign \hat{E}_{i_1 \dots i_M}^{j_N \dots j_1} \left| \Phi \right>

    where :math:`\{a_{i_1} \dots a_{i_M}\}` is the set of annihilators,
    :math:`\{a^\dagger_{j_1} \dots a^\dagger_{j_M}\}` is the set of creators and `sign` is the
    signature resulting from these operations.

    Parameters
    ----------
    sd : int
        Integer that describes the occupation of a Slater determinant as a bitstring.
    annihilators : list/tuple of int
        Indices of the orbitals that will be annihilated, in order.
    creators : list/tuple of int
        Indices of the orbitals that will be created, in order, after annihilating the selected
        orbitals.

    Returns
    -------
    sign : {1, -1}
        Sign of the Slater determinant after excitation.

    Raises
    ------
    ValueError
        If Slater determinant is None.
        If position is not a positive integer.
        If current orbital position is not occupied.

    """
    sign = 1
    for i in annihilators:
        if __debug__ and not occ(sd, i):
            raise ValueError(
                "Given Slater determinant cannot be excited using the given creators "
                "and annihilators."
            )
        # move creator i in the Slater determinant to the front (cancelling out the annihilator)
        sign *= sign_swap(sd, i, 0)
        sd = annihilate(sd, i)

    for i in creators:
        sd = create(sd, i)
        if __debug__ and sd is None:
            raise ValueError(
                "Given Slater determinant cannot be excited using the given creators "
                "and annihilators."
            )
        # move creator i from the first poisition to its position
        sign *= sign_swap(sd, i, 0)

    return sign


def sign_excite_array(occs, annihilators, creators, nspin):
    r"""Return the sign of the Slater determinant after applying excitation operators.

    Parameters
    ----------
    occs : np.ndarray(N,)
        Indices of the spin orbitals that are occupied in the Slater determinant.
    annihilators : np.ndarray(L, M_a)
        Indices of the annihilators that are applied to the Slater determinant.
        First dimension corresponds to different `M_a`-body annihilators.
        Second dimension corresponds to the indices of each one-body annihilator within the
        `M_a`-body annihilator.
        Each `M_a`-body annihilator must have ordered one-body annihilators (smallest to largest).
    creators : np.ndarray(L, M_c)
        Indices of the creators that are applied to the Slater determinant (after the annihilators).
        First dimension corresponds to different `M_c`-body creators.
        Second dimension corresponds to the indices of each one-body creator within the `M_c`-body
        creators.
        Each `M_c`-body annihilator must have ordered one-body creators (smallest to largest).
    nspin : int
        Number of spin orbitals.

    Returns
    -------
    signs : np.ndarray(M_a, M_c)
        Signs of the Slater determinant after applying the excitation operators.

    Notes
    -----
    Assume that we are applying the annihilators then creators from right to left (to a ket). This
    means that the smallest index annihilators are applied first (so they would be ordered from
    largest to smallest from left to right) and the largest index creator are applied first (so they
    would be ordered from smallest to largest from left to right). For example, :math:`a^\dagger_i
    a^\dagger_j a_l a_k` where i < j and k < l. Though it doesn't affect the end result, it will be
    assumed that the creators in the Slater determinant are ordered from largest to smallest (from
    left to right).

    """
    indices = np.zeros(nspin, dtype=int)
    indices[occs] = np.arange(occs.size)

    shared_indices = np.tile(occs, [annihilators.shape[0], 1])

    num_jumps_annihilate = np.zeros(annihilators.shape[0])
    # apply each one-body annihilator (for each M_a body annihilator) in order
    for one_annihilators in annihilators.T:
        # count the number of jumps each one-body annihilator must make to get to the correct
        # position. Note that the annihilator jumps from the side with the largest index to the
        # smallest)
        num_jumps_annihilate += np.sum(one_annihilators[:, None] < shared_indices, axis=1)
        # remove the index that have been removed (the annihilator made it to the correct index and
        # annihilated it)
        shared_indices[np.arange(one_annihilators.size), indices[one_annihilators]] = -1

    # move the one-body creator (for each M_c body creator) in order
    num_jumps_create = np.zeros((annihilators.shape[0], creators.shape[0]))
    for one_creators in creators.T:
        # count the number of jumps each one-body creator must make to get to the correct position.
        # Note that the creator jumps from the side with the largest index to the smallest)
        num_jumps_create += np.sum(shared_indices[:, :, None] > one_creators[None, None, :], axis=1)

    sign = num_jumps_annihilate[:, None] + num_jumps_create
    sign = (-1) ** sign
    # Reverse the ordering of creators. Above, the smallest index creator was applied first, but
    # conventional ordering of creators in an excitation operator has the largest index creators
    # being applied first
    sign *= (-1) ** (creators.shape[1] // 2 % 2)
    return sign


def shared_indices_remove_one_index(indices):
    """Return the indices where one index is removed.

    Parameters
    ----------
    indices : np.ndarray(N,)
        Indices of the orbitals.

    Returns
    -------
    shared_indices : np.ndarray(N, N-1)
        Indices where one of the indices are removed.
        First index of the array is the index of the element removed.
        Second index of the array is the elements are not removed.

    """
    shared_indices = np.tile(indices, [indices.size, 1])
    shared_indices = shared_indices[~np.identity(indices.size, dtype=bool)]
    shared_indices = shared_indices.reshape(indices.size, max(indices.size - 1, 0))
    shared_indices = shared_indices.astype(int)
    return shared_indices


def spatial_to_spin_indices(spatial_indices, nspatial, to_beta=True):
    """Return the spin indices that corresponds to the spatial indices.

    Parameters
    ----------
    spatial_indices : np.ndarray
        Spatial orbital indices.
    nspatial : int
        Number of spatial orbitals.
    to_beta : bool
        True if the produced spin orbitals have spin beta.
        False if the produced spin orbitals have spin alpha.
        By default, beta spin orbitals are produced.

    Returns
    -------
    ind_spatial : int
        Spin orbital indices that corresponds to the spatial indices.

    Raises
    ------
    ValueError
        If `nspin <= 0`.
        If `spin` is not 'alpha' or 'beta'.

    Notes
    -----
    Array analog of function `slater.spin_index`

    """
    # pylint: disable=R1705
    if __debug__:
        if not nspatial > 0:
            raise ValueError("Number of spatial orbitals must be greater than zero.")
        if not np.all(np.logical_and(spatial_indices >= 0, spatial_indices < nspatial)):
            raise ValueError(
                "Spatial orbital index must be greater than or equal to 0 and less than "
                "the number of spatial orbitals."
            )
    if to_beta:
        return spatial_indices + nspatial
    else:
        return spatial_indices


def excite_bulk(sd, occ_indices, vir_indices, excite_order):  # pylint: disable=W0621
    """Return all possible excitations from the given occupied and virtual orbitals.

    Parameters
    ----------
    sd : int
        Slater determinant to which the excitation operators is applied.
    occ_indices : np.ndarray
        Tuple of occupied orbital indices.
    vir_indices : np.ndarray
        Tuple of virtual orbital indices.
    excite_order : int
        Order of excitation.

    Returns
    -------
    excited_sds : np.ndarray(dtype=int)
        Integers that describe the occupations of a excitations as a bitstring.

    Raises
    ------
    ValueError
        If the length of indices is not even (cannot evenly split up the indices in two).

    """
    if __debug__:
        occ_indices = np.array(occ_indices)
        vir_indices = np.array(vir_indices)
        if not (isinstance(excite_order, (int, np.integer)) and excite_order >= 0):
            raise TypeError("Order of excitation must be an integer greater than or equal to zero.")

    # def fromiter(iterator, dtype, ydim, count):
    #     return np.fromiter(
    #         it.chain.from_iterable(iterator), dtype, count=int(count)
    #     ).reshape(-1, ydim)

    if excite_order == 0:
        return np.array([sd], dtype=int)

    occ_indices = np.fromiter(
        it.chain.from_iterable(it.combinations(occ_indices.tolist(), excite_order)),
        int,
        count=excite_order * int(scipy.special.comb(occ_indices.size, excite_order)),
    ).reshape(-1, excite_order)
    occ_indices = np.left_shift(1, occ_indices)
    if excite_order >= 2:
        occ_indices = np.bitwise_or.reduce(occ_indices.T)  # pylint: disable=E1101

    vir_indices = np.fromiter(
        it.chain.from_iterable(it.combinations(vir_indices.tolist(), excite_order)),
        int,
        count=excite_order * int(scipy.special.comb(vir_indices.size, excite_order)),
    ).reshape(-1, excite_order)
    vir_indices = np.left_shift(1, vir_indices)
    if excite_order >= 2:
        vir_indices = np.bitwise_or.reduce(vir_indices.T)  # pylint: disable=E1101

    return np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_indices)[:, None], vir_indices[None, :]))


def excite_bulk_two_ab(sd, occ_alpha, occ_beta, vir_alpha, vir_beta):
    """Return all possible excitations from the given occupied and virtual orbitals.

    Parameters
    ----------
    sd : int
        Slater determinant to which the excitation operators is applied.
    occ_alpha : np.ndarray
        Tuple of occupied alpha orbital indices.
    occ_beta : np.ndarray
        Tuple of occupied beta orbital indices.
    vir_alpha : np.ndarray
        Tuple of virtual alpha orbital indices.
    vir_beta : np.ndarray
        Tuple of virtual beta orbital indices.
    excite_order : int
        Order of excitation.

    Returns
    -------
    excited_sds : np.ndarray(dtype=int)
        Integers that describe the occupations of a excitations as a bitstring.

    Raises
    ------
    ValueError
        If the length of indices is not even (cannot evenly split up the indices in two).

    """
    if __debug__:
        occ_alpha = np.array(occ_alpha)
        occ_beta = np.array(occ_beta)
        vir_alpha = np.array(vir_alpha)
        vir_beta = np.array(vir_beta)
    # pylint: disable=W0621
    occ_indices = np.fromiter(
        it.chain.from_iterable(it.product(occ_alpha.tolist(), occ_beta.tolist())),
        int,
        count=2 * len(occ_alpha) * len(occ_beta),
    ).reshape(-1, 2)
    occ_indices = np.left_shift(1, occ_indices)
    occ_indices = np.bitwise_or.reduce(occ_indices.T)  # pylint: disable=E1101

    vir_indices = np.fromiter(
        it.chain.from_iterable(it.product(vir_alpha.tolist(), vir_beta.tolist())),
        int,
        count=2 * len(vir_alpha) * len(vir_beta),
    ).reshape(-1, 2)
    vir_indices = np.left_shift(1, vir_indices)
    vir_indices = np.bitwise_or.reduce(vir_indices.T)  # pylint: disable=E1101

    return np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_indices)[:, None], vir_indices[None, :]))


def sign_excite_one(occ_indices, vir_indices):  # pylint: disable=W0621
    """Return the signs of Slater determinants after first-order excitation.

    Parameters
    ----------
    occ_indices : np.ndarray
        Orbitals occupied in the given Slater determinant.
    vir_indices : np.ndarray
        Orbitals not occupied in the given Slater determinant.

    Returns
    -------
    signs : np.ndarray
        Signs of the Slater determinants after first-order excitations in the same order as
        `bulk_excite`.

    """
    num_occ = occ_indices.size

    bin_sizes = [0] * (num_occ + 1)
    counter = 0
    for a in vir_indices:
        while counter < num_occ and a > occ_indices[counter]:
            counter += 1
        bin_sizes[counter] += 1

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
    return np.array(output)


def sign_excite_two(occ_indices, vir_indices):  # pylint: disable=W0621,R0912
    """Return the signs of Slater determinants after second-order excitation (same spin).

    Parameters
    ----------
    occ_indices : np.ndarray
        Orbitals occupied in the given Slater determinant.
    vir_indices : np.ndarray
        Orbitals not occupied in the given Slater determinant.

    Returns
    -------
    signs : np.ndarray
        Signs of the Slater determinants after second-order excitations in the same order as
        `bulk_excite`.

    """
    num_occ = len(occ_indices)

    bin_sizes = [0] * (num_occ + 1)
    # cdef long[:] bin_sizes = np.zeros(num_occ + 1, dtype=int)
    # assume occ_indices is ordered
    # assume vir_indices is ordered
    counter = 0
    for a in vir_indices:
        while counter < num_occ and a > occ_indices[counter]:
            counter += 1
        bin_sizes[counter] += 1

    output = []
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

    return np.array(output)


def sign_excite_two_ab(occ_alpha, occ_beta, vir_alpha, vir_beta):  # pylint: disable=R0912
    """Return the signs of Slater determinants after first-order excitations (different spin).

    One occupied alpha orbital and one occupied beta orbital are excited to virtual alpha and beta
    orbitals.

    Parameters
    ----------
    occ_alpha : np.ndarray
        Alpha orbitals occupied in the given Slater determinant.
    occ_beta : np.ndarray
        Beta orbitals occupied in the given Slater determinant.
    vir_alpha : np.ndarray
        Alpha orbitals not occupied in the given Slater determinant.
    vir_beta : np.ndarray
        Beta orbitals not occupied in the given Slater determinant.

    Returns
    -------
    signs : np.ndarray
        Signs of the Slater determinants after second-order excitations in the same order as
        `bulk_excite`.

    """
    num_occ_alpha = len(occ_alpha)
    num_occ_beta = len(occ_beta)

    bin_sizes_alpha = [0] * (num_occ_alpha + 1)
    bin_sizes_beta = [0] * (num_occ_beta + 1)
    # cdef long[:] bin_sizes_alpha = np.zeros(num_occ_alpha + 1, dtype=int)
    # cdef long[:] bin_sizes_beta = np.zeros(num_occ_beta + 1, dtype=int)
    # assume occ_alpha, vir_alpha, occ_beta, vir_beta are ordered

    counter = 0
    for a in vir_alpha:
        while counter < num_occ_alpha and a > occ_alpha[counter]:
            counter += 1
        bin_sizes_alpha[counter] += 1

    counter = 0
    for a in vir_beta:
        while counter < num_occ_beta and a > occ_beta[counter]:
            counter += 1
        bin_sizes_beta[counter] += 1

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
    return np.array(output)
