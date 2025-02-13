"""Functions used to obtain the density matrices.

Functions
---------
add_one_density(matrices, i, j, val, orbtype)
    Adds value to the one electron density matrix appropriately.
add_two_density(matrices, i, j, k, l, val, orbtype)
    Adds value to the two electron density matrix appropriately.
density_matrix(sd_coeffs, civec, nspatial, is_chemist_notation=False, val_threshold=0,
               orbtype='restricted')
    Returns the one and two electron density matrices.
"""

from fanpy.tools import slater

import numpy as np


# FIXME: incredibly slow/bad approach
# TODO: add density_wfn_wfn, density_wfn_sd, density_sd_sd?
def add_one_density(matrices, spin_i, spin_j, val, orbtype):  # pylint: disable=R0912
    r"""Add some value to the appropriate density matrix element.

    .. math::

        \left< \Phi_1 \middle| a_i a_j^\dagger \middle| \Phi_2 \right>

    Parameters
    ----------
    matrices : tuple/list of np.ndarray
        List of one electron density matrices.
        If 1-tuple/list, then restricted or generalized orbitals.
        If 2-tuple/list, then unrestricted orbitals (alpha-alpha and beta-beta components).
    spin_i : int
        Spin orbital index of the density matrix.
    spin_j : int
        Spin orbital index of the density matrix.
    val : float
        Value that will be added to the density matrix.
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital.

    Raises
    ------
    TypeError
        If matrices are not a list of numpy arrays.
        If matrices are not two dimensional.
        If matrices are not square.
    ValueError
        If restricted or generalized orbitals and number of matrices is not one.
        If unrestricted orbitals and number of matrices is not two.
        If orbital type is not one of 'restricted', 'unrestricted', 'generalized'.

    """
    if __debug__:
        if not (isinstance(matrices, list) and all(isinstance(i, np.ndarray) for i in matrices)):
            raise TypeError("Matrices must be given as a list of numpy arrays")

        if any(len(i.shape) != 2 for i in matrices):
            raise TypeError("All matrices must be two dimensional")
        if any(j != matrices[0].shape[0] for i in matrices for j in i.shape):
            raise TypeError("All matrices must be square")

        if orbtype not in ["restricted", "unrestricted", "generalized"]:
            raise ValueError("Orbital type must be one of 'restricted', 'unrestricted', and 'generalized'.")

        if orbtype in ["restricted", "generalized"] and len(matrices) != 1:
            raise ValueError(
                "Density matrix must be given as a list of one numpy array for" " restricted and generalized orbitals"
            )
        if orbtype in ["unrestricted"] and len(matrices) != 2:
            raise ValueError("Density matrix must be given as a list of two numpy arrays for" " unrestricted orbitals")

    if orbtype == "restricted":
        nspatial = matrices[0].shape[0]
        spatial_i = slater.spatial_index(spin_i, nspatial)
        spatial_j = slater.spatial_index(spin_j, nspatial)
        if slater.is_alpha(spin_i, nspatial) == slater.is_alpha(spin_j, nspatial):
            # CHECK: is there a factor of 2 missing here?
            matrices[0][spatial_i, spatial_j] += val

    elif orbtype == "unrestricted":
        nspatial = matrices[0].shape[0]
        spatial_i = slater.spatial_index(spin_i, nspatial)
        spatial_j = slater.spatial_index(spin_j, nspatial)
        # if spins are both alpha
        if slater.is_alpha(spin_i, nspatial) and slater.is_alpha(spin_j, nspatial):
            matrices[0][spatial_i, spatial_j] += val
        # if spins are both beta
        elif not slater.is_alpha(spin_i, nspatial) and not slater.is_alpha(spin_j, nspatial):
            matrices[1][spatial_i, spatial_j] += val

    else:  # if orbtype == "generalized":
        matrices[0][spin_i, spin_j] += val


# FIXME: too many branches
def add_two_density(matrices, spin_i, spin_j, spin_k, spin_l, val, orbtype):
    r"""Add some value to the appropriate one electron density matrix element.

    .. math::

        \left< \Phi_1 \middle| a_i a_j a_k^\dagger a_l^\dagger \middle| \Phi_2 \right>

    Parameters
    ----------
    matrices : tuple/list of np.ndarray
        List of two electron density matrices.
        If 1-tuple/list, then restricted or generalized orbitals.
        If 3-tuple/list, then unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta,
        and beta-beta-beta-beta components).
    spin_i : int
        Spin orbital index of the density matrix.
    spin_j : int
        Spin orbital index of the density matrix.
    spin_k : int
        Spin orbital index of the density matrix.
    spin_l : int
        Spin orbital index of the density matrix.
    val : float
        Value that will be added to the density matrix.
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital.

    Raises
    ------
    TypeError
        If matrices are not a list of numpy arrays.
    ValueError
        If restricted or generalized orbitals and number of matrices is not one.
        If unrestricted orbitals and number of matrices is not two.
        If orbital type is not one of 'restricted', 'unrestricted', 'generalized'.

    Notes
    -----
    Assumes that the spin orbital indices are given with the physicist's notation.

    """
    # pylint: disable=R0912
    if __debug__:
        if not (isinstance(matrices, list) and all(isinstance(i, np.ndarray) for i in matrices)):
            raise TypeError("Matrices must be given as a list of numpy arrays")

        if any(len(i.shape) != 4 for i in matrices):
            raise TypeError("All matrices must be four dimensional")
        if any(j != matrices[0].shape[0] for i in matrices for j in i.shape):
            raise TypeError("All matrices should have the same dimension along all of the axes")

        if orbtype not in ["restricted", "unrestricted", "generalized"]:
            raise ValueError("Orbital type must be one of 'restricted', 'unrestricted', and 'generalized'.")

        if orbtype in ["restricted", "generalized"] and len(matrices) != 1:
            raise ValueError(
                "Density matrix must be given as a list of one numpy array for" " restricted and generalized orbitals"
            )
        if orbtype in ["unrestricted"] and len(matrices) != 3:
            raise ValueError(
                "Density matrix must be given as a list of three numpy arrays for" " unrestricted orbitals"
            )

    if orbtype == "restricted":
        nspatial = matrices[0].shape[0]
        spatial_i = slater.spatial_index(spin_i, nspatial)
        spatial_j = slater.spatial_index(spin_j, nspatial)
        spatial_k = slater.spatial_index(spin_k, nspatial)
        spatial_l = slater.spatial_index(spin_l, nspatial)
        # if i and k have same spin and j and l have same spin
        if slater.is_alpha(spin_i, nspatial) == slater.is_alpha(spin_k, nspatial) and slater.is_alpha(
            spin_j, nspatial
        ) == slater.is_alpha(spin_l, nspatial):
            matrices[0][spatial_i, spatial_j, spatial_k, spatial_l] += val

    elif orbtype == "unrestricted":
        nspatial = matrices[0].shape[0]
        spatial_i = slater.spatial_index(spin_i, nspatial)
        spatial_j = slater.spatial_index(spin_j, nspatial)
        spatial_k = slater.spatial_index(spin_k, nspatial)
        spatial_l = slater.spatial_index(spin_l, nspatial)
        # if all spins are alpha
        if (
            slater.is_alpha(spin_i, nspatial)
            and slater.is_alpha(spin_k, nspatial)
            and slater.is_alpha(spin_j, nspatial)
            and slater.is_alpha(spin_l, nspatial)
        ):
            matrices[0][spatial_i, spatial_j, spatial_k, spatial_l] += val
        # if alpha beta alpha beta
        elif (
            slater.is_alpha(spin_i, nspatial)
            and slater.is_alpha(spin_k, nspatial)
            and not slater.is_alpha(spin_j, nspatial)
            and not slater.is_alpha(spin_l, nspatial)
        ):
            matrices[1][spatial_i, spatial_j, spatial_k, spatial_l] += val
        # if all spins are beta
        elif (
            not slater.is_alpha(spin_i, nspatial)
            and not slater.is_alpha(spin_k, nspatial)
            and not slater.is_alpha(spin_j, nspatial)
            and not slater.is_alpha(spin_l, nspatial)
        ):
            matrices[2][spatial_i, spatial_j, spatial_k, spatial_l] += val

    else:  # if orbtype == "generalized":
        matrices[0][spin_i, spin_j, spin_k, spin_l] += val


# FIXME: make input of Wavefunction and CIWavefunction instead of sd_coeffs, civec, nspatial, ...
# TODO: generalize to arbitrary order density matrix
# FIXME: too many branches, too many statements
def density_matrix(sd_coeffs, civec, nspatial, is_chemist_notation=False, val_threshold=0, orbtype="restricted"):
    r"""Return the first and second order density matrices.

    Second order density matrix uses the Physicist's notation:

    .. math::

        \Gamma_{ijkl} = \left< \Psi \middle| a_i^\dagger a_k^\dagger a_l a_j \middle| \Psi \right>

    Chemist's notation is also implemented

    .. math::

        \Gamma_{ijkl} = \left< \Psi \middle| a_i^\dagger a_j^\dagger a_k a_l \middle| \Psi \right>

    Parameters
    ----------
    sd_coeffs : list of float
        Slater determinant coefficients.
    civec : list of int
        Slater determinant.
    nspatial : int
        Number of spatial orbitals.
    is_chemist_notation : bool
        True if chemist's notation.
        False if physicist's notation.
        Default is Physicist's notation.
    val_threshold : float
        Threshold for truncating the density matrice entries.
        Skips all of the Slater determinants whose maximal sum of contributions to density matrices
        is less than threshold value.
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital.

    Returns
    -------
    one_densities : tuple of np.ndarray
        One electron density matrix.
        For spatial and generalized orbitals, 1-tuple of np.ndarray.
        For unretricted spin orbitals, 2-tuple of np.ndarray.
    two_densities : tuple of np.ndarray
        Two electron density matrix.
        For spatial and generalized orbitals, 1-tuple of np.ndarray.
        For unrestricted orbitals, 3-tuple of np.ndarray.

    Raises
    ------
    TypeError
        If the orbital type is not one of 'restricted', 'unrestricted', 'generalized'

    """
    # pylint: disable=C0103,R0912,R0915
    # TODO: generalize to arbitrary order density matrix
    # sort coefficients and sd's by the magintude of coefficient (useful for truncating)
    sorted_x, sorted_sd = zip(*sorted(zip(sd_coeffs, civec), key=lambda x: abs(x[0]), reverse=True))
    num_sds = len(sorted_sd)

    if __debug__ and orbtype not in ["restricted", "unrestricted", "generalized"]:
        raise ValueError("Orbital type must be one of 'restricted', 'unrestricted', and 'generalized'.")

    # initiate output
    one_densities = []
    two_densities = []
    if orbtype == "restricted":
        one_densities = [np.zeros((nspatial,) * 2)]
        two_densities = [np.zeros((nspatial,) * 4)]
    elif orbtype == "unrestricted":
        one_densities = [np.zeros((nspatial,) * 2) for i in range(2)]
        two_densities = [np.zeros((nspatial,) * 4) for i in range(3)]
    else:  # if orbtype == "generalized":
        one_densities = [np.zeros((2 * nspatial,) * 2)]
        two_densities = [np.zeros((2 * nspatial,) * 4)]

    for count1, sd1 in enumerate(sorted_sd):
        # truncation condition
        if (sorted_x[count1] * (num_sds - count1)) ** 2 < val_threshold:
            break
        for count2, sd2 in enumerate(sorted_sd[count1:]):
            # increatement counter (because enumerate)
            count2 += count1
            # truncation condition
            if abs(sorted_x[count1] * sorted_x[count2]) * (num_sds - count1) ** 2 < val_threshold:
                break

            # orbitals that are not shared by the two determinants
            left_diff, right_diff = slater.diff_orbs(sd1, sd2)
            shared_indices = np.array(slater.shared_orbs(sd1, sd2))

            # moving all the shared orbitals toward one another (in the middle)
            num_transpositions_0 = np.sum(shared_indices[:, np.newaxis] < np.array(left_diff))
            num_transpositions_1 = np.sum(shared_indices[:, np.newaxis] < np.array(right_diff))
            num_transpositions = num_transpositions_0 + num_transpositions_1
            sign = (-1) ** num_transpositions

            # contributing value
            val = sorted_x[count1] * sorted_x[count2] * sign

            # check for number symmetery
            if len(right_diff) != len(left_diff):
                continue

            # FIXME: use symmetry instead
            # if they're the same
            if len(left_diff) == 0:
                for ind, i in enumerate(shared_indices):
                    add_one_density(one_densities, i, i, val, orbtype=orbtype)
                    for j in shared_indices[ind + 1 :]:
                        add_two_density(two_densities, i, j, i, j, val, orbtype=orbtype)
                        add_two_density(two_densities, j, i, j, i, val, orbtype=orbtype)
                        add_two_density(two_densities, i, j, j, i, -val, orbtype=orbtype)
                        add_two_density(two_densities, j, i, i, j, -val, orbtype=orbtype)
            # if single excitation
            elif len(left_diff) == 1:
                (i,) = left_diff
                (k,) = right_diff
                add_one_density(one_densities, i, k, val, orbtype=orbtype)
                add_one_density(one_densities, k, i, val, orbtype=orbtype)
                for j in shared_indices:
                    add_two_density(two_densities, i, j, k, j, val, orbtype=orbtype)
                    add_two_density(two_densities, j, i, j, k, val, orbtype=orbtype)
                    add_two_density(two_densities, k, j, i, j, val, orbtype=orbtype)
                    add_two_density(two_densities, j, k, j, i, val, orbtype=orbtype)
                    add_two_density(two_densities, i, j, j, k, -val, orbtype=orbtype)
                    add_two_density(two_densities, j, i, k, j, -val, orbtype=orbtype)
                    add_two_density(two_densities, j, k, i, j, -val, orbtype=orbtype)
                    add_two_density(two_densities, k, j, j, i, -val, orbtype=orbtype)
            # if double excitation
            elif len(left_diff) == 2:
                i, j = left_diff
                k, l = right_diff  # noqa: E741
                add_two_density(two_densities, i, j, k, l, val, orbtype=orbtype)
                add_two_density(two_densities, j, i, l, k, val, orbtype=orbtype)
                add_two_density(two_densities, k, l, i, j, val, orbtype=orbtype)
                add_two_density(two_densities, l, k, j, i, val, orbtype=orbtype)
                add_two_density(two_densities, i, j, l, k, -val, orbtype=orbtype)
                add_two_density(two_densities, j, i, k, l, -val, orbtype=orbtype)
                add_two_density(two_densities, l, k, i, j, -val, orbtype=orbtype)
                add_two_density(two_densities, k, l, j, i, -val, orbtype=orbtype)
    # change notation if necessary
    if is_chemist_notation:
        two_densities = [np.einsum("ijkl->iklj", i) for i in two_densities]
    return tuple(one_densities), tuple(two_densities)
