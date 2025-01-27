"""Base Hamiltonian for unrestricted orbitals."""

from fanpy.ham.base import BaseHamiltonian

import numpy as np


# FIXME: ordering of the words is not consistent with the UnrestrictedMolecularHamiltonian
class BaseUnrestrictedHamiltonian(BaseHamiltonian):
    """Base class for Hamiltonian with unrestricted orbitals.

    Attributes
    ----------
    one_int : 2-tuple of np.ndarray(K, K)
        One-electron integrals.
    two_int : 3-tuple of np.ndarray(K, K, K, K)
        Two-electron integrals.

    Properties
    ----------
    nspin : int
        Number of spin orbitals.
    nspatial : int
        Number of spatial orbitals.

    Methods
    -------
    __init__(self, one_int, two_int)
        Initialize the Hamiltonian.
    assign_integrals(self, one_int, two_int)
        Assign the one- and two-electron integrals.
    orb_rotate_jacobi(self, jacobi_indices, theta)
        Rotate orbitals using Jacobi matrix.
    orb_rotate_matrix(self, matrix)
        Rotate orbitals with a transformation matrix.

    Abstract Methods
    ----------------
    integrate_sd_wfn(self, sd, wfn, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.
    integrate_sd_sd(self, sd1, sd2, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """

    # pylint: disable=W0223
    def __init__(self, one_int, two_int):
        """Initialize the Hamiltonian.

        Parameters
        ----------
        one_int : 2-tuple of np.ndarray(K, K)
            One electron integrals.
        two_int : 3-tuple of np.ndarray(K, K, K, K)
            Two electron integrals.

        """
        # pylint: disable=W0231
        self.assign_integrals(one_int, two_int)

    @property
    def nspin(self):
        """Return the number of spin orbitals.

        Returns
        -------
        nspin : int
            Number of spin orbitals.

        """
        return self.one_int[0].shape[0] + self.one_int[1].shape[0]

    def assign_integrals(self, one_int, two_int):
        """Assign the one- and two-electron integrals.

        Parameters
        ----------
        one_int : 2-tuple of np.ndarray(K, K)
            One-electron integrals for unrestricted orbitals.
        two_int : 3-tuple of np.ndarray(K, K, K, K)
            Two-electron integrals for unrestricted orbitals.
            Uses physicist's notation.

        Raises
        ------
        TypeError
            If one-electron integrals are not provided as a list/tuple of two numpy array of dtype
            float/complex.
            If two-electron integrals are not provided as a list/tuple of three numpy array of dtype
            float/complex.
        ValueError
            If each block of one-electron integrals are not given as a (two-dimensional) square
            matrix.
            If each block of two-electron integrals are not given as a four-dimensional tensor with
            same dimensionality in each axis.
            If each block of one-electron integrals do not have the same shape.
            If each block of two-electron integrals do not have the same shape.
            If one- and two-electron integrals do not have the same number of orbitals.

        """
        if __debug__:
            if not (
                isinstance(one_int, (list, tuple))
                and len(one_int) == 2
                and all(isinstance(i, np.ndarray) and i.dtype in [float, complex] for i in one_int)
            ):
                raise TypeError(
                    "One-electron integrals must be given as a list/tuple of two numpy "
                    "arrays (with dtype float/complex)."
                )
            if not (
                isinstance(two_int, (list, tuple))
                and len(two_int) == 3
                and all(isinstance(i, np.ndarray) and i.dtype in [float, complex] for i in two_int)
            ):
                raise TypeError(
                    "Two-electron integrals must be given as a list/tuple of three numpy "
                    "arrays (with dtype float/complex)."
                )
            if not all(i.ndim == 2 and i.shape[0] == i.shape[1] for i in one_int):
                raise ValueError("Each block of one-electron integrals be a (two-dimensional) square " "matrix.")
            if not all(i.ndim == 4 and i.shape[0] == i.shape[1] == i.shape[2] == i.shape[3] for i in two_int):
                raise ValueError(
                    "Each block of two-electron integrals must have four-dimensional "
                    "tensor with equal number rows in each axis."
                )
            if one_int[0].shape != one_int[1].shape:
                raise ValueError("Each block of one-electron integrals must have the same shape.")
            if not two_int[0].shape == two_int[1].shape == two_int[2].shape:
                raise ValueError("Each block of two-electron integrals must have the same shape.")
            if one_int[0].shape[0] != two_int[0].shape[0]:
                raise ValueError("One- and two-electron integrals must have the same number of " "orbitals.")

        self.one_int = tuple(one_int)
        self.two_int = tuple(two_int)

    def orb_rotate_jacobi(self, jacobi_indices, theta):
        """Rotate orbitals using Jacobi matrix.

        Parameters
        ----------
        jacobi_indices : 2-tuple of ints
            Indices of the orbitals that will be rotated
        theta : float, np.ndarray of float
            Angle with which the orbitals are rotated

        Raises
        ------
        TypeError
            If indices are not given as a tuple/list of two integers.
            If theta is not a flota or a numpy array of floats.
        ValueError
            If any of the pair of indices given are the same.
            If any of the index is less than 0 or greater than the number of rows.
            If the pair of indices do not have the same spin.

        """
        # pylint: disable=C0103
        num_orbs = self.one_int[0].shape[0]

        if __debug__:
            if not (
                isinstance(jacobi_indices, (tuple, list))
                and len(jacobi_indices) == 2
                and isinstance(jacobi_indices[0], int)
                and isinstance(jacobi_indices[1], int)
            ):
                raise TypeError("Indices must be given a tuple or list of two integers.")
            if jacobi_indices[0] == jacobi_indices[1]:
                raise ValueError("Indices must be different.")
            if not (0 <= jacobi_indices[0] < self.nspin and 0 <= jacobi_indices[1] < self.nspin):
                raise ValueError("Indices must be greater than or equal to 0 and less than the number " "of rows.")
            if jacobi_indices[0] // num_orbs != jacobi_indices[1] // num_orbs:
                raise ValueError("Indices must select from orbitals of same spin.")
            if not (
                isinstance(theta, float) or (isinstance(theta, np.ndarray) and theta.dtype == float and theta.size == 1)
            ):
                raise TypeError("Angle `theta` must be a float or numpy array of one float.")

        p, q = jacobi_indices
        if p > q:
            p, q = q, p
        p, q, spin_index = p % num_orbs, q % num_orbs, p // num_orbs

        # one_electron
        p_col = self.one_int[spin_index][:, p]
        q_col = self.one_int[spin_index][:, q]
        (self.one_int[spin_index][:, p], self.one_int[spin_index][:, q]) = (
            np.cos(theta) * p_col - np.sin(theta) * q_col,
            np.sin(theta) * p_col + np.cos(theta) * q_col,
        )

        p_row = self.one_int[spin_index][p, :]
        q_row = self.one_int[spin_index][q, :]
        (self.one_int[spin_index][p, :], self.one_int[spin_index][q, :]) = (
            np.cos(theta) * p_row - np.sin(theta) * q_row,
            np.sin(theta) * p_row + np.cos(theta) * q_row,
        )

        # two electron
        spin_index *= 2
        # because beta beta component corresponds to the index 2 (rather than 1 in the one-electron
        # case)

        p_slice = self.two_int[spin_index][:, :, :, p]
        q_slice = self.two_int[spin_index][:, :, :, q]
        (self.two_int[spin_index][:, :, :, p], self.two_int[spin_index][:, :, :, q]) = (
            np.cos(theta) * p_slice - np.sin(theta) * q_slice,
            np.sin(theta) * p_slice + np.cos(theta) * q_slice,
        )

        p_slice = self.two_int[spin_index][:, :, p, :]
        q_slice = self.two_int[spin_index][:, :, q, :]
        (self.two_int[spin_index][:, :, p, :], self.two_int[spin_index][:, :, q, :]) = (
            np.cos(theta) * p_slice - np.sin(theta) * q_slice,
            np.sin(theta) * p_slice + np.cos(theta) * q_slice,
        )

        p_slice = self.two_int[spin_index][:, p, :, :]
        q_slice = self.two_int[spin_index][:, q, :, :]
        (self.two_int[spin_index][:, p, :, :], self.two_int[spin_index][:, q, :, :]) = (
            np.cos(theta) * p_slice - np.sin(theta) * q_slice,
            np.sin(theta) * p_slice + np.cos(theta) * q_slice,
        )

        p_slice = self.two_int[spin_index][p, :, :, :]
        q_slice = self.two_int[spin_index][q, :, :, :]
        (self.two_int[spin_index][p, :, :, :], self.two_int[spin_index][q, :, :, :]) = (
            np.cos(theta) * p_slice - np.sin(theta) * q_slice,
            np.sin(theta) * p_slice + np.cos(theta) * q_slice,
        )

        if spin_index == 0:
            p_slice = self.two_int[1][:, :, p, :]
            q_slice = self.two_int[1][:, :, q, :]
            (self.two_int[1][:, :, p, :], self.two_int[1][:, :, q, :]) = (
                np.cos(theta) * p_slice - np.sin(theta) * q_slice,
                np.sin(theta) * p_slice + np.cos(theta) * q_slice,
            )
            p_slice = self.two_int[1][p, :, :, :]
            q_slice = self.two_int[1][q, :, :, :]
            (self.two_int[1][p, :, :, :], self.two_int[1][q, :, :, :]) = (
                np.cos(theta) * p_slice - np.sin(theta) * q_slice,
                np.sin(theta) * p_slice + np.cos(theta) * q_slice,
            )

        else:
            p_slice = self.two_int[1][:, :, :, p]
            q_slice = self.two_int[1][:, :, :, q]
            (self.two_int[1][:, :, :, p], self.two_int[1][:, :, :, q]) = (
                np.cos(theta) * p_slice - np.sin(theta) * q_slice,
                np.sin(theta) * p_slice + np.cos(theta) * q_slice,
            )
            p_slice = self.two_int[1][:, p, :, :]
            q_slice = self.two_int[1][:, q, :, :]
            (self.two_int[1][:, p, :, :], self.two_int[1][:, q, :, :]) = (
                np.cos(theta) * p_slice - np.sin(theta) * q_slice,
                np.sin(theta) * p_slice + np.cos(theta) * q_slice,
            )

    def orb_rotate_matrix(self, matrix):
        r"""Rotate orbitals with a transformation matrix.

        .. math::

            \widetilde{h}_{ab} &= \sum_{ij} C^\dagger_{ai} h_{ij} C_{jb}\\
            \widetilde{g}_{abcd} &= \sum_{ijkl} C^\dagger_{ai} C^\dagger_{bj} g_{ijkl} C_{kc} C_{ld}

        Parameters
        ----------
        matrix : np.ndarray(K, L), 2-list/tuple of np.ndarray(K, L)
            Transformation matrix.
            If two transformation matrices are given, the first transforms the alpha orbitals, and
            the second transforms the beta orbitals.
            If only one transformation matrix is given, then the same transformation is applied to
            both the alpha and beta orbitals.

        Raises
        ------
        TypeError
            If matrix is not a two-diensional numpy array or a 2-list/tuple of two-dimensional numpy
            arrays.
        ValueError
            If shape of matrix does not match up with the shape of the integrals.

        """
        if isinstance(matrix, np.ndarray):
            matrix = (matrix, matrix)

        if __debug__:
            if not (
                isinstance(matrix, (list, tuple))
                and len(matrix) == 2
                and isinstance(matrix[0], np.ndarray)
                and isinstance(matrix[1], np.ndarray)
                and matrix[0].ndim == 2
                and matrix[1].ndim == 2
            ):
                raise TypeError(
                    "Transformation matrix must be given as a numpy array or as list/tuple "
                    "of two two-dimensional numpy arrays."
                )
            if not (matrix[0].shape[0] == self.one_int[0].shape[0] and matrix[1].shape[0] == self.one_int[1].shape[0]):
                raise ValueError("Shape of the transformation matrix must match with the shape of the integrals.")
        # NOTE: don't need to check that matrix matches up with two_int b/c one_int and two_int have
        #       the same number of rows/columns

        new_one_ints = []
        one_int = np.einsum("ij,ia->aj", self.one_int[0], matrix[0])
        one_int = np.einsum("aj,jb->ab", one_int, matrix[0])
        new_one_ints.append(one_int)

        one_int = np.einsum("ij,ia->aj", self.one_int[1], matrix[1])
        one_int = np.einsum("aj,jb->ab", one_int, matrix[1])
        new_one_ints.append(one_int)
        self.one_int = tuple(new_one_ints)

        new_two_ints = []
        two_int = np.einsum("ijkl,ia->ajkl", self.two_int[0], matrix[0])
        two_int = np.einsum("ajkl,jb->abkl", two_int, matrix[0])
        two_int = np.einsum("abkl,kc->abcl", two_int, matrix[0])
        two_int = np.einsum("abcl,ld->abcd", two_int, matrix[0])
        new_two_ints.append(two_int)

        two_int = np.einsum("ijkl,ia->ajkl", self.two_int[1], matrix[0])
        two_int = np.einsum("ajkl,jb->abkl", two_int, matrix[1])
        two_int = np.einsum("abkl,kc->abcl", two_int, matrix[0])
        two_int = np.einsum("abcl,ld->abcd", two_int, matrix[1])
        new_two_ints.append(two_int)

        two_int = np.einsum("ijkl,ia->ajkl", self.two_int[2], matrix[1])
        two_int = np.einsum("ajkl,jb->abkl", two_int, matrix[1])
        two_int = np.einsum("abkl,kc->abcl", two_int, matrix[1])
        two_int = np.einsum("abcl,ld->abcd", two_int, matrix[1])
        new_two_ints.append(two_int)

        self.two_int = tuple(new_two_ints)
