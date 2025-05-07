r"""Hamiltonian used to describe a chemical system expressed wrt generalized orbitals."""

import itertools as it
import os

from fanpy.ham.generalized_base import BaseGeneralizedHamiltonian
from fanpy.tools import math_tools, slater
from fanpy.wfn.composite.lincomb import LinearCombinationWavefunction
from fanpy.wfn.composite.product import ProductWavefunction

import numpy as np

# pylint: disable=C0302


class GeneralizedMolecularHamiltonian(BaseGeneralizedHamiltonian):
    r"""Hamiltonian used to describe a typical chemical system expressed wrt generalized orbitals.

    .. math::

        \hat{H} = \sum_{ij} h_{ij} a^\dagger_i a_j
        + \frac{1}{2} \sum_{ijkl} g_{ijkl} a^\dagger_i a^\dagger_j a_l a_k

    where :math:`h_{ik}` is the one-electron integral and :math:`g_{ijkl}` is the two-electron
    integral in Physicists' notation.

    Attributes
    ----------
    one_int : np.ndarray(K, K)
        One-electron integrals.
    two_int : np.ndarray(K, K, K, K)
        Two-electron integrals.
    params : np.ndarray
        Significant elements of the anti-Hermitian matrix.

    Properties
    ----------
    nspin : int
        Number of spin orbitals.
    nspatial : int
        Number of spatial orbitals.
    nparams : int
        Number of parameters.

    Methods
    -------
    __init__(self, one_int, two_int, params=None, update_prev_params=False)
        Initialize the Hamiltonian
    assign_integrals(self, one_int, two_int)
        Assign the one- and two-electron integrals.
    orb_rotate_jacobi(self, jacobi_indices, theta)
        Rotate orbitals using Jacobi matrix.
    orb_rotate_matrix(self, matrix)
        Rotate orbitals using a transformation matrix.
    assign_params(self, params)
        Transform the integrals with a unitary matrix that corresponds to the given parameters.
    save_params(self, filename)
        Save the parameters associated with the Hamiltonian.
    integrate_sd_wfn(self, sd, wfn, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.
    integrate_sd_sd(self, sd1, sd2, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """

    def __init__(self, one_int, two_int, params=None, update_prev_params=False):
        """Initialize the Hamiltonian.

        Parameters
        ----------
        one_int : np.ndarray(K, K)
            One electron integrals.
        two_int : np.ndarray(K, K, K, K)
            Two electron integrals.

        """
        super().__init__(one_int, two_int)
        self.set_ref_ints()
        self.cache_two_ints()
        self._prev_params = None
        self.update_prev_params = update_prev_params
        self.assign_params(params=params)

    @property
    def nparams(self):
        """Return the number of parameters.

        Returns
        -------
        nparams : int
            Number of parameters.

        """
        return self.params.size

    def set_ref_ints(self):
        """Store the current integrals as the reference from which orbitals will be rotated."""
        self._ref_one_int = np.copy(self.one_int)
        self._ref_two_int = np.copy(self.two_int)

    def cache_two_ints(self):
        """Cache away contractions of the two electron integrals."""
        # store away tensor contractions
        indices = np.arange(self.one_int.shape[0])
        self._cached_two_int_ijij = self.two_int[indices[:, None], indices, indices[:, None], indices]
        self._cached_two_int_ijji = self.two_int[indices[:, None], indices, indices, indices[:, None]]

    def assign_params(self, params=None):
        """Transform the integrals with a unitary matrix that corresponds to the given parameters.

        Parameters
        ----------
        params : {np.ndarray, None}
            Significant elements of the anti-Hermitian matrix. Integrals will be transformed with
            the Unitary matrix that corresponds to the anti-Hermitian matrix.

        Raises
        ------
        ValueError
            If parameters is not a one-dimensional numpy array with K*(K-1)/2 elements, where K is
            the number of orbitals.

        """
        # NOTE: nspin is not used here because RestrictedMolecularHamiltonian inherits this method
        # and it would need to use the number of spatial orbitals instead of the spin orbitals
        # FIXME: this is too hacky. should the dimension of the antihermitian matrix be stored
        # somewhere?
        num_orbs = self.one_int.shape[0]
        num_params = num_orbs * (num_orbs - 1) // 2

        if params is None:
            params = np.zeros(num_params)

        if __debug__ and not (isinstance(params, np.ndarray) and params.ndim == 1 and params.size == num_params):
            raise ValueError(
                "Parameters for orbital rotation must be a one-dimension numpy array "
                "with {0}=K*(K-1)/2 elements, where K is the number of "
                "orbitals.".format(num_params)
            )

        # assign parameters
        self.params = params
        if self._prev_params is None:
            self._prev_params = np.zeros(params.size)
            self._prev_unitary = math_tools.unitary_matrix(self._prev_params)
        params_prev = self._prev_params
        params_diff = params - params_prev
        unitary_prev = self._prev_unitary

        # revert integrals back to original
        self.assign_integrals(np.copy(self._ref_one_int), np.copy(self._ref_two_int))

        # convert antihermitian part to unitary matrix.
        unitary_diff = math_tools.unitary_matrix(params_diff)
        unitary = unitary_prev.dot(unitary_diff)

        # transform integrals
        self.orb_rotate_matrix(unitary)

        if self.update_prev_params:
            self._prev_params = params.copy()
            self._prev_unitary = unitary

        # cache two electron integrals
        self.cache_two_ints()

    def save_params(self, filename):
        """Save parameters associated with the Hamiltonian.

        Since both the parameters and the corresponding unitary matrix are needed to obtain the one-
        and two-electron integrals (i.e. Hamiltonian), they are saved as separate files, using the
        given filename as the root (removing the extension). The unitary matrix is saved by
        appending "_um" to the end of the root.

        Parameters
        ----------
        filename : str

        """
        root, ext = os.path.splitext(filename)
        np.save(filename, self.params)

        unitary_diff = math_tools.unitary_matrix(self.params - self._prev_params)
        unitary = self._prev_unitary.dot(unitary_diff)
        np.save("{}_um{}".format(root, ext), unitary)

    def integrate_sd_sd_decomposed(self, sd1, sd2, deriv=None):  # pylint: disable=R0911
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
        deriv : np.ndarray
            Indices of the Hamiltonian parameter against which the integral is derivatized.
            Default is no derivatization.

        Returns
        -------
        integral : {np.ndarray(3,)}
            Array containing the values of the one electron, coulomb, and exchange components.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.

        """
        if deriv is not None:
            return self._integrate_sd_sd_deriv_decomposed(sd1, sd2, deriv)

        if __debug__ and not (slater.is_sd_compatible(sd1) and slater.is_sd_compatible(sd2)):
            raise TypeError("Slater determinant must be given as an integer.")
        shared_indices = np.array(slater.shared_orbs(sd1, sd2))
        diff_sd1, diff_sd2 = slater.diff_orbs(sd1, sd2)
        # if two Slater determinants do not have the same number of electrons
        if len(diff_sd1) != len(diff_sd2):
            return 0.0, 0.0, 0.0
        diff_order = len(diff_sd1)
        if diff_order > 2:
            return 0.0, 0.0, 0.0

        sign = slater.sign_excite(sd1, diff_sd1, reversed(diff_sd2))

        # two sd's are the same
        if diff_order == 0 and shared_indices.size > 0:
            one_electron, coulomb, exchange = self._integrate_sd_sd_zero(shared_indices)
        # two sd's are different by single excitation
        elif diff_order == 1:
            one_electron, coulomb, exchange = self._integrate_sd_sd_one(diff_sd1, diff_sd2, shared_indices)
        # two sd's are different by double excitation
        else:
            one_electron, coulomb, exchange = self._integrate_sd_sd_two(diff_sd1, diff_sd2)

        return sign * np.array([one_electron, coulomb, exchange])

    def integrate_sd_sd(self, sd1, sd2, deriv=None):  # pylint: disable=R0911
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
        deriv : np.ndarray
            Indices of the Hamiltonian parameter against which the integral is derivatized.
            Default is no derivatization.

        Returns
        -------
        integral : float
            Value of the integral.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.

        """
        decomposed_integral = self.integrate_sd_sd_decomposed(sd1, sd2, deriv=deriv)

        return np.sum(decomposed_integral, axis=0)

    def _integrate_sd_sd_zero(self, shared_indices):
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
        one_electron = np.sum(self.one_int[shared_indices, shared_indices])
        coulomb = np.sum(np.triu(self._cached_two_int_ijij[shared_indices[:, None], shared_indices], k=1))
        exchange = -np.sum(np.triu(self._cached_two_int_ijji[shared_indices[:, None], shared_indices], k=1))
        return one_electron, coulomb, exchange

    def _integrate_sd_sd_one(self, diff_sd1, diff_sd2, shared_indices):
        """Return integrals of the given Slater determinant with its first order excitation.

        Parameters
        ----------
        diff_sd1 : 1-tuple of int
            Index of the orbital that is occupied in the first Slater determinant and not occupied
            in the second.
        diff_sd2 : 1-tuple of int
            Index of the orbital that is occupied in the second Slater determinant and not occupied
            in the first.
        shared_indices : np.ndarray
            Integer indices of the orbitals that are shared between the first and second Slater
            determinants.

        Returns
        -------
        integrals : 3-tuple of float
            Integrals of the given Slater determinant with itself.
            The one-electron (first element), coulomb (second element), and exchange (third element)
            integrals of the given Slater determinant with itself.

        """
        # pylint: disable=C0103
        (a,) = diff_sd1
        (b,) = diff_sd2
        shared_indices = shared_indices.astype(int)
        one_electron = self.one_int[a, b]
        coulomb = np.sum(self.two_int[shared_indices, a, shared_indices, b])
        exchange = -np.sum(self.two_int[shared_indices, a, b, shared_indices])
        return one_electron, coulomb, exchange

    def _integrate_sd_sd_two(self, diff_sd1, diff_sd2):
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
        # pylint: disable=C0103
        a, b = diff_sd1
        c, d = diff_sd2
        coulomb = self.two_int[a, b, c, d]
        exchange = -self.two_int[a, b, d, c]
        return 0, coulomb, exchange

    def param_ind_to_rowcol_ind(self, param_ind):
        r"""Return the row and column indices of the antihermitian matrix from the parameter index.

        Let :math:`n` be the number of columns and rows in the antihermitian matrix and :math:`x`
        and :math:`y` be the row and column indices of the antihermitian matrix, respectively.
        First, we want to convert the index of the parameter, :math:`i`, to the index of the
        flattened antihermitian matrix, :math:`j`:

        .. math::

            j &= i + 1 + 2 + \dots + (x+1)\\
            &= i + \sum_{k=1}^{x+1} k\\
            &= i + \frac{(x+1)(x+2)}{2}\\

        We can find :math:`x` by finding the smallest :math:`x` such that

        .. math::

            ind - (n-1) - (n-2) - \dots - (n-x-1) &< 0\\
            ind - \sum_{k=1}^{x+1} (n-k) &< 0\\
            ind - n(x+1) + \sum_{k=1}^{x+1} k &< 0\\
            ind - n(x+1) + \frac{(x+1)(x+2)}{2} &< 0\\

        Once we find :math:`x`, we can find :math:`j` and then :math:`y`

        .. math::

            y = j \mod n

        Parameters
        ----------
        param_ind : int
            Index of the parameter.

        Returns
        -------
        matrix_indices : 2-tuple of int
            Indices of row and column of the antihermitian matrix that corresponds to the given
            parameter index.

        """
        # pylint: disable=C0103
        # ind = i
        n = self.nspin
        for k in range(n + 1):  # pragma: no branch
            x = k
            if param_ind - n * (x + 1) + (x + 1) * (x + 2) / 2 < 0:
                break
        # ind_flat = j
        ind_flat = param_ind + (x + 1) * (x + 2) / 2
        y = ind_flat % n

        return int(x), int(y)

    # TODO: Much of the following function can be shortened by using impure functions
    # (function with a side effect) instead
    # FIXME: too many branches, too many statements
    def _integrate_sd_sd_deriv_decomposed(self, sd1, sd2, deriv):
        r"""Return derivative of the CI matrix element with respect to the antihermitian elements.

        Parameters
        ----------
        sd1 : int
            Slater Determinant against which the Hamiltonian is integrated.
        sd2 : int
            Slater Determinant against which the Hamiltonian is integrated.
        deriv : np.ndarray
            Indices of the Hamiltonian parameter against which the integral is derivatized.

        Returns
        -------
        d_integral : np.ndarray(3, len(deriv))
            Derivatives of the one electron, coulomb, and exchange integrals with respect
            to the given parameters.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.
        ValueError
            If the given `deriv` contains an integer greater than or equal to 0 and less than the
            number of parameters.

        Notes
        -----
        Integrals are not assumed to be real. The performance benefit (at the moment) for assuming
        real orbitals is not much.

        """
        # pylint: disable=C0103,R0912,R0915
        if __debug__ and not (slater.is_sd_compatible(sd1) and slater.is_sd_compatible(sd2)):
            raise TypeError("Slater determinant must be given as an integer.")
        # NOTE: shared_indices contains spatial orbital indices
        shared_indices = np.array(slater.shared_orbs(sd1, sd2))
        diff_sd1, diff_sd2 = slater.diff_orbs(sd1, sd2)

        # if two Slater determinants do not have the same number of electrons
        if len(diff_sd1) != len(diff_sd2):
            return np.zeros((3, len(deriv)))
        diff_order = len(diff_sd1)
        if diff_order > 2:
            return np.zeros((3, len(deriv)))

        # get sign
        sign = slater.sign_excite(sd1, diff_sd1, reversed(diff_sd2))

        # check deriv
        if __debug__ and not (isinstance(deriv, np.ndarray) and np.all(deriv >= 0) and np.all(deriv < self.nparams)):
            raise ValueError(
                "Derivative indices must be given as a numpy array of integers greater than or "
                "equal to zero and less than the number of parameters, nspatial * (nspatial-1) / 2"
            )

        output = np.zeros((len(deriv), 3))
        for i, deriv_ind in enumerate(deriv):
            # turn deriv into indices of the matrix, (x, y), where x < y
            x, y = self.param_ind_to_rowcol_ind(deriv_ind)

            # two sd's are the same
            if diff_order == 0:
                output[i] = self._integrate_sd_sd_deriv_zero(x, y, shared_indices)
            # two sd's are different by single excitation
            elif diff_order == 1:
                output[i] = self._integrate_sd_sd_deriv_one(diff_sd1, diff_sd2, x, y, shared_indices)
            # two sd's are different by double excitation
            else:
                output[i] = self._integrate_sd_sd_deriv_two(diff_sd1, diff_sd2, x, y)

        return sign * output.T

    def _integrate_sd_sd_deriv(self, sd1, sd2, deriv):
        r"""Return derivative of the CI matrix element with respect to the antihermitian elements.

        Parameters
        ----------
        sd1 : int
            Slater Determinant against which the Hamiltonian is integrated.
        sd2 : int
            Slater Determinant against which the Hamiltonian is integrated.
        deriv : np.ndarray
            Indices of the Hamiltonian parameter against which the integral is derivatized.

        Returns
        -------
        d_integral : np.ndarray(len(deriv))
            Derivatives of the one electron, coulomb, and exchange integrals with respect
            to the given parameters.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.
        ValueError
            If the given `deriv` contains an integer greater than or equal to 0 and less than the
            number of parameters.

        Notes
        -----
        Integrals are not assumed to be real. The performance benefit (at the moment) for assuming
        real orbitals is not much.

        """

        derivatives = self._integrate_sd_sd_deriv_decomposed(sd1, sd2, deriv)
        return np.sum(derivatives)

    def _integrate_sd_sd_deriv_zero(self, x, y, shared_indices):
        """Return the derivative of the integrals of the given Slater determinant with itself.

        Parameters
        ----------
        x : int
            Row of the antihermitian matrix at which the integral will be derivatized.
        y : int
            Column of the antihermitian matrix at which the integral will be derivatized.
        shared_indices : np.ndarray
            Integer indices of the orbitals that are occupied by the Slater determinant.
            Dtype must be int.

        Returns
        -------
        integrals : 3-tuple of float
            The derivatives (with respect to the given parameter) of the one-electron (first
            element), coulomb (second element), and exchange (third element) integrals of the given
            Slater determinant with itself.

        """
        # pylint: disable=C0103
        # remove orbitals x and y from the shared indices
        shared_indices_no_x = shared_indices[shared_indices != x]
        shared_indices_no_y = shared_indices[shared_indices != y]

        one_electron, coulomb, exchange = 0, 0, 0

        if x in shared_indices:
            one_electron -= 2 * np.real(self.one_int[x, y])
            if shared_indices_no_x.size != 0:
                coulomb -= 2 * np.sum(np.real(self.two_int[x, shared_indices_no_x, y, shared_indices_no_x]))
                exchange += 2 * np.sum(np.real(self.two_int[x, shared_indices_no_x, shared_indices_no_x, y]))

        if y in shared_indices:
            one_electron += 2 * np.real(self.one_int[x, y])
            if shared_indices_no_y.size != 0:
                coulomb += 2 * np.sum(np.real(self.two_int[x, shared_indices_no_y, y, shared_indices_no_y]))
                exchange -= 2 * np.sum(np.real(self.two_int[x, shared_indices_no_y, shared_indices_no_y, y]))

        return one_electron, coulomb, exchange

    def _integrate_sd_sd_deriv_one(self, diff_sd1, diff_sd2, x, y, shared_indices):
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
        shared_indices : np.ndarray
            Integer indices of the orbitals that are shared between the first and second Slater
            determinant.
            Dtype must be int.

        Returns
        -------
        integrals : 3-tuple of float
            The derivatives (with respect to the given parameter) of the one-electron (first
            element), coulomb (second element), and exchange (third element) integrals of the given
            Slater determinant with its first order excitation.

        """
        # pylint: disable=C0103
        (a,) = diff_sd1
        (b,) = diff_sd2

        one_electron, coulomb, exchange = 0, 0, 0
        shared_indices = shared_indices.astype(int)

        if x == a:
            one_electron -= self.one_int[y, b]
            coulomb -= np.sum(self.two_int[y, shared_indices, b, shared_indices])
            exchange += np.sum(self.two_int[y, shared_indices, shared_indices, b])
        elif x == b:
            one_electron -= self.one_int[a, y]
            coulomb -= np.sum(self.two_int[a, shared_indices, y, shared_indices])
            exchange += np.sum(self.two_int[a, shared_indices, shared_indices, y])
        elif x in shared_indices:
            # NOTE: we can use the following if we assume that the orbitals are real
            # coulomb -= 2 * self.two_int[x, a, y, b]
            coulomb -= self.two_int[y, a, x, b]
            coulomb -= self.two_int[x, a, y, b]
            exchange += self.two_int[y, a, b, x]
            exchange += self.two_int[x, a, b, y]

        if y == a:
            one_electron += self.one_int[x, b]
            coulomb += np.sum(self.two_int[x, shared_indices, b, shared_indices])
            exchange -= np.sum(self.two_int[x, shared_indices, shared_indices, b])
        elif y == b:
            one_electron += self.one_int[a, x]
            coulomb += np.sum(self.two_int[a, shared_indices, x, shared_indices])
            exchange -= np.sum(self.two_int[a, shared_indices, shared_indices, x])
        if y in shared_indices:
            # NOTE: we can use the following if we assume that the orbitals are real
            # coulomb += 2 * self.two_int[x, a, y, b]
            coulomb += self.two_int[x, a, y, b]
            coulomb += self.two_int[y, a, x, b]
            exchange -= self.two_int[x, a, b, y]
            exchange -= self.two_int[y, a, b, x]

        return one_electron, coulomb, exchange

    def _integrate_sd_sd_deriv_two(self, diff_sd1, diff_sd2, x, y):
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
        # pylint: disable=C0103
        a, b = diff_sd1
        c, d = diff_sd2

        one_electron, coulomb, exchange = 0, 0, 0

        if x == a:
            coulomb -= self.two_int[y, b, c, d]
            exchange += self.two_int[y, b, d, c]
        elif x == b:
            coulomb -= self.two_int[a, y, c, d]
            exchange += self.two_int[a, y, d, c]
        elif x == c:
            coulomb -= self.two_int[a, b, y, d]
            exchange += self.two_int[a, b, d, y]
        elif x == d:
            coulomb -= self.two_int[a, b, c, y]
            exchange += self.two_int[a, b, y, c]

        if y == a:
            coulomb += self.two_int[x, b, c, d]
            exchange -= self.two_int[x, b, d, c]
        elif y == b:
            coulomb += self.two_int[a, x, c, d]
            exchange -= self.two_int[a, x, d, c]
        elif y == c:
            coulomb += self.two_int[a, b, x, d]
            exchange -= self.two_int[a, b, d, x]
        elif y == d:
            coulomb += self.two_int[a, b, c, x]
            exchange -= self.two_int[a, b, x, c]

        return one_electron, coulomb, exchange

    def _integrate_sd_sds_zero(self, occ_indices):
        """Return the integrals of the given Slater determinant with itself.

        Paramters
        ---------
        occ_indices : np.ndarray(N,)
            Indices of the spin orbitals that are occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, 1)
            Integrals of the given Slater determinant with itself.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.

        """
        one_electron = np.sum(self.one_int[occ_indices, occ_indices])
        coulomb = np.sum(np.triu(self._cached_two_int_ijij[occ_indices[:, None], occ_indices], k=1))
        exchange = -np.sum(np.triu(self._cached_two_int_ijji[occ_indices[:, None], occ_indices], k=1))
        return np.array([[one_electron], [coulomb], [exchange]])

    def _integrate_sd_sds_one(self, occ_indices, vir_indices):
        """Return the integrals of the given Slater determinant with its first order excitations.

        Paramters
        ---------
        occ_indices : np.ndarray(N,)
            Indices of the spin orbitals that are occupied in the Slater determinant.
        vir_indices : np.ndarray(K-N,)
            Indices of the spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, M)
            Integrals of the given Slater determinant with its first order excitations.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.
            Second index corresponds to the first order excitations of the given Slater determinant.
            The excitations are ordered by the occupied orbital then the virtual orbital. For
            example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the ordering of the
            excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)].
            `M` is the number of first order excitations of the given Slater determinants.

        """
        shared_indices = slater.shared_indices_remove_one_index(occ_indices)

        sign = slater.sign_excite_one(occ_indices, vir_indices)

        one_electron = self.one_int[occ_indices[:, np.newaxis], vir_indices[np.newaxis, :]]
        coulomb = np.sum(
            self.two_int[
                shared_indices[:, :, np.newaxis],
                occ_indices[:, np.newaxis, np.newaxis],
                shared_indices[:, :, np.newaxis],
                vir_indices[np.newaxis, np.newaxis, :],
            ],
            axis=1,
        )
        exchange = -np.sum(
            self.two_int[
                shared_indices[:, :, np.newaxis],
                occ_indices[:, np.newaxis, np.newaxis],
                vir_indices[np.newaxis, np.newaxis, :],
                shared_indices[:, :, np.newaxis],
            ],
            axis=1,
        )
        return sign[None, :] * np.array([one_electron.ravel(), coulomb.ravel(), exchange.ravel()])

    def _integrate_sd_sds_two(self, occ_indices, vir_indices):
        """Return the integrals of the given Slater determinant with its second order excitations.

        Paramters
        ---------
        occ_indices : np.ndarray(N,)
            Indices of the spin orbitals that are occupied in the Slater determinant.
        vir_indices : np.ndarray(K-N,)
            Indices of the spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, M)
            Integrals of the given Slater determinant with its second order excitations..
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) energy.
            Second index corresponds to the second order excitations of the given Slater
            determinant.
            The excitations are ordered by the occupied orbital then the virtual orbital. For
            example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6], the ordering
            of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1, 3, 4, 5), (1,
            3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)].
            `M` is the number of first order excitations of the given Slater determinants.

        """
        # pylint: disable=C0103
        # FIXME: use method in slater module
        annihilators = np.array(list(it.combinations(occ_indices, 2)))
        a = annihilators[:, 0]
        b = annihilators[:, 1]
        creators = np.array(list(it.combinations(vir_indices, 2)))
        c = creators[:, 0]
        d = creators[:, 1]

        sign = slater.sign_excite_two(occ_indices, vir_indices)

        coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]]
        exchange = -self.two_int[a[:, None], b[:, None], d[None, :], c[None, :]]
        return sign[None, :] * np.array([np.zeros(coulomb.size), coulomb.ravel(), exchange.ravel()])

    def _integrate_sd_sds_deriv_zero(self, occ_indices, vir_indices):
        """Return the derivative of the integrals of the given Slater determinant with itself.

        Paramters
        ---------
        occ_indices : np.ndarray(N,)
            Indices of the spin orbitals that are occupied in the Slater determinant.
        vir_indices : np.ndarray(K-N,)
            Indices of the spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, N_params, 1)
            Integrals of the given Slater determinant with itself.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.
            Second index corresponds to index of the parameter with respect to which the integral is
            derivatived.

        """
        shared_indices = slater.shared_indices_remove_one_index(occ_indices)

        # NOTE: here, we use the following convention for indices:
        # the first index corresponds to the row index of the antihermitian matrix for orbital
        # rotation
        # the second index corresponds to the column index of the antihermitian matrix for orbital
        # rotation
        one_electron = np.zeros((self.nspin, self.nspin))
        coulomb = np.zeros((self.nspin, self.nspin))
        exchange = np.zeros((self.nspin, self.nspin))

        # NOTE: if both x and y are occupied these cancel each other out
        one_electron[occ_indices[:, None], vir_indices[None, :]] -= 2 * np.real(
            self.one_int[occ_indices[:, None], vir_indices[None, :]]
        )
        one_electron[vir_indices[:, None], occ_indices[None, :]] += 2 * np.real(
            self.one_int[vir_indices[:, None], occ_indices[None, :]]
        )

        # if x and y are occupied
        coulomb[occ_indices[:, None], occ_indices[None, :]] -= 2 * np.sum(
            np.real(
                self.two_int[
                    occ_indices[:, None, None],  # x
                    shared_indices[:, :, None],  # shared no x
                    occ_indices[None, None, :],  # y
                    shared_indices[:, :, None],  # shared no x
                ]
            ),
            axis=1,
        )
        coulomb[occ_indices[:, None], occ_indices[None, :]] += 2 * np.sum(
            np.real(
                self.two_int[
                    occ_indices[:, None, None],  # x
                    shared_indices.T[None, :, :],  # shared no y
                    occ_indices[None, None, :],  # y
                    shared_indices.T[None, :, :],  # shared no y
                ]
            ),
            axis=1,
        )
        # if only x is occupied
        coulomb[occ_indices[:, None], vir_indices[None, :]] -= 2 * np.sum(
            np.real(
                self.two_int[
                    occ_indices[:, None, None],  # x
                    shared_indices[:, :, None],  # shared no x
                    vir_indices[None, None, :],  # y
                    shared_indices[:, :, None],  # shared no x
                ]
            ),
            axis=1,
        )
        # if only y is occupied
        coulomb[vir_indices[:, None], occ_indices[None, :]] += 2 * np.sum(
            np.real(
                self.two_int[
                    vir_indices[:, None, None],  # x
                    shared_indices.T[None, :, :],  # shared no y
                    occ_indices[None, None, :],  # y
                    shared_indices.T[None, :, :],  # shared no y
                ]
            ),
            axis=1,
        )

        # if x and y are occupied
        exchange[occ_indices[:, None], occ_indices[None, :]] += 2 * np.sum(
            np.real(
                self.two_int[
                    occ_indices[:, None, None],  # x
                    shared_indices[:, :, None],  # shared no x
                    shared_indices[:, :, None],  # shared no x
                    occ_indices[None, None, :],  # y
                ]
            ),
            axis=1,
        )
        exchange[occ_indices[:, None], occ_indices[None, :]] -= 2 * np.sum(
            np.real(
                self.two_int[
                    occ_indices[:, None, None],  # x
                    shared_indices.T[None, :, :],  # shared no y
                    shared_indices.T[None, :, :],  # shared no y
                    occ_indices[None, None, :],  # y
                ]
            ),
            axis=1,
        )
        # if only x is occupied
        exchange[occ_indices[:, None], vir_indices[None, :]] += 2 * np.sum(
            np.real(
                self.two_int[
                    occ_indices[:, None, None],  # x
                    shared_indices[:, :, None],  # shared no x
                    shared_indices[:, :, None],  # shared no x
                    vir_indices[None, None, :],  # y
                ]
            ),
            axis=1,
        )
        # if only y is occupied
        exchange[vir_indices[:, None], occ_indices[None, :]] -= 2 * np.sum(
            np.real(
                self.two_int[
                    vir_indices[:, None, None],  # x
                    shared_indices.T[None, :, :],  # shared no y
                    shared_indices.T[None, :, :],  # shared no y
                    occ_indices[None, None, :],  # y
                ]
            ),
            axis=1,
        )

        triu_indices = np.triu_indices(self.nspin, k=1)
        return np.array([one_electron[triu_indices], coulomb[triu_indices], exchange[triu_indices]])[:, :, None]

    def _integrate_sd_sds_deriv_one(self, occ_indices, vir_indices):
        """Return derivative of integrals of given Slater determinant with its first excitations.

        Paramters
        ---------
        occ_indices : np.ndarray(N,)
            Indices of the spin orbitals that are occupied in the Slater determinant.
            Data type must be int.
        vir_indices : np.ndarray(N,)
            Indices of the spin orbitals that are not occupied in the Slater determinant.
            Data type must be int.

        Returns
        -------
        integrals : np.ndarray(3, N_params, M)
            Integrals of the given Slater determinant with its first order excitations.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.
            Second index corresponds to index of the parameter with respect to which the integral is
            derivatived.
            Third index corresponds to the first order excitations of the given Slater determinant.
            The excitations are ordered by the occupied orbital then the virtual orbital. For
            example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the ordering of the
            excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)].
            `M` is the number of first order excitations of the given Slater determinants.

        """
        shared_indices = slater.shared_indices_remove_one_index(occ_indices)
        all_indices = np.arange(self.nspin)

        sign = slater.sign_excite_one(occ_indices, vir_indices)

        # NOTE: here, we use the following convention for indices:
        # the first index corresponds to the row index of the antihermitian matrix for orbital
        # rotation
        # the second index corresponds to the column index of the antihermitian matrix for orbital
        # rotation
        # the third index corresponds to the occupied orbital that will be annihilated in the
        # excitation
        # the fourth index corresponds to the virtual orbital that will be created in the excitation
        one_electron = np.zeros((self.nspin, self.nspin, occ_indices.size, vir_indices.size))
        coulomb = np.zeros((self.nspin, self.nspin, occ_indices.size, vir_indices.size))
        exchange = np.zeros((self.nspin, self.nspin, occ_indices.size, vir_indices.size))

        occ_array_indices = np.arange(occ_indices.size)
        vir_array_indices = np.arange(vir_indices.size)

        # FIXME: check if there is redundant operations (i.e. split off the all_indices into
        # occ_indices and vir_indices)
        # x == a
        one_electron[
            occ_indices[:, None, None],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a (occupied index)
            vir_array_indices[None, None, :],  # b (virtual index)
        ] -= self.one_int[
            all_indices[None, :, None], vir_indices[None, None, :]
        ]  # y, b
        coulomb[
            occ_indices[:, None, None],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a (occupied index)
            vir_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                all_indices[None, None, :, None],  # y
                shared_indices[:, :, None, None],  # shared
                vir_indices[None, None, None, :],  # b
                shared_indices[:, :, None, None],  # shared
            ],
            axis=1,
        )
        exchange[
            occ_indices[:, None, None],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a (occupied index)
            vir_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                all_indices[None, None, :, None],  # y
                shared_indices[:, :, None, None],  # shared
                shared_indices[:, :, None, None],  # shared
                vir_indices[None, None, None, :],  # b
            ],
            axis=1,
        )
        # x == b
        one_electron[
            vir_indices[None, None, :],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a (occupied index)
            vir_array_indices[None, None, :],  # b (virtual index)
        ] -= self.one_int[
            occ_indices[:, None, None], all_indices[None, :, None]
        ]  # a, y
        coulomb[
            vir_indices[None, None, :],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a (occupied index)
            vir_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                occ_indices[:, None, None, None],  # a
                shared_indices[:, :, None, None],  # shared
                all_indices[None, None, :, None],  # y
                shared_indices[:, :, None, None],  # shared
            ],
            axis=1,
        )
        exchange[
            vir_indices[None, None, :],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a (occupied index)
            vir_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                occ_indices[:, None, None, None],  # a
                shared_indices[:, :, None, None],  # shared
                shared_indices[:, :, None, None],  # shared
                all_indices[None, None, :, None],  # y
            ],
            axis=1,
        )
        # x in shared
        coulomb[
            shared_indices[:, :, None, None],  # x
            all_indices[None, None, :, None],  # y
            occ_array_indices[:, None, None, None],  # a (occupied index)
            vir_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            all_indices[None, None, :, None],  # y
            occ_indices[:, None, None, None],  # a (occupied index)
            shared_indices[:, :, None, None],  # x
            vir_indices[None, None, None, :],  # b (virtual index)
        ]
        coulomb[
            shared_indices[:, :, None, None],  # x
            all_indices[None, None, :, None],  # y
            occ_array_indices[:, None, None, None],  # a (occupied index)
            vir_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            shared_indices[:, :, None, None],  # x
            occ_indices[:, None, None, None],  # a (occupied index)
            all_indices[None, None, :, None],  # y
            vir_indices[None, None, None, :],  # b (virtual index)
        ]
        exchange[
            shared_indices[:, :, None, None],  # x
            all_indices[None, None, :, None],  # y
            occ_array_indices[:, None, None, None],  # a (occupied index)
            vir_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            all_indices[None, None, :, None],  # y
            occ_indices[:, None, None, None],  # a (occupied index)
            vir_indices[None, None, None, :],  # b (virtual index)
            shared_indices[:, :, None, None],  # x
        ]
        exchange[
            shared_indices[:, :, None, None],  # x
            all_indices[None, None, :, None],  # y
            occ_array_indices[:, None, None, None],  # a (occupied index)
            vir_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            shared_indices[:, :, None, None],  # x
            occ_indices[:, None, None, None],  # a (occupied index)
            vir_indices[None, None, None, :],  # b (virtual index)
            all_indices[None, None, :, None],  # y
        ]

        # y == a
        one_electron[
            all_indices[:, None, None],  # x
            occ_indices[None, :, None],  # y
            occ_array_indices[None, :, None],  # a (occupied index)
            vir_array_indices[None, None, :],  # b (virtual index)
        ] += self.one_int[
            all_indices[:, None, None], vir_indices[None, None, :]
        ]  # x, b
        coulomb[
            all_indices[:, None, None],  # x
            occ_indices[None, :, None],  # y
            occ_array_indices[None, :, None],  # a (occupied index)
            vir_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                all_indices[:, None, None, None],  # x
                shared_indices.T[None, :, :, None],  # shared
                vir_indices[None, None, None, :],  # b
                shared_indices.T[None, :, :, None],  # shared
            ],
            axis=1,
        )
        exchange[
            all_indices[:, None, None],  # x
            occ_indices[None, :, None],  # y
            occ_array_indices[None, :, None],  # a (occupied index)
            vir_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                all_indices[:, None, None, None],  # x
                shared_indices.T[None, :, :, None],  # shared
                shared_indices.T[None, :, :, None],  # shared
                vir_indices[None, None, None, :],  # b
            ],
            axis=1,
        )
        # y == b
        one_electron[
            all_indices[:, None, None],  # x
            vir_indices[None, None, :],  # y
            occ_array_indices[None, :, None],  # a (occupied index)
            vir_array_indices[None, None, :],  # b (virtual index)
        ] += self.one_int[
            occ_indices[None, :, None], all_indices[:, None, None]
        ]  # a, x
        coulomb[
            all_indices[:, None, None],  # x
            vir_indices[None, None, :],  # y
            occ_array_indices[None, :, None],  # a (occupied index)
            vir_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                occ_indices[None, None, :, None],  # a
                shared_indices.T[None, :, :, None],  # shared
                all_indices[:, None, None, None],  # x
                shared_indices.T[None, :, :, None],  # shared
            ],
            axis=1,
        )
        exchange[
            all_indices[:, None, None],  # x
            vir_indices[None, None, :],  # y
            occ_array_indices[None, :, None],  # a (occupied index)
            vir_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                occ_indices[None, None, :, None],  # a
                shared_indices.T[None, :, :, None],  # shared
                shared_indices.T[None, :, :, None],  # shared
                all_indices[:, None, None, None],  # x
            ],
            axis=1,
        )
        # y in shared
        coulomb[
            all_indices[None, None, :, None],  # x
            shared_indices[:, :, None, None],  # y
            occ_array_indices[:, None, None, None],  # a (occupied index)
            vir_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            all_indices[None, None, :, None],  # x
            occ_indices[:, None, None, None],  # a (occupied index)
            shared_indices[:, :, None, None],  # y
            vir_indices[None, None, None, :],  # b (virtual index)
        ]
        coulomb[
            all_indices[None, None, :, None],  # x
            shared_indices[:, :, None, None],  # y
            occ_array_indices[:, None, None, None],  # a (occupied index)
            vir_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            shared_indices[:, :, None, None],  # y
            occ_indices[:, None, None, None],  # a (occupied index)
            all_indices[None, None, :, None],  # x
            vir_indices[None, None, None, :],  # b (virtual index)
        ]
        exchange[
            all_indices[None, None, :, None],  # x
            shared_indices[:, :, None, None],  # y
            occ_array_indices[:, None, None, None],  # a (occupied index)
            vir_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            all_indices[None, None, :, None],  # x
            occ_indices[:, None, None, None],  # a (occupied index)
            vir_indices[None, None, None, :],  # b (virtual index)
            shared_indices[:, :, None, None],  # y
        ]
        exchange[
            all_indices[None, None, :, None],  # x
            shared_indices[:, :, None, None],  # y
            occ_array_indices[:, None, None, None],  # a (occupied index)
            vir_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            shared_indices[:, :, None, None],  # y
            occ_indices[:, None, None, None],  # a (occupied index)
            vir_indices[None, None, None, :],  # b (virtual index)
            all_indices[None, None, :, None],  # x
        ]

        triu_rows, triu_cols = np.triu_indices(self.nspin, k=1)
        return sign[None, None, :] * np.array(
            [
                one_electron[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
                coulomb[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
                exchange[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
            ]
        )

    def _integrate_sd_sds_deriv_two(self, occ_indices, vir_indices):
        """Return derivative of integrals of given Slater determinant with its second excitations.

        Paramters
        ---------
        occ_indices : np.ndarray(N,)
            Indices of the spin orbitals that are occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, N_params, M)
            Integrals of the given Slater determinant with its second order excitations.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.
            Second index corresponds to index of the parameter with respect to which the integral is
            derivatived.
            Third index corresponds to the second order excitations of the given Slater
            determinant.
            The excitations are ordered by the occupied orbital then the virtual orbital. For
            example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6], the ordering
            of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1, 3, 4, 5), (1,
            3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)].
            `M` is the number of first order excitations of the given Slater determinants.

        """
        # pylint: disable=C0103
        all_indices = np.arange(self.nspin)

        # TODO: use method in slater module
        annihilators = np.array(list(it.combinations(occ_indices, 2)))
        a = annihilators[:, 0]
        b = annihilators[:, 1]
        creators = np.array(list(it.combinations(vir_indices, 2)))
        c = creators[:, 0]
        d = creators[:, 1]

        sign = slater.sign_excite_two(occ_indices, vir_indices)

        occ_array_indices = np.arange(a.size)
        vir_array_indices = np.arange(c.size)

        # NOTE: here, we use the following convention for indices:
        # the first index corresponds to the row index of the antihermitian matrix for orbital
        # rotation
        # the second index corresponds to the column index of the antihermitian matrix for orbital
        # rotation
        # the third index corresponds to the occupied orbital that will be annihilated in the
        # excitation
        # the fourth index corresponds to the occupied orbital that will be created in the
        # excitation
        one_electron = np.zeros((self.nspin, self.nspin, a.size, c.size))
        coulomb = np.zeros((self.nspin, self.nspin, a.size, c.size))
        exchange = np.zeros((self.nspin, self.nspin, a.size, c.size))

        # x == a
        coulomb[
            a[:, None, None],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            all_indices[None, :, None],  # y
            b[:, None, None],  # b
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        exchange[
            a[:, None, None],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            all_indices[None, :, None],  # y
            b[:, None, None],  # b
            d[None, None, :],  # d
            c[None, None, :],  # c
        ]
        # x == b
        coulomb[
            b[:, None, None],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            all_indices[None, :, None],  # y
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        exchange[
            b[:, None, None],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            all_indices[None, :, None],  # y
            d[None, None, :],  # d
            c[None, None, :],  # c
        ]
        # x == c
        coulomb[
            c[None, None, :],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_indices[None, :, None],  # y
            d[None, None, :],  # d
        ]
        exchange[
            c[None, None, :],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            d[None, None, :],  # d
            all_indices[None, :, None],  # y
        ]
        # x == d
        coulomb[
            d[None, None, :],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            c[None, None, :],  # c
            all_indices[None, :, None],  # y
        ]
        exchange[
            d[None, None, :],  # x
            all_indices[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_indices[None, :, None],  # y
            c[None, None, :],  # c
        ]

        # y == a
        coulomb[
            all_indices[None, :, None],  # x
            a[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            all_indices[None, :, None],  # x
            b[:, None, None],  # b
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        exchange[
            all_indices[None, :, None],  # x
            a[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            all_indices[None, :, None],  # x
            b[:, None, None],  # b
            d[None, None, :],  # d
            c[None, None, :],  # c
        ]
        # y == b
        coulomb[
            all_indices[None, :, None],  # x
            b[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            all_indices[None, :, None],  # x
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        exchange[
            all_indices[None, :, None],  # x
            b[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            all_indices[None, :, None],  # x
            d[None, None, :],  # d
            c[None, None, :],  # c
        ]
        # y == c
        coulomb[
            all_indices[None, :, None],  # x
            c[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_indices[None, :, None],  # x
            d[None, None, :],  # d
        ]
        exchange[
            all_indices[None, :, None],  # x
            c[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            d[None, None, :],  # d
            all_indices[None, :, None],  # x
        ]
        # y == d
        coulomb[
            all_indices[None, :, None],  # x
            d[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            c[None, None, :],  # c
            all_indices[None, :, None],  # x
        ]
        exchange[
            all_indices[None, :, None],  # x
            d[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_indices[None, :, None],  # x
            c[None, None, :],  # c
        ]

        triu_rows, triu_cols = np.triu_indices(self.nspin, k=1)
        return sign[None, None, :] * np.array(
            [
                one_electron[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
                coulomb[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
                exchange[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
            ]
        )

    def integrate_sd_wfn(self, sd, wfn, wfn_deriv=None, ham_deriv=None, components=False):
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
        wfn_deriv : np.ndarray
            Indices of the wavefunction parameter against which the integrals are derivatized.
            Default is no derivatization.
        ham_deriv : np.ndarray
            Indices of the Hamiltonian parameter against which the integrals are derivatized.
            Default is no derivatization.
        components : {bool, False}
            Option for separating the integrals into the one electron, coulomb, and exchange
            components.
            Default adds the three components together.

        Returns
        -------
        integrals : {float, np.ndarray(3,), np.ndarray(N_derivs,), np.ndarray(3, N_derivs)}
            Integrals or derivative of integrals.
            If `wfn_deriv` or `ham_deriv` is provided, then the derivatives of the integral are
            returned.
            If `components` is False, then the integral is returned.
            If `components` is True, then the one electron, coulomb, and exchange components are
            returned.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.
            If ham_deriv is not a one-dimensional numpy array of integers.
        ValueError
            If both ham_deriv and wfn_deriv is not None.
            If ham_deriv has any indices than is less than 0 or greater than or equal to nparams.

        Notes
        -----
        Selecting only some of the parameter indices will not make the code any faster. The
        integrals are derivatized with respect to all of parameters first and then appropriate
        derivatives are selected afterwards.

        """
        # pylint: disable=C0103,R0912,R0915
        if __debug__:
            if not slater.is_sd_compatible(sd):
                raise TypeError("Slater determinant must be given as an integer.")
            if wfn_deriv is not None and ham_deriv is not None:
                raise ValueError(
                    "Integral can be derivatized with respect to at most one out of the "
                    " wavefunction and Hamiltonian parameters."
                )
            if ham_deriv is not None:
                if not (isinstance(ham_deriv, np.ndarray) and ham_deriv.ndim == 1 and ham_deriv.dtype == int):
                    raise TypeError(
                        "Derivative indices for the Hamiltonian parameters must be given as a "
                        "one-dimensional numpy array of integers."
                    )
                if np.any(ham_deriv < 0) or np.any(ham_deriv >= self.nparams):
                    raise ValueError(
                        "Derivative indices for the Hamiltonian parameters must be greater than or "
                        "equal to 0 and be less than the number of parameters."
                    )

        occ_indices = np.array(slater.occ_indices(sd))
        vir_indices = np.array(slater.vir_indices(sd, self.nspin))

        if wfn_deriv is None and ham_deriv is None:
            shape = (-1,)
            output = np.zeros(3)
        elif wfn_deriv is not None:
            if isinstance(wfn, (LinearCombinationWavefunction, ProductWavefunction)):
                shape = (-1, len(wfn_deriv[1]))
                output = np.zeros((3, len(wfn_deriv[1])))
            else:
                shape = (-1, len(wfn_deriv))
                output = np.zeros((3, len(wfn_deriv)))
        else:  # ham_deriv is not None
            shape = (-1,)
            output = np.zeros((3, len(ham_deriv)))

        overlaps_zero = np.array([wfn.get_overlap(sd, deriv=wfn_deriv)]).reshape(*shape)
        if ham_deriv is not None:
            integrals_zero = self._integrate_sd_sds_deriv_zero(occ_indices, vir_indices)
            output += np.sum(integrals_zero * overlaps_zero, axis=2)[:, ham_deriv]
        else:
            integrals_zero = self._integrate_sd_sds_zero(occ_indices)
            if wfn_deriv is not None:
                integrals_zero = np.expand_dims(integrals_zero, 2)
            output += np.sum(integrals_zero * overlaps_zero, axis=1)

        overlaps_one = np.array(
            [wfn.get_overlap(sd_exc, deriv=wfn_deriv) for sd_exc in slater.excite_bulk(sd, occ_indices, vir_indices, 1)]
        ).reshape(*shape)
        if ham_deriv is not None:
            integrals_one = self._integrate_sd_sds_deriv_one(occ_indices, vir_indices)
            output += np.sum(integrals_one * overlaps_one, axis=2)[:, ham_deriv]
        else:
            integrals_one = self._integrate_sd_sds_one(occ_indices, vir_indices)
            if wfn_deriv is not None:
                integrals_one = np.expand_dims(integrals_one, 2)
            output += np.sum(integrals_one * overlaps_one, axis=1)

        overlaps_two = np.array(
            [wfn.get_overlap(sd_exc, deriv=wfn_deriv) for sd_exc in slater.excite_bulk(sd, occ_indices, vir_indices, 2)]
        ).reshape(*shape)
        if occ_indices.size > 1 and vir_indices.size > 1:
            if ham_deriv is not None:
                integrals_two = self._integrate_sd_sds_deriv_two(occ_indices, vir_indices)
                output += np.sum(integrals_two * overlaps_two, axis=2)[:, ham_deriv]
            else:
                integrals_two = self._integrate_sd_sds_two(occ_indices, vir_indices)
                if wfn_deriv is not None:
                    integrals_two = np.expand_dims(integrals_two, 2)
                output += np.sum(integrals_two * overlaps_two, axis=1)

        if components:
            return output
        return np.sum(output, axis=0)
