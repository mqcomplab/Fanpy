r"""Hamiltonian used to describe a chemical system expressed wrt restricted orbitals."""

import itertools as it

from fanpy.tools import slater
from fanpy.ham.generalized_chemical import GeneralizedMolecularHamiltonian
from fanpy.wfn.composite.lincomb import LinearCombinationWavefunction
from fanpy.wfn.composite.product import ProductWavefunction

import numpy as np

from scipy.special import comb

# pylint: disable=C0302


# FIXME: the methods _integrate_sd_sds_zero, _integrate_sd_sds_one_alpha,
# _integrate_sd_sds_one_beta, _integrate_sd_sds_two_aa, _integrate_sd_sds_two_ab,
# _integrate_sd_sds_two_bb, _integrate_sd_sds_deriv_zero_alpha, _integrate_sd_sds_deriv_zero_beta,
# _integrate_sd_sds_deriv_one_aa, _integrate_sd_sds_deriv_one_ab, _integrate_sd_sds_deriv_one_ba,
# _integrate_sd_sds_deriv_one_bb, _integrate_sd_sds_deriv_two_aaa, _integrate_sd_sds_deriv_two_aab,
# _integrate_sd_sds_deriv_two_bab, _integrate_sd_sds_deriv_two_bbb are practically copied from
# UnrestrictedMolecularHamiltonian. The only difference is that the indices of the one_int, two_int,
# and the _cached_two_int_'s have been removed. We can replace these functions by building the
# appropriate UnrestrictedchemicalHamiltonian, but this class will become redundant in that case.
# Since this class is already written, we keep all of its methods
class RestrictedMolecularHamiltonian(GeneralizedMolecularHamiltonian):
    r"""Hamiltonian used to describe a typical chemical system expressed wrt restricted orbitals.

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
    __init__(self, one_int, two_int)
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

    @property
    def nspin(self):
        """Return the number of spin orbitals.

        Returns
        -------
        nspin : int
            Number of spin orbitals.

        """
        return 2 * self.one_int.shape[0]

    # FIXME: remove sign?
    # FIXME: too many branches, too many statements
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
            Indices of the Hamiltonian parameters against which the integral is derivatized.
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
        # pylint: disable=C0103,R0912,R0915
        if deriv is not None:
            return self._integrate_sd_sd_deriv_decomposed(sd1, sd2, deriv)

        nspatial = self.nspatial

        if __debug__ and not (slater.is_sd_compatible(sd1) and slater.is_sd_compatible(sd2)):
            raise TypeError("Slater determinant must be given as an integer.")
        shared_alpha_sd, shared_beta_sd = slater.split_spin(slater.shared_sd(sd1, sd2), nspatial)
        shared_alpha = np.array(slater.occ_indices(shared_alpha_sd))
        shared_beta = np.array(slater.occ_indices(shared_beta_sd))
        diff_sd1, diff_sd2 = slater.diff_orbs(sd1, sd2)

        # if two Slater determinants do not have the same number of electrons
        if len(diff_sd1) != len(diff_sd2):
            return 0.0, 0.0, 0.0
        diff_order = len(diff_sd1)
        if diff_order > 2:
            return 0.0, 0.0, 0.0

        sign = slater.sign_excite(sd1, diff_sd1, reversed(diff_sd2))

        one_electron, coulomb, exchange = 0, 0, 0
        # two sd's are the same
        if diff_order == 0:
            one_electron, coulomb, exchange = self._integrate_sd_sd_zero(shared_alpha, shared_beta)

        # two sd's are different by single excitation
        elif diff_order == 1:
            one_electron, coulomb, exchange = self._integrate_sd_sd_one(diff_sd1, diff_sd2, shared_alpha, shared_beta)

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
            Indices of the Hamiltonian parameters against which the integral is derivatized.
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
        decomposed_integral = self.integrate_sd_sd_decomposed(sd1, sd2, deriv=deriv)

        return np.sum(decomposed_integral, axis=0)

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
        matrix_indices : 3-tuple of int
            Spin of the orbital and the row and column indices of the antihermitian matrix that
            corresponds to the given parameter index.

        """
        # pylint: disable=C0103
        nspatial = self.nspatial
        # ind = i
        for k in range(nspatial + 1):  # pragma: no branch
            x = k
            if param_ind - nspatial * (x + 1) + (x + 1) * (x + 2) / 2 < 0:
                break
        # ind_flat = j
        ind_flat = param_ind + (x + 1) * (x + 2) / 2
        y = ind_flat % nspatial

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
            If the given `deriv` is not an integer greater than or equal to 0 and less than the
            number of parameters.

        Notes
        -----
        Integrals are not assumed to be real. The performance benefit (at the moment) for assuming
        real orbitals is not much.

        """
        # pylint: disable=C0103
        nspatial = self.nspatial

        if __debug__ and not (slater.is_sd_compatible(sd1) and slater.is_sd_compatible(sd2)):
            raise TypeError("Slater determinant must be given as an integer.")
        # NOTE: following are spatial orbital indices
        shared_alpha, shared_beta = map(
            lambda shared_sd: np.array(slater.occ_indices(shared_sd)),
            slater.split_spin(slater.shared_sd(sd1, sd2), nspatial),
        )
        # NOTE: following are spin orbital indices
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
            # NOTE: x and y are spatial orbitals
            x, y = self.param_ind_to_rowcol_ind(deriv_ind)

            # two sd's are the same
            if diff_order == 0:
                output[i] = self._integrate_sd_sd_deriv_zero(x, y, shared_alpha, shared_beta)

            # two sd's are different by single excitation
            elif diff_order == 1:
                output[i] = self._integrate_sd_sd_deriv_one(diff_sd1, diff_sd2, x, y, shared_alpha, shared_beta)

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
            If the given `deriv` is not an integer greater than or equal to 0 and less than the
            number of parameters.

        Notes
        -----
        Integrals are not assumed to be real. The performance benefit (at the moment) for assuming
        real orbitals is not much.

        """
        # pylint: disable=C0103

        derivatives = self._integrate_sd_sd_deriv_decomposed(sd1, sd2, deriv)
        return np.sum(derivatives)

    def _integrate_sd_sd_zero(self, shared_alpha, shared_beta):
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
        one_electron, coulomb, exchange = 0, 0, 0

        if shared_alpha.size != 0:
            one_electron += np.sum(self.one_int[shared_alpha, shared_alpha])
            coulomb += np.sum(np.triu(self._cached_two_int_ijij[shared_alpha[:, None], shared_alpha], k=1))
            exchange -= np.sum(np.triu(self._cached_two_int_ijji[shared_alpha[:, None], shared_alpha], k=1))
        if shared_beta.size != 0:
            one_electron += np.sum(self.one_int[shared_beta, shared_beta])
            coulomb += np.sum(np.triu(self._cached_two_int_ijij[shared_beta[:, None], shared_beta], k=1))
            exchange -= np.sum(np.triu(self._cached_two_int_ijji[shared_beta[:, None], shared_beta], k=1))
        if shared_alpha.size != 0 and shared_beta.size != 0:
            coulomb += np.sum(self._cached_two_int_ijij[shared_alpha[:, None], shared_beta])

        return one_electron, coulomb, exchange

    def _integrate_sd_sd_one(self, diff_sd1, diff_sd2, shared_alpha, shared_beta):
        """Return integrals of the given Slater determinant with its first order excitation.

        Parameters
        ----------
        diff_sd1 : 1-tuple of int
            Index of the orbital that is occupied in the first Slater determinant and not occupied
            in the second.
        diff_sd2 : 1-tuple of int
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
        one_electron, coulomb, exchange = 0, 0, 0
        nspatial = self.nspatial

        (a,) = diff_sd1
        (b,) = diff_sd2
        # get spatial indices
        spatial_a = slater.spatial_index(a, nspatial)
        spatial_b = slater.spatial_index(b, nspatial)

        if slater.is_alpha(a, nspatial) != slater.is_alpha(b, nspatial):
            return 0.0, 0.0, 0.0

        one_electron += self.one_int[spatial_a, spatial_b]
        if shared_alpha.size != 0:
            coulomb += np.sum(self.two_int[shared_alpha, spatial_a, shared_alpha, spatial_b])
            if slater.is_alpha(a, nspatial):
                exchange -= np.sum(self.two_int[shared_alpha, spatial_a, spatial_b, shared_alpha])
        if shared_beta.size != 0:
            coulomb += np.sum(self.two_int[shared_beta, spatial_a, shared_beta, spatial_b])
            if not slater.is_alpha(a, nspatial):
                exchange -= np.sum(self.two_int[shared_beta, spatial_a, spatial_b, shared_beta])

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
        # pylint:disable=C0103
        one_electron, coulomb, exchange = 0, 0, 0
        nspatial = self.nspatial

        a, b = diff_sd1
        c, d = diff_sd2
        # get spatial indices
        spatial_a = slater.spatial_index(a, nspatial)
        spatial_b = slater.spatial_index(b, nspatial)
        spatial_c = slater.spatial_index(c, nspatial)
        spatial_d = slater.spatial_index(d, nspatial)

        if slater.is_alpha(b, nspatial) == slater.is_alpha(d, nspatial) and slater.is_alpha(
            a, nspatial
        ) == slater.is_alpha(c, nspatial):
            coulomb += self.two_int[spatial_a, spatial_b, spatial_c, spatial_d]
        if slater.is_alpha(b, nspatial) == slater.is_alpha(c, nspatial) and slater.is_alpha(
            a, nspatial
        ) == slater.is_alpha(d, nspatial):
            exchange -= self.two_int[spatial_a, spatial_b, spatial_d, spatial_c]

        return one_electron, coulomb, exchange

    def _integrate_sd_sd_deriv_zero(self, x, y, shared_alpha, shared_beta):
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
        one_electron, coulomb, exchange = 0, 0, 0

        # remove orbitals x and y from the shared indices
        shared_alpha_no_x = shared_alpha[shared_alpha != x]
        shared_alpha_no_y = shared_alpha[shared_alpha != y]
        shared_beta_no_x = shared_beta[shared_beta != x]
        shared_beta_no_y = shared_beta[shared_beta != y]

        if x in shared_alpha:
            one_electron -= 2 * np.real(self.one_int[x, y])
            if shared_beta.size != 0:
                coulomb -= 2 * np.sum(np.real(self.two_int[x, shared_beta, y, shared_beta]))
            if shared_alpha_no_x.size != 0:
                coulomb -= 2 * np.sum(np.real(self.two_int[x, shared_alpha_no_x, y, shared_alpha_no_x]))
                exchange += 2 * np.sum(np.real(self.two_int[x, shared_alpha_no_x, shared_alpha_no_x, y]))
        if x in shared_beta:
            one_electron -= 2 * np.real(self.one_int[x, y])
            if shared_alpha.size != 0:
                coulomb -= 2 * np.sum(np.real(self.two_int[shared_alpha, x, shared_alpha, y]))
            if shared_beta_no_x.size != 0:
                coulomb -= 2 * np.sum(np.real(self.two_int[x, shared_beta_no_x, y, shared_beta_no_x]))
                exchange += 2 * np.sum(np.real(self.two_int[x, shared_beta_no_x, shared_beta_no_x, y]))

        if y in shared_alpha:
            one_electron += 2 * np.real(self.one_int[x, y])
            if shared_beta.size != 0:
                coulomb += 2 * np.sum(np.real(self.two_int[x, shared_beta, y, shared_beta]))
            if shared_alpha_no_y.size != 0:
                coulomb += 2 * np.sum(np.real(self.two_int[x, shared_alpha_no_y, y, shared_alpha_no_y]))
                exchange -= 2 * np.sum(np.real(self.two_int[x, shared_alpha_no_y, shared_alpha_no_y, y]))
        if y in shared_beta:
            one_electron += 2 * np.real(self.one_int[x, y])
            if shared_alpha.size != 0:
                coulomb += 2 * np.sum(np.real(self.two_int[shared_alpha, x, shared_alpha, y]))
            if shared_beta_no_y.size != 0:
                coulomb += 2 * np.sum(np.real(self.two_int[x, shared_beta_no_y, y, shared_beta_no_y]))
                exchange -= 2 * np.sum(np.real(self.two_int[x, shared_beta_no_y, shared_beta_no_y, y]))

        return one_electron, coulomb, exchange

    def _integrate_sd_sd_deriv_one(self, diff_sd1, diff_sd2, x, y, shared_alpha, shared_beta):
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
        one_electron, coulomb, exchange = 0, 0, 0
        nspatial = self.nspatial

        (a,) = diff_sd1
        (b,) = diff_sd2
        spatial_a, spatial_b = map(lambda i: slater.spatial_index(i, nspatial), [a, b])
        spin_a, spin_b = map(lambda i: int(not slater.is_alpha(i, nspatial)), [a, b])

        if spin_a == 0 and spin_b == 0:
            shared_alpha_no_ab = shared_alpha[~np.in1d(shared_alpha, [a, b])]
            shared_beta_no_ab = shared_beta
        elif spin_a == 0 and spin_b == 1:
            shared_alpha_no_ab = shared_alpha[shared_alpha != a]
            shared_beta_no_ab = shared_beta[shared_beta != b]
        elif spin_a == 1 and spin_b == 0:
            shared_alpha_no_ab = shared_alpha[shared_alpha != b]
            shared_beta_no_ab = shared_beta[shared_beta != a]
        else:
            shared_alpha_no_ab = shared_alpha
            shared_beta_no_ab = shared_beta[~np.in1d(shared_beta, [a, b])]

        # selected (spin orbital) x = a
        if x == spatial_a and spin_a == spin_b:
            one_electron -= self.one_int[y, spatial_b]
            # spin of a, b = alpha
            if spin_b == 0:
                if shared_beta_no_ab.size != 0:
                    coulomb -= np.sum(self.two_int[y, shared_beta_no_ab, spatial_b, shared_beta_no_ab])
                if shared_alpha_no_ab.size != 0:
                    coulomb -= np.sum(self.two_int[y, shared_alpha_no_ab, spatial_b, shared_alpha_no_ab])
                    exchange += np.sum(self.two_int[y, shared_alpha_no_ab, shared_alpha_no_ab, spatial_b])
            # spin of a, b = beta
            else:
                if shared_alpha_no_ab.size != 0:
                    coulomb -= np.sum(self.two_int[shared_alpha_no_ab, y, shared_alpha_no_ab, spatial_b])
                if shared_beta_no_ab.size != 0:
                    coulomb -= np.sum(self.two_int[y, shared_beta_no_ab, spatial_b, shared_beta_no_ab])
                    exchange += np.sum(self.two_int[y, shared_beta_no_ab, shared_beta_no_ab, spatial_b])
        # selected (spin orbital) x = b
        elif x == spatial_b and spin_a == spin_b:
            one_electron -= self.one_int[spatial_a, y]
            # spin of a, b = alpha
            if spin_a == 0:
                if shared_beta_no_ab.size != 0:
                    coulomb -= np.sum(self.two_int[spatial_a, shared_beta_no_ab, y, shared_beta_no_ab])
                if shared_alpha_no_ab.size != 0:
                    coulomb -= np.sum(self.two_int[spatial_a, shared_alpha_no_ab, y, shared_alpha_no_ab])
                    exchange += np.sum(self.two_int[spatial_a, shared_alpha_no_ab, shared_alpha_no_ab, y])
            # spin of a, b = beta
            else:
                if shared_alpha_no_ab.size != 0:
                    coulomb -= np.sum(self.two_int[shared_alpha_no_ab, spatial_a, shared_alpha_no_ab, y])
                if shared_beta_no_ab.size != 0:
                    coulomb -= np.sum(self.two_int[spatial_a, shared_beta_no_ab, y, shared_beta_no_ab])
                    exchange += np.sum(self.two_int[spatial_a, shared_beta_no_ab, shared_beta_no_ab, y])
        # non selected (spin orbital) x, spin of a, b = 0
        if spin_a == spin_b == 0:
            if x in shared_alpha:
                coulomb -= self.two_int[x, spatial_a, y, spatial_b]
                coulomb -= self.two_int[x, spatial_b, y, spatial_a]
                exchange += self.two_int[x, spatial_b, spatial_a, y]
                exchange += self.two_int[x, spatial_a, spatial_b, y]
            if x in shared_beta:
                coulomb -= self.two_int[spatial_a, x, spatial_b, y]
                coulomb -= self.two_int[spatial_b, x, spatial_a, y]
        # non selected (spin orbital) x, spin of a, b = 1
        elif spin_a == spin_b == 1:
            if x in shared_beta:
                coulomb -= self.two_int[x, spatial_a, y, spatial_b]
                coulomb -= self.two_int[x, spatial_b, y, spatial_a]
                exchange += self.two_int[x, spatial_b, spatial_a, y]
                exchange += self.two_int[x, spatial_a, spatial_b, y]
            if x in shared_alpha:
                coulomb -= self.two_int[x, spatial_a, y, spatial_b]
                coulomb -= self.two_int[x, spatial_b, y, spatial_a]

        # selected (spin orbital) y = a
        if y == spatial_a and spin_a == spin_b:
            one_electron += self.one_int[x, spatial_b]
            # spin of a, b = alpha
            if spin_b == 0:
                if shared_beta_no_ab.size != 0:
                    coulomb += np.sum(self.two_int[x, shared_beta_no_ab, spatial_b, shared_beta_no_ab])
                if shared_alpha_no_ab.size != 0:
                    coulomb += np.sum(self.two_int[x, shared_alpha_no_ab, spatial_b, shared_alpha_no_ab])
                    exchange -= np.sum(self.two_int[x, shared_alpha_no_ab, shared_alpha_no_ab, spatial_b])
            # spin of a, b = beta
            else:
                if shared_alpha_no_ab.size != 0:
                    coulomb += np.sum(self.two_int[shared_alpha_no_ab, x, shared_alpha_no_ab, spatial_b])
                if shared_beta_no_ab.size != 0:
                    coulomb += np.sum(self.two_int[x, shared_beta_no_ab, spatial_b, shared_beta_no_ab])
                    exchange -= np.sum(self.two_int[x, shared_beta_no_ab, shared_beta_no_ab, spatial_b])
        # selected (spin orbital) x = b
        elif y == spatial_b and spin_a == spin_b:
            one_electron += self.one_int[spatial_a, x]
            # spin of a, b = alpha
            if spin_a == 0:
                if shared_beta_no_ab.size != 0:
                    coulomb += np.sum(self.two_int[spatial_a, shared_beta_no_ab, x, shared_beta_no_ab])
                if shared_alpha_no_ab.size != 0:
                    coulomb += np.sum(self.two_int[spatial_a, shared_alpha_no_ab, x, shared_alpha_no_ab])
                    exchange -= np.sum(self.two_int[spatial_a, shared_alpha_no_ab, shared_alpha_no_ab, x])
            # spin of a, b = beta
            else:
                if shared_alpha_no_ab.size != 0:
                    coulomb += np.sum(self.two_int[shared_alpha_no_ab, spatial_a, shared_alpha_no_ab, x])
                if shared_beta_no_ab.size != 0:
                    coulomb += np.sum(self.two_int[spatial_a, shared_beta_no_ab, x, shared_beta_no_ab])
                    exchange -= np.sum(self.two_int[spatial_a, shared_beta_no_ab, shared_beta_no_ab, x])
        # non selected (spin orbital) x, spin of a, b = 0
        if spin_a == spin_b == 0:
            if y in shared_alpha:
                coulomb += self.two_int[x, spatial_a, y, spatial_b]
                coulomb += self.two_int[x, spatial_b, y, spatial_a]
                exchange -= self.two_int[x, spatial_a, spatial_b, y]
                exchange -= self.two_int[x, spatial_b, spatial_a, y]
            if y in shared_beta:
                coulomb += self.two_int[spatial_a, x, spatial_b, y]
                coulomb += self.two_int[spatial_b, x, spatial_a, y]
        # non selected (spin orbital) x, spin of a, b = 1
        elif spin_a == spin_b == 1:
            if y in shared_beta:
                coulomb += self.two_int[x, spatial_a, y, spatial_b]
                coulomb += self.two_int[x, spatial_b, y, spatial_a]
                exchange -= self.two_int[x, spatial_a, spatial_b, y]
                exchange -= self.two_int[x, spatial_b, spatial_a, y]
            if y in shared_alpha:
                coulomb += self.two_int[x, spatial_a, y, spatial_b]
                coulomb += self.two_int[x, spatial_b, y, spatial_a]

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
        # pylint: disable=C0103,R0912,R0915
        one_electron, coulomb, exchange = 0, 0, 0
        nspatial = self.nspatial

        a, b = diff_sd1
        c, d = diff_sd2
        (spatial_a, spatial_b, spatial_c, spatial_d) = map(lambda i: slater.spatial_index(i, nspatial), [a, b, c, d])
        spin_a, spin_b, spin_c, spin_d = map(lambda i: int(not slater.is_alpha(i, nspatial)), [a, b, c, d])

        if x == spatial_a:
            if spin_a == spin_c == spin_b == spin_d == 0:
                coulomb -= self.two_int[y, spatial_b, spatial_c, spatial_d]
                exchange += self.two_int[y, spatial_b, spatial_d, spatial_c]
            elif spin_a == spin_c == 0 and spin_b == spin_d == 1:
                coulomb -= self.two_int[y, spatial_b, spatial_c, spatial_d]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: d will not be alpha if c is beta
            # elif spin_a == spin_c == 1 and spin_b == spin_d == 0:
            #     coulomb -= self.two_int[spatial_b, y, spatial_d, spatial_c]
            # elif spin_a == spin_d == 0 and spin_b == spin_c == 1:
            #     exchange += self.two_int[y, spatial_b, spatial_d, spatial_c]
            # NOTE: b will not be alpha if a is beta (spin of x = spin of a)
            # elif spin_a == spin_d == 1 and spin_b == spin_c == 0:
            #     exchange += self.two_int[spatial_b, y, spatial_c, spatial_d]
            elif spin_a == spin_c == spin_b == spin_d == 1:
                coulomb -= self.two_int[y, spatial_b, spatial_c, spatial_d]
                exchange += self.two_int[y, spatial_b, spatial_d, spatial_c]
        if x == spatial_b:
            if spin_b == spin_c == spin_a == spin_d == 0:
                exchange += self.two_int[y, spatial_a, spatial_c, spatial_d]
                coulomb -= self.two_int[y, spatial_a, spatial_d, spatial_c]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: b will not be alpha if a is beta (spin of x = spin of b)
            # elif spin_b == spin_c == 0 and spin_a == spin_d == 1:
            #     exchange += self.two_int[y, spatial_a, spatial_c, spatial_d]
            # NOTE: d will not be alpha if c is beta
            # elif spin_b == spin_c == 1 and spin_a == spin_d == 0:
            #     exchange += self.two_int[spatial_a, y, spatial_d, spatial_c]
            # elif spin_b == spin_d == 0 and spin_a == spin_c == 1:
            #     coulomb -= self.two_int[y, spatial_a, spatial_d, spatial_c]
            elif spin_b == spin_d == 1 and spin_a == spin_c == 0:
                coulomb -= self.two_int[spatial_a, y, spatial_c, spatial_d]
            elif spin_b == spin_c == spin_a == spin_d == 1:
                exchange += self.two_int[y, spatial_a, spatial_c, spatial_d]
                coulomb -= self.two_int[y, spatial_a, spatial_d, spatial_c]
        if x == spatial_c:
            if spin_c == spin_a == spin_b == spin_d == 0:
                coulomb -= self.two_int[spatial_a, spatial_b, y, spatial_d]
                exchange += self.two_int[spatial_b, spatial_a, y, spatial_d]
            elif spin_c == spin_a == 0 and spin_b == spin_d == 1:
                coulomb -= self.two_int[spatial_a, spatial_b, y, spatial_d]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # elif spin_c == spin_a == 1 and spin_b == spin_d == 0:
            #     coulomb -= self.two_int[spatial_b, spatial_a, spatial_d, y]
            # elif spin_c == spin_b == 0 and spin_a == spin_d == 1:
            #     exchange += self.two_int[spatial_b, spatial_a, y, spatial_d]
            # NOTE: b will not be alpha if a is beta
            # elif spin_c == spin_b == 1 and spin_a == spin_d == 0:
            #     exchange += self.two_int[spatial_a, spatial_b, spatial_d, y]
            elif spin_c == spin_a == spin_b == spin_d == 1:
                coulomb -= self.two_int[spatial_a, spatial_b, y, spatial_d]
                exchange += self.two_int[spatial_b, spatial_a, y, spatial_d]
        if x == spatial_d:
            if spin_d == spin_a == spin_b == spin_c == 0:
                exchange += self.two_int[spatial_a, spatial_b, y, spatial_c]
                coulomb -= self.two_int[spatial_b, spatial_a, y, spatial_c]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
            # elif spin_d == spin_a == 0 and spin_b == spin_c == 1:
            #     exchange += self.two_int[spatial_a, spatial_b, y, spatial_c]
            # NOTE: b will not be alpha if a is beta
            # elif spin_d == spin_a == 1 and spin_b == spin_c == 0:
            #     exchange += self.two_int[spatial_b, spatial_a, spatial_c, y]
            # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
            # elif spin_d == spin_b == 0 and spin_a == spin_c == 1:
            #     coulomb -= self.two_int[spatial_b, spatial_a, y, spatial_c]
            elif spin_d == spin_b == 1 and spin_a == spin_c == 0:
                coulomb -= self.two_int[spatial_a, spatial_b, spatial_c, y]
            elif spin_d == spin_a == spin_b == spin_c == 1:
                exchange += self.two_int[spatial_a, spatial_b, y, spatial_c]
                coulomb -= self.two_int[spatial_b, spatial_a, y, spatial_c]

        if y == spatial_a:
            if spin_a == spin_c == spin_b == spin_d == 0:
                coulomb += self.two_int[x, spatial_b, spatial_c, spatial_d]
                exchange -= self.two_int[x, spatial_b, spatial_d, spatial_c]
            elif spin_a == spin_c == 0 and spin_b == spin_d == 1:
                coulomb += self.two_int[x, spatial_b, spatial_c, spatial_d]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: d will not be alpha if c is beta
            # elif spin_a == spin_c == 1 and spin_b == spin_d == 0:
            #     coulomb += self.two_int[spatial_b, x, spatial_d, spatial_c]
            # elif spin_a == spin_d == 0 and spin_b == spin_c == 1:
            #     exchange -= self.two_int[x, spatial_b, spatial_d, spatial_c]
            # NOTE: b will not be alpha if a is beta (spin of x = spin of a)
            # elif spin_a == spin_d == 1 and spin_b == spin_c == 0:
            #     exchange -= self.two_int[spatial_b, x, spatial_c, spatial_d]
            elif spin_a == spin_c == spin_b == spin_d == 1:
                coulomb += self.two_int[x, spatial_b, spatial_c, spatial_d]
                exchange -= self.two_int[x, spatial_b, spatial_d, spatial_c]
        if y == spatial_b:
            if spin_b == spin_c == spin_a == spin_d == 0:
                exchange -= self.two_int[x, spatial_a, spatial_c, spatial_d]
                coulomb += self.two_int[x, spatial_a, spatial_d, spatial_c]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: b will not be alpha if a is beta (spin of x = spin of b)
            # elif spin_b == spin_c == 0 and spin_a == spin_d == 1:
            #     exchange -= self.two_int[x, spatial_a, spatial_c, spatial_d]
            # NOTE: d will not be alpha if c is beta
            # elif spin_b == spin_c == 1 and spin_a == spin_d == 0:
            #     exchange -= self.two_int[spatial_a, x, spatial_d, spatial_c]
            # NOTE: b will not be alpha if a is beta (spin of x = spin of b)
            # elif spin_b == spin_d == 0 and spin_a == spin_c == 1:
            #     coulomb += self.two_int[x, spatial_a, spatial_d, spatial_c]
            elif spin_b == spin_d == 1 and spin_a == spin_c == 0:
                coulomb += self.two_int[spatial_a, x, spatial_c, spatial_d]
            elif spin_b == spin_c == spin_a == spin_d == 1:
                exchange -= self.two_int[x, spatial_a, spatial_c, spatial_d]
                coulomb += self.two_int[x, spatial_a, spatial_d, spatial_c]
        if y == spatial_c:
            if spin_c == spin_a == spin_b == spin_d == 0:
                coulomb += self.two_int[spatial_a, spatial_b, x, spatial_d]
                exchange -= self.two_int[spatial_b, spatial_a, x, spatial_d]
            elif spin_c == spin_a == 0 and spin_b == spin_d == 1:
                coulomb += self.two_int[spatial_a, spatial_b, x, spatial_d]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: d will not be alpha if c is beta (spin of x = spin of c)
            # elif spin_c == spin_a == 1 and spin_b == spin_d == 0:
            #     coulomb += self.two_int[spatial_b, spatial_a, spatial_d, x]
            # NOTE: a will not be alpha if b is beta
            # elif spin_c == spin_b == 0 and spin_a == spin_d == 1:
            #     exchange -= self.two_int[spatial_b, spatial_a, x, spatial_d]
            # NOTE: d will not be alpha if c is beta (spin of x = spin of c)
            # elif spin_c == spin_b == 1 and spin_a == spin_d == 0:
            #     exchange -= self.two_int[spatial_a, spatial_b, spatial_d, x]
            elif spin_c == spin_a == spin_b == spin_d == 1:
                coulomb += self.two_int[spatial_a, spatial_b, x, spatial_d]
                exchange -= self.two_int[spatial_b, spatial_a, x, spatial_d]
        if y == spatial_d:
            if spin_d == spin_a == spin_b == spin_c == 0:
                exchange -= self.two_int[spatial_a, spatial_b, x, spatial_c]
                coulomb += self.two_int[spatial_b, spatial_a, x, spatial_c]
            # NOTE: alpha orbitals are ordered before the beta orbitals and slater.diff_orbs
            # returns orbitals in increasing order (which means second index cannot be alpha if
            # the first is beta)
            # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
            # elif spin_d == spin_a == 0 and spin_b == spin_c == 1:
            #     exchange -= self.two_int[spatial_a, spatial_b, x, spatial_c]
            # NOTE: b will not be alpha if a is beta
            # elif spin_d == spin_a == 1 and spin_b == spin_c == 0:
            #     exchange -= self.two_int[spatial_b, spatial_a, spatial_c, x]
            # NOTE: d will not be alpha if c is beta (spin of x = spin of d)
            # elif spin_d == spin_b == 0 and spin_a == spin_c == 1:
            #     coulomb += self.two_int[spatial_b, spatial_a, x, spatial_c]
            elif spin_d == spin_b == 1 and spin_a == spin_c == 0:
                coulomb += self.two_int[spatial_a, spatial_b, spatial_c, x]
            elif spin_d == spin_a == spin_b == spin_c == 1:
                exchange -= self.two_int[spatial_a, spatial_b, x, spatial_c]
                coulomb += self.two_int[spatial_b, spatial_a, x, spatial_c]

        return one_electron, coulomb, exchange

    def _integrate_sd_sds_zero(self, occ_alpha, occ_beta):
        """Return the integrals of the given Slater determinant with itself.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, 1)
            Integrals of the given Slater determinant with itself.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.

        """
        one_electron = np.sum(self.one_int[occ_alpha, occ_alpha])
        one_electron += np.sum(self.one_int[occ_beta, occ_beta])

        coulomb = np.sum(np.triu(self._cached_two_int_ijij[occ_alpha[:, None], occ_alpha], k=1))
        coulomb += np.sum(self._cached_two_int_ijij[occ_alpha[:, None], occ_beta])
        coulomb += np.sum(np.triu(self._cached_two_int_ijij[occ_beta[:, None], occ_beta], k=1))

        exchange = -np.sum(np.triu(self._cached_two_int_ijji[occ_alpha[:, None], occ_alpha], k=1))
        exchange -= np.sum(np.triu(self._cached_two_int_ijji[occ_beta[:, None], occ_beta], k=1))

        return np.array([[one_electron], [coulomb], [exchange]])

    def _integrate_sd_sds_one_alpha(self, occ_alpha, occ_beta, vir_alpha):
        """Return the integrals of the given Slater determinant with its first order excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_alpha : np.ndarray(K-N_a,)
            Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, M)
            Integrals of the given Slater determinant with its first order excitations involving the
            alpha spin orbitals.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.
            Second index corresponds to the first order excitations of the given Slater determinant.
            The excitations are ordered by the occupied orbital then the virtual orbital. For
            example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the ordering of the
            excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number of first order
            excitations of the given Slater determinants.

        """
        shared_alpha = slater.shared_indices_remove_one_index(occ_alpha)

        sign_a = slater.sign_excite_one(occ_alpha, vir_alpha)

        one_electron_a = self.one_int[occ_alpha[:, np.newaxis], vir_alpha[np.newaxis, :]].ravel()

        coulomb_a = np.sum(
            self.two_int[
                shared_alpha[:, :, np.newaxis],
                occ_alpha[:, np.newaxis, np.newaxis],
                shared_alpha[:, :, np.newaxis],
                vir_alpha[np.newaxis, np.newaxis, :],
            ],
            axis=1,
        ).ravel()
        coulomb_a += np.sum(
            self.two_int[
                occ_alpha[:, np.newaxis, np.newaxis],
                occ_beta[np.newaxis, :, np.newaxis],
                vir_alpha[np.newaxis, np.newaxis, :],
                occ_beta[np.newaxis, :, np.newaxis],
            ],
            axis=1,
        ).ravel()

        exchange_a = -np.sum(
            self.two_int[
                shared_alpha[:, :, np.newaxis],
                occ_alpha[:, np.newaxis, np.newaxis],
                vir_alpha[np.newaxis, np.newaxis, :],
                shared_alpha[:, :, np.newaxis],
            ],
            axis=1,
        ).ravel()

        return sign_a[None, :] * np.array([one_electron_a, coulomb_a, exchange_a])

    def _integrate_sd_sds_one_beta(self, occ_alpha, occ_beta, vir_beta):
        """Return the integrals of the given Slater determinant with its first order excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_beta : np.ndarray(K-N_b,)
            Indices of the beta spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, M)
            Integrals of the given Slater determinant with its first order excitations involving the
            beta spin orbitals.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.
            Second index corresponds to the first order excitations of the given Slater determinant.
            The excitations are ordered by the occupied orbital then the virtual orbital. For
            example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the ordering of the
            excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number of first order
            excitations of the given Slater determinants.

        """
        shared_beta = slater.shared_indices_remove_one_index(occ_beta)

        sign_b = slater.sign_excite_one(occ_beta, vir_beta)

        one_electron_b = self.one_int[occ_beta[:, np.newaxis], vir_beta[np.newaxis, :]].ravel()

        coulomb_b = np.sum(
            self.two_int[
                shared_beta[:, :, np.newaxis],
                occ_beta[:, np.newaxis, np.newaxis],
                shared_beta[:, :, np.newaxis],
                vir_beta[np.newaxis, np.newaxis, :],
            ],
            axis=1,
        ).ravel()
        coulomb_b += np.sum(
            self.two_int[
                occ_alpha[np.newaxis, :, np.newaxis],
                occ_beta[:, np.newaxis, np.newaxis],
                occ_alpha[np.newaxis, :, np.newaxis],
                vir_beta[np.newaxis, np.newaxis, :],
            ],
            axis=1,
        ).ravel()

        exchange_b = -np.sum(
            self.two_int[
                shared_beta[:, :, np.newaxis],
                occ_beta[:, np.newaxis, np.newaxis],
                vir_beta[np.newaxis, np.newaxis, :],
                shared_beta[:, :, np.newaxis],
            ],
            axis=1,
        ).ravel()

        return sign_b[None, :] * np.array([one_electron_b, coulomb_b, exchange_b])

    def _integrate_sd_sds_two_aa(self, occ_alpha, occ_beta, vir_alpha):  # pylint: disable=W0613
        """Return the integrals of a Slater determinant with its second order (alpha) excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_alpha : np.ndarray(K-N_a,)
            Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(2, M)
            Integrals of the given Slater determinant with its second order excitations involving
            the alpha spin orbitals.
            First index corresponds to the coulomb (index 0) and exchange (index 1) integrals.
            Second index corresponds to the second order excitations of the given Slater
            determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
            the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
            3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
            the number of first order excitations of the given Slater determinants.

        """
        # pylint: disable=C0103
        annihilators = np.array(list(it.combinations(occ_alpha, 2)))
        a = annihilators[:, 0]
        b = annihilators[:, 1]
        creators = np.array(list(it.combinations(vir_alpha, 2)))
        c = creators[:, 0]
        d = creators[:, 1]

        sign = slater.sign_excite_two(occ_alpha, vir_alpha)

        coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]].ravel()
        exchange = -self.two_int[a[:, None], b[:, None], d[None, :], c[None, :]].ravel()

        return sign[None, :] * np.array([coulomb, exchange])

    def _integrate_sd_sds_two_ab(self, occ_alpha, occ_beta, vir_alpha, vir_beta):
        """Return the integrals of a SD with its second order (alpha and beta) excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_alpha : np.ndarray(K-N_a,)
            Indices of the alpha spin orbitals that are not occupied in the Slater determinant.
        vir_beta : np.ndarray(K-N_b,)
            Indices of the beta spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(M,)
            Coulomb integrals of the given Slater determinant with its second order excitations
            involving both alpha and beta orbitals
            Second index corresponds to the second order excitations of the given Slater
            determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
            the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
            3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
            the number of first order excitations of the given Slater determinants.

        """
        # pylint: disable=C0103
        annihilators = np.array(list(it.product(occ_alpha, occ_beta)))
        a = annihilators[:, 0]
        b = annihilators[:, 1]
        creators = np.array(list(it.product(vir_alpha, vir_beta)))
        c = creators[:, 0]
        d = creators[:, 1]

        sign = slater.sign_excite_two_ab(occ_alpha, occ_beta, vir_alpha, vir_beta)
        coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]].ravel()

        return sign * coulomb

    def _integrate_sd_sds_two_bb(self, occ_alpha, occ_beta, vir_beta):  # pylint: disable=W0613
        """Return the integrals of a Slater determinant with its second order (beta) excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_beta : np.ndarray(K-N_b,)
            Indices of the beta spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(2, M)
            Integrals of the given Slater determinant with its second order excitations involving
            the beta spin orbitals.
            First index corresponds to the coulomb (index 0) and exchange (index 1) integrals.
            Second index corresponds to the second order excitations of the given Slater
            determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
            the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
            3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
            the number of first order excitations of the given Slater determinants.

        """
        # pylint: disable=C0103
        annihilators = np.array(list(it.combinations(occ_beta, 2)))
        a = annihilators[:, 0]
        b = annihilators[:, 1]
        creators = np.array(list(it.combinations(vir_beta, 2)))
        c = creators[:, 0]
        d = creators[:, 1]

        sign = slater.sign_excite_two(occ_beta, vir_beta)
        coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]].ravel()
        exchange = -self.two_int[a[:, None], b[:, None], d[None, :], c[None, :]].ravel()

        return sign[None, :] * np.array([coulomb, exchange])

    def _integrate_sd_sds_deriv_zero_alpha(self, occ_alpha, occ_beta, vir_alpha):
        """Return the derivative wrt alpha parameters of the integrals of the given SD with itself.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_alpha : np.ndarray(K-N_a,)
            Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, N_params, 1)
            Derivatives of the integrals of the given Slater determinant with itself with respect to
            the parameters that correspond to the alpha orbitals.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.
            Second index corresponds to index of the parameter with respect to which the integral is
            derivatived.

        """
        nspatial = self.nspatial
        all_alpha = np.arange(nspatial)
        shared_alpha = slater.shared_indices_remove_one_index(occ_alpha)

        # NOTE: here, we use the following convention for indices:
        # the first index corresponds to the row index of the antihermitian matrix for orbital
        # rotation
        # the second index corresponds to the column index of the antihermitian matrix for orbital
        # rotation
        one_electron_a = np.zeros((nspatial, nspatial))
        coulomb_a = np.zeros((nspatial, nspatial))
        exchange_a = np.zeros((nspatial, nspatial))

        # NOTE: if both x and y are occupied these cancel each other out
        one_electron_a[occ_alpha[:, None], vir_alpha[None, :]] -= 2 * np.real(
            self.one_int[occ_alpha[:, None], vir_alpha[None, :]]
        )
        one_electron_a[vir_alpha[:, None], occ_alpha[None, :]] += 2 * np.real(
            self.one_int[vir_alpha[:, None], occ_alpha[None, :]]
        )

        # if x is occupied and alpha
        coulomb_a[occ_alpha[:, None], all_alpha[None, :]] -= 2 * np.sum(
            np.real(
                self.two_int[
                    occ_alpha[:, None, None],  # x
                    shared_alpha[:, :, None],  # shared alpha no x
                    all_alpha[None, None, :],  # y
                    shared_alpha[:, :, None],  # shared alpha no x
                ]
            ),
            axis=1,
        )
        coulomb_a[occ_alpha[:, None], all_alpha[None, :]] -= 2 * np.sum(
            np.real(
                self.two_int[
                    occ_alpha[:, None, None],  # x
                    occ_beta[None, :, None],  # shared beta no x
                    all_alpha[None, None, :],  # y
                    occ_beta[None, :, None],  # shared beta no x
                ]
            ),
            axis=1,
        )
        # if y is occupied and alpha
        coulomb_a[all_alpha[:, None], occ_alpha[None, :]] += 2 * np.sum(
            np.real(
                self.two_int[
                    all_alpha[:, None, None],  # x
                    shared_alpha.T[None, :, :],  # shared alpha no y
                    occ_alpha[None, None, :],  # y
                    shared_alpha.T[None, :, :],  # shared alpha no y
                ]
            ),
            axis=1,
        )
        coulomb_a[all_alpha[:, None], occ_alpha[None, :]] += 2 * np.sum(
            np.real(
                self.two_int[
                    all_alpha[:, None, None],  # x
                    occ_beta[None, :, None],  # shared beta no y
                    occ_alpha[None, None, :],  # y
                    occ_beta[None, :, None],  # shared beta no y
                ]
            ),
            axis=1,
        )

        # if x is occupied and alpha
        exchange_a[occ_alpha[:, None], all_alpha[None, :]] += 2 * np.sum(
            np.real(
                self.two_int[
                    occ_alpha[:, None, None],  # x
                    shared_alpha[:, :, None],  # shared alpha no x
                    shared_alpha[:, :, None],  # shared alpha no x
                    all_alpha[None, None, :],  # y
                ]
            ),
            axis=1,
        )
        # if y is occupied and alpha
        exchange_a[all_alpha[:, None], occ_alpha[None, :]] -= 2 * np.sum(
            np.real(
                self.two_int[
                    all_alpha[:, None, None],  # x
                    shared_alpha.T[None, :, :],  # shared alpha no y
                    shared_alpha.T[None, :, :],  # shared alpha no y
                    occ_alpha[None, None, :],  # y
                ]
            ),
            axis=1,
        )

        triu_indices = np.triu_indices(nspatial, k=1)
        return np.array([one_electron_a[triu_indices], coulomb_a[triu_indices], exchange_a[triu_indices]])[:, :, None]

    def _integrate_sd_sds_deriv_zero_beta(self, occ_alpha, occ_beta, vir_beta):
        """Return the derivative wrt beta parameters of the integrals of the given SD with itself.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_beta : np.ndarray(K-N_b,)
            Indices of the beta spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, N_params, 1)
            Derivatives of the integrals of the given Slater determinant with itself with respect to
            the parameters that correspond to the beta orbitals.
            First index corresponds to the one-electron (first element), coulomb (second element),
            and exchange (third element) integrals.
            Second index corresponds to index of the parameter with respect to which the integral is
            derivatived.

        """
        nspatial = self.nspatial
        all_beta = np.arange(nspatial)
        shared_beta = slater.shared_indices_remove_one_index(occ_beta)

        # NOTE: here, we use the following convention for indices:
        # the first index corresponds to the row index of the antihermitian matrix for orbital
        # rotation
        # the second index corresponds to the column index of the antihermitian matrix for orbital
        # rotation
        one_electron_b = np.zeros((nspatial, nspatial))
        coulomb_b = np.zeros((nspatial, nspatial))
        exchange_b = np.zeros((nspatial, nspatial))

        # NOTE: if both x and y are occupied these cancel each other out
        one_electron_b[occ_beta[:, None], vir_beta[None, :]] -= 2 * np.real(
            self.one_int[occ_beta[:, None], vir_beta[None, :]]
        )
        one_electron_b[vir_beta[:, None], occ_beta[None, :]] += 2 * np.real(
            self.one_int[vir_beta[:, None], occ_beta[None, :]]
        )

        # if x is occupied and beta
        coulomb_b[occ_beta[:, None], all_beta[None, :]] -= 2 * np.sum(
            np.real(
                self.two_int[
                    occ_beta[:, None, None],  # x
                    shared_beta[:, :, None],  # shared beta no x
                    all_beta[None, None, :],  # y
                    shared_beta[:, :, None],  # shared beta no x
                ]
            ),
            axis=1,
        )
        coulomb_b[occ_beta[:, None], all_beta[None, :]] -= 2 * np.sum(
            np.real(
                self.two_int[
                    occ_alpha[None, :, None],  # shared alpha no x
                    occ_beta[:, None, None],  # x
                    occ_alpha[None, :, None],  # shared alpha no x
                    all_beta[None, None, :],  # y
                ]
            ),
            axis=1,
        )
        # if y is occupied and beta
        coulomb_b[all_beta[:, None], occ_beta[None, :]] += 2 * np.sum(
            np.real(
                self.two_int[
                    all_beta[:, None, None],  # x
                    shared_beta.T[None, :, :],  # shared beta no y
                    occ_beta[None, None, :],  # y
                    shared_beta.T[None, :, :],  # shared beta no y
                ]
            ),
            axis=1,
        )
        coulomb_b[all_beta[:, None], occ_beta[None, :]] += 2 * np.sum(
            np.real(
                self.two_int[
                    occ_alpha[None, :, None],  # shared alpha no y
                    all_beta[:, None, None],  # x
                    occ_alpha[None, :, None],  # shared alpha no y
                    occ_beta[None, None, :],  # y
                ]
            ),
            axis=1,
        )

        # if x is occupied and beta
        exchange_b[occ_beta[:, None], all_beta[None, :]] += 2 * np.sum(
            np.real(
                self.two_int[
                    occ_beta[:, None, None],  # x
                    shared_beta[:, :, None],  # shared beta no x
                    shared_beta[:, :, None],  # shared beta no x
                    all_beta[None, None, :],  # y
                ]
            ),
            axis=1,
        )
        # if y is occupied and beta
        exchange_b[all_beta[:, None], occ_beta[None, :]] -= 2 * np.sum(
            np.real(
                self.two_int[
                    all_beta[:, None, None],  # x
                    shared_beta.T[None, :, :],  # shared beta no y
                    shared_beta.T[None, :, :],  # shared beta no y
                    occ_beta[None, None, :],  # y
                ]
            ),
            axis=1,
        )

        triu_indices = np.triu_indices(nspatial, k=1)
        return np.array([one_electron_b[triu_indices], coulomb_b[triu_indices], exchange_b[triu_indices]])[:, :, None]

    def _integrate_sd_sds_deriv_one_aa(self, occ_alpha, occ_beta, vir_alpha):
        """Return (alpha) derivative of integrals an SD with its (alpha) single excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_alpha : np.ndarray(K-N_a,)
            Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, N_params, M)
            Derivatives of the integrals of the given Slater determinant with its first order
            excitations of alpha orbitals with respect to parameters associated with alpha orbitals.
            First index of the numpy array corresponds to the one-electron (first element), coulomb
            (second elements), and exchange (third element) integrals.
            Second index of the numpy array corresponds to index of the parameter with respect to
            which the integral is derivatived.
            Third index of the numpy array corresponds to the first order excitations of the given
            Slater determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the
            ordering of the excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number
            of first order excitations of the given Slater determinants.

        """
        # pylint: disable=R0915
        nspatial = self.nspatial
        all_alpha = np.arange(nspatial)

        sign_a = slater.sign_excite_one(occ_alpha, vir_alpha)

        occ_alpha_array_indices = np.arange(occ_alpha.size)
        vir_alpha_array_indices = np.arange(vir_alpha.size)

        shared_alpha = slater.shared_indices_remove_one_index(occ_alpha)

        # NOTE: here, we use the following convention for indices:
        # the first index corresponds to the row index of the antihermitian matrix for orbital
        # rotation
        # the second index corresponds to the column index of the antihermitian matrix for orbital
        # rotation
        # the third index corresponds to the occupied orbital that will be annihilated in the
        # excitation
        # the fourth index corresponds to the occupied orbital that will be created in the
        # excitation
        # FIXME: hardcoded parameter shape
        one_electron_a = np.zeros((nspatial, nspatial, occ_alpha.size, vir_alpha.size))
        coulomb_a = np.zeros((nspatial, nspatial, occ_alpha.size, vir_alpha.size))
        exchange_a = np.zeros((nspatial, nspatial, occ_alpha.size, vir_alpha.size))

        # ONE-ELECTRON INTEGRALS
        # x == a
        one_electron_a[
            occ_alpha[:, None, None],  # x
            all_alpha[None, :, None],  # y
            occ_alpha_array_indices[:, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] -= self.one_int[
            all_alpha[None, :, None], vir_alpha[None, None, :]  # y, b
        ]
        # x == b
        one_electron_a[
            vir_alpha[None, None, :],  # x
            all_alpha[None, :, None],  # y
            occ_alpha_array_indices[:, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] -= self.one_int[
            occ_alpha[:, None, None], all_alpha[None, :, None]  # a, y
        ]
        # y == a
        one_electron_a[
            all_alpha[:, None, None],  # x
            occ_alpha[None, :, None],  # y
            occ_alpha_array_indices[None, :, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] += self.one_int[
            all_alpha[:, None, None], vir_alpha[None, None, :]  # x, b
        ]
        # y == b
        one_electron_a[
            all_alpha[:, None, None],  # x
            vir_alpha[None, None, :],  # y
            occ_alpha_array_indices[None, :, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] += self.one_int[
            occ_alpha[None, :, None], all_alpha[:, None, None]  # a, x
        ]

        # COULOMB INTEGRALS
        # x == a
        coulomb_a[
            occ_alpha[:, None, None],  # x
            all_alpha[None, :, None],  # y
            occ_alpha_array_indices[:, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                all_alpha[None, None, :, None],  # y
                shared_alpha[:, :, None, None],  # shared alpha
                vir_alpha[None, None, None, :],  # b
                shared_alpha[:, :, None, None],  # shared alpha
            ],
            axis=1,
        )
        coulomb_a[
            occ_alpha[:, None, None],  # x
            all_alpha[None, :, None],  # y
            occ_alpha_array_indices[:, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                all_alpha[None, None, :, None],  # y
                occ_beta[None, :, None, None],  # shared beta
                vir_alpha[None, None, None, :],  # b
                occ_beta[None, :, None, None],  # shared beta
            ],
            axis=1,
        )
        # x == b
        coulomb_a[
            vir_alpha[None, None, :],  # x
            all_alpha[None, :, None],  # y
            occ_alpha_array_indices[:, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                occ_alpha[:, None, None, None],  # a
                shared_alpha[:, :, None, None],  # shared alpha
                all_alpha[None, None, :, None],  # y
                shared_alpha[:, :, None, None],  # shared alpha
            ],
            axis=1,
        )
        coulomb_a[
            vir_alpha[None, None, :],  # x
            all_alpha[None, :, None],  # y
            occ_alpha_array_indices[:, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                occ_alpha[:, None, None, None],  # a
                occ_beta[None, :, None, None],  # shared beta
                all_alpha[None, None, :, None],  # y
                occ_beta[None, :, None, None],  # shared beta
            ],
            axis=1,
        )
        # x in shared
        coulomb_a[
            shared_alpha[:, :, None, None],  # x
            all_alpha[None, None, :, None],  # y
            occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            shared_alpha[:, :, None, None],  # x
            occ_alpha[:, None, None, None],  # a (occupied index)
            all_alpha[None, None, :, None],  # y
            vir_alpha[None, None, None, :],  # b (virtual index)
        ]
        coulomb_a[
            shared_alpha[:, :, None, None],  # x
            all_alpha[None, None, :, None],  # y
            occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            shared_alpha[:, :, None, None],  # x
            vir_alpha[None, None, None, :],  # b (virtual index)
            all_alpha[None, None, :, None],  # y
            occ_alpha[:, None, None, None],  # a (occupied index)
        ]
        # y == a
        coulomb_a[
            all_alpha[:, None, None],  # x
            occ_alpha[None, :, None],  # y
            occ_alpha_array_indices[None, :, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                all_alpha[:, None, None, None],  # x
                shared_alpha.T[None, :, :, None],  # shared alpha
                vir_alpha[None, None, None, :],  # b
                shared_alpha.T[None, :, :, None],  # shared alpha
            ],
            axis=1,
        )
        coulomb_a[
            all_alpha[:, None, None],  # x
            occ_alpha[None, :, None],  # y
            occ_alpha_array_indices[None, :, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                all_alpha[:, None, None, None],  # x
                occ_beta[None, :, None, None],  # shared beta
                vir_alpha[None, None, None, :],  # b
                occ_beta[None, :, None, None],  # shared beta
            ],
            axis=1,
        )
        # y == b
        coulomb_a[
            all_alpha[:, None, None],  # x
            vir_alpha[None, None, :],  # y
            occ_alpha_array_indices[None, :, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                occ_alpha[None, None, :, None],  # a
                shared_alpha.T[None, :, :, None],  # shared alpha
                all_alpha[:, None, None, None],  # x
                shared_alpha.T[None, :, :, None],  # shared alpha
            ],
            axis=1,
        )
        coulomb_a[
            all_alpha[:, None, None],  # x
            vir_alpha[None, None, :],  # y
            occ_alpha_array_indices[None, :, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                occ_alpha[None, None, :, None],  # a
                occ_beta[None, :, None, None],  # shared beta
                all_alpha[:, None, None, None],  # x
                occ_beta[None, :, None, None],  # shared beta
            ],
            axis=1,
        )
        # y in shared
        coulomb_a[
            all_alpha[None, None, :, None],  # x
            shared_alpha[:, :, None, None],  # y
            occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            all_alpha[None, None, :, None],  # x
            occ_alpha[:, None, None, None],  # a (occupied index)
            shared_alpha[:, :, None, None],  # y
            vir_alpha[None, None, None, :],  # b (virtual index)
        ]
        coulomb_a[
            all_alpha[None, None, :, None],  # x
            shared_alpha[:, :, None, None],  # y
            occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            all_alpha[None, None, :, None],  # x
            vir_alpha[None, None, None, :],  # b (virtual index)
            shared_alpha[:, :, None, None],  # y
            occ_alpha[:, None, None, None],  # a (occupied index)
        ]

        # EXCHANGE INTEGRALS
        # x == a
        exchange_a[
            occ_alpha[:, None, None],  # x
            all_alpha[None, :, None],  # y
            occ_alpha_array_indices[:, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                all_alpha[None, None, :, None],  # y
                shared_alpha[:, :, None, None],  # shared alpha
                shared_alpha[:, :, None, None],  # shared alpha
                vir_alpha[None, None, None, :],  # b
            ],
            axis=1,
        )
        # x == b
        exchange_a[
            vir_alpha[None, None, :],  # x
            all_alpha[None, :, None],  # y
            occ_alpha_array_indices[:, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                occ_alpha[:, None, None, None],  # a
                shared_alpha[:, :, None, None],  # shared alpha
                shared_alpha[:, :, None, None],  # shared alpha
                all_alpha[None, None, :, None],  # y
            ],
            axis=1,
        )
        # x in shared
        exchange_a[
            shared_alpha[:, :, None, None],  # x
            all_alpha[None, None, :, None],  # y
            occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            shared_alpha[:, :, None, None],  # x
            occ_alpha[:, None, None, None],  # a (occupied index)
            vir_alpha[None, None, None, :],  # b (virtual index)
            all_alpha[None, None, :, None],  # y
        ]
        exchange_a[
            shared_alpha[:, :, None, None],  # x
            all_alpha[None, None, :, None],  # y
            occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            shared_alpha[:, :, None, None],  # x
            vir_alpha[None, None, None, :],  # b (virtual index)
            occ_alpha[:, None, None, None],  # a (occupied index)
            all_alpha[None, None, :, None],  # y
        ]
        # y == a
        exchange_a[
            all_alpha[:, None, None],  # x
            occ_alpha[None, :, None],  # y
            occ_alpha_array_indices[None, :, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                all_alpha[:, None, None, None],  # x
                shared_alpha.T[None, :, :, None],  # shared alpha
                shared_alpha.T[None, :, :, None],  # shared alpha
                vir_alpha[None, None, None, :],  # b
            ],
            axis=1,
        )
        # y == b
        exchange_a[
            all_alpha[:, None, None],  # x
            vir_alpha[None, None, :],  # y
            occ_alpha_array_indices[None, :, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                occ_alpha[None, None, :, None],  # a
                shared_alpha.T[None, :, :, None],  # shared alpha
                shared_alpha.T[None, :, :, None],  # shared alpha
                all_alpha[:, None, None, None],  # x
            ],
            axis=1,
        )
        # y in shared
        exchange_a[
            all_alpha[None, None, :, None],  # x
            shared_alpha[:, :, None, None],  # y
            occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            all_alpha[None, None, :, None],  # x
            occ_alpha[:, None, None, None],  # a (occupied index)
            vir_alpha[None, None, None, :],  # b (virtual index)
            shared_alpha[:, :, None, None],  # y
        ]
        exchange_a[
            all_alpha[None, None, :, None],  # x
            shared_alpha[:, :, None, None],  # y
            occ_alpha_array_indices[:, None, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            all_alpha[None, None, :, None],  # x
            vir_alpha[None, None, None, :],  # b (virtual index)
            occ_alpha[:, None, None, None],  # a (occupied index)
            shared_alpha[:, :, None, None],  # y
        ]

        triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
        return sign_a[None, None, :] * np.array(
            [
                one_electron_a[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
                coulomb_a[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
                exchange_a[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
            ]
        )

    def _integrate_sd_sds_deriv_one_ba(self, occ_alpha, occ_beta, vir_alpha):
        """Return (beta) derivative of integrals of the a SD with its (alpha) single excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_alpha : np.ndarray(K-N_a,)
            Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(N_params, M)
            Derivatives of the coulomb integrals of the given Slater determinant with its first
            order excitations of alpha orbitals with respect to parameters associated with beta
            orbitals.
            Second index of the numpy array corresponds to index of the parameter with respect to
            which the integral is derivatived.
            Third index of the numpy array corresponds to the first order excitations of the given
            Slater determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the
            ordering of the excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number
            of first order excitations of the given Slater determinants.

        """
        # pylint: disable=R0915
        nspatial = self.nspatial
        all_beta = np.arange(nspatial)
        occ_alpha_array_indices = np.arange(occ_alpha.size)
        vir_alpha_array_indices = np.arange(vir_alpha.size)

        sign_a = slater.sign_excite_one(occ_alpha, vir_alpha)

        # NOTE: here, we use the following convention for indices:
        # the first index corresponds to the row index of the antihermitian matrix for orbital
        # rotation
        # the second index corresponds to the column index of the antihermitian matrix for orbital
        # rotation
        # the third index corresponds to the occupied orbital that will be annihilated in the
        # excitation
        # the fourth index corresponds to the occupied orbital that will be created in the
        # excitation
        # FIXME: hardcoded parameter shape
        coulomb_ba = np.zeros((nspatial, nspatial, occ_alpha.size, vir_alpha.size))

        #
        coulomb_ba[
            occ_beta[:, None, None, None],  # x
            all_beta[None, None, None, :, None],  # y
            occ_alpha_array_indices[None, None, :, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            occ_alpha[None, None, :, None, None],  # a (occupied index)
            occ_beta[:, None, None, None],  # x
            vir_alpha[None, None, None, None, :],  # b (virtual index)
            all_beta[None, None, None, :, None],  # y
        ]
        coulomb_ba[
            occ_beta[:, None, None, None],  # x
            all_beta[None, None, None, :, None],  # y
            occ_alpha_array_indices[None, None, :, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            vir_alpha[None, None, None, None, :],  # b (virtual index)
            occ_beta[:, None, None, None],  # x
            occ_alpha[None, None, :, None, None],  # a (occupied index)
            all_beta[None, None, None, :, None],  # y
        ]
        #
        coulomb_ba[
            all_beta[None, None, None, :, None],  # x
            occ_beta[:, None, None, None],  # y
            occ_alpha_array_indices[None, None, :, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            occ_alpha[None, None, :, None, None],  # a (occupied index)
            all_beta[None, None, None, :, None],  # x
            vir_alpha[None, None, None, None, :],  # b (virtual index)
            occ_beta[:, None, None, None],  # y
        ]
        coulomb_ba[
            all_beta[None, None, None, :, None],  # x
            occ_beta[:, None, None, None],  # y
            occ_alpha_array_indices[None, None, :, None, None],  # a (occupied index)
            vir_alpha_array_indices[None, None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            vir_alpha[None, None, None, None, :],  # b (virtual index)
            all_beta[None, None, None, :, None],  # x
            occ_alpha[None, None, :, None, None],  # a (occupied index)
            occ_beta[:, None, None, None],  # y
        ]

        triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
        return sign_a[None, :] * coulomb_ba[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1)

    def _integrate_sd_sds_deriv_one_ab(self, occ_alpha, occ_beta, vir_beta):
        """Return (alpha) derivative of integrals of the a SD with its (beta) single excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_beta : np.ndarray(K-N_b,)
            Indices of the beta spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(N_params, M)
            Derivatives of the coulomb integrals of the given Slater determinant with its first
            order excitations of beta orbitals with respect to parameters associated with alpha
            orbitals.
            Second index of the numpy array corresponds to index of the parameter with respect to
            which the integral is derivatived.
            Third index of the numpy array corresponds to the first order excitations of the given
            Slater determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the
            ordering of the excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number
            of first order excitations of the given Slater determinants.

        """
        # pylint: disable=R0915
        nspatial = self.nspatial
        all_alpha = np.arange(nspatial)

        occ_beta_array_indices = np.arange(occ_beta.size)
        vir_beta_array_indices = np.arange(vir_beta.size)

        sign_b = slater.sign_excite_one(occ_beta, vir_beta)

        # NOTE: here, we use the following convention for indices:
        # the first index corresponds to the row index of the antihermitian matrix for orbital
        # rotation
        # the second index corresponds to the column index of the antihermitian matrix for orbital
        # rotation
        # the third index corresponds to the occupied orbital that will be annihilated in the
        # excitation
        # the fourth index corresponds to the occupied orbital that will be created in the
        # excitation
        # FIXME: hardcoded parameter shape
        coulomb_ab = np.zeros((nspatial, nspatial, occ_beta.size, vir_beta.size))

        #
        coulomb_ab[
            occ_alpha[:, None, None, None],  # x
            all_alpha[None, None, :, None],  # y
            occ_beta_array_indices[None, :, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            occ_alpha[:, None, None, None],  # x
            occ_beta[None, :, None, None],  # a (occupied index)
            all_alpha[None, None, :, None],  # y
            vir_beta[None, None, None, :],  # b (virtual index)
        ]
        coulomb_ab[
            occ_alpha[:, None, None, None],  # x
            all_alpha[None, None, :, None],  # y
            occ_beta_array_indices[None, :, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            occ_alpha[:, None, None, None],  # x
            vir_beta[None, None, None, :],  # b (virtual index)
            all_alpha[None, None, :, None],  # y
            occ_beta[None, :, None, None],  # a (occupied index)
        ]
        #
        coulomb_ab[
            all_alpha[None, None, None, :, None],  # x
            occ_alpha[:, None, None, None],  # y
            occ_beta_array_indices[None, :, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            all_alpha[None, None, :, None],  # x
            occ_beta[None, :, None, None],  # a (occupied index)
            occ_alpha[:, None, None, None],  # y
            vir_beta[None, None, None, :],  # b (virtual index)
        ]
        coulomb_ab[
            all_alpha[None, None, :, None],  # x
            occ_alpha[:, None, None, None],  # y
            occ_beta_array_indices[None, :, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            all_alpha[None, None, :, None],  # x
            vir_beta[None, None, None, :],  # b (virtual index)
            occ_alpha[:, None, None, None],  # y
            occ_beta[None, :, None, None],  # a (occupied index)
        ]

        triu_rows, triu_cols = np.triu_indices(nspatial, k=1)

        return sign_b[None, :] * coulomb_ab[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1)

    def _integrate_sd_sds_deriv_one_bb(self, occ_alpha, occ_beta, vir_beta):
        """Return (beta) derivative of integrals an SD with its single excitations of beta orbitals.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_beta : np.ndarray(K-N_b,)
            Indices of the beta spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(3, N_params, M)
            Derivatives of the integrals of the given Slater determinant with its first order
            excitations of beta orbitals with respect to parameters associated with beta orbitals.
            First index of the numpy array corresponds to the one-electron (first element), coulomb
            (second elements), and exchange (third element) integrals.
            Second index of the numpy array corresponds to index of the parameter with respect to
            which the integral is derivatived.
            Third index of the numpy array corresponds to the first order excitations of the given
            Slater determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the
            ordering of the excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number
            of first order excitations of the given Slater determinants.

        """
        # pylint: disable=R0915
        nspatial = self.nspatial
        all_beta = np.arange(nspatial)

        occ_beta_array_indices = np.arange(occ_beta.size)
        vir_beta_array_indices = np.arange(vir_beta.size)

        shared_beta = slater.shared_indices_remove_one_index(occ_beta)

        sign_b = slater.sign_excite_one(occ_beta, vir_beta)

        # NOTE: here, we use the following convention for indices:
        # the first index corresponds to the row index of the antihermitian matrix for orbital
        # rotation
        # the second index corresponds to the column index of the antihermitian matrix for orbital
        # rotation
        # the third index corresponds to the occupied orbital that will be annihilated in the
        # excitation
        # the fourth index corresponds to the occupied orbital that will be created in the
        # excitation
        # FIXME: hardcoded parameter shape
        one_electron_b = np.zeros((nspatial, nspatial, occ_beta.size, vir_beta.size))
        coulomb_b = np.zeros((nspatial, nspatial, occ_beta.size, vir_beta.size))
        exchange_b = np.zeros((nspatial, nspatial, occ_beta.size, vir_beta.size))

        # ONE-ELECTRON INTEGRALS
        # x == a
        one_electron_b[
            occ_beta[:, None, None],  # x
            all_beta[None, :, None],  # y
            occ_beta_array_indices[:, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] -= self.one_int[
            all_beta[None, :, None], vir_beta[None, None, :]  # y, b
        ]
        # x == b
        one_electron_b[
            vir_beta[None, None, :],  # x
            all_beta[None, :, None],  # y
            occ_beta_array_indices[:, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] -= self.one_int[
            occ_beta[:, None, None], all_beta[None, :, None]  # a, y
        ]
        # y == a
        one_electron_b[
            all_beta[:, None, None],  # x
            occ_beta[None, :, None],  # y
            occ_beta_array_indices[None, :, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] += self.one_int[
            all_beta[:, None, None], vir_beta[None, None, :]  # x, b
        ]
        # y == b
        one_electron_b[
            all_beta[:, None, None],  # x
            vir_beta[None, None, :],  # y
            occ_beta_array_indices[None, :, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] += self.one_int[
            occ_beta[None, :, None], all_beta[:, None, None]  # a, x
        ]

        # COULOMB INTEGRALS
        # x == a
        coulomb_b[
            occ_beta[:, None, None],  # x
            all_beta[None, :, None],  # y
            occ_beta_array_indices[:, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                all_beta[None, None, :, None],  # y
                shared_beta[:, :, None, None],  # shared beta
                vir_beta[None, None, None, :],  # b
                shared_beta[:, :, None, None],  # shared beta
            ],
            axis=1,
        )
        coulomb_b[
            occ_beta[:, None, None],  # x
            all_beta[None, :, None],  # y
            occ_beta_array_indices[:, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                occ_alpha[None, :, None, None],  # shared alpha
                all_beta[None, None, :, None],  # y
                occ_alpha[None, :, None, None],  # shared alpha
                vir_beta[None, None, None, :],  # b
            ],
            axis=1,
        )
        # x == b
        coulomb_b[
            vir_beta[None, None, :],  # x
            all_beta[None, :, None],  # y
            occ_beta_array_indices[:, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                occ_beta[:, None, None, None],  # a
                shared_beta[:, :, None, None],  # shared beta
                all_beta[None, None, :, None],  # y
                shared_beta[:, :, None, None],  # shared beta
            ],
            axis=1,
        )
        coulomb_b[
            vir_beta[None, None, :],  # x
            all_beta[None, :, None],  # y
            occ_beta_array_indices[:, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                occ_alpha[None, :, None, None],  # shared alpha
                occ_beta[:, None, None, None],  # a
                occ_alpha[None, :, None, None],  # shared alpha
                all_beta[None, None, :, None],  # y
            ],
            axis=1,
        )
        # x in shared
        coulomb_b[
            shared_beta[:, :, None, None],  # x
            all_beta[None, None, :, None],  # y
            occ_beta_array_indices[:, None, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            shared_beta[:, :, None, None],  # x
            occ_beta[:, None, None, None],  # a (occupied index)
            all_beta[None, None, :, None],  # y
            vir_beta[None, None, None, :],  # b (virtual index)
        ]
        coulomb_b[
            shared_beta[:, :, None, None],  # x
            all_beta[None, None, :, None],  # y
            occ_beta_array_indices[:, None, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            shared_beta[:, :, None, None],  # x
            vir_beta[None, None, None, :],  # b (virtual index)
            all_beta[None, None, :, None],  # y
            occ_beta[:, None, None, None],  # a (occupied index)
        ]
        #
        # y == a
        coulomb_b[
            all_beta[:, None, None],  # x
            occ_beta[None, :, None],  # y
            occ_beta_array_indices[None, :, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                all_beta[:, None, None, None],  # x
                shared_beta.T[None, :, :, None],  # shared beta
                vir_beta[None, None, None, :],  # b
                shared_beta.T[None, :, :, None],  # shared beta
            ],
            axis=1,
        )
        coulomb_b[
            all_beta[:, None, None],  # x
            occ_beta[None, :, None],  # y
            occ_beta_array_indices[None, :, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                occ_alpha[None, :, None, None],  # shared alpha
                all_beta[:, None, None, None],  # x
                occ_alpha[None, :, None, None],  # shared alpha
                vir_beta[None, None, None, :],  # b
            ],
            axis=1,
        )
        # y == b
        coulomb_b[
            all_beta[:, None, None],  # x
            vir_beta[None, None, :],  # y
            occ_beta_array_indices[None, :, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                occ_beta[None, None, :, None],  # a
                shared_beta.T[None, :, :, None],  # shared beta
                all_beta[:, None, None, None],  # x
                shared_beta.T[None, :, :, None],  # shared beta
            ],
            axis=1,
        )
        coulomb_b[
            all_beta[:, None, None],  # x
            vir_beta[None, None, :],  # y
            occ_beta_array_indices[None, :, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                occ_alpha[None, :, None, None],  # shared alpha
                occ_beta[None, None, :, None],  # a
                occ_alpha[None, :, None, None],  # shared alpha
                all_beta[:, None, None, None],  # x
            ],
            axis=1,
        )
        # y in shared
        coulomb_b[
            all_beta[None, None, :, None],  # x
            shared_beta[:, :, None, None],  # y
            occ_beta_array_indices[:, None, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            all_beta[None, None, :, None],  # x
            occ_beta[:, None, None, None],  # a (occupied index)
            shared_beta[:, :, None, None],  # y
            vir_beta[None, None, None, :],  # b (virtual index)
        ]
        coulomb_b[
            all_beta[None, None, :, None],  # x
            shared_beta[:, :, None, None],  # y
            occ_beta_array_indices[:, None, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            all_beta[None, None, :, None],  # x
            vir_beta[None, None, None, :],  # b (virtual index)
            shared_beta[:, :, None, None],  # y
            occ_beta[:, None, None, None],  # a (occupied index)
        ]
        #

        # EXCHANGE INTEGRALS
        # x == a
        exchange_b[
            occ_beta[:, None, None],  # x
            all_beta[None, :, None],  # y
            occ_beta_array_indices[:, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                all_beta[None, None, :, None],  # y
                shared_beta[:, :, None, None],  # shared beta
                shared_beta[:, :, None, None],  # shared beta
                vir_beta[None, None, None, :],  # b
            ],
            axis=1,
        )
        # x == b
        exchange_b[
            vir_beta[None, None, :],  # x
            all_beta[None, :, None],  # y
            occ_beta_array_indices[:, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] += np.sum(
            self.two_int[
                occ_beta[:, None, None, None],  # a
                shared_beta[:, :, None, None],  # shared beta
                shared_beta[:, :, None, None],  # shared beta
                all_beta[None, None, :, None],  # y
            ],
            axis=1,
        )
        # x in shared
        exchange_b[
            shared_beta[:, :, None, None],  # x
            all_beta[None, None, :, None],  # y
            occ_beta_array_indices[:, None, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            shared_beta[:, :, None, None],  # x
            occ_beta[:, None, None, None],  # a (occupied index)
            vir_beta[None, None, None, :],  # b (virtual index)
            all_beta[None, None, :, None],  # y
        ]
        exchange_b[
            shared_beta[:, :, None, None],  # x
            all_beta[None, None, :, None],  # y
            occ_beta_array_indices[:, None, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, None, :],  # b (virtual index)
        ] += self.two_int[
            shared_beta[:, :, None, None],  # x
            vir_beta[None, None, None, :],  # b (virtual index)
            occ_beta[:, None, None, None],  # a (occupied index)
            all_beta[None, None, :, None],  # y
        ]
        # y == a
        exchange_b[
            all_beta[:, None, None],  # x
            occ_beta[None, :, None],  # y
            occ_beta_array_indices[None, :, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                all_beta[:, None, None, None],  # x
                shared_beta.T[None, :, :, None],  # shared beta
                shared_beta.T[None, :, :, None],  # shared beta
                vir_beta[None, None, None, :],  # b
            ],
            axis=1,
        )
        # y == b
        exchange_b[
            all_beta[:, None, None],  # x
            vir_beta[None, None, :],  # y
            occ_beta_array_indices[None, :, None],  # a (occupied index)
            vir_beta_array_indices[None, None, :],  # b (virtual index)
        ] -= np.sum(
            self.two_int[
                occ_beta[None, None, :, None],  # a
                shared_beta.T[None, :, :, None],  # shared beta
                shared_beta.T[None, :, :, None],  # shared beta
                all_beta[:, None, None, None],  # x
            ],
            axis=1,
        )
        # y in shared
        exchange_b[
            all_beta[None, None, :, None],  # x
            shared_beta[:, :, None, None],  # y
            occ_beta_array_indices[:, None, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            all_beta[None, None, :, None],  # x
            occ_beta[:, None, None, None],  # a (occupied index)
            vir_beta[None, None, None, :],  # b (virtual index)
            shared_beta[:, :, None, None],  # y
        ]
        exchange_b[
            all_beta[None, None, :, None],  # x
            shared_beta[:, :, None, None],  # y
            occ_beta_array_indices[:, None, None, None],  # a (occupied index)
            vir_beta_array_indices[None, None, None, :],  # b (virtual index)
        ] -= self.two_int[
            all_beta[None, None, :, None],  # x
            vir_beta[None, None, None, :],  # b (virtual index)
            occ_beta[:, None, None, None],  # a (occupied index)
            shared_beta[:, :, None, None],  # y
        ]

        triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
        return sign_b[None, None, :] * np.array(
            [
                one_electron_b[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
                coulomb_b[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
                exchange_b[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
            ]
        )

    def _integrate_sd_sds_deriv_two_aaa(self, occ_alpha, occ_beta, vir_alpha):  # pylint: disable=W0613
        """Return (alpha) derivatives of integrals of an SD and its (alpha) double excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_alpha : np.ndarray(K-N_a,)
            Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(2, N_params, M)
            Derivatives of the coulomb and exchange integrals (with respect to parameters associated
            with alpha orbitals) of the given Slater determinant with its second order excitations
            involving only alpha orbitals.
            First index of the numpy array corresponds to the one-electron (first element), coulomb
            (second elements), and exchange (third element) integrals.
            Second index corresponds to index of the parameter with respect to which the integral is
            derivatived.
            Third index corresponds to the second order excitations of the given Slater
            determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
            the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
            3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
            the number of first order excitations of the given Slater determinants.

        """
        # pylint: disable=R0915,C0103
        nspatial = self.nspatial
        all_alpha = np.arange(nspatial)

        annihilators = np.array(list(it.combinations(occ_alpha, 2)))
        a = annihilators[:, 0]
        b = annihilators[:, 1]
        creators = np.array(list(it.combinations(vir_alpha, 2)))
        c = creators[:, 0]
        d = creators[:, 1]
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
        coulomb_aa = np.zeros((nspatial, nspatial, a.size, c.size))
        exchange_aa = np.zeros((nspatial, nspatial, a.size, c.size))

        sign_aa = slater.sign_excite_two(occ_alpha, vir_alpha)

        # x == a
        coulomb_aa[
            a[:, None, None],  # x
            all_alpha[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            all_alpha[None, :, None],  # y
            b[:, None, None],  # b
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        exchange_aa[
            a[:, None, None],  # x
            all_alpha[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            all_alpha[None, :, None],  # y
            b[:, None, None],  # b
            d[None, None, :],  # d
            c[None, None, :],  # c
        ]
        # x == b
        coulomb_aa[
            b[:, None, None],  # x
            all_alpha[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            all_alpha[None, :, None],  # y
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        exchange_aa[
            b[:, None, None],  # x
            all_alpha[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            all_alpha[None, :, None],  # y
            d[None, None, :],  # d
            c[None, None, :],  # c
        ]
        # x == c
        coulomb_aa[
            c[None, None, :],  # x
            all_alpha[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_alpha[None, :, None],  # y
            d[None, None, :],  # d
        ]
        exchange_aa[
            c[None, None, :],  # x
            all_alpha[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            d[None, None, :],  # d
            all_alpha[None, :, None],  # y
        ]
        # x == d
        coulomb_aa[
            d[None, None, :],  # x
            all_alpha[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            c[None, None, :],  # c
            all_alpha[None, :, None],  # y
        ]
        exchange_aa[
            d[None, None, :],  # x
            all_alpha[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_alpha[None, :, None],  # y
            c[None, None, :],  # c
        ]

        # y == a
        coulomb_aa[
            all_alpha[None, :, None],  # x
            a[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            all_alpha[None, :, None],  # x
            b[:, None, None],  # b
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        exchange_aa[
            all_alpha[None, :, None],  # x
            a[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            all_alpha[None, :, None],  # x
            b[:, None, None],  # b
            d[None, None, :],  # d
            c[None, None, :],  # c
        ]
        # y == b
        coulomb_aa[
            all_alpha[None, :, None],  # x
            b[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            all_alpha[None, :, None],  # x
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        exchange_aa[
            all_alpha[None, :, None],  # x
            b[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            all_alpha[None, :, None],  # x
            d[None, None, :],  # d
            c[None, None, :],  # c
        ]
        # y == c
        coulomb_aa[
            all_alpha[None, :, None],  # x
            c[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_alpha[None, :, None],  # x
            d[None, None, :],  # d
        ]
        exchange_aa[
            all_alpha[None, :, None],  # x
            c[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            d[None, None, :],  # d
            all_alpha[None, :, None],  # x
        ]
        # y == d
        coulomb_aa[
            all_alpha[None, :, None],  # x
            d[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            c[None, None, :],  # c
            all_alpha[None, :, None],  # x
        ]
        exchange_aa[
            all_alpha[None, :, None],  # x
            d[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_alpha[None, :, None],  # x
            c[None, None, :],  # c
        ]

        triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
        return sign_aa[None, None, :] * np.array(
            [
                coulomb_aa[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
                exchange_aa[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
            ]
        )

    def _integrate_sd_sds_deriv_two_aab(self, occ_alpha, occ_beta, vir_alpha, vir_beta):
        """Return (alpha) derivatives of integrals of an SD and its (alpha beta) double excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_alpha : np.ndarray(K-N_a,)
            Indices of the alpha spin orbitals that are not occupied in the Slater determinant.
        vir_beta : np.ndarray(K-N_b,)
            Indices of the beta spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(N_params, M)
            Derivatives of the coulomb integrals (with respect to parameters associated with alpha
            orbitals) of the given Slater determinant with its second order excitations involving
            alpha and beta orbitals.
            First index corresponds to index of the parameter with respect to which the integral is
            derivatived.
            Second index corresponds to the second order excitations of the given Slater
            determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
            the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
            3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
            the number of first order excitations of the given Slater determinants.

        """
        # pylint: disable=R0915,C0103
        nspatial = self.nspatial
        all_alpha = np.arange(nspatial)

        annihilators = np.array(list(it.product(occ_alpha, occ_beta)))
        a = annihilators[:, 0]
        b = annihilators[:, 1]
        creators = np.array(list(it.product(vir_alpha, vir_beta)))
        c = creators[:, 0]
        d = creators[:, 1]
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
        coulomb_ab = np.zeros((nspatial, nspatial, a.size, c.size))

        sign_ab = slater.sign_excite_two_ab(occ_alpha, occ_beta, vir_alpha, vir_beta)

        # x == a
        coulomb_ab[
            a[:, None, None],  # x
            all_alpha[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            all_alpha[None, :, None],  # y
            b[:, None, None],  # b
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        # x == c
        coulomb_ab[
            c[None, None, :],  # x
            all_alpha[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_alpha[None, :, None],  # y
            d[None, None, :],  # d
        ]

        # y == a
        coulomb_ab[
            all_alpha[None, :, None],  # x
            a[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            all_alpha[None, :, None],  # x
            b[:, None, None],  # b
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        # y == c
        coulomb_ab[
            all_alpha[None, :, None],  # x
            c[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_alpha[None, :, None],  # x
            d[None, None, :],  # d
        ]

        triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
        return sign_ab[None, :] * coulomb_ab[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1)

    def _integrate_sd_sds_deriv_two_bab(self, occ_alpha, occ_beta, vir_alpha, vir_beta):
        """Return (beta) derivatives of integrals of an SD and its (alpha, beta) double excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_alpha : np.ndarray(K-N_a,)
            Indices of the alpha spin orbitals that are not occupied in the Slater determinant.
        vir_beta : np.ndarray(K-N_b,)
            Indices of the beta spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(N_params, M)
            Derivatives of the coulomb integrals (with respect to parameters associated with beta
            orbitals) of the given Slater determinant with its second order excitations involving
            alpha and beta orbitals.
            First index corresponds to index of the parameter with respect to which the integral is
            derivatived.
            Second index corresponds to the second order excitations of the given Slater
            determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
            the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
            3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
            the number of first order excitations of the given Slater determinants.

        """
        # pylint: disable=R0915,C0103
        nspatial = self.nspatial
        all_beta = np.arange(nspatial)

        annihilators = np.array(list(it.product(occ_alpha, occ_beta)))
        a = annihilators[:, 0]
        b = annihilators[:, 1]
        creators = np.array(list(it.product(vir_alpha, vir_beta)))
        c = creators[:, 0]
        d = creators[:, 1]
        occ_array_indices = np.arange(a.size)
        vir_array_indices = np.arange(c.size)

        # NOTE: here, we use the following convention for indices:
        # the first index corresponds to the row index of the antihermitian matrix for orbital
        # rotation
        # the second index corresponds to the column index of the antihermitian matrix for orbital
        # rotation
        # the third index corresponds to the occupied orbital that will be annihilated in the
        # excitation
        # the fourth index corresponds to the virtual orbital that will be created in the excitation
        coulomb_ba = np.zeros((nspatial, nspatial, b.size, d.size))

        sign_ab = slater.sign_excite_two_ab(occ_alpha, occ_beta, vir_alpha, vir_beta)

        # x == b
        coulomb_ba[
            b[:, None, None],  # x
            all_beta[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            all_beta[None, :, None],  # y
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        # x == d
        coulomb_ba[
            d[None, None, :],  # x
            all_beta[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            c[None, None, :],  # c
            all_beta[None, :, None],  # y
        ]

        # y == b
        coulomb_ba[
            all_beta[None, :, None],  # x
            b[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            all_beta[None, :, None],  # x
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        # y == d
        coulomb_ba[
            all_beta[None, :, None],  # x
            d[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            c[None, None, :],  # c
            all_beta[None, :, None],  # x
        ]

        triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
        return sign_ab[None, :] * coulomb_ba[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1)

    def _integrate_sd_sds_deriv_two_bbb(self, occ_alpha, occ_beta, vir_beta):  # pylint: disable=W0613
        """Return (beta) derivatives of integrals of an SD and its (beta) double excitations.

        Paramters
        ---------
        occ_alpha : np.ndarray(N_a,)
            Indices of the alpha spin orbitals that are occupied in the Slater determinant.
        occ_beta : np.ndarray(N_b,)
            Indices of the beta spin orbitals that are occupied in the Slater determinant.
        vir_beta : np.ndarray(K-N_b,)
            Indices of the beta spin orbitals that are not occupied in the Slater determinant.

        Returns
        -------
        integrals : np.ndarray(2, N_params, M)
            Derivatives of the coulomb and exchange integrals (with respect to parameters associated
            with beta orbitals) of the given Slater determinant with its second order excitations
            involving only beta orbitals.
            First index of the numpy array corresponds to the one-electron (first element), coulomb
            (second elements), and exchange (third element) integrals.
            Second index corresponds to index of the parameter with respect to which the integral is
            derivatived.
            Third index corresponds to the second order excitations of the given Slater
            determinant. The excitations are ordered by the occupied orbital then the virtual
            orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
            the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
            3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
            the number of first order excitations of the given Slater determinants.

        """
        # pylint: disable=R0915,C0103
        nspatial = self.nspatial
        all_beta = np.arange(nspatial)

        annihilators = np.array(list(it.combinations(occ_beta, 2)))
        a = annihilators[:, 0]
        b = annihilators[:, 1]
        creators = np.array(list(it.combinations(vir_beta, 2)))
        c = creators[:, 0]
        d = creators[:, 1]
        occ_array_indices = np.arange(a.size)
        vir_array_indices = np.arange(c.size)

        sign_bb = slater.sign_excite_two(occ_beta, vir_beta)

        # NOTE: here, we use the following convention for indices:
        # the first index corresponds to the row index of the antihermitian matrix for orbital
        # rotation of the beta orbitals
        # the second index corresponds to the column index of the antihermitian matrix for orbital
        # rotation of the beta orbitals
        # the third index corresponds to the occupied orbitals that will be annihilated in the
        # excitation
        # the fourth index corresponds to the virtual orbitals that will be created in the
        # excitation
        coulomb_bb = np.zeros((nspatial, nspatial, a.size, c.size))
        exchange_bb = np.zeros((nspatial, nspatial, a.size, c.size))

        # x == a
        coulomb_bb[
            a[:, None, None],  # x
            all_beta[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            all_beta[None, :, None],  # y
            b[:, None, None],  # b
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        exchange_bb[
            a[:, None, None],  # x
            all_beta[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            all_beta[None, :, None],  # y
            b[:, None, None],  # b
            d[None, None, :],  # d
            c[None, None, :],  # c
        ]
        # x == b
        coulomb_bb[
            b[:, None, None],  # x
            all_beta[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            all_beta[None, :, None],  # y
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        exchange_bb[
            b[:, None, None],  # x
            all_beta[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            all_beta[None, :, None],  # y
            d[None, None, :],  # d
            c[None, None, :],  # c
        ]
        # x == c
        coulomb_bb[
            c[None, None, :],  # x
            all_beta[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_beta[None, :, None],  # y
            d[None, None, :],  # d
        ]
        exchange_bb[
            c[None, None, :],  # x
            all_beta[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            d[None, None, :],  # d
            all_beta[None, :, None],  # y
        ]
        # x == d
        coulomb_bb[
            d[None, None, :],  # x
            all_beta[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            c[None, None, :],  # c
            all_beta[None, :, None],  # y
        ]
        exchange_bb[
            d[None, None, :],  # x
            all_beta[None, :, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_beta[None, :, None],  # y
            c[None, None, :],  # c
        ]

        # y == a
        coulomb_bb[
            all_beta[None, :, None],  # x
            a[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            all_beta[None, :, None],  # x
            b[:, None, None],  # b
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        exchange_bb[
            all_beta[None, :, None],  # x
            a[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            all_beta[None, :, None],  # x
            b[:, None, None],  # b
            d[None, None, :],  # d
            c[None, None, :],  # c
        ]
        # y == b
        coulomb_bb[
            all_beta[None, :, None],  # x
            b[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            all_beta[None, :, None],  # x
            c[None, None, :],  # c
            d[None, None, :],  # d
        ]
        exchange_bb[
            all_beta[None, :, None],  # x
            b[:, None, None],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            all_beta[None, :, None],  # x
            d[None, None, :],  # d
            c[None, None, :],  # c
        ]
        # y == c
        coulomb_bb[
            all_beta[None, :, None],  # x
            c[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_beta[None, :, None],  # x
            d[None, None, :],  # d
        ]
        exchange_bb[
            all_beta[None, :, None],  # x
            c[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            d[None, None, :],  # d
            all_beta[None, :, None],  # x
        ]
        # y == d
        coulomb_bb[
            all_beta[None, :, None],  # x
            d[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] += self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            c[None, None, :],  # c
            all_beta[None, :, None],  # x
        ]
        exchange_bb[
            all_beta[None, :, None],  # x
            d[None, None, :],  # y
            occ_array_indices[:, None, None],  # a, b (occupied index)
            vir_array_indices[None, None, :],  # c, d (virtual index)
        ] -= self.two_int[
            a[:, None, None],  # a
            b[:, None, None],  # b
            all_beta[None, :, None],  # x
            c[None, None, :],  # c
        ]

        triu_rows, triu_cols = np.triu_indices(nspatial, k=1)
        return sign_bb[None, None, :] * np.array(
            [
                coulomb_bb[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
                exchange_bb[triu_rows, triu_cols, :, :].reshape(triu_rows.size, -1),
            ]
        )

    def integrate_sd_wfn(
        self, sd, wfn, wfn_deriv=None, ham_deriv=None, components=False
    ):  # pylint: disable=R0912,R0915
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
        # pylint: disable=C0103
        if __debug__:
            if not slater.is_sd_compatible(sd):
                raise TypeError("Slater determinant must be given as an integer.")
            if wfn_deriv is not None and ham_deriv is not None:
                raise ValueError(
                    "Integral can be derivatized with respect to the wavefunction or Hamiltonian "
                    "parameters, but not both."
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
        if self.nspin > 64:
            sd = int(sd)
        nspatial = self.nspatial
        occ_indices = np.array(slater.occ_indices(sd))
        vir_indices = np.array(slater.vir_indices(sd, self.nspin))
        # FIXME: hardcode slater determinant structure
        occ_alpha = occ_indices[occ_indices < nspatial]
        vir_alpha = vir_indices[vir_indices < nspatial]
        occ_beta = occ_indices[occ_indices >= nspatial]
        vir_beta = vir_indices[vir_indices >= nspatial]

        if self.nspin <= 64:

            def fromiter(iterator, dtype, ydim, count):
                return np.fromiter(it.chain.from_iterable(iterator), dtype, count=int(count)).reshape(-1, ydim)

            occ_one_alpha = fromiter(it.combinations(occ_alpha.tolist(), 1), int, 1, count=len(occ_alpha))
            occ_one_alpha = np.left_shift(1, occ_one_alpha[:, 0])

            occ_one_beta = fromiter(it.combinations(occ_beta.tolist(), 1), int, 1, count=len(occ_beta))
            occ_one_beta = np.left_shift(1, occ_one_beta[:, 0])

            vir_one_alpha = fromiter(it.combinations(vir_alpha.tolist(), 1), int, 1, count=len(vir_alpha))
            vir_one_alpha = np.left_shift(1, vir_one_alpha[:, 0])

            vir_one_beta = fromiter(it.combinations(vir_beta.tolist(), 1), int, 1, count=len(vir_beta))
            vir_one_beta = np.left_shift(1, vir_one_beta[:, 0])

            occ_two_aa = fromiter(
                it.combinations(occ_alpha.tolist(), 2), int, ydim=2, count=2 * comb(len(occ_alpha), 2)
            )
            occ_two_aa = np.left_shift(1, occ_two_aa)
            occ_two_aa = np.bitwise_or(occ_two_aa[:, 0], occ_two_aa[:, 1])

            occ_two_ab = fromiter(
                it.product(occ_alpha.tolist(), occ_beta.tolist()), int, 2, count=2 * len(occ_alpha) * len(occ_beta)
            )
            occ_two_ab = np.left_shift(1, occ_two_ab)
            occ_two_ab = np.bitwise_or(occ_two_ab[:, 0], occ_two_ab[:, 1])

            occ_two_bb = fromiter(it.combinations(occ_beta.tolist(), 2), int, 2, count=2 * comb(len(occ_beta), 2))
            occ_two_bb = np.left_shift(1, occ_two_bb)
            occ_two_bb = np.bitwise_or(occ_two_bb[:, 0], occ_two_bb[:, 1])

            vir_two_aa = fromiter(it.combinations(vir_alpha.tolist(), 2), int, 2, count=2 * comb(len(vir_alpha), 2))
            vir_two_aa = np.left_shift(1, vir_two_aa)
            vir_two_aa = np.bitwise_or(vir_two_aa[:, 0], vir_two_aa[:, 1])

            vir_two_ab = fromiter(
                it.product(vir_alpha.tolist(), vir_beta.tolist()), int, 2, count=2 * len(vir_alpha) * len(vir_beta)
            )
            vir_two_ab = np.left_shift(1, vir_two_ab)
            vir_two_ab = np.bitwise_or(vir_two_ab[:, 0], vir_two_ab[:, 1])

            vir_two_bb = fromiter(it.combinations(vir_beta.tolist(), 2), int, 2, count=2 * comb(len(vir_beta), 2))
            vir_two_bb = np.left_shift(1, vir_two_bb)
            vir_two_bb = np.bitwise_or(vir_two_bb[:, 0], vir_two_bb[:, 1])

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

        if self.nspin <= 64:
            overlaps_zero = np.array([[wfn.get_overlap(sd, deriv=wfn_deriv)]]).reshape(*shape)
            overlaps_one_alpha = np.array(
                [
                    wfn.get_overlap(sd_exc, deriv=wfn_deriv)
                    for sd_exc in np.ravel(
                        np.bitwise_or(np.bitwise_xor(sd, occ_one_alpha)[:, None], vir_one_alpha[None, :])
                    )
                ]
            ).reshape(*shape)
            overlaps_one_beta = np.array(
                [
                    wfn.get_overlap(sd_exc, deriv=wfn_deriv)
                    for sd_exc in np.ravel(
                        np.bitwise_or(np.bitwise_xor(sd, occ_one_beta)[:, None], vir_one_beta[None, :])
                    )
                ]
            ).reshape(*shape)
            overlaps_two_aa = np.array(
                [
                    wfn.get_overlap(sd_exc, deriv=wfn_deriv)
                    for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_aa)[:, None], vir_two_aa[None, :]))
                ]
            ).reshape(*shape)
            overlaps_two_ab = np.array(
                [
                    wfn.get_overlap(sd_exc, deriv=wfn_deriv)
                    for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_ab)[:, None], vir_two_ab[None, :]))
                ]
            ).reshape(*shape)
            overlaps_two_bb = np.array(
                [
                    wfn.get_overlap(sd_exc, deriv=wfn_deriv)
                    for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_bb)[:, None], vir_two_bb[None, :]))
                ]
            ).reshape(*shape)
        else:
            overlaps_zero = np.array([[wfn.get_overlap(sd, deriv=wfn_deriv)]]).reshape(*shape)
            overlaps_one_alpha = np.array(
                [
                    wfn.get_overlap(slater.excite(sd, *occ, *vir), deriv=wfn_deriv)
                    for occ in it.combinations(occ_alpha.tolist(), 1)
                    for vir in it.combinations(vir_alpha.tolist(), 1)
                ]
            ).reshape(*shape)

            overlaps_one_beta = np.array(
                [
                    wfn.get_overlap(slater.excite(sd, *occ, *vir), deriv=wfn_deriv)
                    for occ in it.combinations(occ_beta.tolist(), 1)
                    for vir in it.combinations(vir_beta.tolist(), 1)
                ]
            ).reshape(*shape)

            overlaps_two_aa = np.array(
                [
                    wfn.get_overlap(slater.excite(sd, *occ, *vir), deriv=wfn_deriv)
                    for occ in it.combinations(occ_alpha.tolist(), 2)
                    for vir in it.combinations(vir_alpha.tolist(), 2)
                ]
            ).reshape(*shape)
            overlaps_two_ab = np.array(
                [
                    wfn.get_overlap(slater.excite(sd, *occ, *vir), deriv=wfn_deriv)
                    for occ in it.product(occ_alpha.tolist(), occ_beta.tolist())
                    for vir in it.product(vir_alpha.tolist(), vir_beta.tolist())
                ]
            ).reshape(*shape)
            overlaps_two_bb = np.array(
                [
                    wfn.get_overlap(slater.excite(sd, *occ, *vir), deriv=wfn_deriv)
                    for occ in it.combinations(occ_beta.tolist(), 2)
                    for vir in it.combinations(vir_beta.tolist(), 2)
                ]
            ).reshape(*shape)

        # FIXME: hardcode slater determinant structure
        occ_beta -= nspatial
        vir_beta -= nspatial

        if ham_deriv is not None:
            integrals_zero = self._integrate_sd_sds_deriv_zero_alpha(occ_alpha, occ_beta, vir_alpha)
            integrals_zero += self._integrate_sd_sds_deriv_zero_beta(occ_alpha, occ_beta, vir_beta)
            output += np.squeeze(integrals_zero * overlaps_zero, axis=2)[:, ham_deriv]
        else:
            integrals_zero = self._integrate_sd_sds_zero(occ_alpha, occ_beta)
            if wfn_deriv is not None:
                integrals_zero = np.expand_dims(integrals_zero, 2)
            output += np.sum(integrals_zero * overlaps_zero, axis=1)

        if ham_deriv is not None:
            integrals_one_aa = self._integrate_sd_sds_deriv_one_aa(occ_alpha, occ_beta, vir_alpha)
            output += np.sum(integrals_one_aa * overlaps_one_alpha, axis=2)[:, ham_deriv]
            integrals_one_ba = self._integrate_sd_sds_deriv_one_ba(occ_alpha, occ_beta, vir_alpha)
            output[1, :] += np.sum(integrals_one_ba * overlaps_one_alpha, axis=1)[ham_deriv]
        else:
            integrals_one_alpha = self._integrate_sd_sds_one_alpha(occ_alpha, occ_beta, vir_alpha)
            if wfn_deriv is not None:
                integrals_one_alpha = np.expand_dims(integrals_one_alpha, 2)
            output += np.sum(integrals_one_alpha * overlaps_one_alpha, axis=1)

        # FIXME: hardcode slater determinant structure
        if ham_deriv is not None:
            integrals_one_bb = self._integrate_sd_sds_deriv_one_bb(occ_alpha, occ_beta, vir_beta)
            output += np.sum(integrals_one_bb * overlaps_one_beta, axis=2)[:, ham_deriv]
            integrals_one_ab = self._integrate_sd_sds_deriv_one_ab(occ_alpha, occ_beta, vir_beta)
            output[1, :] += np.sum(integrals_one_ab * overlaps_one_beta, axis=1)[ham_deriv]
        else:
            integrals_one_beta = self._integrate_sd_sds_one_beta(occ_alpha, occ_beta, vir_beta)
            if wfn_deriv is not None:
                integrals_one_beta = np.expand_dims(integrals_one_beta, 2)
            output += np.sum(integrals_one_beta * overlaps_one_beta, axis=1)

        if occ_alpha.size > 1 and vir_alpha.size > 1:
            if ham_deriv is not None:
                integrals_two_aaa = self._integrate_sd_sds_deriv_two_aaa(occ_alpha, occ_beta, vir_alpha)
                output[1:, :] += np.sum(integrals_two_aaa * overlaps_two_aa, axis=2)[:, ham_deriv]
            else:
                integrals_two_aa = self._integrate_sd_sds_two_aa(occ_alpha, occ_beta, vir_alpha)
                if wfn_deriv is not None:
                    integrals_two_aa = np.expand_dims(integrals_two_aa, 2)
                output[1:] += np.sum(integrals_two_aa * overlaps_two_aa, axis=1)

        # FIXME: hardcode slater determinant structure
        if occ_alpha.size > 0 and occ_beta.size > 0 and vir_alpha.size > 0 and vir_beta.size > 0:
            if ham_deriv is not None:
                integrals_two_aab = self._integrate_sd_sds_deriv_two_aab(occ_alpha, occ_beta, vir_alpha, vir_beta)
                output[1, :] += np.sum(integrals_two_aab * overlaps_two_ab, axis=1)[ham_deriv]
                integrals_two_bab = self._integrate_sd_sds_deriv_two_bab(occ_alpha, occ_beta, vir_alpha, vir_beta)
                output[1, :] += np.sum(integrals_two_bab * overlaps_two_ab, axis=1)[ham_deriv]
            else:
                integrals_two_ab = self._integrate_sd_sds_two_ab(occ_alpha, occ_beta, vir_alpha, vir_beta)
                if wfn_deriv is not None:
                    integrals_two_ab = np.expand_dims(integrals_two_ab, 1)
                output[1] += np.sum(integrals_two_ab * overlaps_two_ab, axis=0)

        # FIXME: hardcode slater determinant structure
        if occ_beta.size > 1 and vir_beta.size > 1:
            if ham_deriv is not None:
                integrals_two_bbb = self._integrate_sd_sds_deriv_two_bbb(occ_alpha, occ_beta, vir_beta)
                output[1:, :] += np.sum(integrals_two_bbb * overlaps_two_bb, axis=2)[:, ham_deriv]
            else:
                integrals_two_bb = self._integrate_sd_sds_two_bb(occ_alpha, occ_beta, vir_beta)
                if wfn_deriv is not None:
                    integrals_two_bb = np.expand_dims(integrals_two_bb, 2)
                output[1:] += np.sum(integrals_two_bb * overlaps_two_bb, axis=1)

        if components:
            return output

        return np.sum(output, axis=0)
