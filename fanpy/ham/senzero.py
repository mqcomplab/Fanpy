"""Seniority-zero Hamiltonian object that interacts with the wavefunction."""

from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.tools import slater

import numpy as np


class SeniorityZeroHamiltonian(RestrictedMolecularHamiltonian):
    r"""Hamiltonian that involves only the zero-seniority terms.

    # FIXME: fix up eqns
    .. math::

        \def\i{\imath}
        \def\j{\jmath}
        \hat{H} =
        \sum_{i} \left(
            h_{ii} a^\dagger_i a_i + h_{\bar{\i}\bar{\i}} a^\dagger_{\bar{\i}} a_{\bar{\i}}
        \right)
        + \sum_{i<j} \left(
            g_{ijij} a^\dagger_i a^\dagger_j a_j a_i
            + g_{i\bar{\j}i\bar{\j}} a^\dagger_i a^\dagger_{\bar{\j}} a_{\bar{\j}} a_i
            + g_{\bar{\i}j\bar{\i}j} a^\dagger_{\bar{\i}} a^\dagger_j  a_j a_{\bar{\i}}
            + g_{\bar{\i}\bar{\j}\bar{\i}\bar{\j}}
            a^\dagger_{\bar{\i}} a^\dagger_{\bar{\j}} a_{\bar{\j}} a_{\bar{\i}}
            + g_{i\bar{\j}\bar{\j}i} a^\dagger_i a^\dagger_{\bar{\j}} a_{\bar{\j} a_i}
        \right)

    where :math:`i` and :math:`\bar{\i}` are the indices that correspond to the ith alpha- and beta-
    spin orbtials, :math:`h_{ik}` is the one-electron integral and :math:`g_{ijkl}` is the two-
    electron integral.

    Seniority zero means that there are no unpaired electrons.

    Attributes
    ----------
    params : np.ndarray
        Significant elements of the anti-Hermitian matrix.
    one_int : {1- or 2-tuple np.ndarray(K, K)}
        One electron integrals for restricted, unrestricted, or generalized orbitals.
        1-tuple for spatial (restricted) and generalized orbitals.
        2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components).
    two_int : {1- or 3-tuple np.ndarray(K, K)}
        Two electron integrals for restricted, unrestricted, or generalized orbitals.
        Uses the physicist's notation.
        1-tuple for spatial (restricted) and generalized orbitals.
        3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
        beta-beta-beta-beta components).

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
    assign_orbtype(self, orbtype=None)
        Assign the orbital type.
    assign_integrals(self, one_int, two_int)
        Assign the one- and two-electron integrals.
    assign_params(self, params)
        Transform the integrals with a unitary matrix that corresponds to the given parameters.
    orb_rotate_jacobi(self, jacobi_indices, theta)
        Rotate orbitals using Jacobi matrix.
    orb_rotate_matrix(self, matrix)
        Rotate orbitals using a transformation matrix.
    integrate_sd_wfn(self, sd, wfn, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.
    integrate_sd_sd(self, sd1, sd2, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """

    def integrate_sd_wfn(self, sd, wfn, wfn_deriv=None, ham_deriv=None, components=False):
        r"""Integrate the Hamiltonian with against a wavefunction and Slater determinant.

        .. math::

            \left< \Phi \middle| \hat{H} \middle| \Psi \right>
            = \sum_{\mathbf{m} \in S_\Phi}
              f(\mathbf{m}) \left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>

        where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
        :math:`\Phi` is the Slater determinant. The :math:`S_{\Phi}` is the set of Slater
        determinants for which :math:`\left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>` is
        not zero, which are the :math:`\Phi` and its first order orbital-pair excitations for a
        chemical Hamiltonian.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.
            If ham_deriv is not a one-dimensional numpy array of integers.
        ValueError
            If both ham_deriv and wfn_deriv is not None.
            If ham_deriv has any indices than is less than 0 or greater than or equal to nparams.

        """
        nspatial = self.nspatial

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
        if slater.get_seniority(sd, nspatial) != 0:
            if components:
                return 0.0, 0.0, 0.0
            return 0.0

        sd_spatial = slater.split_spin(sd, nspatial)[0]
        occ_spatial_indices = slater.occ_indices(sd_spatial)
        vir_spatial_indices = slater.vir_indices(sd_spatial, nspatial)

        coeff = wfn.get_overlap(sd, deriv=wfn_deriv)
        integral = coeff * self.integrate_sd_sd(sd, sd, deriv=ham_deriv, components=components)
        for i in occ_spatial_indices:
            for a in vir_spatial_indices:  # pylint: disable=C0103
                sd_m = slater.excite(
                    sd,
                    slater.spin_index(i, nspatial, "alpha"),
                    slater.spin_index(i, nspatial, "beta"),
                    slater.spin_index(a, nspatial, "beta"),
                    slater.spin_index(a, nspatial, "alpha"),
                )
                coeff = wfn.get_overlap(sd_m, deriv=wfn_deriv)
                integral += coeff * self.integrate_sd_sd(sd, sd_m, deriv=ham_deriv, components=components)

        return integral

    def integrate_sd_sd(self, sd1, sd2, deriv=None, components=False):  # pylint: disable=R0911
        r"""Integrate the Hamiltonian with against two Slater determinants.

        .. math::

            H_{\mathbf{m}\mathbf{n}} =
            \left< \mathbf{m} \middle| \hat{H} \middle| \mathbf{n} \right>

        In the first summation involving :math:`h_{ij}`, only the terms where :math:`\mathbf{m}` and
        :math:`\mathbf{n}` are the same will contribute to the integral. In the second summation
        involving :math:`g_{ijkl}`, only the terms where :math:`\mathbf{m}` and :math:`\mathbf{n}`
        are different by at most single pair-wise excitation will contribute to the integral.

        Parameters
        ----------
        sd1 : int
            Seniority-zero Slater Determinant.
        sd2 : int
            Seniority-zero Slater Determinant.
        deriv : {int, None}
            Index of the Hamiltonian parameter against which the integral is derivatized.
            Default is no derivatization.
        components : {bool, False}
            Option for separating the integrals into the one electron, coulomb, and exchange
            components.
            Default adds the three components together.

        Returns
        -------
        integral : {float, np.ndarray(3,)}
            Values of the integrals.
            If `components` is False, then the value of the integral is returned.
            If `components` is True, then the value of the one electron, coulomb, and exchange
            components are returned.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.
        NotImplementedError
            If `deriv` is not `None`.

        """
        # pylint: disable=C0103
        if deriv is not None:
            raise NotImplementedError(
                "Orbital rotation is not implemented properly: you cannot "
                "take the derivative of CI matrix elements with respect to "
                "orbital rotation coefficients."
            )

        nspatial = self.nspatial

        if __debug__ and not (slater.is_sd_compatible(sd1) and slater.is_sd_compatible(sd2)):
            raise TypeError("Slater determinant must be given as an integer.")

        # if the Slater determinants are not seniority zero
        if not slater.get_seniority(sd1, nspatial) == slater.get_seniority(sd2, nspatial) == 0:
            if components:
                return 0.0, 0.0, 0.0
            return 0.0

        sd1_spatial = slater.split_spin(sd1, nspatial)[0]
        sd2_spatial = slater.split_spin(sd2, nspatial)[0]
        shared_indices = np.array(slater.shared_orbs(sd1_spatial, sd2_spatial))
        diff_sd1, diff_sd2 = slater.diff_orbs(sd1_spatial, sd2_spatial)

        # if two Slater determinants do not have the same number of electrons
        if len(diff_sd1) != len(diff_sd2):
            if components:
                return 0.0, 0.0, 0.0
            return 0.0

        diff_order = len(diff_sd1)
        # if two Slater determinants are greater than double (spatial orbital) excitation away
        if diff_order > 1:
            if components:
                return 0.0, 0.0, 0.0
            return 0.0

        sign = 1

        one_electron, coulomb, exchange = 0.0, 0.0, 0.0
        # two sd's are the same
        if diff_order == 0:
            one_electron = 2 * np.sum(self.one_int[shared_indices, shared_indices])
            coulomb = 2 * np.sum(np.triu(self._cached_two_int_ijij[shared_indices[:, None], shared_indices], k=1))
            coulomb += np.sum(self._cached_two_int_ijij[shared_indices[:, None], shared_indices])
            exchange = -2 * np.sum(np.triu(self._cached_two_int_ijji[shared_indices[:, None], shared_indices], k=1))
        # two sd's are different by double excitation
        else:
            (a,) = diff_sd1
            (b,) = diff_sd2
            spatial_a = slater.spatial_index(a, nspatial)
            spatial_b = slater.spatial_index(b, nspatial)

            coulomb = self.two_int[spatial_a, spatial_a, spatial_b, spatial_b]

        if components:
            return sign * np.array([one_electron, coulomb, exchange])
        return sign * (one_electron + coulomb + exchange)
