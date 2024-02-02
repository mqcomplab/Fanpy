"""Hamiltonian that will be used to make the Schrodinger equation."""
import abc
import itertools as it

from fanpy.tools import slater

import numpy as np


class BaseHamiltonian:
    """Hamiltonian for a Schrodinger equation.

    Properties
    ----------
    nspatial : int
        Number of spatial orbitals.

    Methods
    -------
    __init__(self, one_int, two_int)
        Initialize the Hamiltonian.
    integrate_sd_wfn(self, sd, wfn, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.

    Abstract Properties
    -------------------
    nspin : int
        Number of spin orbitals.

    Abstract Methods
    ----------------
    assign_integrals(self, one_int, two_int)
        Assign the one- and two-electron integrals.
    integrate_sd_sd(self, sd1, sd2, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """

    def __init__(self):
        """Initialize the Hamiltonian."""

    @abc.abstractproperty
    def nspin(self):
        """Return the number of spin orbitals.

        Returns
        -------
        nspin : int
            Number of spin orbitals.

        """

    @property
    def nspatial(self):
        """Return the number of spatial orbitals.

        Returns
        -------
        nspatial : int
            Number of spatial orbitals.

        """
        return self.nspin // 2

    def assign_integrals(self, one_int, two_int):
        """Assign the basis set representations of the Hamiltonian (integrals).

        Parameters
        ----------
        one_int : np.ndarray(K, K)
            One electron integrals.
        two_int : np.ndarray(K, K, K, K)
            Two electron integrals.
            Uses physicist's notation.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    # FIXME: need to speed up
    def integrate_sd_wfn(
        self, sd, wfn, wfn_deriv=None, ham_deriv=None, orders=(1, 2), components=False
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
        sd : int
            Slater Determinant against which the Hamiltonian is integrated.
        wfn : Wavefunction
            Wavefunction against which the Hamiltonian is integrated.
            Needs to have the following in `__dict__`: `get_overlap`.
        wfn_deriv : np.ndarray
            Indices of the wavefunction parameter against which the integral is derivatized.
            Default is no derivatization.
        ham_deriv : np.ndarray
            Indices of the Hamiltonian parameter against which the integral is derivatized.
            Default is no derivatization.
        components : {bool, False}
            Option for separating the integrals into the one electron, coulomb, and exchange
            components.
            Default adds the three components together.

        Returns
        -------
        integral : {float, np.ndarray}
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
                if not (
                    isinstance(ham_deriv, np.ndarray)
                    and ham_deriv.ndim == 1
                    and ham_deriv.dtype == int
                ):
                    raise TypeError(
                        "Derivative indices for the Hamiltonian parameters must be given as a "
                        "one-dimensional numpy array of integers."
                    )
                # pylint: disable=E1101
                if np.any(ham_deriv < 0) or np.any(ham_deriv >= self.nparams):
                    raise ValueError(
                        "Derivative indices for the Hamiltonian parameters must be greater than or "
                        "equal to 0 and be less than the number of parameters."
                    )

        occ_indices = slater.occ_indices(sd)
        vir_indices = slater.vir_indices(sd, self.nspin)

        coeff = wfn.get_overlap(sd, deriv=wfn_deriv)
        integral = coeff * self.integrate_sd_sd(sd, sd, deriv=ham_deriv, components=components)
        for order in orders:
            for annihilators in it.combinations(occ_indices, order):
                for creators in it.combinations(vir_indices, order):
                    sd_m = slater.excite(sd, *annihilators, *creators)
                    coeff = wfn.get_overlap(sd_m, deriv=wfn_deriv)
                    integral += coeff * self.integrate_sd_sd(
                        sd, sd_m, deriv=ham_deriv, components=components
                    )

        return integral

    @abc.abstractmethod
    def integrate_sd_sd(self, sd1, sd2, deriv=False, components=False):
        r"""Integrate the Hamiltonian with against two Slater determinants.

        .. math::

            H_{ij} = \left< \Phi_i \middle| \hat{H} \middle| \Phi_j \right>

        where :math:`\hat{H}` is the Hamiltonian operator, and :math:`\Phi_i` and :math:`\Phi_j` are
        Slater determinants.

        Parameters
        ----------
        sd1 : int
            Slater Determinant against which the Hamiltonian is integrated.
        sd2 : int
            Slater Determinant against which the Hamiltonian is integrated.
        deriv : np.ndarray
            Indices of the Hamiltonian parameter against which the integral is derivatized.
            Default is no derivatization.
        components : {bool, False}
            Option for separating the integrals into the one electron, coulomb, and exchange
            components.
            Default adds the three components together.

        Returns
        -------
        integral : {float, np.ndarray}
            If `components` is False, then the value of the integral is returned.
            If `components` is True, then the value of the one electron, coulomb, and exchange
            components are returned.

        """
