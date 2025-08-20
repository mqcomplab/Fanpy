"""Utility function for constructing Hamiltonian instances."""

import itertools as it
import numpy as np

from fanpy.ham.base import BaseHamiltonian
from fanpy.tools import slater


def ham_factory(integrate_sd_sd_decomposed, integrals, nspin, orders=(1, 2), integrate_sd_wfn=None):
    r"""Return the instance of the Hamiltonian class with the given integrals.

    Parameters
    ----------
    integrate_sd_sd_decomposed(sd1, sd2, integrals) : function
        Function that returns the integrals components
        :math:`\left< \mathbf{m}_1 \middle| \hat{H} \middle| \mathbf{m}_2 \right>` from the given
        Slater determinants and the basis set representation of the Hamiltonian.
        `sd1` and `sd2` are integers whose bitstring describes the occupation of the Slater
        determinant. See `fanpy.tools.slater` for more details.
        `integrals` is a numpy array that contains the parameters of the wavefunction.
    integrals
        Basis set representation of the Hamiltonian.
    orders : tuple of integers
        Tuple of integers of the orders of excitations involved in the Hamiltonian.
        Does nothing if `integrate_sd_wfn` is provided.
        Default is the one- and two-body operators.
    integrate_sd_wfn(self, sd, wfn, wfn_deriv=None) : function
        Method for integrating Slater determinant with a wavefunction.
        `sd` is the Slater determinant.
        `wfn` is the wavefunction instance.
        `wfn_deriv` is a numpy array of integer indices of the wavefunction parameters with respect
        to which the integral is derivatized.

    """

    class GeneratedHamiltonian(BaseHamiltonian):
        """Generated Hamiltonian using the given integrate_sd_sd and integrate_sd_wfn."""

        def __init__(self, integrals, nspin):
            """Initialize the Hamiltonian.

            Parameters
            ----------
            integrals
                Basis set representation of the Hamiltonian.

            """
            super().__init__()
            self.assign_integrals(integrals)
            self._nspin = nspin

        @property
        def nspin(self):
            """Return the number of spin orbitals.

            Returns
            -------
            nspin : int
                Number of spin orbitals.

            """
            return self._nspin

        def assign_integrals(self, integrals):
            """Assign the basis set representations of the Hamiltonian (integrals).

            Parameters
            ----------
            integrals
                Basis set representation of the Hamiltonian.

            Raises
            ------
            NotImplementedError

            """
            self.integrals = integrals

        def integrate_sd_wfn(self, sd, wfn, wfn_deriv=None, orders=orders):
            r"""Integrate the Hamiltonian with against a wavefunction and Slater determinant.

            .. math::

                \left< \Phi \middle| \hat{H} \middle| \Psi \right>
                = \sum_{\mathbf{m} \in S_\Phi}
                f(\mathbf{m}) \left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>

            where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
            :math:`\Phi` is the Slater determinant. The :math:`S_{\Phi}` is the set of Slater
            determinants for which :math:`\left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>`
            is not zero, which are the :math:`\Phi` and its first and second order excitations for a
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
            orders : tuple of int
                Orders of excitations involved in the Hamiltonian.
                Default is provided as an argument by `ham_factory`.

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
                If ham_deriv has any indices than is less than 0 or greater than or equal to
                nparams.

            Note
            ----
            The TypeError and ValueError are only raised if the integrate_sd_wfn is None.
            Otherwise, it depends on the user defined integrate_sd_wfn if these errors are
            raised. 
            """
            # pylint: disable=C0103
            if integrate_sd_wfn is None:
                # pylint: disable=C0103
                occ_indices = slater.occ_indices(sd)
                vir_indices = slater.vir_indices(sd, self.nspin)

                coeff = wfn.get_overlap(sd, deriv=wfn_deriv)
                integral = coeff * self.integrate_sd_sd(sd, sd)
                for order in orders:
                    for annihilators in it.combinations(occ_indices, order):
                        for creators in it.combinations(vir_indices, order):
                            sd_m = slater.excite(sd, *annihilators, *creators)
                            coeff = wfn.get_overlap(sd_m, deriv=wfn_deriv)
                            integral += coeff * self.integrate_sd_sd(sd, sd_m)

                return integral
            return integrate_sd_wfn(self, sd, wfn, wfn_deriv=wfn_deriv)

        def integrate_sd_sd_decomposed(self, sd1, sd2):
            r"""Integrate the Hamiltonian with against two Slater determinants.

            .. math::

                H_{ij} = \left< \Phi_i \middle| \hat{H} \middle| \Phi_j \right>

            where :math:`\hat{H}` is the Hamiltonian operator, and :math:`\Phi_i` and :math:`\Phi_j`
            are Slater determinants.

            Parameters
            ----------
            sd1 : int
                Slater Determinant against which the Hamiltonian is integrated.
            sd2 : int
                Slater Determinant against which the Hamiltonian is integrated.

            Returns
            -------
            integral : np.ndarray
                The value of the one electron, coulomb, and exchange components 
                are returned.

            """
            return integrate_sd_sd_decomposed(sd1, sd2, self.integrals)

        def integrate_sd_sd(self, sd1, sd2):
            r"""Integrate the Hamiltonian with against two Slater determinants.

            .. math::

                H_{ij} = \left< \Phi_i \middle| \hat{H} \middle| \Phi_j \right>

            where :math:`\hat{H}` is the Hamiltonian operator, and :math:`\Phi_i` and :math:`\Phi_j`
            are Slater determinants.

            Parameters
            ----------
            sd1 : int
                Slater Determinant against which the Hamiltonian is integrated.
            sd2 : int
                Slater Determinant against which the Hamiltonian is integrated.

            Returns
            -------
            integral : float
                The value of the integral is returned. Sum of one electron, coulomb, and exchange components. 
            """

            decomposed_integral = self.integrate_sd_sd_decomposed(sd1, sd2)

            return np.sum(decomposed_integral, axis=0)

    return GeneratedHamiltonian(integrals, nspin)
