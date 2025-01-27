"""Energy of the Schrodinger equation integrated against a reference wavefunction."""

from fanpy.eqn.base import BaseSchrodinger
from fanpy.tools import sd_list, slater
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np


class EnergyOneSideProjection(BaseSchrodinger):
    r"""Energy evaluated by projecting against a reference wavefunction.

    .. math::

        E = \frac{\left< \Phi \middle| \hat{H} \middle| \Psi \right>}
                 {\left< \Phi \middle| \Psi \right>}

    where :math:`\Phi` is some reference wavefunction that can be a CI wavefunction

    .. math::

        \left| \Phi \right> = \sum_{\mathbf{m} \in S} c_{\mathbf{m}} \left| \mathbf{m} \right>

    or a projected form of wavefunction :math:`\Psi`

    .. math::

        \left| \Phi \right> = \sum_{\mathbf{m} \in S}
                              \left< \Psi \middle| \mathbf{m} \middle> \middle| \mathbf{m} \right>

    where :math:`S` is the projection space.

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited state).
    ham : BaseHamiltonian
        Hamiltonian that defines the system under study.
    indices_component_params : ComponentParameterIndices
        Indices of the component parameters that are active in the objective.
    step_print : bool
        Option to print relevant information when the objective is evaluated.
    step_save : bool
        Option to save parameters when the objective is evaluated.
    tmpfile : str
        Name of the file that will store the parameters used by the objective method.
        By default, the parameter values are not stored.
        If a file name is provided, then parameters are stored upon execution of the objective
        method.
    refwfn : {tuple of int, CIWavefunction, None}
        Wavefunction against which the Schrodinger equation is integrated.
        Tuple of Slater determinants will be interpreted as a projection space, and the reference
        wavefunction will be the given wavefunction truncated to the given projection space.

    Properties
    ----------
    indices_objective_params : dict
        Indices of the (active) objective parameters that corresponds to each component.
    all_params : np.ndarray
        All of the parameters associated with the objective.
    active_params : np.ndarray
        Parameters that are selected for optimization.
    active_nparams : int
        Number of active parameters in the objective.
    num_eqns : int
        Number of equations in the objective.

    Methods
    -------
    __init__(self, wfn, ham, param_selection=None, optimize_orbitals=False, tmpfile="", refwfn=None)
        Initialize the objective.
    assign_params(self, params)
        Assign the parameters to the wavefunction and/or Hamiltonian.
    save_params(self)
        Save all of the parameters to the temporary file.
    wrapped_get_overlap(self, sd, deriv=False)
        Wrap `get_overlap` to be derivatized with respect to the (active) parameters of the
        objective.
    wrapped_integrate_sd_wfn(self, sd, deriv=False)
        Wrap `integrate_sd_wfn` to be derivatized wrt the (active) parameters of the objective.
    wrapped_integrate_sd_sd(self, sd1, sd2, deriv=False)
        Wrap `integrate_sd_sd` to be derivatized wrt the (active) parameters of the objective.
    get_energy_one_proj(self, refwfn, deriv=False)
        Return the energy with respect to a reference wavefunction.
    get_energy_two_proj(self, pspace_l, pspace_r=None, pspace_norm=None, deriv=False)
        Return the energy after projecting out both sides.
    assign_refwfn(self, refwfn=None)
        Assign the reference wavefunction.
    objective(self, params) : float
        Return the energy integrated against the reference wavefunction.
    gradient(self, params) : np.ndarray
        Return the gradient of the energy integrated against the reference wavefunction.

    """

    # pylint: disable=W0223
    def __init__(
        self,
        wfn,
        ham,
        param_selection=None,
        optimize_orbitals=False,
        step_print=True,
        step_save=True,
        tmpfile="",
        refwfn=None,
    ):
        """Initialize the objective instance.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction.
        ham : BaseHamiltonian
            Hamiltonian that defines the system under study.
        param_selection : tuple/list of 2-tuple/list
            Selection of the parameters that will be used in the objective.
            First element of each entry is a component of the objective: a wavefunction,
            Hamiltonian, or `ParamContainer` instance.
            Second element of each entry is a numpy index array (boolean or indices) that will
            select the parameters from each component that will be used in the objective.
            Default selects the wavefunction parameters.
        optimize_orbitals : bool
            Option to optimize orbitals.
            If Hamiltonian parameters are not selected, all of the orbital optimization parameters
            are optimized.
            If Hamiltonian parameters are selected, then only optimize the selected parameters.
            Default is no orbital optimization.
        step_print : bool
            Option to print relevant information when the objective is evaluated.
            Default is True.
        step_save : bool
            Option to save parameters with every evaluation of the objective.
            Default is True
        tmpfile : str
            Name of the file that will store the parameters used by the objective method.
            By default, the parameter values are not stored.
            If a file name is provided, then parameters are stored upon execution of the objective
            method.
        refwfn : {tuple/list of int, tuple/list of CIWavefunction, None}
            Wavefunction against which the Schrodinger equation is integrated.
            Tuple of Slater determinants will be interpreted as a projection space, and the
            reference wavefunction will be the given wavefunction truncated to the given projection
            space.
            By default, the given wavefunction is used as the reference by using a complete
            projection space.

        Raises
        ------
        TypeError
            If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
            If Hamiltonian is not an instance (or instance of a child) of BaseHamiltonian.
            If tmpfile is not a string.
        ValueError
            If wavefunction and Hamiltonian do not have the same number of spin orbitals.

        """
        super().__init__(
            wfn,
            ham,
            param_selection=param_selection,
            optimize_orbitals=optimize_orbitals,
            step_print=step_print,
            step_save=step_save,
            tmpfile=tmpfile,
        )
        self.assign_refwfn(refwfn)

    @property
    def num_eqns(self):
        """Return the number of equations in the objective.

        Returns
        -------
        num_eqns : int
            Number of equations in the objective.

        """
        return 1

    def assign_refwfn(self, refwfn=None):  # pylint: disable=R0912
        """Assign the reference wavefunction.

        Parameters
        ----------
        refwfn : {tuple/list of int, tuple/list of CIWavefunction, None}
            Wavefunction against which the Schrodinger equation is integrated.
            Tuple of Slater determinants will be interpreted as a projection space, and the
            reference wavefunction will be the given wavefunction truncated to the given projection
            space.
            By default, the given wavefunction is used as the reference by using a complete
            projection space.

        Raises
        ------
        TypeError
            If reference wavefunction is not a list or a tuple.
            If projection space (for the reference wavefunction) must be given as a list/tuple of
            Slater determinants.
        ValueError
            If given Slater determinant in projection space (for the reference wavefunction) does
            not have the same number of electrons as the wavefunction.
            If given Slater determinant in projection space (for the reference wavefunction) does
            not have the same number of spin orbitals as the wavefunction.
            If given reference wavefunction does not have the same number of electrons as the
            wavefunction.
            If given reference wavefunction does not have the same number of spin orbitals as the
            wavefunction.

        """
        if refwfn is None:
            self.refwfn = tuple(
                sd_list.sd_list(self.wfn.nelec, self.wfn.nspin, spin=self.wfn.spin, seniority=self.wfn.seniority)
            )
            # break out of function
            return

        if slater.is_sd_compatible(refwfn):
            refwfn = [refwfn]

        if __debug__:
            if isinstance(refwfn, (list, tuple)):
                for sd in refwfn:  # pylint: disable=C0103
                    if slater.is_sd_compatible(sd):
                        occs = slater.occ_indices(sd)
                        if len(occs) != self.wfn.nelec:
                            raise ValueError(
                                "Given Slater determinant does not have the same number of"
                                " electrons as the given wavefunction."
                            )
                        if any(i >= self.wfn.nspin for i in occs):
                            raise ValueError(
                                "Given Slater determinant does not have the same number of"
                                " spin orbitals as the given wavefunction."
                            )
                    else:
                        raise TypeError(
                            "Projection space (for the reference wavefunction) must only "
                            "contain Slater determinants."
                        )
                refwfn = tuple(refwfn)
            elif isinstance(refwfn, CIWavefunction):
                if refwfn.nelec != self.wfn.nelec:
                    raise ValueError(
                        "Given reference wavefunction does not have the same number of "
                        "electrons as the given wavefunction."
                    )
                if refwfn.nspin != self.wfn.nspin:
                    raise ValueError(
                        "Given reference wavefunction does not have the same number of "
                        "spin orbitals as the given wavefunction."
                    )
            else:
                raise TypeError("Projection space must be given as a list or a tuple.")

        self.refwfn = refwfn

    def objective(self, params, assign=True, normalize=False, save=True):
        """Return the energy integrated against the reference wavefunction.

        See `BaseSchrodinger.get_energy_one_proj` for details.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        objective : float
            Energy with respect to the reference.

        """
        params = np.array(params)
        # Assign params
        if assign:
            self.assign_params(params)
        # Normalize
        if normalize and hasattr(self.wfn, "normalize"):
            self.wfn.normalize(self.refwfn)
        # Save params
        if save and self.step_save:
            self.save_params()

        energy = self.get_energy_one_proj(self.refwfn)

        if self.step_print:
            print("(Mid Optimization) Electronic energy: {}".format(energy))
        else:
            self.print_queue["Electronic energy"] = energy

        return energy

    def gradient(self, params, assign=True, normalize=True, save=True):
        """Return the gradient of the energy integrated against the reference wavefunction.

        See `BaseSchrodinger.get_energy_one_proj` for details.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        gradient : np.array(N,)
            Derivative of the energy with respect to the reference.

        """
        params = np.array(params)
        # Assign params
        if assign:
            self.assign_params(params)
        # Normalize
        if normalize and hasattr(self.wfn, "normalize"):
            self.wfn.normalize(self.refwfn)
        # Save params
        if save:
            self.save_params()

        grad = self.get_energy_one_proj(self.refwfn, True)

        grad_norm = np.linalg.norm(grad)
        if self.step_print:
            print("(Mid Optimization) Norm of the gradient of the energy: {}".format(grad_norm))
        else:
            self.print_queue["Norm of the gradient of the energy"] = grad_norm

        return grad
