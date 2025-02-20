"""Local energy used in orbital space variational quantum Monte Carlo."""

from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.tools import slater

import numpy as np


class LocalEnergy(EnergyOneSideProjection):
    r"""Local energy used in orbital space variational quantum Monte Carlo.

    .. math::

        E_{\mathrm{L}} = \sum_{\mathbf{m} \in S}
        \frac{\left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>}
        {\left< \mathbf{m} \middle| \Psi \right>}

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
    pspace : {tuple of int, None}
        Slater determinants from which the energy is calculated.

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
    __init__(self, wfn, ham, param_selection=None, optimize_orbitals=False, tmpfile="", pspace=None)
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
        Return the local energy.
    gradient(self, params) : np.ndarray
        Return the gradient of the local energy.

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
        pspace=None,
    ):
        r"""Initialize the objective instance.

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
        pspace : {tuple/list of int, None}
            Projection space in terms of which the wavefunction is expressed.
            By default, the largest projection space is used.

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
        self.assign_pspace(pspace)

    def assign_pspace(self, pspace=None):
        r"""Assign the projection space.

        Parameters
        ----------
        pspace : {tuple/list of int, None}
            Projection space with respect to which the wavefunction is expressed.
            By default, the largest space is used.

        Raises
        ------
        TypeError
            If projection space is not a list or a tuple of integers.
        ValueError
            If given state in projection space does not have the same number of electrons as the
            wavefunction.
            If given state in projection space does not have the same number of spin orbitals as the
            wavefunction.

        """
        if __debug__ and not (
            pspace is None or (isinstance(pspace, (list, tuple)) and all(slater.is_sd_compatible(sd) for sd in pspace))
        ):
            raise TypeError("Projection space must be given as a list/tuple of integers.")
        super().assign_refwfn(pspace)
        self.pspace = self.refwfn

    def objective(self, params):
        """Return the energy of the wavefunction integrated against the reference wavefunction.

        See `BaseSchrodinger.get_energy_one_proj` for details.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        objective : float
            Value of the objective.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Save params
        if self.step_save:
            self.save_params()

        overlaps = np.array([self.wrapped_get_overlap(sd) for sd in self.pspace])
        integrals = np.array([self.wrapped_integrate_sd_wfn(sd) for sd in self.pspace])
        energy = np.sum(integrals / overlaps)

        if self.step_print:
            print("(Mid Optimization) Electronic energy: {}".format(energy))
        else:
            self.print_queue["Electronic energy"] = energy

        return energy

    def gradient(self, params):
        """Return the gradient of the objective.

        See `BaseSchrodinger.get_energy_one_proj` for details.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        gradient : np.array(N,)
            Derivative of the objective with respect to each of the parameters.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)

        grad = self.get_energy_one_proj(self.refwfn, True)
        overlaps = np.array([self.wrapped_get_overlap(sd) for sd in self.pspace])
        integrals = np.array([self.wrapped_integrate_sd_wfn(sd) for sd in self.pspace])
        d_overlaps = np.array([self.wrapped_get_overlap(sd, True) for sd in self.pspace])
        d_integrals = np.array([self.wrapped_integrate_sd_wfn(sd, True) for sd in self.pspace])
        grad = d_integrals / overlaps[:, None]
        grad -= integrals[:, None] / overlaps[:, None] ** 2 * d_overlaps
        grad = np.sum(grad, axis=0)

        grad_norm = np.linalg.norm(grad)
        if self.step_print:
            print("(Mid Optimization) Norm of the gradient of the energy: {}".format(grad_norm))
        else:
            self.print_queue["Norm of the gradient of the energy"] = grad_norm

        return grad
