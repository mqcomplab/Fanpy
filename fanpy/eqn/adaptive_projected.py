"""Schrodinger equation as a system of equations."""

from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.eqn.constraints.norm import NormConstraint
from fanpy.eqn.utils import ParamContainer
from fanpy.tools import sd_list, slater
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np


class AdaptiveProjectedSchrodinger(ProjectedSchrodinger):
    # TODO: Update docstring to reflect the changes in the class.
    r"""Schrodinger equation as a system of equations.

    .. math::

        w_1 \left(
            \left< \mathbf{m}_1 \middle| \hat{H} \middle| \Psi \right> -
            E \left< \mathbf{m}_1 \middle| \Psi \right>
        \right) &= 0\\
        w_2 \left(
            \left< \mathbf{m}_2 \middle| \hat{H} \middle| \Psi \right> -
            E \left< \mathbf{m}_2 \middle| \Psi \right>
        \right) &= 0\\
        &\vdots\\
        w_M \left(
            \left< \mathbf{m}_M \middle| \hat{H} \middle| \Psi \right> -
            E \left< \mathbf{m}_M \middle| \Psi \right>
        \right) &= 0\\
        w_{M+1} f_{\mathrm{constraint}_1} &= 0\\
        &\vdots

    where :math:`M` is the number of Slater determinant onto which the wavefunction is
    projected.

    The energy can be a fixed constant, a variable parameter, or computed from the given
    wavefunction and Hamiltonian according to the following equation:

    .. math::

        E = \frac{\left< \Phi_\mathrm{ref} \middle| \hat{H} \middle| \Psi \right>}
                    {\left< \Phi_\mathrm{ref} \middle| \Psi \right>}

    where :math:`\Phi_{\mathrm{ref}}` is a linear combination of Slater determinants,
    :math:`\sum_{\mathbf{m} \in S} c_{\mathbf{m}} \left| \mathbf{m} \right>` or
    the wavefunction truncated to a given set of Slater determinants,
    :math:`\sum_{\mathbf{m} \in S} \left< \mathbf{m} \middle| \Psi \right> \left|\mathbf{m}\right>`.

    Additionally, the normalization constraint is added with respect to the reference state.

    .. math::

        f_{\mathrm{constraint}} = \left< \Phi_\mathrm{ref} \middle| \Psi \right> - 1

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system.
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
        If a file name is provided, then parameters are stored upon execution of the objective
        method.
    pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
        States onto which the Schrodinger equation is projected.
        By default, the largest space is used.
    refwfn : {tuple/list of int, CIWavefunction}
        State with respect to which the energy and the norm are computed.
        If a list/tuple of Slater determinants are given, then the reference state is the given
        wavefunction truncated by the provided Slater determinants.
        Default is ground state HF.
    eqn_weights : np.ndarray
        Weights of each equation.
        By default, all equations are given weight of 1 except for the normalization constraint,
        which is weighed by the number of equations.
    energy : ParamContainer
        Energy used in the Schrodinger equation.
        Used to store/cache the energy.
    energy_type : {'fixed', 'variable', 'compute'}
        Type of the energy used in the Schrodinger equation.
        If 'fixed', the energy of the Schrodinger equation is fixed at the given value.
        If 'variable', the energy of the Schrodinger equation is optimized as a parameter.
        If 'compute', the energy of the Schrodinger equation is computed on-the-fly with respect to
        the reference.

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
    params : {np.ndarray(K, )}
        Parameters of the objective at the current state.
    nproj : int
        Number of states onto which the Schrodinger equation is projected.

    Methods
    -------
    __init__(self, wfn, ham, param_selection=None, optimize_orbitals=False, tmpfile="")
        Initialize the objective.
    assign_params(self, params)
        Assign the parameters to the wavefunction and/or hamiltonian.
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
    assign_pspace(self, pspace=None)
        Assign the projection space.
    assign_refwfn(self, refwfn=None)
        Assign the reference wavefunction.
    assign_constraints(self, constraints=None)
        Assign the constraints on the objective.
    assign_eqn_weights(self, eqn_weights=None)
        Assign the weights of each equation.
    objective(self, params) : np.ndarray(self.num_eqns, )
        Return the values of the system of equations.
    jacobian(self, params) : np.ndarray(self.num_eqns, self.nparams.size)
        Return the Jacobian of the objective function.

    """

    # pylint: disable=W0223
    def __init__(
        self,
        wfn,
        ham,
        param_selection=None,
        optimize_orbitals=False,
        step_print=True,
        adaptive_step_print=True,
        step_save=True,
        tmpfile="",
        initial_pspace=None,
        final_pspace=None,
        refwfn=None,
        eqn_weights=None,
        energy_type="compute",
        energy=None,
        constraints=None,
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
        optimize_orbitals : {bool, False}
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
        tmpfile : {str, ''}
            Name of the file that will store the parameters used by the objective method.
            By default, the parameter values are not stored.
            If a file name is provided, then parameters are stored upon execution of the objective
            method.
        pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the Schrodinger equation is projected.
            By default, the largest space is used.
        refwfn : {tuple/list of int, CIWavefunction, None}
            State with respect to which the energy and the norm are computed.
            If a list/tuple of Slater determinants are given, then the reference state is the given
            wavefunction truncated by the provided Slater determinants.
            Default is ground state HF.
        eqn_weights : np.ndarray
            Weights of each equation.
            By default, all equations are given weight of 1 except for the normalization constraint,
            which is weighed by the number of equations.
        energy_type : {'fixed', 'variable', 'compute'}
            Type of the energy used in the Schrodinger equation.
            If 'fixed', the energy of the Schrodinger equation is fixed at the given value.
            If 'variable', the energy of the Schrodinger equation is optimized as a parameter.
            If 'compute', the energy of the Schrodinger equation is computed on-the-fly with respect
            to the reference.
            By default, the energy is computed on-the-fly.
        energy : {float, None}
            Energy of the Schrodinger equation.
            If not provided, energy is computed with respect to the reference.
            By default, energy is computed with respect to the reference.
            Note that this parameter is not used at all if `energy_type` is 'compute'.
        constraints : list/tuple of BaseSchrodinger
            Constraints that will be imposed on the optimization process.
            By default, the normalization constraint used.

        Raises
        ------
        TypeError
            If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
            If Hamiltonian is not an instance (or instance of a child) of BaseHamiltonian.
            If tmpfile is not a string.
            If `energy` is not a number.
        ValueError
            If wavefunction and Hamiltonian do not have the same number of spin orbitals.
            If `energy_type` is not one of 'fixed', 'variable', or 'compute'.

        """
        # TODO: Development of init method.

        self.assign_initial_pspace(pspace=initial_pspace)
        self.pspace = self.initial_pspace
        self.final_pspace = final_pspace

        self.is_adaptive_step_converged = False
        self.adaptive_step_print = adaptive_step_print

        super().__init__(
            wfn,
            ham,
            param_selection=param_selection,
            optimize_orbitals=optimize_orbitals,
            step_print=step_print,
            step_save=step_save,
            tmpfile=tmpfile,
            pspace=initial_pspace,
            refwfn=refwfn,
            eqn_weights=eqn_weights,
            energy_type=energy_type,
            energy=energy,
            constraints=constraints,
        )

    @property
    def nproj(self):
        """Return the size of the current projection space.

        Returns
        -------
        nproj : int
            Number of Slater determinants onto which the Schrodinger equation is projected.

        """
        return len(self.current_pspace)

    def assign_initial_pspace(self, pspace=None):
        """Assign the initial projection space.

        Parameters
        ----------
        pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the Schrodinger equation is projected.
            By default, the first and second-order excitation is used.

        Raises
        ------
        TypeError
            If list/tuple of incompatible objects.
            If not list/tuple or None.

        """
        if pspace is None:
            pspace = sd_list.sd_list(
                self.wfn.nelec,
                self.wfn.nspin,
                spin=self.wfn.spin,
                seniority=self.wfn.seniority,
                exc_orders=[1],
            )

        if __debug__ and not (
            isinstance(pspace, (list, tuple))
            and all(slater.is_sd_compatible(state) or isinstance(state, CIWavefunction) for state in pspace)
        ):
            raise TypeError(
                "Initial projection space must be given as a list/tuple of Slater determinants "
                "or `CIWavefunction`. See `tools.slater` for compatible Slater determinant "
                "formats."
            )

        self.initial_pspace = tuple(pspace)

    def update_current_pspace(self, *kwargs):
        """Update the current projection space.

        This method is called after each evaluation of the objective function.
        It can be used to adaptively change the projection space based on the optimization
        progress or other criteria.

        Parameters
        ----------
        residuals : np.ndarray
            Residuals of the objective function, used to determine current projection performance.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("General adaptive projection space update is not implemented yet.")

    def check_adaptive_step_convergence(self, **kwargs):
        """Check if the adaptive convergence criteria are met.

        This method can be used to determine if the optimization has converged based on the
        the projection space construction method.

        Returns
        -------
        bool
            True if the convergence criteria are met, False otherwise.

        """
        raise NotImplementedError("General adaptive projection space convergence criteria is not implemented yet.")


class PruningAdaptiveProjectedSchrodinger(AdaptiveProjectedSchrodinger):
    # TODO: Update docstring to reflect the changes in the class.
    r"""Schrodinger equation as a system of equations.

    .. math::

        w_1 \left(
            \left< \mathbf{m}_1 \middle| \hat{H} \middle| \Psi \right> -
            E \left< \mathbf{m}_1 \middle| \Psi \right>
        \right) &= 0\\
        w_2 \left(
            \left< \mathbf{m}_2 \middle| \hat{H} \middle| \Psi \right> -
            E \left< \mathbf{m}_2 \middle| \Psi \right>
        \right) &= 0\\
        &\vdots\\
        w_M \left(
            \left< \mathbf{m}_M \middle| \hat{H} \middle| \Psi \right> -
            E \left< \mathbf{m}_M \middle| \Psi \right>
        \right) &= 0\\
        w_{M+1} f_{\mathrm{constraint}_1} &= 0\\
        &\vdots

    where :math:`M` is the number of Slater determinant onto which the wavefunction is
    projected.

    The energy can be a fixed constant, a variable parameter, or computed from the given
    wavefunction and Hamiltonian according to the following equation:

    .. math::

        E = \frac{\left< \Phi_\mathrm{ref} \middle| \hat{H} \middle| \Psi \right>}
                    {\left< \Phi_\mathrm{ref} \middle| \Psi \right>}

    where :math:`\Phi_{\mathrm{ref}}` is a linear combination of Slater determinants,
    :math:`\sum_{\mathbf{m} \in S} c_{\mathbf{m}} \left| \mathbf{m} \right>` or
    the wavefunction truncated to a given set of Slater determinants,
    :math:`\sum_{\mathbf{m} \in S} \left< \mathbf{m} \middle| \Psi \right> \left|\mathbf{m}\right>`.

    Additionally, the normalization constraint is added with respect to the reference state.

    .. math::

        f_{\mathrm{constraint}} = \left< \Phi_\mathrm{ref} \middle| \Psi \right> - 1

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system.
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
        If a file name is provided, then parameters are stored upon execution of the objective
        method.
    pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
        States onto which the Schrodinger equation is projected.
        By default, the largest space is used.
    refwfn : {tuple/list of int, CIWavefunction}
        State with respect to which the energy and the norm are computed.
        If a list/tuple of Slater determinants are given, then the reference state is the given
        wavefunction truncated by the provided Slater determinants.
        Default is ground state HF.
    eqn_weights : np.ndarray
        Weights of each equation.
        By default, all equations are given weight of 1 except for the normalization constraint,
        which is weighed by the number of equations.
    energy : ParamContainer
        Energy used in the Schrodinger equation.
        Used to store/cache the energy.
    energy_type : {'fixed', 'variable', 'compute'}
        Type of the energy used in the Schrodinger equation.
        If 'fixed', the energy of the Schrodinger equation is fixed at the given value.
        If 'variable', the energy of the Schrodinger equation is optimized as a parameter.
        If 'compute', the energy of the Schrodinger equation is computed on-the-fly with respect to
        the reference.

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
    params : {np.ndarray(K, )}
        Parameters of the objective at the current state.
    nproj : int
        Number of states onto which the Schrodinger equation is projected.

    Methods
    -------
    __init__(self, wfn, ham, param_selection=None, optimize_orbitals=False, tmpfile="")
        Initialize the objective.
    assign_params(self, params)
        Assign the parameters to the wavefunction and/or hamiltonian.
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
    assign_pspace(self, pspace=None)
        Assign the projection space.
    assign_refwfn(self, refwfn=None)
        Assign the reference wavefunction.
    assign_constraints(self, constraints=None)
        Assign the constraints on the objective.
    assign_eqn_weights(self, eqn_weights=None)
        Assign the weights of each equation.
    objective(self, params) : np.ndarray(self.num_eqns, )
        Return the values of the system of equations.
    jacobian(self, params) : np.ndarray(self.num_eqns, self.nparams.size)
        Return the Jacobian of the objective function.

    """

    # pylint: disable=W0223
    def __init__(
        self,
        wfn,
        ham,
        param_selection=None,
        optimize_orbitals=False,
        step_print=True,
        adaptive_step_print=True,
        step_save=True,
        tmpfile="",
        initial_pspace=None,
        refwfn=None,
        eqn_weights=None,
        energy_type="compute",
        energy=None,
        constraints=None,
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
        optimize_orbitals : {bool, False}
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
        tmpfile : {str, ''}
            Name of the file that will store the parameters used by the objective method.
            By default, the parameter values are not stored.
            If a file name is provided, then parameters are stored upon execution of the objective
            method.
        initial_pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the Schrodinger equation is projected.
            By default, the largest space is used.
        refwfn : {tuple/list of int, CIWavefunction, None}
            State with respect to which the energy and the norm are computed.
            If a list/tuple of Slater determinants are given, then the reference state is the given
            wavefunction truncated by the provided Slater determinants.
            Default is ground state HF.
        eqn_weights : np.ndarray
            Weights of each equation.
            By default, all equations are given weight of 1 except for the normalization constraint,
            which is weighed by the number of equations.
        energy_type : {'fixed', 'variable', 'compute'}
            Type of the energy used in the Schrodinger equation.
            If 'fixed', the energy of the Schrodinger equation is fixed at the given value.
            If 'variable', the energy of the Schrodinger equation is optimized as a parameter.
            If 'compute', the energy of the Schrodinger equation is computed on-the-fly with respect
            to the reference.
            By default, the energy is computed on-the-fly.
        energy : {float, None}
            Energy of the Schrodinger equation.
            If not provided, energy is computed with respect to the reference.
            By default, energy is computed with respect to the reference.
            Note that this parameter is not used at all if `energy_type` is 'compute'.
        constraints : list/tuple of BaseSchrodinger
            Constraints that will be imposed on the optimization process.
            By default, the normalization constraint used.

        Raises
        ------
        TypeError
            If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
            If Hamiltonian is not an instance (or instance of a child) of BaseHamiltonian.
            If tmpfile is not a string.
            If `energy` is not a number.
        ValueError
            If wavefunction and Hamiltonian do not have the same number of spin orbitals.
            If `energy_type` is not one of 'fixed', 'variable', or 'compute'.

        """
        super().__init__(
            wfn,
            ham,
            param_selection=param_selection,
            optimize_orbitals=optimize_orbitals,
            step_print=step_print,
            adaptive_step_print=adaptive_step_print,
            step_save=step_save,
            tmpfile=tmpfile,
            pspace=None,
            initial_pspace=initial_pspace,
            refwfn=refwfn,
            eqn_weights=eqn_weights,
            energy_type=energy_type,
            energy=energy,
            constraints=constraints,
        )

    def assign_initial_pspace(self, pspace=None):
        """_summary_

        Parameters
        ----------
        pspace : tuple
            Initial projection space to be pruned.

        """
        if pspace is None:
            pspace = sd_list.sd_list(
                self.wfn.nelec,
                self.wfn.nspin,
                spin=self.wfn.spin,
                seniority=self.wfn.seniority,
                exc_orders=None,
            )

        self.initial_pspace = pspace

    def update_current_pspace(self, **kwargs):
        """Update the current projection space.

        This method is called after each evaluation of the objective function.
        It can be used to adaptively change the projection space based on the optimization
        progress or other criteria.

        Parameters
        ----------
        residuals : np.ndarray
            Residuals of the objective function, used to determine current projection performance.

        """
        residuals = kwargs.get("residuals", None)
        residuals_threshold = kwargs.get("residuals_threshold", 1e-8)

        if residuals is None:
            raise ValueError("Residuals must be provided to update the current projection space.")

        updated_pspace_indices = np.transpose(np.argwhere(np.abs(residuals[:-1]) < residuals_threshold))[0]

        print(f"Projection space updated: {len(updated_pspace_indices)} -> {len(self.pspace)}.")
        self.pspace = tuple(np.asarray(self.pspace)[updated_pspace_indices])

    def check_adaptive_step_convergence(self, **kwargs):
        """Check if the adaptive convergence criteria are met.

        This method can be used to determine if the optimization has converged based on the
        the projection space construction method.

        Returns
        -------
        bool
            True if the convergence criteria are met, False otherwise.

        """
        residuals = kwargs.get("residuals", None)
        residuals_threshold = kwargs.get("residuals_threshold", 1e-8)

        if residuals is None:
            raise ValueError("Residuals must be provided to update the current projection space.")

        if not np.any(residuals[:-1] < residuals_threshold):
            self.is_adaptive_step_converged = True
