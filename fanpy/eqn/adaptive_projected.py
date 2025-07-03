"""Schrodinger equation as a system of equations."""

from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.tools import sd_list, slater
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np


# TODO: Check if residuals can be negative. If not, remove useless abs steps.
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

        self._is_adaptive_procedure_done = False
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
    def is_adaptive_procedure_done(self):
        """Return flag to check adaptive procedure completion.

        Returns
        -------
        _is_adaptive_procedure_done : boolean
            If True, the conditions to adaptive procedure completion were achieved.

        """
        return self._is_adaptive_procedure_done

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

    def rebuild_restricted_ham_sds(self, pspace):
        """Rebuild the projection space by restoring missing Slater determinant pairs
           in the context of a restricted orbital basis.

        In a restricted orbital basis, alpha and beta spin-orbitals share the same spatial orbitals.
        During adaptive procedures, it is possible to unintentionally remove one determinant
        from a pair of spin-equivalent excitations (e.g., '0b00110101' and '0b01010011'),
        leading to an incomplete representation of the excitation space.

        This method iterates over the current projection space and adds any missing
        spin-complementary determinants to ensure all relevant excitation pairs are included.

        Parameters
        ----------
        pspace : {tuple/list of int}
            States onto which the Schrodinger equation is projected.

        Returns
        -------
        corrected_pspace : tuple of int
            Corrected projection space, including any previously missing spin-partner determinants.

        """

        if isinstance(self.ham, RestrictedMolecularHamiltonian):
            formatted_sds = [slater.split_spin(sd, self.wfn.nspatial) for sd in pspace]

            # Use a set for faster lookup of existing spin pairs
            existing_pairs = set(formatted_sds)
            corrected_pspace = []

            for sd_alpha, sd_beta in formatted_sds:
                if sd_alpha != sd_beta and (sd_beta, sd_alpha) not in existing_pairs:
                    new_sd = slater.combine_spin(sd_beta, sd_alpha, self.wfn.nspatial)
                    corrected_pspace.append(new_sd)

            # Append only new determinants
            pspace = tuple(pspace) + tuple(corrected_pspace)

        return pspace

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
        residuals_threshold = kwargs.get("residuals_thresholds", [1e-8])[0]

        if residuals is None:
            raise ValueError("Residuals must be provided to update the current projection space.")

        abs_residuals = np.abs(residuals[:-1])
        below_threshold_indices = np.where(abs_residuals < residuals_threshold)[0]
        n_below_threshold = len(below_threshold_indices)

        if n_below_threshold < self.active_nparams:
            below_threshold_indices = np.argsort(abs_residuals)[: self.active_nparams]

        updated_pspace = tuple(np.asarray(self.pspace)[below_threshold_indices])

        # Check if equivalent SDs pairs in restricted orbital basis need to be fixed
        updated_pspace = self.rebuild_restricted_ham_sds(updated_pspace)

        if self.adaptive_step_print:
            print(f"(Adaptive Optimization) Projection space updated: {len(self.pspace)} -> {len(updated_pspace)}.")

        self.pspace = updated_pspace

        # Reassing equation weights
        self.assign_eqn_weights()

    def check_adaptive_procedure(self, **kwargs):
        """Check if the adaptive procedure completion criterias are met.

        This method can be used to determine if the optimization has converged based on the
        the projection space construction method.

        Returns
        -------
        bool
            True if the convergence criteria are met or procedure is completed, False otherwise.

        """
        residuals = kwargs.get("residuals", None)
        residuals_thresholds = kwargs.get("residuals_thresholds", [1e-8])

        if residuals is None:
            raise ValueError("Residuals must be provided to update the current projection space.")

        if not residuals_thresholds:
            raise ValueError("Residuals thresholds list must be provided to update the current projection space.")

        # Check if the current residuals meet the convergence thresholds
        while residuals_thresholds:
            threshold = residuals_thresholds[0]
            if np.any(np.abs(residuals[:-1]) > threshold):
                break

            # Threshold met: remove it and optionally print
            residuals_thresholds.pop(0)
            if self.adaptive_step_print:
                print(f"(Adaptive Optimization) Adaptive convergence criteria met: residuals < {threshold}")

        # If all thresholds have been met, mark procedure as done
        if not residuals_thresholds:
            self._is_adaptive_procedure_done = True
            if self.adaptive_step_print:
                print("(Adaptive Optimization) All residuals thresholds were applied. Adaptive procedure done.")

        # If the number of projections is the same as the number of parameters, mark procedure as done
        if len(residuals) - 1 == self.active_nparams:
            self._is_adaptive_procedure_done = True
            if self.adaptive_step_print:
                print(
                    "(Adaptive Optimization) Projection space reached the active number of parameters. Adaptive procedure done."
                )


class ExtendingAdaptiveProjectedSchrodinger(AdaptiveProjectedSchrodinger):
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
        extension_pspace=None,
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

        self.assign_initial_pspace(pspace=initial_pspace)
        self.assign_external_pspace(pspace=extension_pspace)

        super().__init__(
            wfn,
            ham,
            param_selection=param_selection,
            optimize_orbitals=optimize_orbitals,
            step_print=step_print,
            adaptive_step_print=adaptive_step_print,
            step_save=step_save,
            tmpfile=tmpfile,
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
            Initial projection space to be refined.

        """
        if pspace is None:
            pspace = sd_list.sd_list(
                self.wfn.nelec,
                self.wfn.nspin,
                spin=self.wfn.spin,
                seniority=self.wfn.seniority,
                exc_orders=[1, 2],
            )

        self.initial_pspace = pspace

    def assign_external_pspace(self, pspace=None):
        """_summary_

        Parameters
        ----------
        pspace : tuple
            External projection space to be refined.

        """
        if pspace is None:
            pspace = sd_list.sd_list(
                self.wfn.nelec,
                self.wfn.nspin,
                spin=self.wfn.spin,
                seniority=self.wfn.seniority,
                exc_orders=None,
            )

        # Remove duplicates from the external projection space
        pspace = list(set(pspace) - set(self.initial_pspace))

        self.external_pspace = pspace

    def update_current_pspace(self, **kwargs):
        """Update the current projection space.

        This method is called after each evaluation of the objective function.
        It can be used to adaptively change the projection space based on the optimization
        progress or other criteria.

        Parameters
        ----------
        residuals_thresholds : np.ndarray
            Description.

        """
        residuals_thresholds = kwargs.get("residuals_thresholds", [1e-8])

        get_overlap = self.wrapped_get_overlap
        integrate_sd_wfn = self.wrapped_integrate_sd_wfn
        energy = self.energy.params

        external_residuals = [(integrate_sd_wfn(sd) - energy * get_overlap(sd)) ** 2 for sd in self.external_pspace]

        abs_residuals = np.abs(external_residuals[:-1])

        # Check if the current residuals meet the convergence thresholds
        threshold = residuals_thresholds.pop(0)
        below_threshold_indices = np.where(abs_residuals < threshold)[0]

        updated_pspace = tuple(self.pspace + list(np.asarray(self.external_pspace)[below_threshold_indices]))

        # Check if equivalent SDs pairs in restricted orbital basis need to be fixed
        updated_pspace = self.rebuild_restricted_ham_sds(updated_pspace)

        if self.adaptive_step_print:
            print(f"(Adaptive Optimization) Projection space updated: {len(self.pspace)} -> {len(updated_pspace)}.")

        self.pspace = updated_pspace
        self.external_pspace = tuple(set(self.external_pspace) - set(updated_pspace))

        # Reassing equation weights
        self.assign_eqn_weights()

    def check_adaptive_procedure(self, **kwargs):
        """Check if the adaptive procedure completion criterias are met.

        This method can be used to determine if the optimization has converged based on the
        the projection space construction method.

        Returns
        -------
        bool
            True if the convergence criteria are met or procedure is completed, False otherwise.

        """
        residuals_thresholds = kwargs.get("residuals_thresholds", [1e-8])

        if not residuals_thresholds:
            raise ValueError("Residuals thresholds list must be provided to update the current projection space.")

        # If the number of external projections is zero, mark procedure as done
        if len(self.external_pspace) == 0:
            self._is_adaptive_procedure_done = True
            if self.adaptive_step_print:
                print(
                    "(Adaptive Optimization) All expansion projections added to the projection space. Adaptive procedure done."
                )

        # If all thresholds have been met, mark procedure as done
        if not residuals_thresholds:
            self._is_adaptive_procedure_done = True
            if self.adaptive_step_print:
                print("(Adaptive Optimization) All residuals thresholds were applied. Adaptive procedure done.")
