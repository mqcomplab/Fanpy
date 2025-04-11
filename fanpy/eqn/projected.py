"""Schrodinger equation as a system of equations."""

from fanpy.eqn.base import BaseSchrodinger
from fanpy.eqn.constraints.norm import NormConstraint
from fanpy.eqn.utils import ParamContainer
from fanpy.tools import sd_list, slater
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np


class ProjectedSchrodinger(BaseSchrodinger):
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
        step_save=True,
        tmpfile="",
        pspace=None,
        pspace_exc_orders=None,
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
        pspace_exc_orders : list/tuple of int
            Excitation orders of the Slater determinants that will be used in the projection space.
            By default, if the pspace was not assigned, the first and second-order excitation is used.
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
            step_save=step_save,
            tmpfile=tmpfile,
        )
        self.assign_pspace(pspace)
        self.assign_refwfn(refwfn)

        self.energy_type = energy_type
        if energy is None:
            energy = self.get_energy_one_proj(self.refwfn)

        if __debug__:
            if not np.issubdtype(np.array(energy).dtype, np.number):
                raise TypeError("Energy must be a number.")
            if energy_type not in ["fixed", "variable", "compute"]:
                raise ValueError("`energy_type` must be one of 'fixed', 'variable', or 'compute'.")

        self.energy = ParamContainer(energy)
        if energy_type in ["fixed", "variable"]:
            self.indices_component_params[self.energy] = [energy_type == "variable"]

        self.assign_constraints(constraints)
        self.assign_eqn_weights(eqn_weights)

    @property
    def num_eqns(self):
        """Return the number of equations in the objective.

        Returns
        -------
        num_eqns : int
            Number of equations in the objective.

        """
        return self.nproj + sum(cons.num_eqns for cons in self.constraints)

    @property
    def nproj(self):
        """Return the size of the projection space.

        Returns
        -------
        nproj : int
            Number of Slater determinants onto which the Schrodinger equation is projected.

        """
        return len(self.pspace)

    def assign_pspace(self, pspace=None):
        """Assign the projection space.

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
            if self.pspace_exc_orders is None:
                self.pspace_exc_orders = [1, 2]

            pspace = sd_list.sd_list(
                self.wfn.nelec,
                self.wfn.nspin,
                spin=self.wfn.spin,
                seniority=self.wfn.seniority,
                exc_orders=self.pspace_exc_orders,
            )

        if __debug__ and not (
            isinstance(pspace, (list, tuple))
            and all(slater.is_sd_compatible(state) or isinstance(state, CIWavefunction) for state in pspace)
        ):
            raise TypeError(
                "Projection space must be given as a list/tuple of Slater determinants or "
                "`CIWavefunction`. See `tools.slater` for compatible Slater determinant "
                "formats."
            )

        self.pspace = tuple(pspace)

    def assign_refwfn(self, refwfn=None):
        """Assign the reference wavefunction.

        Parameters
        ----------
        refwfn : {tuple/list of int, CIWavefunction, None}
            State with respect to which the energy and the norm are computed.
            If a list/tuple of Slater determinants are given, then the reference state is the given
            wavefunction truncated by the provided Slater determinants.
            Default is ground-state Slater determinant.

        Raises
        ------
        TypeError
            If reference wavefunction is not a CIWavefunction, int, or list/tuple of int.

        """
        if refwfn is None:
            refwfn = (slater.ground(self.wfn.nelec, self.wfn.nspin),)
        elif slater.is_sd_compatible(refwfn):
            refwfn = (refwfn,)
        elif isinstance(refwfn, list):
            refwfn = tuple(refwfn)

        if __debug__ and not (
            isinstance(refwfn, (CIWavefunction))
            or (isinstance(refwfn, (list, tuple)) and all(slater.is_sd_compatible(sd) for sd in refwfn))
        ):
            raise TypeError(
                "Reference wavefunction must be given as a Slater determinant, a list/tuple of"
                " Slater determinants, or a CIWavefunction. See `tools.slater` for compatible"
                " representations of the Slater determinants."
            )

        self.refwfn = refwfn

    def assign_constraints(self, constraints=None):
        """Assign the constraints on the objective.

        Parameters
        ----------
        constraints : {list/tuple of BaseSchrodinger, BaseSchrodinger, None}
            Constraints that will be imposed on the optimization process.
            By default, the normalization constraint used.

        Raises
        ------
        TypeError
            If constraints are not given as a BaseSchrodinger instance or list/tuple of
            BaseSchrodinger instances.
        ValueError
            If a constraint does not have the same param_selection as the objective class.

        """
        if constraints is None:
            constraints = [NormConstraint(self.wfn, refwfn=self.refwfn, param_selection=self.indices_component_params)]
        elif isinstance(constraints, BaseSchrodinger):
            constraints = [constraints]

        if __debug__:
            if not (
                isinstance(constraints, (list, tuple))
                and all(isinstance(constraint, BaseSchrodinger) for constraint in constraints)
            ):
                raise TypeError(
                    "Constraints must be given as a BaseSchrodinger instance or list/tuple of "
                    "BaseSchrodinger instances."
                )
            if not all(
                constraint.indices_component_params == self.indices_component_params for constraint in constraints
            ):
                raise ValueError("Constraint must have the same active parameters as the objective.")

        self.constraints = constraints

    def assign_eqn_weights(self, eqn_weights=None):
        """Assign the weights of each equation.

        Parameters
        ----------
        eqn_weights : {np.ndarray, None}
            Weights of each equation.
            By default, all equations are given weight of 1 except for the normalization constraint,
            which is weighed by the number of equations.

        Raises
        ------
        TypeError
            If eqn_weights is not a numpy array.
        ValueError
            If eqn_weights do not have the correct shape.

        """
        num_constraints = sum(cons.num_eqns for cons in self.constraints)
        if eqn_weights is None:
            eqn_weights = np.ones(self.nproj + num_constraints)
            eqn_weights[self.nproj :] *= self.nproj
        if __debug__:
            if not isinstance(eqn_weights, np.ndarray):
                raise TypeError("Weights of the equations must be given as a numpy array.")
            if eqn_weights.shape != (self.nproj + num_constraints,):
                raise ValueError(
                    "Weights of the equations must be given as a one-dimensional array of shape, "
                    "{0}.".format((self.nproj + num_constraints,))
                )
        self.eqn_weights = eqn_weights

    def objective(self, params, return_energy=False, assign=True, normalize=True, save=False):
        r"""Return the Projected Schrodinger equation evaluated at the given parameters.

        Parameters
        ----------
        params : np.ndarray(N,)
            Parameters of the projected Schrodinger equation.

        Returns
        -------
        objective : np.ndarray(nproj+nconstraints, )
            Output of the function that will be optimized.

        """
        params = np.array(params)
        # Assign params to wavefunction, hamiltonian, and other involved objects.
        if assign:
            self.assign_params(params)
        # Normalize
        if normalize and hasattr(self.wfn, "normalize"):
            self.wfn.normalize(self.refwfn)
        # Save params
        if save and self.step_save:
            self.save_params()

        get_overlap = self.wrapped_get_overlap
        integrate_sd_wfn = self.wrapped_integrate_sd_wfn

        # reference values
        if self.energy_type in ["variable", "fixed"]:
            energy = self.energy.params
        else:  # if self.energy_type == "compute":
            # define reference
            if isinstance(self.refwfn, CIWavefunction):
                ref_sds = self.refwfn.sds
                ref_coeffs = self.refwfn.params
            else:
                ref_sds = self.refwfn
                ref_coeffs = np.array([get_overlap(i) for i in ref_sds])

            norm = np.sum(ref_coeffs * np.array([get_overlap(i) for i in ref_sds]))
            # energy = self.get_energy_one_proj(self.refwfn)
            energy = np.sum(ref_coeffs * np.array([integrate_sd_wfn(i) for i in ref_sds])) / norm
            # can be replaced with
            # energy = self.get_energy_one_proj(self.refwfn)
            self.energy.assign_params(energy)

        # objective
        obj = np.empty(self.num_eqns)
        # <SD|H|Psi> - E<SD|Psi> == 0
        obj[: self.nproj] = np.array([integrate_sd_wfn(i) for i in self.pspace]) - energy * np.array(
            [get_overlap(i) for i in self.pspace]
        )
        # Add constraints
        if self.nproj < self.num_eqns:
            obj[self.nproj :] = np.hstack([cons.objective(params) for cons in self.constraints])
        # weigh equations
        obj *= self.eqn_weights

        residuals = obj**2
        cost = np.sum(residuals)
        cost_constraints = np.sum(residuals[self.nproj :])
        if self.step_print:
            print("(Mid Optimization) Electronic energy: {}".format(energy))
            print("(Mid Optimization) Cost: {}".format(cost))
            if self.constraints:
                print("(Mid Optimization) Cost from constraints: {}".format(cost_constraints))
        else:
            self.print_queue["Electronic energy"] = energy
            self.print_queue["Cost"] = cost
            self.print_queue["Cost from constraints"] = cost_constraints

        if return_energy:
            return obj, energy
        else:
            return obj

    def jacobian(
        self, params, return_d_energy=False, assign=False, normalize=False, save=True
    ):  # pylint: disable=R0915
        r"""Return the Jacobian of the projected Schrodinger equation evaluated at the given params.

        If :math:`(f_1(\vec{x}), f_2(\vec{x}), \dots)` is the objective function, the Jacobian is

        .. math::

            J_{ij}(\vec{x}) = \frac{\partial f_i(x_1, \dots, x_N)}{\partial x_j}

        where :math:`\vec{x}` is the parameters at a given iteration.

        Parameters
        ----------
        params : np.ndarray(N,)
            Parameters of the projected Schrodinger equation.

        Returns
        -------
        jac : np.ndarray(nproj+nconstraints, nparams)
            Value of the Jacobian :math:`J_{ij}`.

        Notes
        -----
        Much of the code is copied from `BaseSchrodinger.get_energy_one_proj` to compute the energy.
        It is not called because the norm is still needed for the constraint (meaning that much of
        the code will be copied over anyways), and the derivative of the energy uses some of the
        the same matrices.

        """
        params = np.array(params)
        # Assign params
        if assign:
            self.assign_params(params)
        # Normalize
        # commented out because trf would already normalized
        if normalize and hasattr(self.wfn, "normalize"):
            self.wfn.normalize(self.refwfn)
        # Save params
        if save:
            self.save_params()

        get_overlap = self.wrapped_get_overlap
        integrate_sd_wfn = self.wrapped_integrate_sd_wfn

        # define reference
        if isinstance(self.refwfn, CIWavefunction):
            ref_sds = np.array(self.refwfn.sds)
            ref_coeffs = self.refwfn.params[:, np.newaxis]

            d_ref_coeffs = np.zeros((self.refwfn.nparams, params.size), dtype=float)
            inds_component = self.indices_component_params[self.refwfn]
            if inds_component.size > 0:
                inds_objective = self.indices_objective_params[self.refwfn]
                d_ref_coeffs[inds_component, inds_objective] = 1.0
        else:
            ref_sds = np.array(self.refwfn)
            ref_coeffs = np.array([[get_overlap(i)] for i in ref_sds])
            d_ref_coeffs = np.array([get_overlap(i, True) for i in ref_sds])

        # overlaps of each Slater determinant in reference <SD_i | Psi>
        ref_sds_olps = np.array([[get_overlap(i)] for i in ref_sds])
        d_ref_sds_olps = np.array([get_overlap(i, True) for i in ref_sds])
        # NOTE: d_ref_olps and d_ref_ints are two dimensional matrices (axis 0 corresponds to the
        # reference Slater determinants and 1 to the index of the parameter with respect to which
        # the value is derivatized).

        # norm <ref | Psi>
        d_norm = np.sum(d_ref_coeffs * ref_sds_olps, axis=0)
        d_norm += np.sum(ref_coeffs * d_ref_sds_olps, axis=0)

        # energy
        if self.energy_type in ["variable", "fixed"]:
            energy = self.energy.params
            d_energy = np.zeros(params.size)
            inds_component = self.indices_component_params[self.energy]
            if inds_component.size > 0:
                inds_objective = self.indices_objective_params[self.energy]
                d_energy[inds_objective] = 1.0
        else:  # if self.energy_type == "compute":
            # norm
            norm = np.sum(ref_coeffs * ref_sds_olps)
            # integral <SD | H | Psi>
            ref_sds_ints = np.array([[integrate_sd_wfn(i)] for i in ref_sds])
            d_ref_sds_ints = np.array([integrate_sd_wfn(i, True) for i in ref_sds])
            # integral <ref | H | Psi>
            ref_int = np.sum(ref_coeffs * ref_sds_ints)
            d_ref_int = np.sum(d_ref_coeffs * ref_sds_ints, axis=0)
            d_ref_int += np.sum(ref_coeffs * d_ref_sds_ints, axis=0)

            energy = ref_int / norm
            d_energy = (d_ref_int - d_norm * energy) / norm
            self.energy.assign_params(energy)

        # reshape for broadcasting
        pspace = np.array(self.pspace)

        # jacobian
        jac = np.empty((self.num_eqns, params.size))
        jac[: self.nproj, :] = np.array([integrate_sd_wfn(i, True) for i in pspace])
        jac[: self.nproj, :] -= energy * np.array([get_overlap(i, True) for i in pspace])
        jac[: self.nproj, :] -= d_energy[np.newaxis, :] * np.array([[get_overlap(i)] for i in pspace])
        # Add constraints
        if self.nproj < self.num_eqns:
            jac[self.nproj :] = np.vstack([cons.gradient(params) for cons in self.constraints])
        # weigh equations
        jac *= self.eqn_weights[:, np.newaxis]

        jac_norm = np.linalg.norm(jac)
        if self.step_print:
            print("(Mid Optimization) Norm of the Jacobian: {}".format(jac_norm))
        else:
            self.print_queue["Norm of the Jacobian"] = jac_norm

        return jac
