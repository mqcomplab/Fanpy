"""Schrodinger equation as a least-squares problem."""

from fanpy.eqn.projected import ProjectedSchrodinger

import numpy as np


# FIXME: change name
# FIXME: inherited jacobian
class LeastSquaresEquations(ProjectedSchrodinger):
    r"""Projected Schrodinger equation as a least squares problem.

    .. math::

        0 = w_1^2 \left(
            \left< \mathbf{m}_1 \middle| \hat{H} \middle| \Psi \right> -
            E \left< \mathbf{m}_1 \middle| \Psi \right>
        \right)^2 +
        w_2^2 \left(
            \left< \mathbf{m}_2 \middle| \hat{H} \middle| \Psi \right> -
            E \left< \mathbf{m}_2 \middle| \Psi \right>
        \right)^2 +
        \dots
        w_M^2 \left(
            \left< \mathbf{m}_M \middle| \hat{H} \middle| \Psi \right> -
            E \left< \mathbf{m}_M \middle| \Psi \right>
        \right)^2 +
        w_{M+1}^2 f_{\mathrm{constraint}_1}^2
        \dots

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
    objective(self, params) : float
        Return the squared sum of the values of the system of equations.
    gradient(self, params) : np.ndarray
        Return the gradient of the objective function.

    """

    @property
    def num_eqns(self):
        """Return the number of equations in the objective.

        Returns
        -------
        num_eqns : int
            Number of equations in the objective.

        """
        return 1

    def objective(self, params):
        r"""Return the projected Schrodinger equation as a sum of squared residuals.

        Parameters
        ----------
        params : np.ndarray(N,)
            Parameters of the projected Schrodinger equation.

        Returns
        -------
        objective : float
            Output of the function that will be optimized.

        """
        # FIXME: monkey patched property
        # The property num_eqns need to be that of ProjectedSchrodinger rather than the
        # LeastSquaresEquations for ProjectedSchrodinger.objective to function properly. This occurs
        # because the number of equations goes to 1 in LeastSquaresEquations. We can add more
        # structure to differentiate between inherited and changed values, but this would require a
        # rewrite of the other parts of code. Since this problem occurs only in this module, we will
        # just patch in the fix here.
        orig_num_eqns = LeastSquaresEquations.num_eqns
        LeastSquaresEquations.num_eqns = ProjectedSchrodinger.num_eqns

        system_eqns = super().objective(params)

        # patch back
        LeastSquaresEquations.num_eqns = orig_num_eqns

        return np.sum(system_eqns**2)

    def gradient(self, params):
        r"""Gradient of the projected Schrodinger equation as sum of squared resduals.

        If the system of equations, :math:`\{f_1(\vec{x}), f_2(\vec{x}), \dots, f_K(\vec{x})\}`,
        corresponds to the projected Schrodinger equation and the corresponding sum of squared
        residuals is

        .. math::

            f(\vec{x}) = f_1(\vec{x})^2 + f_2(\vec{x})^2 + \dots + f_K(\vec{x})^2

        then the gradient is

        .. math::

            \nabla f(\vec{x})_j &= \frac{\partial f(\vec{x})}{\partial x_j}\\
            &= 2 f_1(\vec{x}) \frac{\partial f_1(\vec{x})}{\partial x_j}
            + 2 f_2(\vec{x}) \frac{\partial f_2(\vec{x})}{\partial x_j}
            \dots
            + 2 f_K(\vec{x}) \frac{\partial f_K(\vec{x})}{\partial x_j}
            + 2 f_{constraint}(\vec{x}) \frac{\partial f_{constraint}(\vec{x})}{\partial x_j}

        where :math:`\vec{x}` is the parameters at a given iteration.

        Parameters
        ----------
        params : np.ndarray(N,)
            Parameters of the projected Schrodinger equation.

        Returns
        -------
        grad : np.ndarray(nparams)
            Value of the gradient at the given parameter values.

        """
        # FIXME: monkey patched property
        # The property num_eqns need to be that of ProjectedSchrodinger rather than the
        # LeastSquaresEquations for ProjectedSchrodinger.objective to function properly. This occurs
        # because the number of equations goes to 1 in LeastSquaresEquations. We can add more
        # structure to differentiate between inherited and changed values, but this would require a
        # rewrite of the other parts of code. Since this problem occurs only in this module, we will
        # just patch in the fix here.
        orig_num_eqns = LeastSquaresEquations.num_eqns
        LeastSquaresEquations.num_eqns = ProjectedSchrodinger.num_eqns

        orig_step_print = self.step_print
        self.step_print = False

        system_eqns = super().objective(params)
        # FIXME: There is probably a better way to implement this. In jacobian method, all of the
        # integrals needed to create the system_eqns is already available. This implementation would
        # repeat certain operations (but might not be that bad with caching?)
        system_eqns = super().objective(params)[:, np.newaxis]
        system_eqns_jac = super().jacobian(params)

        # patch back
        LeastSquaresEquations.num_eqns = orig_num_eqns
        self.step_print = orig_step_print

        grad = 2 * np.sum(system_eqns * system_eqns_jac, axis=0)

        grad_norm = np.linalg.norm(grad)
        if self.step_print:
            print("(Mid Optimization) Norm of the gradient: {}".format(grad_norm))
        else:
            self.print_queue["Norm of the gradient"] = grad_norm

        return grad
