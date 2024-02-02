import os
import numpy as np
from numpy import asarray, Inf, isinf
# from scipy.optimize import (
#     _epsilon, _check_unknown_options, wrap_function, approx_fprime, vecnorm, _line_search_wolfe12,
#     _LineSearchError, _status_message, OptimizeResult
# )
import scipy.optimize
from scipy.optimize._trustregion_constr.tr_interior_point import BarrierSubproblem
from scipy.optimize._trustregion_constr.equality_constrained_sqp import (
    default_scaling, projections, norm, modified_dogleg, projected_cg, box_intersections
)
from scipy.optimize._trustregion_constr.minimize_trustregion_constr import (
    HessianLinearOperator, BFGS, strict_bounds, ScalarFunction, NonlinearConstraint,
    LinearConstraint, PreparedConstraint, initial_constraints_as_canonical, CanonicalConstraint,
    LagrangianHessian, OptimizeResult, time, update_state_sqp, BasicReport, SQPReport,
    update_state_ip, IPReport, TERMINATION_MESSAGES
    )
from scipy.optimize._minimize import warn, MemoizeJac, standardize_bounds, standardize_constraints
from scipy.optimize._differentiable_functions import VectorFunction

from fanpy.solver.wrappers import wrap_scipy
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.eqn.constraints.norm import NormConstraint
from fanpy.tools import math_tools
#from fanpy.upgrades import speedup_sd, speedup_sign, speedup_objective
from fanpy.upgrades import speedup_sign


def equality_constrained_sqp(fun_and_constr, grad_and_jac, lagr_hess,
                             x0, fun0, grad0, constr0,
                             jac0, stop_criteria,
                             state,
                             initial_penalty,
                             initial_trust_radius,
                             factorization_method,
                             trust_lb=None,
                             trust_ub=None,
                             scaling=default_scaling,
                             schrodinger=None):
    """Solve nonlinear equality-constrained problem using trust-region SQP.

    Solve optimization problem:

        minimize fun(x)
        subject to: constr(x) = 0

    using Byrd-Omojokun Trust-Region SQP method described in [1]_. Several
    implementation details are based on [2]_ and [3]_, p. 549.

    References
    ----------
    .. [1] Lalee, Marucha, Jorge Nocedal, and Todd Plantenga. "On the
           implementation of an algorithm for large-scale equality
           constrained optimization." SIAM Journal on
           Optimization 8.3 (1998): 682-706.
    .. [2] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
           "An interior point algorithm for large-scale nonlinear
           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
    .. [3] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    """
    # XXXX: define useful variables
    if (
        schrodinger.wfn in schrodinger.indices_component_params and
        schrodinger.indices_component_params[schrodinger.wfn].size
    ):
        wfn = schrodinger.wfn
        param_indices = schrodinger.indices_objective_params[wfn]
        param_indices = np.where(param_indices)[0]
        wfn_indices = schrodinger.indices_component_params[wfn]
    else:
        wfn = None
    if (
        schrodinger.ham in schrodinger.indices_component_params and
        schrodinger.indices_component_params[schrodinger.ham].size
    ):
        ham = schrodinger.ham
        ham_indices = schrodinger.indices_objective_params[ham]
        ham_indices = np.where(ham_indices)[0]
        ham.update_prev_params = False
    else:
        ham = None
    counter = 0

    PENALTY_FACTOR = 0.3  # Rho from formula (3.51), reference [2]_, p.891.
    LARGE_REDUCTION_RATIO = 0.9
    INTERMEDIARY_REDUCTION_RATIO = 0.3
    SUFFICIENT_REDUCTION_RATIO = 1e-8  # Eta from reference [2]_, p.892.
    TRUST_ENLARGEMENT_FACTOR_L = 7.0
    TRUST_ENLARGEMENT_FACTOR_S = 2.0
    MAX_TRUST_REDUCTION = 0.5
    MIN_TRUST_REDUCTION = 0.1
    SOC_THRESHOLD = 0.1
    TR_FACTOR = 0.8  # Zeta from formula (3.21), reference [2]_, p.885.
    BOX_FACTOR = 0.5

    n, = np.shape(x0)  # Number of parameters

    # Set default lower and upper bounds.
    if trust_lb is None:
        trust_lb = np.full(n, -np.inf)
    if trust_ub is None:
        trust_ub = np.full(n, np.inf)

    # Initial values
    x = np.copy(x0)
    trust_radius = initial_trust_radius
    penalty = initial_penalty
    # Compute Values
    f = fun0
    c = grad0
    b = constr0
    A = jac0
    S = scaling(x)
    # Get projections
    Z, LS, Y = projections(A, factorization_method)
    # Compute least-square lagrange multipliers
    v = -LS.dot(c)
    # Compute Hessian
    H = lagr_hess(x, v)

    # Update state parameters
    optimality = norm(c + A.T.dot(v), np.inf)
    constr_violation = norm(b, np.inf) if len(b) > 0 else 0
    cg_info = {'niter': 0, 'stop_cond': 0,
               'hits_boundary': False}

    last_iteration_failed = False
    while not stop_criteria(state, x, last_iteration_failed,
                            optimality, constr_violation,
                            trust_radius, penalty, cg_info):
        print('Iterating trust region')
        # Normal Step - `dn`
        # minimize 1/2*||A dn + b||^2
        # subject to:
        # ||dn|| <= TR_FACTOR * trust_radius
        # BOX_FACTOR * lb <= dn <= BOX_FACTOR * ub.
        dn = modified_dogleg(A, Y, b,
                             TR_FACTOR*trust_radius,
                             BOX_FACTOR*trust_lb,
                             BOX_FACTOR*trust_ub)

        # Tangential Step - `dt`
        # Solve the QP problem:
        # minimize 1/2 dt.T H dt + dt.T (H dn + c)
        # subject to:
        # A dt = 0
        # ||dt|| <= sqrt(trust_radius**2 - ||dn||**2)
        # lb - dn <= dt <= ub - dn
        c_t = H.dot(dn) + c
        b_t = np.zeros_like(b)
        trust_radius_t = np.sqrt(trust_radius**2 - np.linalg.norm(dn)**2)
        lb_t = trust_lb - dn
        ub_t = trust_ub - dn
        dt, cg_info = projected_cg(H, c_t, Z, Y, b_t,
                                   trust_radius_t,
                                   lb_t, ub_t)

        # Compute update (normal + tangential steps).
        d = dn + dt

        # Compute second order model: 1/2 d H d + c.T d + f.
        quadratic_model = 1/2*(H.dot(d)).dot(d) + c.T.dot(d)
        # Compute linearized constraint: l = A d + b.
        linearized_constr = A.dot(d)+b
        # Compute new penalty parameter according to formula (3.52),
        # reference [2]_, p.891.
        vpred = norm(b) - norm(linearized_constr)
        # Guarantee `vpred` always positive,
        # regardless of roundoff errors.
        vpred = max(1e-16, vpred)
        previous_penalty = penalty
        if quadratic_model > 0:
            new_penalty = quadratic_model / ((1-PENALTY_FACTOR)*vpred)
            penalty = max(penalty, new_penalty)
        # Compute predicted reduction according to formula (3.52),
        # reference [2]_, p.891.
        predicted_reduction = -quadratic_model + penalty*vpred

        # Compute merit function at current point
        merit_function = f + penalty*norm(b)
        # Evaluate function and constraints at trial point
        x_next = x + S.dot(d)
        f_next, b_next = fun_and_constr(x_next)
        # Compute merit function at trial point
        merit_function_next = f_next + penalty*norm(b_next)
        # Compute actual reduction according to formula (3.54),
        # reference [2]_, p.892.
        actual_reduction = merit_function - merit_function_next
        # Compute reduction ratio
        reduction_ratio = actual_reduction / predicted_reduction

        # Second order correction (SOC), reference [2]_, p.892.
        if reduction_ratio < SUFFICIENT_REDUCTION_RATIO and \
           norm(dn) <= SOC_THRESHOLD * norm(dt):
            # Compute second order correction
            y = -Y.dot(b_next)
            # Make sure increment is inside box constraints
            _, t, intersect = box_intersections(d, y, trust_lb, trust_ub)
            # Compute tentative point
            x_soc = x + S.dot(d + t*y)
            f_soc, b_soc = fun_and_constr(x_soc)
            # Recompute actual reduction
            merit_function_soc = f_soc + penalty*norm(b_soc)
            actual_reduction_soc = merit_function - merit_function_soc
            # Recompute reduction ratio
            reduction_ratio_soc = actual_reduction_soc / predicted_reduction
            if intersect and reduction_ratio_soc >= SUFFICIENT_REDUCTION_RATIO:
                x_next = x_soc
                f_next = f_soc
                b_next = b_soc
                reduction_ratio = reduction_ratio_soc

        # Readjust trust region step, formula (3.55), reference [2]_, p.892.
        if reduction_ratio >= LARGE_REDUCTION_RATIO:
            trust_radius = max(TRUST_ENLARGEMENT_FACTOR_L * norm(d),
                               trust_radius)
        elif reduction_ratio >= INTERMEDIARY_REDUCTION_RATIO:
            trust_radius = max(TRUST_ENLARGEMENT_FACTOR_S * norm(d),
                               trust_radius)
        # Reduce trust region step, according to reference [3]_, p.696.
        elif reduction_ratio < SUFFICIENT_REDUCTION_RATIO:
            trust_reduction = ((1-SUFFICIENT_REDUCTION_RATIO) /
                               (1-reduction_ratio))
            new_trust_radius = trust_reduction * norm(d)
            if new_trust_radius >= MAX_TRUST_REDUCTION * trust_radius:
                trust_radius *= MAX_TRUST_REDUCTION
            elif new_trust_radius >= MIN_TRUST_REDUCTION * trust_radius:
                trust_radius = new_trust_radius
            else:
                trust_radius *= MIN_TRUST_REDUCTION

        # Update iteratio
        if reduction_ratio >= SUFFICIENT_REDUCTION_RATIO:
            print('amount of change', np.linalg.norm(x - x_next))
            x = x_next

            # XXXX: Update parameters of Hamiltonian
            if ham:
                ham_params = x[ham_indices]
                params_diff = ham_params - ham._prev_params
                unitary = math_tools.unitary_matrix(params_diff)
                ham._prev_params = ham_params.copy()
                ham._prev_unitary = ham._prev_unitary.dot(unitary)
            # # XXXX: Normalize wavefunction parameters
            # if wfn:
            #     x[param_indices] = wfn.params.flatten()[wfn_indices]
            # XXXX: Save parameters
            schrodinger.save_params()
            if 'energy' in schrodinger.adapt_type:
                schrodinger.energy_objective.adapt_pspace()
            if 'norm' in schrodinger.adapt_type:
                schrodinger.wfn.update_pspace_norm(schrodinger.energy_objective.refwfn)

            f, b = f_next, b_next
            c, A = grad_and_jac(x)
            S = scaling(x)
            # Get projections
            Z, LS, Y = projections(A, factorization_method)
            # Compute least-square lagrange multipliers
            v = -LS.dot(c)
            # Compute Hessian
            H = lagr_hess(x, v)
            # Set Flag
            last_iteration_failed = False
            # Otimality values
            optimality = norm(c + A.T.dot(v), np.inf)
            constr_violation = norm(b, np.inf) if len(b) > 0 else 0

            # XXXX: prevent crazy energy optimization when sampling
            # if constr_violation > 0.1:
            #     counter = 0
            # else:
            #     counter += 1
            #     if schrodinger.adapt_type and counter > 5:
            #         return x, state
        else:
            print('no change')
            penalty = previous_penalty
            last_iteration_failed = True
        print('modified fun', f)
        print('modified cost', 0.5 * np.sum(b[:schrodinger.nproj] ** 2), np.max(np.abs(b)))
        print('norm cost', (b[schrodinger.nproj:schrodinger.num_eqns]) ** 2)

    return x, state


def tr_interior_point(fun, grad, lagr_hess, n_vars, n_ineq, n_eq,
                      constr, jac, x0, fun0, grad0,
                      constr_ineq0, jac_ineq0, constr_eq0,
                      jac_eq0, stop_criteria,
                      enforce_feasibility, xtol, state,
                      initial_barrier_parameter,
                      initial_tolerance,
                      initial_penalty,
                      initial_trust_radius,
                      factorization_method,
                      schrodinger, objective, constraints):
    """Trust-region interior points method.

    Solve problem:
        minimize fun(x)
        subject to: constr_ineq(x) <= 0
                    constr_eq(x) = 0
    using trust-region interior point method described in [1]_.
    """
    # BOUNDARY_PARAMETER controls the decrease on the slack
    # variables. Represents ``tau`` from [1]_ p.885, formula (3.18).
    BOUNDARY_PARAMETER = 0.995
    # BARRIER_DECAY_RATIO controls the decay of the barrier parameter
    # and of the subproblem toloerance. Represents ``theta`` from [1]_ p.879.
    BARRIER_DECAY_RATIO = 0.2
    # TRUST_ENLARGEMENT controls the enlargement on trust radius
    # after each iteration
    TRUST_ENLARGEMENT = 5

    # Default enforce_feasibility
    if enforce_feasibility is None:
        enforce_feasibility = np.zeros(n_ineq, bool)
    # Initial Values
    barrier_parameter = initial_barrier_parameter
    tolerance = initial_tolerance
    trust_radius = initial_trust_radius
    # Define initial value for the slack variables
    s0 = np.maximum(-1.5*constr_ineq0, np.ones(n_ineq))

    # XXXX: tweak objective update to fit fanpy api
    # old_function_and_constraints = BarrierSubproblem.function_and_constraints
    # old_gradient_and_jacobian = BarrierSubproblem.gradient_and_jacobian

    # def function_and_constraints(self, z):
    #     # schrodinger.assign_params(self.get_variables(z))
    #     # schrodinger.wfn.normalize()
    #     return old_function_and_constraints(self, z)

    # def gradient_and_jacobian(self, z):
    #     # schrodinger.assign_params(self.get_variables(z))
    #     # schrodinger.wfn.normalize()
    #     return old_gradient_and_jacobian(self, z)

    # BarrierSubproblem.function_and_constraints = function_and_constraints
    # BarrierSubproblem.gradient_and_jacobian = gradient_and_jacobian

    # Define barrier subproblem
    subprob = BarrierSubproblem(
        x0, s0, fun, grad, lagr_hess, n_vars, n_ineq, n_eq, constr, jac,
        barrier_parameter, tolerance, enforce_feasibility,
        stop_criteria, xtol, fun0, grad0, constr_ineq0, jac_ineq0,
        constr_eq0, jac_eq0)

    # Define initial parameter for the first iteration.
    z = np.hstack((x0, s0))
    fun0_subprob, constr0_subprob = subprob.fun0, subprob.constr0
    grad0_subprob, jac0_subprob = subprob.grad0, subprob.jac0
    # Define trust region bounds
    trust_lb = np.hstack((np.full(subprob.n_vars, -np.inf),
                          np.full(subprob.n_ineq, -BOUNDARY_PARAMETER)))
    trust_ub = np.full(subprob.n_vars+subprob.n_ineq, np.inf)

    # XXXX: counter
    counter = 0

    # Solves a sequence of barrier problems
    while True:
        # Solve SQP subproblem
        z, state = equality_constrained_sqp(
            subprob.function_and_constraints,
            subprob.gradient_and_jacobian,
            subprob.lagrangian_hessian,
            z, fun0_subprob, grad0_subprob,
            constr0_subprob, jac0_subprob, subprob.stop_criteria,
            state, initial_penalty, trust_radius,
            factorization_method, trust_lb, trust_ub, subprob.scaling, schrodinger)
        # XXXX: make sure that the optimized successively for sampled ones
        # if schrodinger.adapt_type:
        #     if subprob.terminate:
        #         counter += 1
        #     else:
        #         counter = 0
        #     if counter > 10:
        #         break
        # elif subprob.terminate:
        #     break
        if subprob.terminate:
            break

        # Update parameters
        trust_radius = max(initial_trust_radius,
                           TRUST_ENLARGEMENT*state.tr_radius)
        # TODO: Use more advanced strategies from [2]_
        # to update this parameters.
        barrier_parameter *= BARRIER_DECAY_RATIO
        tolerance *= BARRIER_DECAY_RATIO
        # Update Barrier Problem
        subprob.update(barrier_parameter, tolerance)

        # XXXX: Save parameters
        # schrodinger.save_params()

        # XXXX: Adapt objective
        if schrodinger.adapt_type:
            print('Adapting projection space')
        if 'pspace' in schrodinger.adapt_type:
            schrodinger.adapt_pspace()
        if 'energy' in schrodinger.adapt_type:
            schrodinger.energy_objective.adapt_pspace()
        if 'norm' in schrodinger.adapt_type:
            schrodinger.wfn.update_pspace_norm(schrodinger.energy_objective.refwfn)
        print('Iterating subproblem')
        # XXXX: reset cached function values
        if schrodinger.adapt_type:
            objective._update_x_impl(subprob.get_variables(z))
            for constraint in constraints:
                if isinstance(constraint.fun, VectorFunction):
                    constraint.fun._update_x_impl(subprob.get_variables(z))

        # Compute initial values for next iteration
        fun0_subprob, constr0_subprob = subprob.function_and_constraints(z)
        grad0_subprob, jac0_subprob = subprob.gradient_and_jacobian(z)

    # Get x and s
    x = subprob.get_variables(z)
    return x, state


def _minimize_trustregion_constr(fun, x0, args, grad,
                                 hess, hessp, bounds, constraints,
                                 xtol=1e-8, gtol=1e-8,
                                 barrier_tol=1e-8,
                                 sparse_jacobian=None,
                                 callback=None, maxiter=1000,
                                 verbose=0, finite_diff_rel_step=None,
                                 initial_constr_penalty=1.0, initial_tr_radius=1.0,
                                 initial_barrier_parameter=0.1,
                                 initial_barrier_tolerance=0.1,
                                 factorization_method=None,
                                 disp=False,
                                 schrodinger=None):
    """Minimize a scalar function subject to constraints.

    Parameters
    ----------
    gtol : float, optional
        Tolerance for termination by the norm of the Lagrangian gradient.
        The algorithm will terminate when both the infinity norm (i.e. max
        abs value) of the Lagrangian gradient and the constraint violation
        are smaller than ``gtol``. Default is 1e-8.
    xtol : float, optional
        Tolerance for termination by the change of the independent variable.
        The algorithm will terminate when ``tr_radius < xtol``, where
        ``tr_radius`` is the radius of the trust region used in the algorithm.
        Default is 1e-8.
    barrier_tol : float, optional
        Threshold on the barrier parameter for the algorithm termination.
        When inequality constraints are present the algorithm will terminate
        only when the barrier parameter is less than `barrier_tol`.
        Default is 1e-8.
    sparse_jacobian : {bool, None}, optional
        Determines how to represent Jacobians of the constraints. If bool,
        then Jacobians of all the constraints will be converted to the
        corresponding format. If None (default), then Jacobians won't be
        converted, but the algorithm can proceed only if they all have the
        same format.
    initial_tr_radius: float, optional
        Initial trust radius. The trust radius gives the maximum distance
        between solution points in consecutive iterations. It reflects the
        trust the algorithm puts in the local approximation of the optimization
        problem. For an accurate local approximation the trust-region should be
        large and for an  approximation valid only close to the current point it
        should be a small one. The trust radius is automatically updated throughout
        the optimization process, with ``initial_tr_radius`` being its initial value.
        Default is 1 (recommended in [1]_, p. 19).
    initial_constr_penalty : float, optional
        Initial constraints penalty parameter. The penalty parameter is used for
        balancing the requirements of decreasing the objective function
        and satisfying the constraints. It is used for defining the merit function:
        ``merit_function(x) = fun(x) + constr_penalty * constr_norm_l2(x)``,
        where ``constr_norm_l2(x)`` is the l2 norm of a vector containing all
        the constraints. The merit function is used for accepting or rejecting
        trial points and ``constr_penalty`` weights the two conflicting goals
        of reducing objective function and constraints. The penalty is automatically
        updated throughout the optimization  process, with
        ``initial_constr_penalty`` being its  initial value. Default is 1
        (recommended in [1]_, p 19).
    initial_barrier_parameter, initial_barrier_tolerance: float, optional
        Initial barrier parameter and initial tolerance for the barrier subproblem.
        Both are used only when inequality constraints are present. For dealing with
        optimization problems ``min_x f(x)`` subject to inequality constraints
        ``c(x) <= 0`` the algorithm introduces slack variables, solving the problem
        ``min_(x,s) f(x) + barrier_parameter*sum(ln(s))`` subject to the equality
        constraints  ``c(x) + s = 0`` instead of the original problem. This subproblem
        is solved for increasing values of ``barrier_parameter`` and with decreasing
        tolerances for the termination, starting with ``initial_barrier_parameter``
        for the barrier parameter and ``initial_barrier_tolerance`` for the
        barrier subproblem  barrier. Default is 0.1 for both values (recommended in [1]_ p. 19).
    factorization_method : string or None, optional
        Method to factorize the Jacobian of the constraints. Use None (default)
        for the auto selection or one of:

            - 'NormalEquation' (requires scikit-sparse)
            - 'AugmentedSystem'
            - 'QRFactorization'
            - 'SVDFactorization'

        The methods 'NormalEquation' and 'AugmentedSystem' can be used only
        with sparse constraints. The projections required by the algorithm
        will be computed using, respectively, the the normal equation  and the
        augmented system approaches explained in [1]_. 'NormalEquation'
        computes the Cholesky factorization of ``A A.T`` and 'AugmentedSystem'
        performs the LU factorization of an augmented system. They usually
        provide similar results. 'AugmentedSystem' is used by default for
        sparse matrices.

        The methods 'QRFactorization' and 'SVDFactorization' can be used
        only with dense constraints. They compute the required projections
        using, respectively, QR and SVD factorizations. The 'SVDFactorization'
        method can cope with Jacobian matrices with deficient row rank and will
        be used whenever other factorization methods fail (which may imply the
        conversion of sparse matrices to a dense format when required).
        By default 'QRFactorization' is used for dense matrices.
    finite_diff_rel_step : None or array_like, optional
        Relative step size for the finite difference approximation.
    maxiter : int, optional
        Maximum number of algorithm iterations. Default is 1000.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.
            * 3 : display progress during iterations (more complete report).

    disp : bool, optional
        If True (default) then `verbose` will be set to 1 if it was 0.

    Returns
    -------
    `OptimizeResult` with the fields documented below. Note the following:

        1. All values corresponding to the constraints are ordered as they
           were passed to the solver. And values corresponding to `bounds`
           constraints are put *after* other constraints.
        2. All numbers of function, Jacobian or Hessian evaluations correspond
           to numbers of actual Python function calls. It means, for example,
           that if a Jacobian is estimated by finite differences then the
           number of Jacobian evaluations will be zero and the number of
           function evaluations will be incremented by all calls during the
           finite difference estimation.

    x : ndarray, shape (n,)
        Solution found.
    optimality : float
        Infinity norm of the Lagrangian gradient at the solution.
    constr_violation : float
        Maximum constraint violation at the solution.
    fun : float
        Objective function at the solution.
    grad : ndarray, shape (n,)
        Gradient of the objective function at the solution.
    lagrangian_grad : ndarray, shape (n,)
        Gradient of the Lagrangian function at the solution.
    nit : int
        Total number of iterations.
    nfev : integer
        Number of the objective function evaluations.
    ngev : integer
        Number of the objective function gradient evaluations.
    nhev : integer
        Number of the objective function Hessian evaluations.
    cg_niter : int
        Total number of the conjugate gradient method iterations.
    method : {'equality_constrained_sqp', 'tr_interior_point'}
        Optimization method used.
    constr : list of ndarray
        List of constraint values at the solution.
    jac : list of {ndarray, sparse matrix}
        List of the Jacobian matrices of the constraints at the solution.
    v : list of ndarray
        List of the Lagrange multipliers for the constraints at the solution.
        For an inequality constraint a positive multiplier means that the upper
        bound is active, a negative multiplier means that the lower bound is
        active and if a multiplier is zero it means the constraint is not
        active.
    constr_nfev : list of int
        Number of constraint evaluations for each of the constraints.
    constr_njev : list of int
        Number of Jacobian matrix evaluations for each of the constraints.
    constr_nhev : list of int
        Number of Hessian evaluations for each of the constraints.
    tr_radius : float
        Radius of the trust region at the last iteration.
    constr_penalty : float
        Penalty parameter at the last iteration, see `initial_constr_penalty`.
    barrier_tolerance : float
        Tolerance for the barrier subproblem at the last iteration.
        Only for problems with inequality constraints.
    barrier_parameter : float
        Barrier parameter at the last iteration. Only for problems
        with inequality constraints.
    execution_time : float
        Total execution time.
    message : str
        Termination message.
    status : {0, 1, 2, 3}
        Termination status:

            * 0 : The maximum number of function evaluations is exceeded.
            * 1 : `gtol` termination condition is satisfied.
            * 2 : `xtol` termination condition is satisfied.
            * 3 : `callback` function requested termination.

    cg_stop_cond : int
        Reason for CG subproblem termination at the last iteration:

            * 0 : CG subproblem not evaluated.
            * 1 : Iteration limit was reached.
            * 2 : Reached the trust-region boundary.
            * 3 : Negative curvature detected.
            * 4 : Tolerance was satisfied.

    References
    ----------
    .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.
           Trust region methods. 2000. Siam. pp. 19.
    """
    x0 = np.atleast_1d(x0).astype(float)
    n_vars = np.size(x0)
    if hess is None:
        if callable(hessp):
            hess = HessianLinearOperator(hessp, n_vars)
        else:
            hess = BFGS()
    if disp and verbose == 0:
        verbose = 1

    if bounds is not None:
        finite_diff_bounds = strict_bounds(bounds.lb, bounds.ub,
                                           bounds.keep_feasible, n_vars)
    else:
        finite_diff_bounds = (-np.inf, np.inf)

    # Define Objective Function
    objective = ScalarFunction(fun, x0, args, grad, hess,
                               finite_diff_rel_step, finite_diff_bounds)

    # XXXX: overwrite objective update to fit fanpy API
    def update_x(x):
        objective._update_grad()
        objective.x_prev = objective.x
        objective.g_prev = objective.g

        objective.x = np.atleast_1d(x).astype(float)
        schrodinger.assign_params(x)
        # schrodinger.wfn.normalize()
        # schrodinger.save_params()

        objective.f_updated = False
        objective.g_updated = False
        objective.H_updated = False
        objective._update_hess()

    objective._update_x_impl = update_x

    # Put constraints in list format when needed
    if isinstance(constraints, (NonlinearConstraint, LinearConstraint)):
        constraints = [constraints]

    # XXXX: add constraints on energy
    if schrodinger.energy_bounds != (-np.inf, np.inf):
        constraints.append(
            scipy.optimize.NonlinearConstraint(
                objective.fun, schrodinger.energy_bounds[0], schrodinger.energy_bounds[1],
                jac=lambda x: objective.grad(x).reshape(1, -1),
                hess=objective.H
            )
        )

    # Prepare constraints.
    prepared_constraints = [
        PreparedConstraint(c, x0, sparse_jacobian, finite_diff_bounds)
        for c in constraints]

    # Check that all constraints are either sparse or dense.
    n_sparse = sum(c.fun.sparse_jacobian for c in prepared_constraints)
    if 0 < n_sparse < len(prepared_constraints):
        raise ValueError("All constraints must have the same kind of the "
                         "Jacobian --- either all sparse or all dense. "
                         "You can set the sparsity globally by setting "
                         "`sparse_jacobian` to either True of False.")
    if prepared_constraints:
        sparse_jacobian = n_sparse > 0

    if bounds is not None:
        if sparse_jacobian is None:
            sparse_jacobian = True
        prepared_constraints.append(PreparedConstraint(bounds, x0,
                                                       sparse_jacobian))

    # Concatenate initial constraints to the canonical form.
    c_eq0, c_ineq0, J_eq0, J_ineq0 = initial_constraints_as_canonical(
        n_vars, prepared_constraints, sparse_jacobian)

    # Prepare all canonical constraints and concatenate it into one.
    canonical_all = [CanonicalConstraint.from_PreparedConstraint(c)
                     for c in prepared_constraints]

    if len(canonical_all) == 0:
        canonical = CanonicalConstraint.empty(n_vars)
    elif len(canonical_all) == 1:
        canonical = canonical_all[0]
    else:
        canonical = CanonicalConstraint.concatenate(canonical_all,
                                                    sparse_jacobian)

    # Generate the Hessian of the Lagrangian.
    lagrangian_hess = LagrangianHessian(n_vars, objective.hess, canonical.hess)

    # Choose appropriate method
    if canonical.n_ineq == 0:
        method = 'equality_constrained_sqp'
    else:
        method = 'tr_interior_point'

    # Construct OptimizeResult
    state = OptimizeResult(
        nit=0, nfev=0, njev=0, nhev=0,
        cg_niter=0, cg_stop_cond=0,
        fun=objective.f, grad=objective.g,
        lagrangian_grad=np.copy(objective.g),
        constr=[c.fun.f for c in prepared_constraints],
        jac=[c.fun.J for c in prepared_constraints],
        constr_nfev=[0 for c in prepared_constraints],
        constr_njev=[0 for c in prepared_constraints],
        constr_nhev=[0 for c in prepared_constraints],
        v=[c.fun.v for c in prepared_constraints],
        method=method)

    # Start counting
    start_time = time.time()

    # Define stop criteria
    if method == 'equality_constrained_sqp':
        def stop_criteria(state, x, last_iteration_failed,
                          optimality, constr_violation,
                          tr_radius, constr_penalty, cg_info):
            state = update_state_sqp(state, x, last_iteration_failed,
                                     objective, prepared_constraints,
                                     start_time, tr_radius, constr_penalty,
                                     cg_info)
            if verbose == 2:
                BasicReport.print_iteration(state.nit,
                                            state.nfev,
                                            state.cg_niter,
                                            state.fun,
                                            state.tr_radius,
                                            state.optimality,
                                            state.constr_violation)
            elif verbose > 2:
                SQPReport.print_iteration(state.nit,
                                          state.nfev,
                                          state.cg_niter,
                                          state.fun,
                                          state.tr_radius,
                                          state.optimality,
                                          state.constr_violation,
                                          state.constr_penalty,
                                          state.cg_stop_cond)
            state.status = None
            state.niter = state.nit  # Alias for callback (backward-compatibility)
            if callback is not None and callback(np.copy(state.x), state):
                state.status = 3
            elif state.optimality < gtol and state.constr_violation < gtol:
                state.status = 1
            elif state.tr_radius < xtol:
                state.status = 2
            elif state.nit > maxiter:
                state.status = 0
            return state.status in (0, 1, 2, 3)
    elif method == 'tr_interior_point':
        def stop_criteria(state, x, last_iteration_failed, tr_radius,
                          constr_penalty, cg_info, barrier_parameter,
                          barrier_tolerance):
            state = update_state_ip(state, x, last_iteration_failed,
                                    objective, prepared_constraints,
                                    start_time, tr_radius, constr_penalty,
                                    cg_info, barrier_parameter, barrier_tolerance)
            if verbose == 2:
                BasicReport.print_iteration(state.nit,
                                            state.nfev,
                                            state.cg_niter,
                                            state.fun,
                                            state.tr_radius,
                                            state.optimality,
                                            state.constr_violation)
            elif verbose > 2:
                IPReport.print_iteration(state.nit,
                                         state.nfev,
                                         state.cg_niter,
                                         state.fun,
                                         state.tr_radius,
                                         state.optimality,
                                         state.constr_violation,
                                         state.constr_penalty,
                                         state.barrier_parameter,
                                         state.cg_stop_cond)
            state.status = None
            state.niter = state.nit  # Alias for callback (backward-compatibility)
            if callback is not None and callback(np.copy(state.x), state):
                state.status = 3
            elif state.optimality < gtol and state.constr_violation < gtol:
                state.status = 1
            elif (state.tr_radius < xtol
                  and state.barrier_parameter < barrier_tol):
                state.status = 2
            elif state.nit > maxiter:
                state.status = 0
            return state.status in (0, 1, 2, 3)

    if verbose == 2:
        BasicReport.print_header()
    elif verbose > 2:
        if method == 'equality_constrained_sqp':
            SQPReport.print_header()
        elif method == 'tr_interior_point':
            IPReport.print_header()

    # Call inferior function to do the optimization
    if method == 'equality_constrained_sqp':
        def fun_and_constr(x):
            f = objective.fun(x)
            c_eq, _ = canonical.fun(x)
            return f, c_eq

        def grad_and_jac(x):
            g = objective.grad(x)
            J_eq, _ = canonical.jac(x)
            return g, J_eq

        _, result = equality_constrained_sqp(
            fun_and_constr, grad_and_jac, lagrangian_hess,
            x0, objective.f, objective.g,
            c_eq0, J_eq0,
            stop_criteria, state,
            initial_constr_penalty, initial_tr_radius,
            factorization_method, schrodinger=schrodinger)

    elif method == 'tr_interior_point':
        _, result = tr_interior_point(
            objective.fun, objective.grad, lagrangian_hess,
            n_vars, canonical.n_ineq, canonical.n_eq,
            canonical.fun, canonical.jac,
            x0, objective.f, objective.g,
            c_ineq0, J_ineq0, c_eq0, J_eq0,
            stop_criteria,
            canonical.keep_feasible,
            xtol, state, initial_barrier_parameter,
            initial_barrier_tolerance,
            initial_constr_penalty, initial_tr_radius,
            factorization_method, schrodinger, objective, prepared_constraints)

    # Status 3 occurs when the callback function requests termination,
    # this is assumed to not be a success.
    result.success = True if result.status in (1, 2) else False
    result.message = TERMINATION_MESSAGES[result.status]

    # Alias (for backward compatibility with 1.1.0)
    result.niter = result.nit

    if verbose == 2:
        BasicReport.print_footer()
    elif verbose > 2:
        if method == 'equality_constrained_sqp':
            SQPReport.print_footer()
        elif method == 'tr_interior_point':
            IPReport.print_footer()
    if verbose >= 1:
        print(result.message)
        print("Number of iterations: {}, function evaluations: {}, "
              "CG iterations: {}, optimality: {:.2e}, "
              "constraint violation: {:.2e}, execution time: {:4.2} s."
              .format(result.nit, result.nfev, result.cg_niter,
                      result.optimality, result.constr_violation,
                      result.execution_time))
    return result


def minimize_rewritten(fun, x0, args=(), method=None, jac=None, hess=None,
             hessp=None, bounds=None, constraints=(), tol=None,
             callback=None, options=None):
    """Minimization of scalar function of one or more variables.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where x is an 1-D array with shape (n,) and `args`
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
    method : str or callable, optional
        Type of solver.  Should be one of

            - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
            - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
            - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
            - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
            - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
            - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
            - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
            - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
            - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
            - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
            - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
            - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
            - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
            - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
            - custom - a callable object (added in version 0.14.0),
              see below for description.

        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
        depending if the problem has constraints or bounds.
    jac : {callable,  '2-point', '3-point', 'cs', bool}, optional
        Method for computing the gradient vector. Only for CG, BFGS,
        Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov,
        trust-exact and trust-constr. If it is a callable, it should be a
        function that returns the gradient vector:

            ``jac(x, *args) -> array_like, shape (n,)``

        where x is an array with shape (n,) and `args` is a tuple with
        the fixed parameters. Alternatively, the keywords
        {'2-point', '3-point', 'cs'} select a finite
        difference scheme for numerical estimation of the gradient. Options
        '3-point' and 'cs' are available only to 'trust-constr'.
        If `jac` is a Boolean and is True, `fun` is assumed to return the
        gradient along with the objective function. If False, the gradient
        will be estimated using '2-point' finite difference estimation.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy},  optional
        Method for computing the Hessian matrix. Only for Newton-CG, dogleg,
        trust-ncg,  trust-krylov, trust-exact and trust-constr. If it is
        callable, it should return the  Hessian matrix:

            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``

        where x is a (n,) ndarray and `args` is a tuple with the fixed
        parameters. LinearOperator and sparse matrix returns are
        allowed only for 'trust-constr' method. Alternatively, the keywords
        {'2-point', '3-point', 'cs'} select a finite difference scheme
        for numerical estimation. Or, objects implementing
        `HessianUpdateStrategy` interface can be used to approximate
        the Hessian. Available quasi-Newton methods implementing
        this interface are:

            - `BFGS`;
            - `SR1`.

        Whenever the gradient is estimated via finite-differences,
        the Hessian cannot be estimated with options
        {'2-point', '3-point', 'cs'} and needs to be
        estimated using one of the quasi-Newton strategies.
        Finite-difference options {'2-point', '3-point', 'cs'} and
        `HessianUpdateStrategy` are available only for 'trust-constr' method.
    hessp : callable, optional
        Hessian of objective function times an arbitrary vector p. Only for
        Newton-CG, trust-ncg, trust-krylov, trust-constr.
        Only one of `hessp` or `hess` needs to be given.  If `hess` is
        provided, then `hessp` will be ignored.  `hessp` must compute the
        Hessian times an arbitrary vector:

            ``hessp(x, p, *args) ->  ndarray shape (n,)``

        where x is a (n,) ndarray, p is an arbitrary vector with
        dimension (n,) and `args` is a tuple with the fixed
        parameters.
    bounds : sequence or `Bounds`, optional
        Bounds on variables for L-BFGS-B, TNC, SLSQP and
        trust-constr methods. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.

    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition (only for COBYLA, SLSQP and trust-constr).
        Constraints for 'trust-constr' are defined as a single object or a
        list of objects specifying constraints to the optimization problem.
        Available constraints are:

            - `LinearConstraint`
            - `NonlinearConstraint`

        Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
        Each dictionary with fields:

            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of `fun` (only for SLSQP).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.

        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be non-negative.
        Note that COBYLA only supports inequality constraints.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform. Depending on the
                method each iteration may use several function evaluations.
            disp : bool
                Set to True to print convergence messages.

        For method-specific options, see :func:`show_options()`.
    callback : callable, optional
        Called after each iteration. For 'trust-constr' it is a callable with
        the signature:

            ``callback(xk, OptimizeResult state) -> bool``

        where ``xk`` is the current parameter vector. and ``state``
        is an `OptimizeResult` object, with the same fields
        as the ones from the return.  If callback returns True
        the algorithm execution is terminated.
        For all the other methods, the signature is:

            ``callback(xk)``

        where ``xk`` is the current parameter vector.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    See also
    --------
    minimize_scalar : Interface to minimization algorithms for scalar
        univariate functions
    show_options : Additional options accepted by the solvers

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is *BFGS*.

    **Unconstrained minimization**

    Method :ref:`Nelder-Mead <optimize.minimize-neldermead>` uses the
    Simplex algorithm [1]_, [2]_. This algorithm is robust in many
    applications. However, if numerical computation of derivative can be
    trusted, other algorithms using the first and/or second derivatives
    information might be preferred for their better performance in
    general.

    Method :ref:`Powell <optimize.minimize-powell>` is a modification
    of Powell's method [3]_, [4]_ which is a conjugate direction
    method. It performs sequential one-dimensional minimizations along
    each vector of the directions set (`direc` field in `options` and
    `info`), which is updated at each iteration of the main
    minimization loop. The function need not be differentiable, and no
    derivatives are taken.

    Method :ref:`CG <optimize.minimize-cg>` uses a nonlinear conjugate
    gradient algorithm by Polak and Ribiere, a variant of the
    Fletcher-Reeves method described in [5]_ pp.  120-122. Only the
    first derivatives are used.

    Method :ref:`BFGS <optimize.minimize-bfgs>` uses the quasi-Newton
    method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS) [5]_
    pp. 136. It uses the first derivatives only. BFGS has proven good
    performance even for non-smooth optimizations. This method also
    returns an approximation of the Hessian inverse, stored as
    `hess_inv` in the OptimizeResult object.

    Method :ref:`Newton-CG <optimize.minimize-newtoncg>` uses a
    Newton-CG algorithm [5]_ pp. 168 (also known as the truncated
    Newton method). It uses a CG method to the compute the search
    direction. See also *TNC* method for a box-constrained
    minimization with a similar algorithm. Suitable for large-scale
    problems.

    Method :ref:`dogleg <optimize.minimize-dogleg>` uses the dog-leg
    trust-region algorithm [5]_ for unconstrained minimization. This
    algorithm requires the gradient and Hessian; furthermore the
    Hessian is required to be positive definite.

    Method :ref:`trust-ncg <optimize.minimize-trustncg>` uses the
    Newton conjugate gradient trust-region algorithm [5]_ for
    unconstrained minimization. This algorithm requires the gradient
    and either the Hessian or a function that computes the product of
    the Hessian with a given vector. Suitable for large-scale problems.

    Method :ref:`trust-krylov <optimize.minimize-trustkrylov>` uses
    the Newton GLTR trust-region algorithm [14]_, [15]_ for unconstrained
    minimization. This algorithm requires the gradient
    and either the Hessian or a function that computes the product of
    the Hessian with a given vector. Suitable for large-scale problems.
    On indefinite problems it requires usually less iterations than the
    `trust-ncg` method and is recommended for medium and large-scale problems.

    Method :ref:`trust-exact <optimize.minimize-trustexact>`
    is a trust-region method for unconstrained minimization in which
    quadratic subproblems are solved almost exactly [13]_. This
    algorithm requires the gradient and the Hessian (which is
    *not* required to be positive definite). It is, in many
    situations, the Newton method to converge in fewer iteraction
    and the most recommended for small and medium-size problems.

    **Bound-Constrained minimization**

    Method :ref:`L-BFGS-B <optimize.minimize-lbfgsb>` uses the L-BFGS-B
    algorithm [6]_, [7]_ for bound constrained minimization.

    Method :ref:`TNC <optimize.minimize-tnc>` uses a truncated Newton
    algorithm [5]_, [8]_ to minimize a function with variables subject
    to bounds. This algorithm uses gradient information; it is also
    called Newton Conjugate-Gradient. It differs from the *Newton-CG*
    method described above as it wraps a C implementation and allows
    each variable to be given upper and lower bounds.

    **Constrained Minimization**

    Method :ref:`COBYLA <optimize.minimize-cobyla>` uses the
    Constrained Optimization BY Linear Approximation (COBYLA) method
    [9]_, [10]_, [11]_. The algorithm is based on linear
    approximations to the objective function and each constraint. The
    method wraps a FORTRAN implementation of the algorithm. The
    constraints functions 'fun' may return either a single number
    or an array or list of numbers.

    Method :ref:`SLSQP <optimize.minimize-slsqp>` uses Sequential
    Least SQuares Programming to minimize a function of several
    variables with any combination of bounds, equality and inequality
    constraints. The method wraps the SLSQP Optimization subroutine
    originally implemented by Dieter Kraft [12]_. Note that the
    wrapper handles infinite values in bounds by converting them into
    large floating values.

    Method :ref:`trust-constr <optimize.minimize-trustconstr>` is a
    trust-region algorithm for constrained optimization. It swiches
    between two implementations depending on the problem definition.
    It is the most versatile constrained minimization algorithm
    implemented in SciPy and the most appropriate for large-scale problems.
    For equality constrained problems it is an implementation of Byrd-Omojokun
    Trust-Region SQP method described in [17]_ and in [5]_, p. 549. When
    inequality constraints  are imposed as well, it swiches to the trust-region
    interior point  method described in [16]_. This interior point algorithm,
    in turn, solves inequality constraints by introducing slack variables
    and solving a sequence of equality-constrained barrier problems
    for progressively smaller values of the barrier parameter.
    The previously described equality constrained SQP method is
    used to solve the subproblems with increasing levels of accuracy
    as the iterate gets closer to a solution.

    **Finite-Difference Options**

    For Method :ref:`trust-constr <optimize.minimize-trustconstr>`
    the gradient and the Hessian may be approximated using
    three finite-difference schemes: {'2-point', '3-point', 'cs'}.
    The scheme 'cs' is, potentially, the most accurate but it
    requires the function to correctly handles complex inputs and to
    be differentiable in the complex plane. The scheme '3-point' is more
    accurate than '2-point' but requires twice as much operations.

    **Custom minimizers**

    It may be useful to pass a custom minimization method, for example
    when using a frontend to this method such as `scipy.optimize.basinhopping`
    or a different library.  You can simply pass a callable as the ``method``
    parameter.

    The callable is called as ``method(fun, x0, args, **kwargs, **options)``
    where ``kwargs`` corresponds to any other parameters passed to `minimize`
    (such as `callback`, `hess`, etc.), except the `options` dict, which has
    its contents also passed as `method` parameters pair by pair.  Also, if
    `jac` has been passed as a bool type, `jac` and `fun` are mangled so that
    `fun` returns just the function values and `jac` is converted to a function
    returning the Jacobian.  The method shall return an `OptimizeResult`
    object.

    The provided `method` callable must be able to accept (and possibly ignore)
    arbitrary parameters; the set of parameters accepted by `minimize` may
    expand in future versions and then these parameters will be passed to
    the method.  You can find an example in the scipy.optimize tutorial.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] Nelder, J A, and R Mead. 1965. A Simplex Method for Function
        Minimization. The Computer Journal 7: 308-13.
    .. [2] Wright M H. 1996. Direct search methods: Once scorned, now
        respectable, in Numerical Analysis 1995: Proceedings of the 1995
        Dundee Biennial Conference in Numerical Analysis (Eds. D F
        Griffiths and G A Watson). Addison Wesley Longman, Harlow, UK.
        191-208.
    .. [3] Powell, M J D. 1964. An efficient method for finding the minimum of
       a function of several variables without calculating derivatives. The
       Computer Journal 7: 155-162.
    .. [4] Press W, S A Teukolsky, W T Vetterling and B P Flannery.
       Numerical Recipes (any edition), Cambridge University Press.
    .. [5] Nocedal, J, and S J Wright. 2006. Numerical Optimization.
       Springer New York.
    .. [6] Byrd, R H and P Lu and J. Nocedal. 1995. A Limited Memory
       Algorithm for Bound Constrained Optimization. SIAM Journal on
       Scientific and Statistical Computing 16 (5): 1190-1208.
    .. [7] Zhu, C and R H Byrd and J Nocedal. 1997. L-BFGS-B: Algorithm
       778: L-BFGS-B, FORTRAN routines for large scale bound constrained
       optimization. ACM Transactions on Mathematical Software 23 (4):
       550-560.
    .. [8] Nash, S G. Newton-Type Minimization Via the Lanczos Method.
       1984. SIAM Journal of Numerical Analysis 21: 770-778.
    .. [9] Powell, M J D. A direct search optimization method that models
       the objective and constraint functions by linear interpolation.
       1994. Advances in Optimization and Numerical Analysis, eds. S. Gomez
       and J-P Hennart, Kluwer Academic (Dordrecht), 51-67.
    .. [10] Powell M J D. Direct search algorithms for optimization
       calculations. 1998. Acta Numerica 7: 287-336.
    .. [11] Powell M J D. A view of algorithms for optimization without
       derivatives. 2007.Cambridge University Technical Report DAMTP
       2007/NA03
    .. [12] Kraft, D. A software package for sequential quadratic
       programming. 1988. Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace
       Center -- Institute for Flight Mechanics, Koln, Germany.
    .. [13] Conn, A. R., Gould, N. I., and Toint, P. L.
       Trust region methods. 2000. Siam. pp. 169-200.
    .. [14] F. Lenders, C. Kirches, A. Potschka: "trlib: A vector-free
       implementation of the GLTR method for iterative solution of
       the trust region problem", https://arxiv.org/abs/1611.04718
    .. [15] N. Gould, S. Lucidi, M. Roma, P. Toint: "Solving the
       Trust-Region Subproblem using the Lanczos Method",
       SIAM J. Optim., 9(2), 504--525, (1999).
    .. [16] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal. 1999.
        An interior point algorithm for large-scale nonlinear  programming.
        SIAM Journal on Optimization 9.4: 877-900.
    .. [17] Lalee, Marucha, Jorge Nocedal, and Todd Plantega. 1998. On the
        implementation of an algorithm for large-scale equality constrained
        optimization. SIAM Journal on Optimization 8.3: 682-706.

    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function. This
    function (and its respective derivatives) is implemented in `rosen`
    (resp. `rosen_der`, `rosen_hess`) in the `scipy.optimize`.

    >>> from scipy.optimize import minimize, rosen, rosen_der

    A simple application of the *Nelder-Mead* method is:

    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    >>> res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
    >>> res.x
    array([ 1.,  1.,  1.,  1.,  1.])

    Now using the *BFGS* algorithm, using the first derivative and a few
    options:

    >>> res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
    ...                options={'gtol': 1e-6, 'disp': True})
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 26
             Function evaluations: 31
             Gradient evaluations: 31
    >>> res.x
    array([ 1.,  1.,  1.,  1.,  1.])
    >>> print(res.message)
    Optimization terminated successfully.
    >>> res.hess_inv
    array([[ 0.00749589,  0.01255155,  0.02396251,  0.04750988,  0.09495377],  # may vary
           [ 0.01255155,  0.02510441,  0.04794055,  0.09502834,  0.18996269],
           [ 0.02396251,  0.04794055,  0.09631614,  0.19092151,  0.38165151],
           [ 0.04750988,  0.09502834,  0.19092151,  0.38341252,  0.7664427 ],
           [ 0.09495377,  0.18996269,  0.38165151,  0.7664427,   1.53713523]])


    Next, consider a minimization problem with several constraints (namely
    Example 16.4 from [5]_). The objective function is:

    >>> fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2

    There are three constraints defined as:

    >>> cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
    ...         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
    ...         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

    And variables must be positive, hence the following bounds:

    >>> bnds = ((0, None), (0, None))

    The optimization problem is solved using the SLSQP method as:

    >>> res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
    ...                constraints=cons)

    It should converge to the theoretical solution (1.4 ,1.7).

    """
    x0 = np.asarray(x0)
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)

    if not isinstance(args, tuple):
        args = (args,)

    if method is None:
        # Select automatically
        if constraints:
            method = 'SLSQP'
        elif bounds is not None:
            method = 'L-BFGS-B'
        else:
            method = 'BFGS'

    if callable(method):
        meth = "_custom"
    else:
        meth = method.lower()

    if options is None:
        options = {}
    # check if optional parameters are supported by the selected method
    # - jac
    if meth in ('nelder-mead', 'powell', 'cobyla') and bool(jac):
        warn('Method %s does not use gradient information (jac).' % method,
             RuntimeWarning)
    # - hess
    if meth not in ('newton-cg', 'dogleg', 'trust-ncg', 'trust-constr',
                    'trust-krylov', 'trust-exact', '_custom') and hess is not None:
        warn('Method %s does not use Hessian information (hess).' % method,
             RuntimeWarning)
    # - hessp
    if meth not in ('newton-cg', 'dogleg', 'trust-ncg', 'trust-constr',
                    'trust-krylov', '_custom') \
       and hessp is not None:
        warn('Method %s does not use Hessian-vector product '
             'information (hessp).' % method, RuntimeWarning)
    # - constraints or bounds
    if (meth in ('nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'dogleg',
                 'trust-ncg') and (bounds is not None or np.any(constraints))):
        warn('Method %s cannot handle constraints nor bounds.' % method,
             RuntimeWarning)
    if meth in ('l-bfgs-b', 'tnc') and np.any(constraints):
        warn('Method %s cannot handle constraints.' % method,
             RuntimeWarning)
    if meth == 'cobyla' and bounds is not None:
        warn('Method %s cannot handle bounds.' % method,
             RuntimeWarning)
    # - callback
    if (meth in ('cobyla',) and callback is not None):
        warn('Method %s does not support callback.' % method, RuntimeWarning)
    # - return_all
    if (meth in ('l-bfgs-b', 'tnc', 'cobyla', 'slsqp') and
            options.get('return_all', False)):
        warn('Method %s does not support the return_all option.' % method,
             RuntimeWarning)

    # check gradient vector
    if meth == 'trust-constr':
        if type(jac) is bool:
            if jac:
                fun = MemoizeJac(fun)
                jac = fun.derivative
            else:
                jac = '2-point'
        elif jac is None:
            jac = '2-point'
        elif not callable(jac) and jac not in ('2-point', '3-point', 'cs'):
            raise ValueError("Unsupported jac definition.")
    else:
        if jac in ('2-point', '3-point', 'cs'):
            if jac in ('3-point', 'cs'):
                warn("Only 'trust-constr' method accept %s "
                     "options for 'jac'. Using '2-point' instead." % jac)
            jac = None
        elif not callable(jac):
            if bool(jac):
                fun = MemoizeJac(fun)
                jac = fun.derivative
            else:
                jac = None

    # set default tolerances
    if tol is not None:
        options = dict(options)
        if meth == 'nelder-mead':
            options.setdefault('xatol', tol)
            options.setdefault('fatol', tol)
        if meth in ('newton-cg', 'powell', 'tnc'):
            options.setdefault('xtol', tol)
        if meth in ('powell', 'l-bfgs-b', 'tnc', 'slsqp'):
            options.setdefault('ftol', tol)
        if meth in ('bfgs', 'cg', 'l-bfgs-b', 'tnc', 'dogleg',
                    'trust-ncg', 'trust-exact', 'trust-krylov'):
            options.setdefault('gtol', tol)
        if meth in ('cobyla', '_custom'):
            options.setdefault('tol', tol)
        if meth == 'trust-constr':
            options.setdefault('xtol', tol)
            options.setdefault('gtol', tol)
            options.setdefault('barrier_tol', tol)

    if meth == '_custom':
        # custom method called before bounds and constraints are 'standardised'
        # custom method should be able to accept whatever bounds/constraints
        # are provided to it.
        return method(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
                      bounds=bounds, constraints=constraints,
                      callback=callback, **options)

    if bounds is not None:
        bounds = standardize_bounds(bounds, x0, meth)

    if constraints is not None:
        constraints = standardize_constraints(constraints, x0, meth)

    if meth == 'trust-constr':
        return _minimize_trustregion_constr(fun, x0, args, jac, hess, hessp,
                                            bounds, constraints,
                                            callback=callback, **options)
    else:
        raise ValueError('Unknown solver %s' % method)


class EnergyOneSideProjectionSystem(EnergyOneSideProjection):
    def __init__(self, objective):
        super().__init__(
            objective.wfn, objective.ham, tmpfile=objective.tmpfile,
            param_selection=objective.indices_component_params, refwfn=objective.refwfn
        )
        self._objective = objective
        self.refwfn = objective.refwfn

    @property
    def refwfn(self):
        return self._refwfn

    @refwfn.setter
    def refwfn(self, val):
        self._refwfn = val
        if isinstance(self._objective, list):
            for obj in self._objective:
                obj.refwfn = val
        self._objective.refwfn = val

    def assign_refwfn(self, refwfn=None):
        pass

    def objective(self, params, parallel=None):
        return super().objective(
            params, assign=False, normalize=False, save=False
        )

    def gradient(self, params, parallel=None):
        return super().gradient(
            params, assign=False, normalize=False, save=False
        )

    def adapt_pspace_energy(self):
        probable_sds, olps = zip(*self.wfn.probable_sds.items())

        print('Adapt energy pspace')
        self.refwfn = sorted(
            self.wfn.probable_sds, key=lambda sd: abs(self.wfn.probable_sds[sd]), reverse=True
        )[:self.sample_size]


class ProjectedSchrodingerSystem(ProjectedSchrodinger):
    def __init__(self, objective, norm_constraint=True):
        super().__init__(
            objective.wfn,
            objective.ham,
            tmpfile=objective.tmpfile,
            param_selection=objective.indices_component_params,
            pspace=objective.pspace,
            refwfn=objective.refwfn,
            eqn_weights=objective.eqn_weights,
            energy_type=objective.energy_type,
            energy=objective.energy.params[0],
            constraints=[],
        )
        if objective.constraints:
            raise ValueError(
                'No constraints allowed'
            )
        if norm_constraint:
            self.constraints = [NormConstraintSystem(objective)]
            self.eqn_weights = np.hstack([self.eqn_weights, 1])

    def objective(self, params):
        return super().objective(
            # params, return_energy=False, assign=False, normalize=False, save=False
            params, return_energy=False, assign=False, normalize=False, save=False
        )

    def jacobian(self, params):
        return super().jacobian(
            params, return_d_energy=False, assign=False, normalize=False, save=False
        )

    def adapt_pspace(self):
        probable_sds, olps = zip(*self.wfn.probable_sds.items())
        prob = np.array(olps) ** 2
        prob /= np.sum(prob)

        # repetition
        # pspace = np.random.choice(probable_sds, size=self.sample_size, p=prob, replace=True)
        # pspace_count = Counter(pspace)
        # weights = []
        # pspace = []
        # total_count = sum(pspace_count.values())
        # for sd, count in pspace_count.items():
        #     pspace.append(sd)
        #     weights.append(count / total_count)
        # weights = np.array(weights)

        # no repetition
        pspace = np.random.choice(
            probable_sds, size=min(self.sample_size, len(probable_sds)), p=prob, replace=False
        )
        weights = np.array([self.wfn.probable_sds[sd]**2 for sd in pspace])
        weights /= np.sum(weights)

        if hasattr(self, 'weight_type'):
            if self.weight_type == 'ones':
                weights = np.ones(len(pspace))

        print('Adapt pspace')
        print(len(pspace), len(probable_sds), max(olps), 'pspace')
        if len(pspace) < self.sample_size:
            print("Not large enough pspace. Duplicating some functions")
            n = self.sample_size // len(pspace)
            indices = np.zeros(len(pspace), dtype=bool)
            indices[:self.sample_size - n * len(pspace)] = True
            indices = np.tile(indices, n + 1)[:self.sample_size]

            pspace = np.tile(pspace, n + 1)[:self.sample_size]
            weights = np.tile(weights, n + 1)[:self.sample_size]
            weights[indices] /= (n + 1)
            weights[~indices] /= n

        self.pspace = list(pspace)
        if self.constraints:
            self.eqn_weights = np.hstack([weights, self.eqn_weights[-len(self.constraints):]])
        else:
            self.eqn_weights = np.array(weights)

    def save_params(self):
        super().save_params()
        if self.tmpfile != "":
            # np.save(self.tmpfile, self.param_selection.all_params)
            header, ext = os.path.splitext(self.tmpfile)
            try:
                np.save(header + '_pspace' + ext, self.pspace)
            except (OSError, AttributeError):
                pass
            try:
                np.save(header + '_refwfn' + ext, self.refwfn)
            except (OSError, AttributeError):
                pass
            try:
                np.save(header + '_pspace_norm' + ext, self.wfn.pspace_norm)
            except (OSError, AttributeError):
                pass
            try:
                np.save(header + '_eqn_weights' + ext, self.eqn_weights)
            except (OSError, AttributeError):
                pass


class NormConstraintSystem(NormConstraint):
    def __init__(self, objective):
        self.wfn = objective.wfn
        self.indices_component_params = objective.indices_component_params
        self.tmpfile = ''

    @property
    def refwfn(self):
        return list(self.wfn.pspace_norm)

    def objective(self, params):
        get_overlap = self.wrapped_get_overlap
        overlaps = np.array([get_overlap(i) for i in self.refwfn])
        print(np.sort(np.abs(overlaps))[:-5:-1], len(self.refwfn), len(self.wfn.probable_sds), 'norm')
        return np.sum(overlaps ** 2) - 1

    def gradient(self, params):
        get_overlap = self.wrapped_get_overlap
        overlaps = np.array([get_overlap(i) for i in self.refwfn])
        d_overlaps = np.array([get_overlap(sd, True) for sd in self.refwfn])
        d_norm = 2 * np.sum(overlaps[:, None] * d_overlaps, axis=0)
        return d_norm


def minimize(
        objective, norm_constraint=True, constraint_bounds=(-0.1, 0.1), energy_bounds=(-np.inf, np.inf), energy_bound=-np.inf,
        save_file="", **kwargs
):
    objective, old_objective = ProjectedSchrodingerSystem(
        objective, norm_constraint=norm_constraint
    ), objective
    objective.assign_params(old_objective.active_params)

    objective.step_print = True
    objective.ham.update_prev_params = False
    if objective.wfn.olp_threshold >= 1:
        objective.wfn.olp_threshold = 0.01
    try:
        objective.weight_type = old_objective.weight_type
    except AttributeError:
        objective.weight_type = 'ones'
    try:
        objective.sample_size = old_objective.sample_size
    except AttributeError:
        objective.sample_size = len(objective.pspace)
    try:
        objective.adapt_type = old_objective.adapt_type
    except AttributeError:
        # objective.adapt_type = ['pspace', 'norm', 'energy']
        objective.adapt_type = []
    try:
        objective.wfn.pspace_norm = old_objective.wfn.pspace_norm
    except AttributeError:
        objective.wfn.pspace_norm = set(objective.refwfn)

    energy = EnergyOneSideProjectionSystem(objective)
    objective.energy_objective = energy
    energy.sample_size = objective.sample_size

    if old_objective.constraints and not (
        len(old_objective.constraints) == 1 and
        isinstance(old_objective.constraints[0], NormConstraint)
    ):
        raise ValueError(
            'Only norm constraint is upported (and there cannot be more than one norm '
            'constraint)'
        )

    if objective.sample_size != len(objective.pspace):
        raise ValueError('Starting pspace must have the same size as the sample size.')

    objective.wfn.normalize(objective.refwfn)

    kwargs["method"] = "trust-constr"
    kwargs["jac"] = energy.gradient
    kwargs.setdefault("options", {})
    kwargs["options"].setdefault('gtol', 1e-8)
    kwargs["options"].setdefault('xtol', 1e-10)
    kwargs["options"].setdefault('maxiter', 1000)
    kwargs["options"].setdefault('disp', 2)
    kwargs["options"].setdefault('verbose', 3)
    kwargs["options"]['factorization_method'] = 'SVDFactorization'
    kwargs["options"]['schrodinger'] = objective
    # kwargs.setdefault("bounds", ((-2, 2) for _ in range(objective.params.size)))
    constraints = []
    # if energy_bound != -np.inf:
    #     constraints.append(
    #         scipy.optimize.NonlinearConstraint(
    #             energy.objective, energy_bound, 0,
    #             jac=lambda x: energy.gradient(x).reshape(1, -1),
    #         )
    #     )

    lb, ub = constraint_bounds
    lb = np.ones(objective.num_eqns) * constraint_bounds[0]
    ub = np.ones(objective.num_eqns) * constraint_bounds[1]
    if objective.constraints:
        lb[-1] = -1e-5
        ub[-1] = 1e-5

    constraints.append(
        scipy.optimize.NonlinearConstraint(objective.objective, lb, ub, jac=objective.jacobian)
    )
    if energy_bounds == (-np.inf, np.inf) and energy_bound != -np.inf:
        energy_bounds = (energy_bound, 0)
    objective.energy_bounds = energy_bounds

    # kwargs = {"method": "COBYLA", "jac": objective.gradient}
    # kwargs.setdefault("options", {"gtol": 1e-8})
    # energy = EnergyOneSideProjection(
    #     objective.wfn, objective.ham, tmpfile=objective.tmpfile,
    #     param_selection=objective.param_selection, refwfn=objective.pspace
    # )
    # lb, ub = constraint_bounds
    # constraints = scipy.optimize.NonlinearConstraint(
    #     objective.objective, lb, ub, jac=objective.jacobian
    # )

    output = wrap_scipy(minimize_rewritten)(
        energy, constraints=constraints, **kwargs
    )
    output["function"] = output["internal"].fun
    output["energy"] = output["function"]

    return output
