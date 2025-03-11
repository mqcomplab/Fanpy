"""Generic interface for least-squares minimization."""

from warnings import warn

import numpy as np
from numpy.linalg import norm

from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import _minpack, OptimizeResult
from scipy.optimize._numdiff import approx_derivative, group_columns

from scipy.optimize._lsq.dogbox import dogbox
from scipy.optimize._lsq.common import EPS, in_bounds, make_strictly_feasible


from scipy.linalg import svd, qr
from scipy.sparse.linalg import lsmr

from scipy.optimize._lsq.common import (
    step_size_to_bound,
    find_active_constraints,
    in_bounds,
    make_strictly_feasible,
    intersect_trust_region,
    solve_lsq_trust_region,
    solve_trust_region_2d,
    minimize_quadratic_1d,
    build_quadratic_1d,
    evaluate_quadratic,
    right_multiplied_operator,
    regularized_lsq_operator,
    CL_scaling_vector,
    compute_grad,
    compute_jac_scale,
    check_termination,
    update_tr_radius,
    scale_for_robust_loss_function,
    print_header_nonlinear,
    print_iteration_nonlinear,
)


TERMINATION_MESSAGES = {
    -1: "Improper input parameters status returned from `leastsq`",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied.",
}


FROM_MINPACK_TO_COMMON = {
    0: -1,  # Improper input parameters from MINPACK.
    1: 2,
    2: 3,
    3: 4,
    4: 1,
    5: 0,
    # There are 6, 7, 8 for too small tolerance parameters,
    # but we guard against it by checking ftol, xtol, gtol beforehand.
}


def call_minpack(fun, x0, jac, ftol, xtol, gtol, max_nfev, x_scale, diff_step):
    n = x0.size

    if diff_step is None:
        epsfcn = EPS
    else:
        epsfcn = diff_step**2

    # Compute MINPACK's `diag`, which is inverse of our `x_scale` and
    # ``x_scale='jac'`` corresponds to ``diag=None``.
    if isinstance(x_scale, str) and x_scale == "jac":
        diag = None
    else:
        diag = 1 / x_scale

    full_output = True
    col_deriv = False
    factor = 100.0

    if jac is None:
        if max_nfev is None:
            # n squared to account for Jacobian evaluations.
            max_nfev = 100 * n * (n + 1)
        x, info, status = _minpack._lmdif(fun, x0, (), full_output, ftol, xtol, gtol, max_nfev, epsfcn, factor, diag)
    else:
        if max_nfev is None:
            max_nfev = 100 * n
        x, info, status = _minpack._lmder(
            fun, jac, x0, (), full_output, col_deriv, ftol, xtol, gtol, max_nfev, factor, diag
        )

    f = info["fvec"]

    if callable(jac):
        J = jac(x)
    else:
        J = np.atleast_2d(approx_derivative(fun, x))

    cost = 0.5 * np.dot(f, f)
    g = J.T.dot(f)
    g_norm = norm(g, ord=np.inf)

    nfev = info["nfev"]
    njev = info.get("njev", None)

    status = FROM_MINPACK_TO_COMMON[status]
    active_mask = np.zeros_like(x0, dtype=int)

    return OptimizeResult(
        x=x,
        cost=cost,
        fun=f,
        jac=J,
        grad=g,
        optimality=g_norm,
        active_mask=active_mask,
        nfev=nfev,
        njev=njev,
        status=status,
    )


def prepare_bounds(bounds, n):
    lb, ub = [np.asarray(b, dtype=float) for b in bounds]
    if lb.ndim == 0:
        lb = np.resize(lb, n)

    if ub.ndim == 0:
        ub = np.resize(ub, n)

    return lb, ub


def check_tolerance(ftol, xtol, gtol, method):
    def check(tol, name):
        if tol is None:
            tol = 0
        elif tol < EPS:
            warn(
                "Setting `{}` below the machine epsilon ({:.2e}) effectively "
                "disables the corresponding termination condition.".format(name, EPS)
            )
        return tol

    ftol = check(ftol, "ftol")
    xtol = check(xtol, "xtol")
    gtol = check(gtol, "gtol")

    if method == "lm" and (ftol < EPS or xtol < EPS or gtol < EPS):
        raise ValueError("All tolerances must be higher than machine epsilon " "({:.2e}) for method 'lm'.".format(EPS))
    elif ftol < EPS and xtol < EPS and gtol < EPS:
        raise ValueError("At least one of the tolerances must be higher than " "machine epsilon ({:.2e}).".format(EPS))

    return ftol, xtol, gtol


def check_x_scale(x_scale, x0):
    if isinstance(x_scale, str) and x_scale == "jac":
        return x_scale

    try:
        x_scale = np.asarray(x_scale, dtype=float)
        valid = np.all(np.isfinite(x_scale)) and np.all(x_scale > 0)
    except (ValueError, TypeError):
        valid = False

    if not valid:
        raise ValueError("`x_scale` must be 'jac' or array_like with " "positive numbers.")

    if x_scale.ndim == 0:
        x_scale = np.resize(x_scale, x0.shape)

    if x_scale.shape != x0.shape:
        raise ValueError("Inconsistent shapes between `x_scale` and `x0`.")

    return x_scale


def check_jac_sparsity(jac_sparsity, m, n):
    if jac_sparsity is None:
        return None

    if not issparse(jac_sparsity):
        jac_sparsity = np.atleast_2d(jac_sparsity)

    if jac_sparsity.shape != (m, n):
        raise ValueError("`jac_sparsity` has wrong shape.")

    return jac_sparsity, group_columns(jac_sparsity)


# Loss functions.


def huber(z, rho, cost_only):
    mask = z <= 1
    rho[0, mask] = z[mask]
    rho[0, ~mask] = 2 * z[~mask] ** 0.5 - 1
    if cost_only:
        return
    rho[1, mask] = 1
    rho[1, ~mask] = z[~mask] ** -0.5
    rho[2, mask] = 0
    rho[2, ~mask] = -0.5 * z[~mask] ** -1.5


def soft_l1(z, rho, cost_only):
    t = 1 + z
    rho[0] = 2 * (t**0.5 - 1)
    if cost_only:
        return
    rho[1] = t**-0.5
    rho[2] = -0.5 * t**-1.5


def cauchy(z, rho, cost_only):
    rho[0] = np.log1p(z)
    if cost_only:
        return
    t = 1 + z
    rho[1] = 1 / t
    rho[2] = -1 / t**2


def arctan(z, rho, cost_only):
    rho[0] = np.arctan(z)
    if cost_only:
        return
    t = 1 + z**2
    rho[1] = 1 / t
    rho[2] = -2 * z / t**2


IMPLEMENTED_LOSSES = dict(linear=None, huber=huber, soft_l1=soft_l1, cauchy=cauchy, arctan=arctan)


def construct_loss_function(m, loss, f_scale):
    if loss == "linear":
        return None

    if not callable(loss):
        loss = IMPLEMENTED_LOSSES[loss]
        rho = np.empty((3, m))

        def loss_function(f, cost_only=False):
            z = (f / f_scale) ** 2
            loss(z, rho, cost_only=cost_only)
            if cost_only:
                return 0.5 * f_scale**2 * np.sum(rho[0])
            rho[0] *= f_scale**2
            rho[2] /= f_scale**2
            return rho

    else:

        def loss_function(f, cost_only=False):
            z = (f / f_scale) ** 2
            rho = loss(z)
            if cost_only:
                return 0.5 * f_scale**2 * np.sum(rho[0])
            rho[0] *= f_scale**2
            rho[2] /= f_scale**2
            return rho

    return loss_function


def least_squares(
    fun,
    x0,
    jac="2-point",
    bounds=(-np.inf, np.inf),
    method="trf",
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    x_scale=1.0,
    loss="linear",
    f_scale=1.0,
    diff_step=None,
    tr_solver=None,
    tr_options={},
    jac_sparsity=None,
    max_nfev=None,
    verbose=0,
    vtol=0,
    args=(),
    kwargs={},
):
    """Solve a nonlinear least-squares problem with bounds on the variables.

    Given the residuals f(x) (an m-D real function of n real
    variables) and the loss function rho(s) (a scalar function), `least_squares`
    finds a local minimum of the cost function F(x)::

        minimize F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1)
        subject to lb <= x <= ub

    The purpose of the loss function rho(s) is to reduce the influence of
    outliers on the solution.

    Parameters
    ----------
    fun : callable
        Function which computes the vector of residuals, with the signature
        ``fun(x, *args, **kwargs)``, i.e., the minimization proceeds with
        respect to its first argument. The argument ``x`` passed to this
        function is an ndarray of shape (n,) (never a scalar, even for n=1).
        It must allocate and return a 1-D array_like of shape (m,) or a scalar.
        If the argument ``x`` is complex or the function ``fun`` returns
        complex residuals, it must be wrapped in a real function of real
        arguments, as shown at the end of the Examples section.
    x0 : array_like with shape (n,) or float
        Initial guess on independent variables. If float, it will be treated
        as a 1-D array with one element.
    jac : {'2-point', '3-point', 'cs', callable}, optional
        Method of computing the Jacobian matrix (an m-by-n matrix, where
        element (i, j) is the partial derivative of f[i] with respect to
        x[j]). The keywords select a finite difference scheme for numerical
        estimation. The scheme '3-point' is more accurate, but requires
        twice as many operations as '2-point' (default). The scheme 'cs'
        uses complex steps, and while potentially the most accurate, it is
        applicable only when `fun` correctly handles complex inputs and
        can be analytically continued to the complex plane. Method 'lm'
        always uses the '2-point' scheme. If callable, it is used as
        ``jac(x, *args, **kwargs)`` and should return a good approximation
        (or the exact value) for the Jacobian as an array_like (np.atleast_2d
        is applied), a sparse matrix (csr_matrix preferred for performance) or
        a `scipy.sparse.linalg.LinearOperator`.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each array must match the size of `x0` or be a scalar, in the latter
        case a bound will be the same for all variables. Use ``np.inf`` with
        an appropriate sign to disable bounds on all or some variables.
    method : {'trf', 'dogbox', 'lm'}, optional
        Algorithm to perform minimization.

            * 'trf' : Trust Region Reflective algorithm, particularly suitable
              for large sparse problems with bounds. Generally robust method.
            * 'dogbox' : dogleg algorithm with rectangular trust regions,
              typical use case is small problems with bounds. Not recommended
              for problems with rank-deficient Jacobian.
            * 'lm' : Levenberg-Marquardt algorithm as implemented in MINPACK.
              Doesn't handle bounds and sparse Jacobians. Usually the most
              efficient method for small unconstrained problems.

        Default is 'trf'. See Notes for more information.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function. Default
        is 1e-8. The optimization process is stopped when ``dF < ftol * F``,
        and there was an adequate agreement between a local quadratic model and
        the true model in the last step.

        If None and 'method' is not 'lm', the termination by this condition is
        disabled. If 'method' is 'lm', this tolerance must be higher than
        machine epsilon.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        Default is 1e-8. The exact condition depends on the `method` used:

            * For 'trf' and 'dogbox' : ``norm(dx) < xtol * (xtol + norm(x))``.
            * For 'lm' : ``Delta < xtol * norm(xs)``, where ``Delta`` is
              a trust-region radius and ``xs`` is the value of ``x``
              scaled according to `x_scale` parameter (see below).

        If None and 'method' is not 'lm', the termination by this condition is
        disabled. If 'method' is 'lm', this tolerance must be higher than
        machine epsilon.
    gtol : float or None, optional
        Tolerance for termination by the norm of the gradient. Default is 1e-8.
        The exact condition depends on a `method` used:

            * For 'trf' : ``norm(g_scaled, ord=np.inf) < gtol``, where
              ``g_scaled`` is the value of the gradient scaled to account for
              the presence of the bounds [STIR]_.
            * For 'dogbox' : ``norm(g_free, ord=np.inf) < gtol``, where
              ``g_free`` is the gradient with respect to the variables which
              are not in the optimal state on the boundary.
            * For 'lm' : the maximum absolute value of the cosine of angles
              between columns of the Jacobian and the residual vector is less
              than `gtol`, or the residual vector is zero.

        If None and 'method' is not 'lm', the termination by this condition is
        disabled. If 'method' is 'lm', this tolerance must be higher than
        machine epsilon.
    x_scale : array_like or 'jac', optional
        Characteristic scale of each variable. Setting `x_scale` is equivalent
        to reformulating the problem in scaled variables ``xs = x / x_scale``.
        An alternative view is that the size of a trust region along jth
        dimension is proportional to ``x_scale[j]``. Improved convergence may
        be achieved by setting `x_scale` such that a step of a given size
        along any of the scaled variables has a similar effect on the cost
        function. If set to 'jac', the scale is iteratively updated using the
        inverse norms of the columns of the Jacobian matrix (as described in
        [JJMore]_).
    loss : str or callable, optional
        Determines the loss function. The following keyword values are allowed:

            * 'linear' (default) : ``rho(z) = z``. Gives a standard
              least-squares problem.
            * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
              approximation of l1 (absolute value) loss. Usually a good
              choice for robust least squares.
            * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
              similarly to 'soft_l1'.
            * 'cauchy' : ``rho(z) = ln(1 + z)``. Severely weakens outliers
              influence, but may cause difficulties in optimization process.
            * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
              a single residual, has properties similar to 'cauchy'.

        If callable, it must take a 1-D ndarray ``z=f**2`` and return an
        array_like with shape (3, m) where row 0 contains function values,
        row 1 contains first derivatives and row 2 contains second
        derivatives. Method 'lm' supports only 'linear' loss.
    f_scale : float, optional
        Value of soft margin between inlier and outlier residuals, default
        is 1.0. The loss function is evaluated as follows
        ``rho_(f**2) = C**2 * rho(f**2 / C**2)``, where ``C`` is `f_scale`,
        and ``rho`` is determined by `loss` parameter. This parameter has
        no effect with ``loss='linear'``, but for other `loss` values it is
        of crucial importance.
    max_nfev : None or int, optional
        Maximum number of function evaluations before the termination.
        If None (default), the value is chosen automatically:

            * For 'trf' and 'dogbox' : 100 * n.
            * For 'lm' :  100 * n if `jac` is callable and 100 * n * (n + 1)
              otherwise (because 'lm' counts function calls in Jacobian
              estimation).

    diff_step : None or array_like, optional
        Determines the relative step size for the finite difference
        approximation of the Jacobian. The actual step is computed as
        ``x * diff_step``. If None (default), then `diff_step` is taken to be
        a conventional "optimal" power of machine epsilon for the finite
        difference scheme used [NR]_.
    tr_solver : {None, 'exact', 'lsmr'}, optional
        Method for solving trust-region subproblems, relevant only for 'trf'
        and 'dogbox' methods.

            * 'exact' is suitable for not very large problems with dense
              Jacobian matrices. The computational complexity per iteration is
              comparable to a singular value decomposition of the Jacobian
              matrix.
            * 'lsmr' is suitable for problems with sparse and large Jacobian
              matrices. It uses the iterative procedure
              `scipy.sparse.linalg.lsmr` for finding a solution of a linear
              least-squares problem and only requires matrix-vector product
              evaluations.

        If None (default), the solver is chosen based on the type of Jacobian
        returned on the first iteration.
    tr_options : dict, optional
        Keyword options passed to trust-region solver.

            * ``tr_solver='exact'``: `tr_options` are ignored.
            * ``tr_solver='lsmr'``: options for `scipy.sparse.linalg.lsmr`.
              Additionally,  ``method='trf'`` supports  'regularize' option
              (bool, default is True), which adds a regularization term to the
              normal equation, which improves convergence if the Jacobian is
              rank-deficient [Byrd]_ (eq. 3.4).

    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines the sparsity structure of the Jacobian matrix for finite
        difference estimation, its shape must be (m, n). If the Jacobian has
        only few non-zero elements in *each* row, providing the sparsity
        structure will greatly speed up the computations [Curtis]_. A zero
        entry means that a corresponding element in the Jacobian is identically
        zero. If provided, forces the use of 'lsmr' trust-region solver.
        If None (default), then dense differencing will be used. Has no effect
        for 'lm' method.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations (not supported by 'lm'
              method).

    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun` and `jac`. Both empty by default.
        The calling signature is ``fun(x, *args, **kwargs)`` and the same for
        `jac`.

    Returns
    -------
    result : OptimizeResult
        `OptimizeResult` with the following fields defined:

            x : ndarray, shape (n,)
                Solution found.
            cost : float
                Value of the cost function at the solution.
            fun : ndarray, shape (m,)
                Vector of residuals at the solution.
            jac : ndarray, sparse matrix or LinearOperator, shape (m, n)
                Modified Jacobian matrix at the solution, in the sense that J^T J
                is a Gauss-Newton approximation of the Hessian of the cost function.
                The type is the same as the one used by the algorithm.
            grad : ndarray, shape (m,)
                Gradient of the cost function at the solution.
            optimality : float
                First-order optimality measure. In unconstrained problems, it is
                always the uniform norm of the gradient. In constrained problems,
                it is the quantity which was compared with `gtol` during iterations.
            active_mask : ndarray of int, shape (n,)
                Each component shows whether a corresponding constraint is active
                (that is, whether a variable is at the bound):

                    *  0 : a constraint is not active.
                    * -1 : a lower bound is active.
                    *  1 : an upper bound is active.

                Might be somewhat arbitrary for 'trf' method as it generates a
                sequence of strictly feasible iterates and `active_mask` is
                determined within a tolerance threshold.
            nfev : int
                Number of function evaluations done. Methods 'trf' and 'dogbox' do
                not count function calls for numerical Jacobian approximation, as
                opposed to 'lm' method.
            njev : int or None
                Number of Jacobian evaluations done. If numerical Jacobian
                approximation is used in 'lm' method, it is set to None.
            status : int
                The reason for algorithm termination:

                    * -1 : improper input parameters status returned from MINPACK.
                    *  0 : the maximum number of function evaluations is exceeded.
                    *  1 : `gtol` termination condition is satisfied.
                    *  2 : `ftol` termination condition is satisfied.
                    *  3 : `xtol` termination condition is satisfied.
                    *  4 : Both `ftol` and `xtol` termination conditions are satisfied.

            message : str
                Verbal description of the termination reason.
            success : bool
                True if one of the convergence criteria is satisfied (`status` > 0).

    See Also
    --------
    leastsq : A legacy wrapper for the MINPACK implementation of the
              Levenberg-Marquadt algorithm.
    curve_fit : Least-squares minimization applied to a curve-fitting problem.

    Notes
    -----
    Method 'lm' (Levenberg-Marquardt) calls a wrapper over least-squares
    algorithms implemented in MINPACK (lmder, lmdif). It runs the
    Levenberg-Marquardt algorithm formulated as a trust-region type algorithm.
    The implementation is based on paper [JJMore]_, it is very robust and
    efficient with a lot of smart tricks. It should be your first choice
    for unconstrained problems. Note that it doesn't support bounds. Also,
    it doesn't work when m < n.

    Method 'trf' (Trust Region Reflective) is motivated by the process of
    solving a system of equations, which constitute the first-order optimality
    condition for a bound-constrained minimization problem as formulated in
    [STIR]_. The algorithm iteratively solves trust-region subproblems
    augmented by a special diagonal quadratic term and with trust-region shape
    determined by the distance from the bounds and the direction of the
    gradient. This enhancements help to avoid making steps directly into bounds
    and efficiently explore the whole space of variables. To further improve
    convergence, the algorithm considers search directions reflected from the
    bounds. To obey theoretical requirements, the algorithm keeps iterates
    strictly feasible. With dense Jacobians trust-region subproblems are
    solved by an exact method very similar to the one described in [JJMore]_
    (and implemented in MINPACK). The difference from the MINPACK
    implementation is that a singular value decomposition of a Jacobian
    matrix is done once per iteration, instead of a QR decomposition and series
    of Givens rotation eliminations. For large sparse Jacobians a 2-D subspace
    approach of solving trust-region subproblems is used [STIR]_, [Byrd]_.
    The subspace is spanned by a scaled gradient and an approximate
    Gauss-Newton solution delivered by `scipy.sparse.linalg.lsmr`. When no
    constraints are imposed the algorithm is very similar to MINPACK and has
    generally comparable performance. The algorithm works quite robust in
    unbounded and bounded problems, thus it is chosen as a default algorithm.

    Method 'dogbox' operates in a trust-region framework, but considers
    rectangular trust regions as opposed to conventional ellipsoids [Voglis]_.
    The intersection of a current trust region and initial bounds is again
    rectangular, so on each iteration a quadratic minimization problem subject
    to bound constraints is solved approximately by Powell's dogleg method
    [NumOpt]_. The required Gauss-Newton step can be computed exactly for
    dense Jacobians or approximately by `scipy.sparse.linalg.lsmr` for large
    sparse Jacobians. The algorithm is likely to exhibit slow convergence when
    the rank of Jacobian is less than the number of variables. The algorithm
    often outperforms 'trf' in bounded problems with a small number of
    variables.

    Robust loss functions are implemented as described in [BA]_. The idea
    is to modify a residual vector and a Jacobian matrix on each iteration
    such that computed gradient and Gauss-Newton Hessian approximation match
    the true gradient and Hessian approximation of the cost function. Then
    the algorithm proceeds in a normal way, i.e., robust loss functions are
    implemented as a simple wrapper over standard least-squares algorithms.

    .. versionadded:: 0.17.0

    References
    ----------
    .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,
              and Conjugate Gradient Method for Large-Scale Bound-Constrained
              Minimization Problems," SIAM Journal on Scientific Computing,
              Vol. 21, Number 1, pp 1-23, 1999.
    .. [NR] William H. Press et. al., "Numerical Recipes. The Art of Scientific
            Computing. 3rd edition", Sec. 5.7.
    .. [Byrd] R. H. Byrd, R. B. Schnabel and G. A. Shultz, "Approximate
              solution of the trust region problem by minimization over
              two-dimensional subspaces", Math. Programming, 40, pp. 247-263,
              1988.
    .. [Curtis] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
                sparse Jacobian matrices", Journal of the Institute of
                Mathematics and its Applications, 13, pp. 117-120, 1974.
    .. [JJMore] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation
                and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
                Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
    .. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region
                Dogleg Approach for Unconstrained and Bound Constrained
                Nonlinear Optimization", WSEAS International Conference on
                Applied Mathematics, Corfu, Greece, 2004.
    .. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization,
                2nd edition", Chapter 4.
    .. [BA] B. Triggs et. al., "Bundle Adjustment - A Modern Synthesis",
            Proceedings of the International Workshop on Vision Algorithms:
            Theory and Practice, pp. 298-372, 1999.

    Examples
    --------
    In this example we find a minimum of the Rosenbrock function without bounds
    on independent variables.

    >>> def fun_rosenbrock(x):
    ...     return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])

    Notice that we only provide the vector of the residuals. The algorithm
    constructs the cost function as a sum of squares of the residuals, which
    gives the Rosenbrock function. The exact minimum is at ``x = [1.0, 1.0]``.

    >>> from scipy.optimize import least_squares
    >>> x0_rosenbrock = np.array([2, 2])
    >>> res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)
    >>> res_1.x
    array([ 1.,  1.])
    >>> res_1.cost
    9.8669242910846867e-30
    >>> res_1.optimality
    8.8928864934219529e-14

    We now constrain the variables, in such a way that the previous solution
    becomes infeasible. Specifically, we require that ``x[1] >= 1.5``, and
    ``x[0]`` left unconstrained. To this end, we specify the `bounds` parameter
    to `least_squares` in the form ``bounds=([-np.inf, 1.5], np.inf)``.

    We also provide the analytic Jacobian:

    >>> def jac_rosenbrock(x):
    ...     return np.array([
    ...         [-20 * x[0], 10],
    ...         [-1, 0]])

    Putting this all together, we see that the new solution lies on the bound:

    >>> res_2 = least_squares(fun_rosenbrock, x0_rosenbrock, jac_rosenbrock,
    ...                       bounds=([-np.inf, 1.5], np.inf))
    >>> res_2.x
    array([ 1.22437075,  1.5       ])
    >>> res_2.cost
    0.025213093946805685
    >>> res_2.optimality
    1.5885401433157753e-07

    Now we solve a system of equations (i.e., the cost function should be zero
    at a minimum) for a Broyden tridiagonal vector-valued function of 100000
    variables:

    >>> def fun_broyden(x):
    ...     f = (3 - x) * x + 1
    ...     f[1:] -= x[:-1]
    ...     f[:-1] -= 2 * x[1:]
    ...     return f

    The corresponding Jacobian matrix is sparse. We tell the algorithm to
    estimate it by finite differences and provide the sparsity structure of
    Jacobian to significantly speed up this process.

    >>> from scipy.sparse import lil_matrix
    >>> def sparsity_broyden(n):
    ...     sparsity = lil_matrix((n, n), dtype=int)
    ...     i = np.arange(n)
    ...     sparsity[i, i] = 1
    ...     i = np.arange(1, n)
    ...     sparsity[i, i - 1] = 1
    ...     i = np.arange(n - 1)
    ...     sparsity[i, i + 1] = 1
    ...     return sparsity
    ...
    >>> n = 100000
    >>> x0_broyden = -np.ones(n)
    ...
    >>> res_3 = least_squares(fun_broyden, x0_broyden,
    ...                       jac_sparsity=sparsity_broyden(n))
    >>> res_3.cost
    4.5687069299604613e-23
    >>> res_3.optimality
    1.1650454296851518e-11

    Let's also solve a curve fitting problem using robust loss function to
    take care of outliers in the data. Define the model function as
    ``y = a + b * exp(c * t)``, where t is a predictor variable, y is an
    observation and a, b, c are parameters to estimate.

    First, define the function which generates the data with noise and
    outliers, define the model parameters, and generate data:

    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> def gen_data(t, a, b, c, noise=0., n_outliers=0, seed=None):
    ...     rng = default_rng(seed)
    ...
    ...     y = a + b * np.exp(t * c)
    ...
    ...     error = noise * rng.standard_normal(t.size)
    ...     outliers = rng.integers(0, t.size, n_outliers)
    ...     error[outliers] *= 10
    ...
    ...     return y + error
    ...
    >>> a = 0.5
    >>> b = 2.0
    >>> c = -1
    >>> t_min = 0
    >>> t_max = 10
    >>> n_points = 15
    ...
    >>> t_train = np.linspace(t_min, t_max, n_points)
    >>> y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)

    Define function for computing residuals and initial estimate of
    parameters.

    >>> def fun(x, t, y):
    ...     return x[0] + x[1] * np.exp(x[2] * t) - y
    ...
    >>> x0 = np.array([1.0, 1.0, 0.0])

    Compute a standard least-squares solution:

    >>> res_lsq = least_squares(fun, x0, args=(t_train, y_train))

    Now compute two solutions with two different robust loss functions. The
    parameter `f_scale` is set to 0.1, meaning that inlier residuals should
    not significantly exceed 0.1 (the noise level used).

    >>> res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1,
    ...                             args=(t_train, y_train))
    >>> res_log = least_squares(fun, x0, loss='cauchy', f_scale=0.1,
    ...                         args=(t_train, y_train))

    And, finally, plot all the curves. We see that by selecting an appropriate
    `loss`  we can get estimates close to optimal even in the presence of
    strong outliers. But keep in mind that generally it is recommended to try
    'soft_l1' or 'huber' losses first (if at all necessary) as the other two
    options may cause difficulties in optimization process.

    >>> t_test = np.linspace(t_min, t_max, n_points * 10)
    >>> y_true = gen_data(t_test, a, b, c)
    >>> y_lsq = gen_data(t_test, *res_lsq.x)
    >>> y_soft_l1 = gen_data(t_test, *res_soft_l1.x)
    >>> y_log = gen_data(t_test, *res_log.x)
    ...
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t_train, y_train, 'o')
    >>> plt.plot(t_test, y_true, 'k', linewidth=2, label='true')
    >>> plt.plot(t_test, y_lsq, label='linear loss')
    >>> plt.plot(t_test, y_soft_l1, label='soft_l1 loss')
    >>> plt.plot(t_test, y_log, label='cauchy loss')
    >>> plt.xlabel("t")
    >>> plt.ylabel("y")
    >>> plt.legend()
    >>> plt.show()

    In the next example, we show how complex-valued residual functions of
    complex variables can be optimized with ``least_squares()``. Consider the
    following function:

    >>> def f(z):
    ...     return z - (0.5 + 0.5j)

    We wrap it into a function of real variables that returns real residuals
    by simply handling the real and imaginary parts as independent variables:

    >>> def f_wrap(x):
    ...     fx = f(x[0] + 1j*x[1])
    ...     return np.array([fx.real, fx.imag])

    Thus, instead of the original m-D complex function of n complex
    variables we optimize a 2m-D real function of 2n real variables:

    >>> from scipy.optimize import least_squares
    >>> res_wrapped = least_squares(f_wrap, (0.1, 0.1), bounds=([0, 0], [1, 1]))
    >>> z = res_wrapped.x[0] + res_wrapped.x[1]*1j
    >>> z
    (0.49999999999925893+0.49999999999925893j)

    """
    if method not in ["trf", "dogbox", "lm"]:
        raise ValueError("`method` must be 'trf', 'dogbox' or 'lm'.")

    if jac not in ["2-point", "3-point", "cs"] and not callable(jac):
        raise ValueError("`jac` must be '2-point', '3-point', 'cs' or " "callable.")

    if tr_solver not in [None, "exact", "lsmr"]:
        raise ValueError("`tr_solver` must be None, 'exact' or 'lsmr'.")

    if loss not in IMPLEMENTED_LOSSES and not callable(loss):
        raise ValueError("`loss` must be one of {0} or a callable.".format(IMPLEMENTED_LOSSES.keys()))

    if method == "lm" and loss != "linear":
        raise ValueError("method='lm' supports only 'linear' loss function.")

    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    if len(bounds) != 2:
        raise ValueError("`bounds` must contain 2 elements.")

    if max_nfev is not None and max_nfev <= 0:
        raise ValueError("`max_nfev` must be None or positive integer.")

    if np.iscomplexobj(x0):
        raise ValueError("`x0` must be real.")

    x0 = np.atleast_1d(x0).astype(float)

    if x0.ndim > 1:
        raise ValueError("`x0` must have at most 1 dimension.")

    lb, ub = prepare_bounds(bounds, x0.shape[0])

    if method == "lm" and not np.all((lb == -np.inf) & (ub == np.inf)):
        raise ValueError("Method 'lm' doesn't support bounds.")

    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")

    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each " "upper bound.")

    if not in_bounds(x0, lb, ub):
        raise ValueError("`x0` is infeasible.")

    x_scale = check_x_scale(x_scale, x0)

    ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol, method)

    def fun_wrapped(x):
        return np.atleast_1d(fun(x, *args, **kwargs))

    if method == "trf":
        x0 = make_strictly_feasible(x0, lb, ub)

    f0 = fun_wrapped(x0)

    if f0.ndim != 1:
        raise ValueError("`fun` must return at most 1-d array_like. " "f0.shape: {0}".format(f0.shape))

    if not np.all(np.isfinite(f0)):
        raise ValueError("Residuals are not finite in the initial point.")

    n = x0.size
    m = f0.size

    if method == "lm" and m < n:
        raise ValueError(
            "Method 'lm' doesn't work when the number of " "residuals is less than the number of variables."
        )

    loss_function = construct_loss_function(m, loss, f_scale)
    if callable(loss):
        rho = loss_function(f0)
        if rho.shape != (3, m):
            raise ValueError("The return value of `loss` callable has wrong " "shape.")
        initial_cost = 0.5 * np.sum(rho[0])
    elif loss_function is not None:
        initial_cost = loss_function(f0, cost_only=True)
    else:
        initial_cost = 0.5 * np.dot(f0, f0)

    if callable(jac):
        J0 = jac(x0, *args, **kwargs)

        if issparse(J0):
            J0 = J0.tocsr()

            def jac_wrapped(x, _=None):
                return jac(x, *args, **kwargs).tocsr()

        elif isinstance(J0, LinearOperator):

            def jac_wrapped(x, _=None):
                return jac(x, *args, **kwargs)

        else:
            J0 = np.atleast_2d(J0)

            def jac_wrapped(x, _=None):
                return np.atleast_2d(jac(x, *args, **kwargs))

    else:  # Estimate Jacobian by finite differences.
        if method == "lm":
            if jac_sparsity is not None:
                raise ValueError("method='lm' does not support " "`jac_sparsity`.")

            if jac != "2-point":
                warn("jac='{0}' works equivalently to '2-point' " "for method='lm'.".format(jac))

            J0 = jac_wrapped = None
        else:
            if jac_sparsity is not None and tr_solver == "exact":
                raise ValueError("tr_solver='exact' is incompatible " "with `jac_sparsity`.")

            jac_sparsity = check_jac_sparsity(jac_sparsity, m, n)

            def jac_wrapped(x, f):
                J = approx_derivative(
                    fun,
                    x,
                    rel_step=diff_step,
                    method=jac,
                    f0=f,
                    bounds=bounds,
                    args=args,
                    kwargs=kwargs,
                    sparsity=jac_sparsity,
                )
                if J.ndim != 2:  # J is guaranteed not sparse.
                    J = np.atleast_2d(J)

                return J

            J0 = jac_wrapped(x0, f0)

    if J0 is not None:
        if J0.shape != (m, n):
            raise ValueError(
                "The return value of `jac` has wrong shape: expected {0}, " "actual {1}.".format((m, n), J0.shape)
            )

        if not isinstance(J0, np.ndarray):
            if method == "lm":
                raise ValueError("method='lm' works only with dense " "Jacobian matrices.")

            if tr_solver == "exact":
                raise ValueError("tr_solver='exact' works only with dense " "Jacobian matrices.")

        jac_scale = isinstance(x_scale, str) and x_scale == "jac"
        if isinstance(J0, LinearOperator) and jac_scale:
            raise ValueError("x_scale='jac' can't be used when `jac` " "returns LinearOperator.")

        if tr_solver is None:
            if isinstance(J0, np.ndarray):
                tr_solver = "exact"
            else:
                tr_solver = "lsmr"

    if method == "trf":
        result = trf(
            fun_wrapped,
            jac_wrapped,
            x0,
            f0,
            J0,
            lb,
            ub,
            ftol,
            xtol,
            gtol,
            max_nfev,
            x_scale,
            loss_function,
            tr_solver,
            tr_options.copy(),
            verbose,
            vtol=vtol,
        )
    else:
        raise ValueError("Only trf supported")

    result.message = TERMINATION_MESSAGES[result.status]
    result.success = result.status > 0

    if verbose >= 1:
        print(result.message)
        print(
            "Function evaluations {0}, initial cost {1:.4e}, final cost "
            "{2:.4e}, first-order optimality {3:.2e}.".format(result.nfev, initial_cost, result.cost, result.optimality)
        )

    return result


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

"""Trust Region Reflective algorithm for least-squares optimization.

The algorithm is based on ideas from paper [STIR]_. The main idea is to
account for the presence of the bounds by appropriate scaling of the variables (or,
equivalently, changing a trust-region shape). Let's introduce a vector v:

           | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf
    v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf
           | 1,           otherwise

where g is the gradient of a cost function and lb, ub are the bounds. Its
components are distances to the bounds at which the anti-gradient points (if
this distance is finite). Define a scaling matrix D = diag(v**0.5).
First-order optimality conditions can be stated as

    D^2 g(x) = 0.

Meaning that components of the gradient should be zero for strictly interior
variables, and components must point inside the feasible region for variables
on the bound.

Now consider this system of equations as a new optimization problem. If the
point x is strictly interior (not on the bound), then the left-hand side is
differentiable and the Newton step for it satisfies

    (D^2 H + diag(g) Jv) p = -D^2 g

where H is the Hessian matrix (or its J^T J approximation in least squares),
Jv is the Jacobian matrix of v with components -1, 1 or 0, such that all
elements of matrix C = diag(g) Jv are non-negative. Introduce the change
of the variables x = D x_h (_h would be "hat" in LaTeX). In the new variables,
we have a Newton step satisfying

    B_h p_h = -g_h,

where B_h = D H D + C, g_h = D g. In least squares B_h = J_h^T J_h, where
J_h = J D. Note that J_h and g_h are proper Jacobian and gradient with respect
to "hat" variables. To guarantee global convergence we formulate a
trust-region problem based on the Newton step in the new variables:

    0.5 * p_h^T B_h p + g_h^T p_h -> min, ||p_h|| <= Delta

In the original space B = H + D^{-1} C D^{-1}, and the equivalent trust-region
problem is

    0.5 * p^T B p + g^T p -> min, ||D^{-1} p|| <= Delta

Here, the meaning of the matrix D becomes more clear: it alters the shape
of a trust-region, such that large steps towards the bounds are not allowed.
In the implementation, the trust-region problem is solved in "hat" space,
but handling of the bounds is done in the original space (see below and read
the code).

The introduction of the matrix D doesn't allow to ignore bounds, the algorithm
must keep iterates strictly feasible (to satisfy aforementioned
differentiability), the parameter theta controls step back from the boundary
(see the code for details).

The algorithm does another important trick. If the trust-region solution
doesn't fit into the bounds, then a reflected (from a firstly encountered
bound) search direction is considered. For motivation and analysis refer to
[STIR]_ paper (and other papers of the authors). In practice, it doesn't need
a lot of justifications, the algorithm simply chooses the best step among
three: a constrained trust-region step, a reflected step and a constrained
Cauchy step (a minimizer along -g_h in "hat" space, or -D^2 g in the original
space).

Another feature is that a trust-region radius control strategy is modified to
account for appearance of the diagonal C matrix (called diag_h in the code).

Note that all described peculiarities are completely gone as we consider
problems without bounds (the algorithm becomes a standard trust-region type
algorithm very similar to ones implemented in MINPACK).

The implementation supports two methods of solving the trust-region problem.
The first, called 'exact', applies SVD on Jacobian and then solves the problem
very accurately using the algorithm described in [JJMore]_. It is not
applicable to large problem. The second, called 'lsmr', uses the 2-D subspace
approach (sometimes called "indefinite dogleg"), where the problem is solved
in a subspace spanned by the gradient and the approximate Gauss-Newton step
found by ``scipy.sparse.linalg.lsmr``. A 2-D trust-region problem is
reformulated as a 4th order algebraic equation and solved very accurately by
``numpy.roots``. The subspace approach allows to solve very large problems
(up to couple of millions of residuals on a regular PC), provided the Jacobian
matrix is sufficiently sparse.

References
----------
.. [STIR] Branch, M.A., T.F. Coleman, and Y. Li, "A Subspace, Interior,
      and Conjugate Gradient Method for Large-Scale Bound-Constrained
      Minimization Problems," SIAM Journal on Scientific Computing,
      Vol. 21, Number 1, pp 1-23, 1999.
.. [JJMore] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
    and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
"""


def trf(
    fun,
    jac,
    x0,
    f0,
    J0,
    lb,
    ub,
    ftol,
    xtol,
    gtol,
    max_nfev,
    x_scale,
    loss_function,
    tr_solver,
    tr_options,
    verbose,
    vtol=0,
):
    # For efficiency, it makes sense to run the simplified version of the
    # algorithm when no bounds are imposed. We decided to write the two
    # separate functions. It violates the DRY principle, but the individual
    # functions are kept the most readable.
    if np.all(lb == -np.inf) and np.all(ub == np.inf):
        return trf_no_bounds(
            fun,
            jac,
            x0,
            f0,
            J0,
            ftol,
            xtol,
            gtol,
            max_nfev,
            x_scale,
            loss_function,
            tr_solver,
            tr_options,
            verbose,
            vtol=vtol,
        )
    else:
        return trf_bounds(
            fun,
            jac,
            x0,
            f0,
            J0,
            lb,
            ub,
            ftol,
            xtol,
            gtol,
            max_nfev,
            x_scale,
            loss_function,
            tr_solver,
            tr_options,
            verbose,
            vtol=vtol,
        )


def select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta):
    """Select the best step according to Trust Region Reflective algorithm."""
    if in_bounds(x + p, lb, ub):
        p_value = evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
        return p, p_h, -p_value

    p_stride, hits = step_size_to_bound(x, p, lb, ub)

    # Compute the reflected direction.
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h

    # Restrict trust-region step, such that it hits the bound.
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p

    # Reflected direction will cross first either feasible region or trust
    # region boundary.
    _, to_tr = intersect_trust_region(p_h, r_h, Delta)
    to_bound, _ = step_size_to_bound(x_on_bound, r, lb, ub)

    # Find lower and upper bounds on a step size along the reflected
    # direction, considering the strict feasibility requirement. There is no
    # single correct way to do that, the chosen approach seems to work best
    # on test problems.
    r_stride = min(to_bound, to_tr)
    if r_stride > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride
        if r_stride == to_bound:
            r_stride_u = theta * to_bound
        else:
            r_stride_u = to_tr
    else:
        r_stride_l = 0
        r_stride_u = -1

    # Check if reflection step is available.
    if r_stride_l <= r_stride_u:
        a, b, c = build_quadratic_1d(J_h, g_h, r_h, s0=p_h, diag=diag_h)
        r_stride, r_value = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c=c)
        r_h *= r_stride
        r_h += p_h
        r = r_h * d
    else:
        r_value = np.inf

    # Now correct p_h to make it strictly interior.
    p *= theta
    p_h *= theta
    p_value = evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)

    ag_h = -g_h
    ag = d * ag_h

    to_tr = Delta / norm(ag_h)
    to_bound, _ = step_size_to_bound(x, ag, lb, ub)
    if to_bound < to_tr:
        ag_stride = theta * to_bound
    else:
        ag_stride = to_tr

    a, b = build_quadratic_1d(J_h, g_h, ag_h, diag=diag_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride)
    ag_h *= ag_stride
    ag *= ag_stride

    if p_value < r_value and p_value < ag_value:
        return p, p_h, -p_value
    elif r_value < p_value and r_value < ag_value:
        return r, r_h, -r_value
    else:
        return ag, ag_h, -ag_value


def trf_bounds(
    fun,
    jac,
    x0,
    f0,
    J0,
    lb,
    ub,
    ftol,
    xtol,
    gtol,
    max_nfev,
    x_scale,
    loss_function,
    tr_solver,
    tr_options,
    verbose,
    vtol=0,
):
    x = x0.copy()

    f = f0
    f_true = f.copy()
    nfev = 1

    J = J0
    njev = 1
    m, n = J.shape

    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
        J, f = scale_for_robust_loss_function(J, f, rho)
    else:
        cost = 0.5 * np.dot(f, f)

    g = compute_grad(J, f)

    jac_scale = isinstance(x_scale, str) and x_scale == "jac"
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    v, dv = CL_scaling_vector(x, g, lb, ub)
    v[dv != 0] *= scale_inv[dv != 0]
    Delta = norm(x0 * scale_inv / v**0.5)
    if Delta == 0:
        Delta = 1.0

    g_norm = norm(g * v, ord=np.inf)

    f_augmented = np.zeros((m + n))
    if tr_solver == "exact":
        J_augmented = np.empty((m + n, n))
    elif tr_solver == "lsmr":
        reg_term = 0.0
        regularize = tr_options.pop("regularize", True)

    if max_nfev is None:
        max_nfev = x0.size * 100

    alpha = 0.0  # "Levenberg-Marquardt" parameter

    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None

    if verbose == 2:
        print_header_nonlinear()

    while True:
        v, dv = CL_scaling_vector(x, g, lb, ub)

        g_norm = norm(g * v, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if cost < vtol:
            termination_status = 3
            print(
                "Terminating optimization because cost is less than threshold. It will say it terminated by xtol though"
            )

        if verbose == 2:
            print_iteration_nonlinear(iteration, nfev, cost, actual_reduction, step_norm, g_norm)

        if termination_status is not None or nfev == max_nfev:
            break

        # Now compute variables in "hat" space. Here, we also account for
        # scaling introduced by `x_scale` parameter. This part is a bit tricky,
        # you have to write down the formulas and see how the trust-region
        # problem is formulated when the two types of scaling are applied.
        # The idea is that first we apply `x_scale` and then apply Coleman-Li
        # approach in the new variables.

        # v is recomputed in the variables after applying `x_scale`, note that
        # components which were identically 1 not affected.
        v[dv != 0] *= scale_inv[dv != 0]

        # Here, we apply two types of scaling.
        d = v**0.5 * scale

        # C = diag(g * scale) Jv
        diag_h = g * dv * scale

        # After all this has been done, we continue normally.

        # "hat" gradient.
        g_h = d * g

        f_augmented[:m] = f
        if tr_solver == "exact":
            J_augmented[:m] = J * d
            J_h = J_augmented[:m]  # Memory view.
            J_augmented[m:] = np.diag(diag_h**0.5)
            U, s, V = svd(J_augmented, full_matrices=False)
            V = V.T
            uf = U.T.dot(f_augmented)
        elif tr_solver == "lsmr":
            J_h = right_multiplied_operator(J, d)

            if regularize:
                a, b = build_quadratic_1d(J_h, g_h, -g_h, diag=diag_h)
                to_tr = Delta / norm(g_h)
                ag_value = minimize_quadratic_1d(a, b, 0, to_tr)[1]
                reg_term = -ag_value / Delta**2

            lsmr_op = regularized_lsq_operator(J_h, (diag_h + reg_term) ** 0.5)
            gn_h = lsmr(lsmr_op, f_augmented, **tr_options)[0]
            S = np.vstack((g_h, gn_h)).T
            S, _ = qr(S, mode="economic")
            JS = J_h.dot(S)  # LinearOperator does dot too.
            B_S = np.dot(JS.T, JS) + np.dot(S.T * diag_h, S)
            g_S = S.T.dot(g_h)

        # theta controls step back step ratio from the bounds.
        theta = max(0.995, 1 - g_norm)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            if tr_solver == "exact":
                p_h, alpha, n_iter = solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=alpha)
            elif tr_solver == "lsmr":
                p_S, _ = solve_trust_region_2d(B_S, g_S, Delta)
                p_h = S.dot(p_S)

            p = d * p_h  # Trust-region solution in the original space.
            step, step_h, predicted_reduction = select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta)

            x_new = make_strictly_feasible(x + step, lb, ub, rstep=0)
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step_h)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            # Usual trust-region step quality estimation.
            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new
            Delta_new, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction, step_h_norm, step_h_norm > 0.95 * Delta
            )

            step_norm = norm(step)
            termination_status = check_termination(actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

        if actual_reduction > 0:
            x = x_new

            f = f_new
            f_true = f.copy()

            cost = cost_new

            J = jac(x, f)
            njev += 1

            if loss_function is not None:
                rho = loss_function(f)
                J, f = scale_for_robust_loss_function(J, f, rho)

            g = compute_grad(J, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

    if termination_status is None:
        termination_status = 0

    active_mask = find_active_constraints(x, lb, ub, rtol=xtol)
    return OptimizeResult(
        x=x,
        cost=cost,
        fun=f_true,
        jac=J,
        grad=g,
        optimality=g_norm,
        active_mask=active_mask,
        nfev=nfev,
        njev=njev,
        status=termination_status,
    )


def trf_no_bounds(
    fun, jac, x0, f0, J0, ftol, xtol, gtol, max_nfev, x_scale, loss_function, tr_solver, tr_options, verbose, vtol=0
):
    x = x0.copy()

    f = f0
    f_true = f.copy()
    nfev = 1

    J = J0
    njev = 1
    m, n = J.shape

    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
        J, f = scale_for_robust_loss_function(J, f, rho)
    else:
        cost = 0.5 * np.dot(f, f)

    g = compute_grad(J, f)

    jac_scale = isinstance(x_scale, str) and x_scale == "jac"
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    Delta = norm(x0 * scale_inv)
    if Delta == 0:
        Delta = 1.0

    if tr_solver == "lsmr":
        reg_term = 0
        damp = tr_options.pop("damp", 0.0)
        regularize = tr_options.pop("regularize", True)

    if max_nfev is None:
        max_nfev = x0.size * 100

    alpha = 0.0  # "Levenberg-Marquardt" parameter

    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None

    if verbose == 2:
        print_header_nonlinear()

    while True:
        g_norm = norm(g, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if cost < vtol:
            termination_status = 3
            print(
                f"Terminating optimization because cost is less than threshold {vtol}. "
                "Because this condition was hacked in, it'll say termination by xtol"
            )

        if verbose == 2:
            print_iteration_nonlinear(iteration, nfev, cost, actual_reduction, step_norm, g_norm)

        if termination_status is not None or nfev == max_nfev:
            break

        d = scale
        g_h = d * g

        if tr_solver == "exact":
            J_h = J * d
            U, s, V = svd(J_h, full_matrices=False)
            V = V.T
            uf = U.T.dot(f)
        elif tr_solver == "lsmr":
            J_h = right_multiplied_operator(J, d)

            if regularize:
                a, b = build_quadratic_1d(J_h, g_h, -g_h)
                to_tr = Delta / norm(g_h)
                ag_value = minimize_quadratic_1d(a, b, 0, to_tr)[1]
                reg_term = -ag_value / Delta**2

            damp_full = (damp**2 + reg_term) ** 0.5
            gn_h = lsmr(J_h, f, damp=damp_full, **tr_options)[0]
            S = np.vstack((g_h, gn_h)).T
            S, _ = qr(S, mode="economic")
            JS = J_h.dot(S)
            B_S = np.dot(JS.T, JS)
            g_S = S.T.dot(g_h)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            if tr_solver == "exact":
                step_h, alpha, n_iter = solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=alpha)
            elif tr_solver == "lsmr":
                p_S, _ = solve_trust_region_2d(B_S, g_S, Delta)
                step_h = S.dot(p_S)

            predicted_reduction = -evaluate_quadratic(J_h, g_h, step_h)
            step = d * step_h
            x_new = x + step
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step_h)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            # Usual trust-region step quality estimation.
            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new

            Delta_new, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction, step_h_norm, step_h_norm > 0.95 * Delta
            )

            step_norm = norm(step)
            termination_status = check_termination(actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

        if actual_reduction > 0:
            x = x_new

            f = f_new
            f_true = f.copy()

            cost = cost_new

            J = jac(x, f)
            njev += 1

            if loss_function is not None:
                rho = loss_function(f)
                J, f = scale_for_robust_loss_function(J, f, rho)

            g = compute_grad(J, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

    if termination_status is None:
        termination_status = 0

    active_mask = np.zeros_like(x)
    return OptimizeResult(
        x=x,
        cost=cost,
        fun=f_true,
        jac=J,
        grad=g,
        optimality=g_norm,
        active_mask=active_mask,
        nfev=nfev,
        njev=njev,
        status=termination_status,
    )
