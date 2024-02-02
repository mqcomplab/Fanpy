import os

import numpy as np
from numpy import asarray, Inf, isinf
import scipy.optimize.optimize

from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.tools import math_tools
from fanpy.solver.equation import minimize


import numpy
from scipy.optimize._optimize import (
    _epsilon, _check_unknown_options, wrap_function, approx_fprime, vecnorm, _line_search_wolfe12,
    _LineSearchError, _status_message, OptimizeResult
)


def fanpy_minimize_bfgs(fun, x0, args=(), jac=None, callback=None,
                   gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None,
                   disp=False, return_all=False,
                   **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.

    """
    schrodinger = unknown_options.pop('schrodinger')
    if (
        schrodinger.wfn in schrodinger.indices_component_params and
        schrodinger.indices_component_params[schrodinger.wfn].size
    ):
        wfn = schrodinger.wfn
        param_indices = schrodinger.indices_objective_params[wfn]
        wfn_indices = schrodinger.indices_component_params[wfn]
    else:
        wfn = None
    if (
        schrodinger.ham in schrodinger.indices_component_params and
        schrodinger.indices_component_params[schrodinger.ham].size
    ):
        ham = schrodinger.ham
        ham.update_prev_params = False
    else:
        ham = None


    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200
    func_calls, f = wrap_function(f, args)

    old_fval = f(x0)
    print(
        "(Mid Optimization) Electronic Energy: {}".format(schrodinger.print_queue['Electronic energy'])
    )

    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)
    gfk = myfprime(x0)
    k = 0
    N = len(x0)
    I = numpy.eye(N, dtype=int)
    Hk = I

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        pk = -numpy.dot(Hk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                          old_fval, old_old_fval, amin=1e-100, amax=1e100)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        xkp1 = xk + alpha_k * pk
        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)


        yk = gfkp1 - gfk
        gfk = gfkp1
        if callback is not None:
            callback(xk)
        k += 1

        # update parameters
        if ham:
            params_diff = ham.params - ham._prev_params
            unitary = math_tools.unitary_matrix(params_diff)
            ham._prev_params = ham.params.copy()
            ham._prev_unitary = ham._prev_unitary.dot(unitary)
        # save parameters
        schrodinger.save_params()
        # print
        print(
            "(Mid Optimization) Electronic Energy: {}".format(schrodinger.print_queue['Electronic energy'])
        )

        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        if not numpy.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        try:  # this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (numpy.dot(yk, sk))
        except ZeroDivisionError:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        if isinf(rhok):  # this is patch for numpy
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        A1 = I - sk[:, numpy.newaxis] * yk[numpy.newaxis, :] * rhok
        A2 = I - yk[:, numpy.newaxis] * sk[numpy.newaxis, :] * rhok
        Hk = numpy.dot(A1, numpy.dot(Hk, A2)) + (rhok * sk[:, numpy.newaxis] *
                                                 sk[numpy.newaxis, :])

    fval = old_fval

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % func_calls[0])
        print("         Gradient evaluations: %d" % grad_calls[0])

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=func_calls[0],
                            njev=grad_calls[0], status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


scipy.optimize.optimize._minimize_bfgs = fanpy_minimize_bfgs


class EnergyOneSideProjectionBFGS(EnergyOneSideProjection):
    def __init__(self, objective):
        super().__init__(
            objective.wfn, objective.ham, tmpfile=objective.tmpfile,
            param_selection=objective.indices_component_params, refwfn=objective.refwfn
        )

    def objective(self, params, parallel=None):
        return super().objective(
            params, assign=True, normalize=False, save=False
        )

    def gradient(self, params, parallel=None):
        return super().gradient(
            params, assign=True, normalize=False, save=False
        )


from fanpy.solver.wrappers import wrap_scipy


def scipy_minimize(fun, x0, args=(), method=None, jac=None, hess=None,
                   hessp=None, bounds=None, constraints=(), tol=None,
                   callback=None, options=None):
    x0 = np.asarray(x0)
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)

    if not isinstance(args, tuple):
        args = (args,)

    if options is None:
        options = {}

    return fanpy_minimize_bfgs(fun, x0, args, jac, callback, **options)


def bfgs_minimize(objective, **kwargs):
    objective = EnergyOneSideProjectionBFGS(objective)

    # objective.print_energy = False
    objective.step_print = False
    objective.ham.update_prev_params = False

    # objective.wfn.normalize()
    kwargs['method'] = 'bfgs'
    kwargs['jac'] = objective.gradient
    kwargs.setdefault('options', {"gtol": 1e-8})
    kwargs["options"]['schrodinger'] = objective


    output = wrap_scipy(scipy_minimize)(objective, **kwargs)
    output["function"] = output["internal"].fun
    output["energy"] = output["function"]

    # save parameters
    objective.save_params()

    return output

