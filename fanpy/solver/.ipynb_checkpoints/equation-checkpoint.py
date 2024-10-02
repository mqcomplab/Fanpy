"""Solvers for single equations."""
from fanpy.eqn.base import BaseSchrodinger
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.energy_twoside import EnergyTwoSideProjection
from fanpy.eqn.least_squares import LeastSquaresEquations
from fanpy.solver.wrappers import wrap_scipy

import numpy as np


def cma(objective, **kwargs):
    """Solve an equation using Covariance Matrix Adaptation Evolution Strategy.

    See module `cma` for details.

    Parameters
    ----------
    objective : BaseSchrodinger
        Instance that contains the function that will be optimized.
    kwargs : dict
        Keyword arguments to `cma.fmin`. See its documentation for details.
        By default, 'sigma0' is set to 0.01 and 'options' to `{'ftarget': None, 'timeout': np.inf,
        'tolfun': 1e-11, 'verb_filenameprefix': 'outcmaes', 'verb_log': 0}`.
        The 'sigma0' is the initial standard deviation. The optimum is expected to be within
        `3*sigma0` of the initial guess.
        The 'ftarget' is the termination condition for the function value upper limit.
        The 'timeout' is the termination condition for the time. It is provided in seconds and must
        be provided as a string.
        The 'tolfun' is the termination condition for the change in function value.
        The 'verb_filenameprefix' is the prefix of the logger files that will be written to disk.
        The 'verb_log' is the verbosity of the logger files that will be written to disk. `0` means
        that no logs will be saved.
        See `cma.evolution_strategy.cma_default_options` for more options.

    Returns
    -------
    Dictionary with the following keys and values:
    success : bool
        True if optimization succeeded.
    params : np.ndarray
        Parameters at the end of the optimization.
    energy : float
        Energy after optimization.
        Only available for objectives that are EnergyOneSideProjection, EnergyTwoSideProjection, and
        LeastSquaresEquations instances.
    message : str
        Termination reason.
    internal : list
        Returned value of the `cma.fmin`.

    Raises
    ------
    TypeError
        If objective is not BaseSchrodinger instance.
    ValueError
        If objective has more than one equation.

    """
    import cma as solver  # pylint: disable=C0415

    if not isinstance(objective, BaseSchrodinger):
        raise TypeError("Objective must be a BaseSchrodinger instance.")
    if objective.num_eqns != 1:
        raise ValueError("Objective must contain only one equation.")

    # disable hamiltonian update because algorithm is stochastic
    objective.ham.update_prev_params = False
    # disable print with each iteration b/c solver prints
    objective.step_print = False

    kwargs.setdefault("sigma0", 0.01)
    kwargs.setdefault("options", {})
    kwargs["options"].setdefault("ftarget", None)
    kwargs["options"].setdefault("timeout", np.inf)
    kwargs["options"].setdefault("tolfun", 1e-11)
    kwargs["options"].setdefault("verb_log", 0)

    if objective.active_params.size == 1:
        raise ValueError("CMA solver cannot be used on objectives with only one parameter.")

    results = solver.fmin(objective.objective, objective.active_params, **kwargs)

    output = {}
    output["success"] = results[-3] != {}
    output["params"] = results[0]
    output["function"] = results[1]

    if isinstance(objective, LeastSquaresEquations):
        output["energy"] = objective.energy.params
    elif isinstance(  # pragma: no branch
        objective, (EnergyOneSideProjection, EnergyTwoSideProjection)
    ):
        output["energy"] = results[1]

    if output["success"]:  # pragma: no branch
        output["message"] = "Following termination conditions are satisfied:" + "".join(
            " {0}: {1},".format(key, val) for key, val in results[-3].items()
        )
        output["message"] = output["message"][:-1] + "."
    else:  # pragma: no cover
        output["message"] = "Optimization did not succeed."

    output["internal"] = results

    objective.assign_params(results[0])
    objective.save_params()

    return output


def minimize(objective, use_gradient=True, **kwargs):
    """Solve an equation using `scipy.optimize.minimize`.

    See module `scipy.optimize.minimize` for details.

    Parameters
    ----------
    objective : BaseSchrodinger
        Instance that contains the function that will be optimized.
    use_gradient : bool
        Option to use gradient.
        Default is True.
    kwargs : dict
        Keyword arguments to `scipy.optimize.minimize`. See its documentation for details.
        By default, if the objective has a gradient,  'method' is 'BFGS', 'jac' is the gradient
        of the objective, and 'options' is `{'gtol': 1e-8}`.
        By default, if the objective does not have a gradient, 'method' is 'Powell' and 'options' is
        `{'xtol': 1e-9, 'ftol': 1e-9}}`.

    Returns
    -------
    Dictionary with the following keys and values:
    success : bool
        True if optimization succeeded.
    params : np.ndarray
        Parameters at the end of the optimization.
    energy : float
        Energy after optimization.
        Only available for objectives that are EnergyOneSideProjection, EnergyTwoSideProjection, and
        LeastSquaresEquations instances.
    message : str
        Message returned by the optimizer.
    internal : list
        Returned value of the `scipy.optimize.minimize`.

    Raises
    ------
    TypeError
        If objective is not BaseSchrodinger instance.
    ValueError
        If objective has more than one equation.

    """
    import scipy.optimize  # pylint: disable=C0415

    if not isinstance(objective, BaseSchrodinger):
        raise TypeError("Objective must be a BaseSchrodinger instance.")
    if objective.num_eqns != 1:
        raise ValueError("Objective must contain only one equation.")

    if use_gradient:
        kwargs.setdefault("method", "BFGS")
        kwargs.setdefault("jac", objective.gradient)
        kwargs.setdefault("options", {})
        kwargs["options"].setdefault("gtol", 1e-8)
    else:
        kwargs.setdefault("method", "Powell")
        kwargs.setdefault("options", {})
        kwargs["options"].setdefault("xtol", 1e-9)
        kwargs["options"].setdefault("ftol", 1e-9)

    ham = objective.ham
    ham.update_prev_params = False

    def update_iteration(*args):  # pylint: disable=W0613
        """Clean up at the end of each iteration."""
        # update hamiltonian
        if objective.indices_component_params[ham].size > 0:
            ham.update_prev_prams = True
            ham.assign_params(ham.params)
            ham.update_prev_prams = False
        # save parameters
        objective.save_params()
        # print
        for key, value in objective.print_queue.items():
            print("(Mid Optimization) {}: {}".format(key, value))

    kwargs.setdefault("callback", update_iteration)

    output = wrap_scipy(scipy.optimize.minimize)(objective, **kwargs)
    output["function"] = output["internal"].fun
    if isinstance(objective, LeastSquaresEquations):
        output["energy"] = objective.energy.params
    elif isinstance(  # pragma: no branch
        objective, (EnergyOneSideProjection, EnergyTwoSideProjection)
    ):
        output["energy"] = output["function"]

    return output
