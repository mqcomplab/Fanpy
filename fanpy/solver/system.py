"""Solvers for system of Schrodinger equations."""

from fanpy.eqn.least_squares import ProjectedSchrodinger
from fanpy.solver.wrappers import wrap_scipy

import numpy as np


def least_squares(objective, **kwargs):
    """Solve the system of Schrodinger equations as a least squares problem.

    This function wraps around `scipy.optimize.least_squares`.

    Parameters
    ----------
    objective : ProjectedSchrodinger
        Objective that describes the system of equations.
    kwargs : dict
        Keyword arguments to `scipy.optimize.least_squares`. See its documentation for details.
        By default, 'xtol', 'ftol', and 'gtol' are set to `1e-15`, 'max_nfev' is 1000 times the
        number of active parameters, and `jac` is the Jacobian provided by the objective.
        If Jacobian from the objective is not used, then `3-point` scheme should be used for
        parameters that are floats and `cs` scheme for parameters that are complex.

    Returns
    -------
    Dictionary with the following keys and values:
    success : bool
        True if optimization succeeded.
    params : np.ndarray
        Parameters at the end of the optimization.
    energy : float
        Energy of the system after optimization
    residuals  : np.ndarray
        Value of each equation in the system.
    message : str
        Message returned by the optimizer.
    internal : OptimizeResults
        Returned value of the `scipy.optimize.least_squares`.

    Raises
    ------
    TypeError
        If objective is not ProjectedSchrodinger instance.

    """
    from scipy.optimize import least_squares as solver  # pylint: disable=C0415

    if not isinstance(objective, ProjectedSchrodinger):
        raise TypeError("Given objective must be an instance of ProjectedSchrodinger.")

    kwargs.setdefault("xtol", 1.0e-15)
    kwargs.setdefault("ftol", 1.0e-15)
    kwargs.setdefault("gtol", 1.0e-15)
    kwargs.setdefault("max_nfev", 1000 * objective.active_params.size)

    ham = objective.ham
    ham.update_prev_params = False
    objective.step_print = False

    def hacked_jacobian(params):
        """Clean up at the end of each iteration and return Jacobian afterwards.

        Since `scipy.optimize.least_squares` does not have a callback option, clean up process is
        performed via Jacobian, which is evaluated at the end of each iteration.

        """
        # update hamiltonian
        if objective.indices_component_params[ham].size > 0:
            ham.update_prev_params = True
            ham.assign_params(ham.params)
            ham.update_prev_params = False
        # save parameters
        objective.save_params()
        # print
        for key, value in objective.print_queue.items():
            print("(Mid Optimization) {}: {}".format(key, value))

        return objective.jacobian(params)

    kwargs.setdefault("jac", hacked_jacobian)

    output = wrap_scipy(solver)(objective, verbose=2, **kwargs)
    output["energy"] = objective.energy.params[0]
    output["residuals"] = output["internal"].fun
    return output


def root(objective, **kwargs):
    """Solve the system of Schrodinger equations by finding its root.

    This function wraps around `scipy.optimize.root`.

    Parameters
    ----------
    objective : ProjectedSchrodinger
        Objective that describes the system of equations.
    kwargs : dict
        Keyword arguments to `scipy.optimize.least_squares`. See its documentation for details.
        By default, 'method' is 'hybr', 'jac' is the Jacobian provided, and 'options' is set to
        `{'xtol': 1e-9}`.
        by the objective.
        If Jacobian from the objective is not used, then the preferred keyword argument for 'method'
        is 'krylov' and 'option' is `{'fatol': 1.0e-9, 'xatol': 1.0e-7}`.

    Returns
    -------
    Dictionary with the following keys and values:
    success : bool
        True if optimization succeeded.
    params : np.ndarray
        Parameters at the end of the optimization.
    energy : float
        Energy of the system after optimization
    message : str
        Message returned by the optimizer.
    internal : OptimizeResults
        Returned value of the `scipy.optimize.root`.

    Raises
    ------
    TypeError
        If objective is not ProjectedSchrodinger instance.

    """
    from scipy.optimize import root as solver  # pylint: disable=C0415

    if not isinstance(objective, ProjectedSchrodinger):
        raise TypeError("Given objective must be an instance of ProjectedSchrodinger.")
    if objective.num_eqns < objective.active_params.size:
        raise ValueError(
            "Given objective must be greater than or equal to the number of equations as the number" " of parameters."
        )
    if objective.num_eqns > objective.active_params.size:
        print("Too many equations for root solver. Excess equations will be truncated.")
        num_constraints = len(objective.constraints)
        objective.pspace = objective.pspace[: objective.active_params.size - num_constraints]
        objective.eqn_weights = np.hstack(
            [
                objective.eqn_weights[: objective.active_params.size - num_constraints],
                objective.eqn_weights[-num_constraints:],
            ]
        )

    kwargs.setdefault("method", "hybr")
    kwargs.setdefault("options", {})
    kwargs["options"].setdefault("xtol", 1.0e-9)

    ham = objective.ham
    ham.update_prev_params = False
    objective.step_print = False

    def hacked_jacobian(params):
        """Clean up at the end of each iteration and return Jacobian afterwards.

        Since some methods in `scipy.optimize.root` does not have a callback option, clean up
        process is performed via Jacobian, which is evaluated at the end of each iteration.

        """
        # update hamiltonian
        if objective.indices_component_params[ham].size > 0:
            ham.update_prev_params = True
            ham.assign_params(ham.params)
            ham.update_prev_params = False
        # save parameters
        objective.save_params()
        # print
        for key, value in objective.print_queue.items():
            print("(Mid Optimization) {}: {}".format(key, value))

        return objective.jacobian(params)

    kwargs.setdefault("jac", hacked_jacobian)

    output = wrap_scipy(solver)(objective, **kwargs)
    output["energy"] = objective.energy.params
    return output
