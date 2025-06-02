"""
Wrapper for adaptive solvers in FanPy.
"""

from fanpy.eqn.adaptive_projected import AdaptiveProjectedSchrodinger

import numpy as np


def adaptive_solver(solver, objective, update_objective_parameters=True, **kwargs):
    """Wrapper for adaptive solvers in FanPy.

    Parameters
    ----------
    solver : callable
        The numerical solver to be used.
    objective : AdaptiveProjectedSchrodinger
        The objective function to be optimized, which should be an instance of
        `AdaptiveProjectedSchrodinger`.
    **kwargs : dict
        Keyword arguments to be passed to the solver.

    Returns
    -------
    dict
        A dictionary containing the results of the solver.

    """

    if not isinstance(objective, AdaptiveProjectedSchrodinger):
        raise TypeError("Given objective must be an instance of AdaptiveProjectedSchrodinger.")

    def print_adaptive_step(_objective, _results):
        """Prints the current adaptive step information."""
        energy = _results["energy"]
        cost = np.sum(_results["residuals"])

        print("\n(Adaptive Optimization) Electronic energy: {}".format(energy))
        print("(Adaptive Optimization) Cost: {}".format(cost))
        if _objective.constraints:
            cost_constraints = np.sum(_results["residuals"][_objective.nproj :])
            print("(Adaptive Optimization) Cost from constraints: {}".format(cost_constraints))

    # Check adaptive objective kwargs
    adaptive_kwargs = {}
    adaptive_kwargs["residuals_thresholds"] = kwargs.pop("residuals_thresholds", [])

    # Initial evaluation of the objective function using the initial p-space
    results = solver(objective, **kwargs)

    if objective.adaptive_step_print:
        print_adaptive_step(objective, results)

    # Update the objective parameters based on the previous results
    if update_objective_parameters:
        objective.wfn.assign_params(results["params"])

    adaptive_kwargs["residuals"] = results["residuals"]

    # Reset Adaptive convergence and completion flags
    objective._is_adaptive_procedure_done = False
    objective._is_adaptive_procedure_converged = False

    # Run the adaptive optimization loop until convergence criteria are met
    while (not objective.is_adaptive_procedure_done) and (not objective.is_adaptive_procedure_converged):

        objective.update_current_pspace(**adaptive_kwargs)
        results = solver(objective, **kwargs)

        if objective.adaptive_step_print:
            print_adaptive_step(objective, results)
        adaptive_kwargs["residuals"] = results["residuals"]

        objective.check_adaptive_step_convergence(**adaptive_kwargs)
        objective.check_adaptive_procedure(**adaptive_kwargs)

        # Update the objective parameters based on the previous results
        if update_objective_parameters:
            objective.wfn.assign_params(results["params"])

    return results
