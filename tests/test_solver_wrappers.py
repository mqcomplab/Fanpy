"""Test for fanpy.solver.wrappers"""

import numpy as np
import scipy as sp
import pytest

from fanpy.solver.wrappers import wrap_scipy, wrap_skopt



class DummyObjective:
    """A dummy objective for testing purposes."""
    def __init__(self):
        self.active_params = np.array([0.0])
        self.objective = lambda x: np.sum(x)**2 + 2 * np.sum(x) + 1

@pytest.mark.parametrize("method", ["BFGS", "Nelder-Mead", "trust-constr", "Powell"])
def test_wrap_scipy_minimize(method):
    # set up dummy objective 
    objective = DummyObjective()

    ## scipy.optimize.minimize
    wrapped_minimize = wrap_scipy(sp.optimize.minimize)

    # optimize with BFGS
    results = wrapped_minimize(objective, method=method)
    assert isinstance(results, dict)
    assert "success" in results
    # Check that the result is close to the known minimum within a reasonable tolerance 
    assert np.allclose(results["params"], np.array([-1.0]), atol=1e-5)
    assert "message" in results.keys()
    assert "internal" in results.keys() # these are the raw scipy results 


def test_wrap_scipy_least_squares():
    # set up dummy objective 
    objective = DummyObjective()

    ## scipy.optimize.least_squares
    wrapped_least_squares = wrap_scipy(sp.optimize.least_squares)

    # optimize with least_squares
    results = wrapped_least_squares(objective)
    assert isinstance(results, dict)
    assert "success" in results
    # Check that the result is close to the known minimum within a reasonable tolerance 
    # Note: lstsq is less precise, it usually gives 0.999 instead of -1.0
    assert np.allclose(results["params"], np.array([-1.0]), atol=1e-3)
    assert "message" in results.keys()
    assert "internal" in results.keys() # these are the raw scipy results

def test_wrap_scipy_root():
    # set up dummy objective 
    objective = DummyObjective()

    ## scipy.optimize.root
    wrapped_root = wrap_scipy(sp.optimize.root)

    # optimize with root
    results = wrapped_root(objective, method="hybr")

    assert isinstance(results, dict)
    assert "success" in results
    # Check that the result is close to the known minimum within a reasonable tolerance 
    assert np.allclose(results["params"], np.array([-1.0]), atol=1e-5)
    assert "message" in results.keys()
    assert "internal" in results.keys() # these are the raw scipy results

@pytest.mark.parametrize("method", ["gp_minimize", "forest_minimize", "dummy_minimize"])
def test_wrap_skopt(method):
    import skopt

    # set up dummy objective 
    objective = DummyObjective()

    # get the skopt function
    skopt_func = getattr(skopt, method)
    wrapped_skopt = wrap_skopt(skopt_func)

    # optimize with skopt
    results = wrapped_skopt(objective, 
                            dimensions=[(-1.5, 1.5)], 
                            random_state=42)

    assert isinstance(results, dict)
    # Note: skopt does not get precise results. Therefore, we do not check for accuracy of the minimum
    assert type(results["params"]) is list
    assert "internal" in results.keys()

