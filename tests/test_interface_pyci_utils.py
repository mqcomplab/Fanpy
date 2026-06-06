""" Test for PyCI utils"""

import numpy as np
import pytest

from fanpy.interface.pyci_utils import ConstraintAdapter
from fanpy.eqn.base import BaseSchrodinger
from fanpy.eqn.constraints.energy import EnergyConstraint
from interface_utils import FakeHamiltonian, FakeWavefunction

class FakeConstraint(BaseSchrodinger):
    """ Fake constraint class for testing purposes"""
    def __init__(self):
        pass

    def objective(self, x):
        """ Returns lenght of input as the objective. 
         Parameters
         ----------
         x : np.ndarray
            input array that should only contain wfn params
        
        Returns
        -------
        result : int
            length of input array
        """
        
        result = len(x)
        return result
    
    def gradient(self, x):
        """ fake gradient for testing purposes. 
        
        Parameters
        ----------
        x : np.ndarray
            input array that should only contain wfn params
        
        Returns
        -------
        result : np.ndarray
            Ones array of length of the input array. 
        """

        return np.ones(len(x))
    
def test_init_check():
    with pytest.raises(TypeError):
        ConstraintAdapter("not a constraint")
    
def test_objective():
    """ Make sure objective gets correct number of parameters"""
    fake_const = FakeConstraint()
    adapted_const = ConstraintAdapter(fake_const)
    x = np.ones(5)
    assert adapted_const.objective(x) == len(x) - 1

def test_gradient():
    """ Check gradient dimension and energy padding. """
    fake_const = FakeConstraint()
    adapted_const = ConstraintAdapter(fake_const)
    x = np.ones(5)
    adapted_grad = adapted_const.gradient(x)
    assert adapted_grad[-1] == 0
    assert np.allclose(adapted_grad[:-1], np.ones(4))

def test_integration():
    """ Make sure Constraint adapter works with Energy constraint. """
    params = np.ones(6)
    fake_wfn = FakeWavefunction(4, 8, params)
    one_int = np.ones((4, 4))
    two_int = np.ones((4, 4, 4, 4))
    fake_ham = FakeHamiltonian(one_int, two_int)
    e_const = EnergyConstraint(fake_wfn, fake_ham)
    e_const_adapted = ConstraintAdapter(e_const)

    # check if objective runs
    obj = e_const_adapted.objective(np.ones(len(params)+1))

    # check if gradient runs
    grad = e_const_adapted.gradient(np.ones(len(params)+1))
    assert np.allclose(grad.shape, (len(params)+1, ))
    assert grad[-1] == 0

