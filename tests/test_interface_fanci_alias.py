"""Unit tests for fanpy.interface.fanci.alias"""

import numpy as np
import pytest

from fanpy.interface.fanci.alias import Alias

# Note: the Alias class is used in the legacy interface
# Thus it is expected that it will be depricated in the future
# These tests act as a sanity check for Alias, but they do not test 
# if Alias is implemented properly.


def test_init():
    """Check init type checks and if Alias initializes correctly."""
    #### INPUT CHECKS ###

    # wrong data types 
    with pytest.raises(TypeError):
        Alias("not a pvec")
    
    # out of range data
    pvec = np.asarray([2.0, 0.5])
    with pytest.raises(ValueError):
        Alias(pvec)
    pvec_neg = np.asarray([-0.5, 0.2])
    with pytest.raises(ValueError):
        Alias(pvec_neg)

    # warning if 2d array
    pvec = np.asarray([[0.1, 0.2], [0.3, 0.5]])
    with pytest.warns():
        a = Alias(pvec)
        assert a.n == 4

    #### NORMAL INIT ###
    a = Alias(np.asarray([0.3, 0.25, 0.15, 0.2, 0.1]))
    assert a.n == 5 # check size of input vector calculated properly

@pytest.mark.parametrize("n", [100, 0, -1])
def test_call_value_checks(n):
    """check if errors for out of range n are raised appropriately."""
    pvec = np.asarray([0.3, 0.25, 0.15, 0.2, 0.1])
    a = Alias(pvec)
    with pytest.raises(ValueError):
        a(n)

def test_call_type_check():
    """make sure we cannot pass floats to the while loop in __call__"""
    pvec = np.asarray([0.3, 0.25, 0.15, 0.2, 0.1])
    a = Alias(pvec)
    with pytest.raises(TypeError):
        a(0.1)

@pytest.mark.parametrize("n", [1, 3, 5])
def test_call(n):
    """Check if we generate the expected amount of random idx and they are within the range"""
    pvec = np.asarray([0.3, 0.25, 0.15, 0.2, 0.1])
    a = Alias(pvec)
    idx = a(n)
    assert idx.size == n
    assert np.all((idx >= 0) & (idx < pvec.size))