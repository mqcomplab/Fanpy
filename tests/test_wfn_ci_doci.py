"""Test fanpy.wavefunction.doci."""
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.ham.senzero import SeniorityZeroHamiltonian
from fanpy.solver.equation import cma
from fanpy.wfn.ci.doci import DOCI

import numpy as np

import pytest

from utils import find_datafile, skip_init


def test_assign_nelec():
    """Test DOCI.assign_nelec."""
    test = skip_init(DOCI)
    # int
    test.assign_nelec(2)
    assert test.nelec == 2
    # check errors
    with pytest.raises(TypeError):
        test.assign_nelec(None)
    with pytest.raises(TypeError):
        test.assign_nelec(2.0)
    with pytest.raises(TypeError):
        test.assign_nelec("2")
    with pytest.raises(ValueError):
        test.assign_nelec(0)
    with pytest.raises(ValueError):
        test.assign_nelec(-2)
    with pytest.raises(ValueError):
        test.assign_nelec(1)
    with pytest.raises(ValueError):
        test.assign_nelec(3)


def test_assign_spin():
    """Test DOCI.assign_spin."""
    test = skip_init(DOCI)
    test.assign_spin()
    assert test.spin == 0
    test.assign_spin(0)
    assert test.spin == 0
    with pytest.raises(ValueError):
        test.assign_spin(0.5)
    with pytest.raises(ValueError):
        test.assign_spin(1)
    with pytest.raises(ValueError):
        test.assign_spin(True)


def test_assign_seniority():
    """Test DOCI.ssign_seniority."""
    test = skip_init(DOCI)
    test.assign_seniority()
    assert test.seniority == 0
    test.assign_seniority(0)
    assert test.seniority == 0
    with pytest.raises(ValueError):
        test.assign_seniority(1)
    with pytest.raises(ValueError):
        test.assign_seniority(True)
