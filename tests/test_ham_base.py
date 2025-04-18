"""Test fanpy.ham.base."""

from fanpy.ham.base import BaseHamiltonian
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np

import pytest


class TempBaseHamiltonian(BaseHamiltonian):
    """Empty container class."""

    def __init__(self):
        """Fake init."""

    @property
    def nspin(self):
        """Return number of spin orbitals."""
        return 10

    def integrate_sd_sd(self, sd1, sd2, deriv=None):
        """Integrate slater determinants."""
        return 1


def test_assign_integrals():
    """Test BaseHamiltonian.assign_integrals."""
    test = TempBaseHamiltonian()
    with pytest.raises(NotImplementedError):
        test.assign_integrals(None, None)


def test_integrate_sd_wfn():
    """Test BaseHamiltonian.integrate_sd_wfn."""
    ham = TempBaseHamiltonian()
    wfn = CIWavefunction(4, 10)
    wfn.assign_params(np.random.rand(wfn.nparams))
    with pytest.raises(TypeError):
        ham.integrate_sd_wfn("1", wfn)
    with pytest.raises(ValueError):
        ham.integrate_sd_wfn(0b0001100011, wfn, wfn_deriv=np.array([1, 2]), ham_deriv=np.array([0]))
    with pytest.raises(TypeError):
        ham.integrate_sd_wfn(0b0001100011, wfn, ham_deriv=[1])
    with pytest.raises(ValueError):
        ham.integrate_sd_wfn(0b0001100011, wfn, ham_deriv=np.array([-1]))
    ham.integrate_sd_wfn(0b0001100011, wfn)


def test_nspatial():
    """Test BaseHamiltonian.nspatial."""
    test = TempBaseHamiltonian()
    assert test.nspatial == 5
