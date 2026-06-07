"""Test fanpy.wfn.utils."""
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.tools import slater
from fanpy.wfn.geminal.ap1rog import AP1roG
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.utils import wfn_factory

import numpy as np

import pytest

from utils import find_datafile, skip_init


def test_wfn_utils():
    """Test wfn.utils.wfn_factory."""

    def olp(sd, params):
        """Test overlap."""
        return np.sum(params)

    def olp_deriv(sd, params):
        """Test overlap deriv."""
        return params

    def assign_params(self, params):
        """Test assign_params."""
        self.params = np.array(params)

    params = np.random.rand(100)

    wfn = wfn_factory(olp, olp_deriv, 3, 6, params)
    assert wfn.nspin == 6
    assert np.allclose(wfn.params, params)
    assert np.allclose(wfn.get_overlap(0b000111), np.sum(params))
    assert np.allclose(wfn.get_overlap(0b000111, np.arange(50)), params[:50])

    wfn = wfn_factory(olp, olp_deriv, 3, 6, params.tolist(), assign_params=assign_params)
    assert wfn.nspin == 6
    assert np.allclose(wfn.params, params)
    assert np.allclose(wfn.get_overlap(0b000111), np.sum(params))
    assert np.allclose(wfn.get_overlap(0b000111, np.arange(50)), params[:50])