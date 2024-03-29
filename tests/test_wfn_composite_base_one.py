"""Test wfn.wavefunction.composite.base_one."""
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.composite.base_one import BaseCompositeOneWavefunction

import numpy as np

import pytest

from utils import disable_abstract, skip_init


class TempWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abstract class structure."""

    _spin = None
    _seniority = None

    def get_overlap(self):
        """Do nothing."""
        pass

    @property
    def spin(self):
        """Return the spin of the wavefunction."""
        return self._spin

    @property
    def seniority(self):
        """Return the seniority of the wavefunction."""
        return self._seniority

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction."""
        if params is None:
            params = np.identity(10)

        super().assign_params(params=params, add_noise=add_noise)


def test_assign_wfn():
    """Test BaseCompositeOneWavefunction.assign_wfn."""
    test = skip_init(disable_abstract(BaseCompositeOneWavefunction))
    with pytest.raises(TypeError):
        BaseCompositeOneWavefunction.assign_wfn(test, 1)
    with pytest.raises(TypeError):
        BaseCompositeOneWavefunction.assign_wfn(test, (TempWavefunction(4, 10),))
    test.nelec = 4
    with pytest.raises(ValueError):
        BaseCompositeOneWavefunction.assign_wfn(test, TempWavefunction(5, 10))
    test.memory = np.inf
    BaseCompositeOneWavefunction.assign_wfn(test, TempWavefunction(4, 10))
    assert test.wfn.nelec == 4
    assert test.wfn.nspin == 10


def test_init():
    """Test BaseCompositeOneWavefunction.__init__."""
    wfn_one = TempWavefunction(4, 10)
    wfn = disable_abstract(BaseCompositeOneWavefunction)(
        4, 10, wfn_one, params=np.array([0]), enable_cache=True
    )
    assert wfn.nelec == 4
    assert wfn.nspin == 10
    assert wfn.wfn == wfn_one
    assert np.allclose(wfn.params, 0)
    assert wfn._cache_fns["overlap"]
    assert wfn._cache_fns["overlap derivative"]

    wfn = disable_abstract(BaseCompositeOneWavefunction)(
        4, 10, wfn_one, params=np.array([0]), enable_cache=False
    )
    with pytest.raises(AttributeError):
        wfn._cache_fns["overlap"]
    with pytest.raises(AttributeError):
        wfn._cache_fns["overlap derivative"]


def test_save_params(tmp_path):
    """Test BaseCompositeOneWavefunction.save_params."""
    wfn_one = TempWavefunction(4, 10)
    wfn_one.assign_params(np.random.rand(10))
    wfn = disable_abstract(BaseCompositeOneWavefunction)(
        4, 10, wfn_one, params=np.random.rand(7), enable_cache=True
    )
    wfn.save_params(str(tmp_path / "temp.npy"))
    assert np.allclose(np.load(str(tmp_path / "temp.npy")), wfn.params)
    assert np.allclose(np.load(str(tmp_path / "temp_TempWavefunction.npy")), wfn.wfn.params)
