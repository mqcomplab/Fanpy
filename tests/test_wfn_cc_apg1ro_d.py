"""Test fanpy.wavefunction.cc.apg1ro_d."""
import pytest
from fanpy.wfn.cc.apg1ro_d import APG1roD


class TempAPG1roD(APG1roD):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_exops():
    """Test APG1roD.assign_exops."""
    test = TempAPG1roD()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    test.assign_ranks()
    with pytest.raises(TypeError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_exops()
    assert test.exops == {(0, 4, 2, 3): 0, (0, 4, 2, 6): 1, (0, 4, 2, 7): 2, (0, 4, 3, 6): 3,
                          (0, 4, 3, 7): 4, (0, 4, 6, 7): 5, (1, 5, 2, 3): 6, (1, 5, 2, 6): 7,
                          (1, 5, 2, 7): 8, (1, 5, 3, 6): 9, (1, 5, 3, 7): 10, (1, 5, 6, 7): 11}
