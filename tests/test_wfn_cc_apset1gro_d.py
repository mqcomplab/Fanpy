"""Test fanpy.wavefunction.cc.apset1rog_d."""
import pytest
from fanpy.wfn.cc.apset1rog_d import APset1roGD
from fanpy.tools.slater import ground, vir_indices


class TempAPset1roGD(APset1roGD):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_exops_default_inds():
    """Test APset1roGD.assign_exops."""
    test = TempAPset1roGD()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    with pytest.raises(ValueError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_exops()
    assert test.exops == {(0, 4, 2, 6): 0, (0, 4, 2, 7): 1, (0, 4, 3, 6): 2, (0, 4, 3, 7): 3,
                          (1, 5, 2, 6): 4, (1, 5, 2, 7): 5, (1, 5, 3, 6): 6, (1, 5, 3, 7): 7}
    
def test_assign_exops_non_default_inds():
    test = TempAPset1roGD()
    nelec = 6
    nspin = 14
    test.assign_nelec(nelec)
    test.assign_nspin(nspin)
    test.assign_refwfn()
    ground_state = ground(nelec, nspin)
    vir_idx = vir_indices(ground_state, nspin).tolist()
    alpha_idx = vir_idx[:len(vir_idx)//2]
    beta_idx = vir_idx[len(vir_idx)//2:]
    # trigger non-default case, but with same alpha beta subsets
    # as for the default case. This allows easier testing. 
    test.assign_exops([alpha_idx, beta_idx])

    test_default = TempAPset1roGD()
    test_default.assign_nelec(nelec)
    test_default.assign_nspin(nspin)
    test_default.assign_refwfn()
    test_default.assign_exops()

    assert test_default.exops == test.exops

def test_assign_exops_errors():
    test = TempAPset1roGD()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()

    # not having two sets:
    with pytest.raises(TypeError, match="`indices` must have exactly 2 elements"):
        test.assign_exops([[ 2, 3 ], [6], [7, 5]])

    # non int inds: 
    with pytest.raises(TypeError, match="The elements of `indices` must be lists of non-negative ints"):
        test.assign_exops([[ 2, 3 ], [ 6, 7.5]])

    # negative inds:
    with pytest.raises(ValueError, match="All `indices` must be lists of non-negative ints"):
        test.assign_exops([[ -2, 3 ], [ 6, 7]])

    # set has occ orbitals
    with pytest.raises(ValueError, match="`indices` cannot correspond to occupied spin-orbitals"):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])

    # not a disjoint set
    with pytest.raises(ValueError, match="The sets of creation operators must be disjoint"):
        test.assign_exops([[ 3, 6], [ 6, 7]])