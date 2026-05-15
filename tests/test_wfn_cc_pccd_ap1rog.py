"""Test fanpy.wavefunction.cc.pccd_ap1rog."""
import pytest
import numpy as np
from fanpy.tools import slater
from fanpy.wfn.cc.pccd_ap1rog import PCCD


class TempPCCD(PCCD):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_nelec():
    """Test PCCD.assign_nelec."""
    test = TempPCCD()
    test.assign_nelec(4)
    assert test.nelec == 4
    with pytest.raises(TypeError):
        test.assign_nelec(4.0)
    with pytest.raises(ValueError):
        test.assign_nelec(-4)
    with pytest.raises(ValueError):
        test.assign_nelec(5)


def test_assign_ranks():
    """Test PCCD.assign_ranks."""
    test = TempPCCD()
    with pytest.raises(ValueError):
        test.assign_ranks([1, 2])
    test.assign_nelec(2)
    with pytest.raises(ValueError):
        test.assign_ranks([3])
    test.assign_nelec(4)
    test.assign_ranks()
    assert test.ranks == [2]


def test_assign_exops():
    """Test PCCD.assign_exops."""
    test = TempPCCD()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    with pytest.raises(TypeError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_exops()
    assert test.exops == {(0, 4, 2, 6): 0, (0, 4, 3, 7): 1, (1, 5, 2, 6): 2, (1, 5, 3, 7): 3}


def test_assign_refwfn():
    """Test PCCD.assign_refwfn."""
    test = TempPCCD()
    test.assign_nelec(4)
    test.assign_nspin(8)
    with pytest.raises(TypeError):
        test.assign_refwfn("This is not a gmpy2 instance")
    with pytest.raises(ValueError):
        test.assign_refwfn(0b00010001)
    with pytest.raises(ValueError):
        test.assign_refwfn(0b0001100011)
    with pytest.raises(ValueError):
        test.assign_refwfn(0b11000011)
    test.assign_refwfn()
    assert test.refwfn == (0b00110011)


def test_assign_refwfn_sen_0_check():
    """Test seniority-0 check in PCCD.assign_refwfn."""
    test = TempPCCD()
    test.assign_nelec(4)
    test.assign_nspin(16)
    with pytest.raises(ValueError):
        test.assign_refwfn(0b11000011)
    with pytest.raises(ValueError):
        test.assign_refwfn(0b0000010100000011)
    test.assign_refwfn()
    assert test.refwfn == (0b0000001100000011)


def test_assign_s_type():
    """Test PCCD.assign_s_type."""
    test = TempPCCD()

    test.assign_s_type("free")
    assert test.s_type == "free"

    test.assign_s_type("sen-o")
    assert test.s_type == "sen-o"

    test.assign_s_type("sen-v")
    assert test.s_type == "sen-v"

    test.assign_s_type("sen-ov")
    assert test.s_type == "sen-ov"


def test_assign_s_type_invalid():
    """Test invalid s_type."""
    test = TempPCCD()

    with pytest.raises(ValueError):
        test.assign_s_type("bad-option")


def test_init_s_type():
    """Test initialization of s_type."""
    test = PCCD(4, 8, s_type="sen-v")
    assert test.s_type == "sen-v"


def tests_type_effct_on_pCCD_overlap():
    """Test s_type effect on pCCD overlap.
    pCCD doesn't have singles so it shouldn't enter 
    into the sen-x logics of singles.
    """

    sd = 0b10100011

    test_free = PCCD(4, 8, s_type="free")
    test_seno = PCCD(4, 8, s_type="sen-o")

    olp_free = test_free.get_overlap(sd)
    olp_seno = test_seno.get_overlap(sd)

    assert olp_free == olp_seno


def test_olp_double_derivative_shape():
    """Test overlap Hessian shape."""

    test = PCCD(4, 8)

    hess = test._olp_double_derivative(test.refwfn)

    assert hess.shape == (test.nparams, test.nparams)


def test_olp_double_derivative_symmetric():
    """Test overlap Hessian symmetry."""

    test = PCCD(4, 8)

    sd = 0b11001100

    hess = test._olp_double_derivative(sd)

    assert np.allclose(hess, hess.T)


def test_olp_double_derivative_zero_diagonal():
    """Test Hessian diagonal vanishes."""

    test = PCCD(4, 8)

    sd = 0b11001100

    hess = test._olp_double_derivative(sd)

    assert np.allclose(np.diag(hess), 0)


def test_olp_double_derivative_finite_difference():
    """Test overlap Hessian with finite differences."""

    test = PCCD(4, 8)

    sd = 0b11001100

    h = 1e-7

    analytic = test._olp_double_derivative(sd)

    numerical = np.zeros_like(analytic)

    orig = test.params.copy()

    for j in range(test.nparams):

        test.params = orig.copy()
        test.params[j] += h
        plus = test._olp_deriv(sd)

        test.params = orig.copy()
        test.params[j] -= h
        minus = test._olp_deriv(sd)

        numerical[:, j] = (plus - minus) / (2 * h)

    test.params = orig

    assert np.allclose(analytic, numerical, atol=1e-5)
