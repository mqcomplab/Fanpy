"""Test fanpy.wavefunction.cc.pccd_ap1rog."""
import pytest
import numpy as np
from fanpy.tools import slater
from fanpy.wfn.cc.pccd_ap1rog import PCCD
from fanpy.wfn.cc.ap1rog_generalized import AP1roGSDGeneralized
from fanpy.wfn.cc.apset1rog_sd import APset1roGSD


class TempPCCD(PCCD):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}

######### ASSIGN METHODS #########

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

######### OVERLAP #########

def tests_type_effct_on_pCCD_overlap():
    """Test s_type effect on pCCD overlap.
    pCCD doesn't have singles so it shouldn't enter 
    into the sen-x logics of singles.
    """

    sd = 0b10100011

    test_free = PCCD(4, 8, s_type="free")
    params = np.random.rand(test_free.nparams)
    test_free.assign_params(params)
    test_seno = PCCD(4, 8, s_type="sen-o")
    test_seno.assign_params(params)

    olp_free = test_free.get_overlap(sd)
    olp_seno = test_seno.get_overlap(sd)

    assert olp_free == olp_seno

def test_olp_identity_is_one():
    """Test that the reference determinant has unit overlap."""
    test = PCCD(4, 8)

    assert test._olp(test.refwfn) == pytest.approx(1.0)

def test_olp_pair_excitation_matches_parameter():
    """Test that a single pair excitation returns the matching amplitude."""
    test = PCCD(4, 8)
    test.assign_params(np.array([0.25, -0.5, 0.75, -1.25]))

    sd = slater.excite(test.refwfn, 0, 4, 2, 6)
    sign = slater.sign_excite(test.refwfn, [0, 4], [2, 6])

    assert test._olp(sd) == pytest.approx(sign * test.params[test.get_ind((0, 4, 2, 6))])

def test_olp_broken_pair_is_zero():
    """Test that seniority-breaking determinants have zero overlap in pCCD."""
    test = PCCD(4, 8)
    test.assign_params(np.array([0.25, -0.5, 0.75, -1.25]))

    sd = slater.excite(test.refwfn, 0, 2)

    assert test._olp(sd) == pytest.approx(0.0)

def test_olp_single_excitation_matches_parameter():
    """Test that a single pair excitation returns the matching amplitude."""
    test = AP1roGSDGeneralized(4, 8)
    test.assign_params(np.random.rand(test.nparams))

    sd = slater.excite(test.refwfn, 0, 2)
    sign = slater.sign_excite(test.refwfn, [0], [2])

    assert test._olp(sd) == pytest.approx(sign * test.params[test.get_ind((0, 2))])

@pytest.mark.parametrize("s_type", ["sen-o", "sen-v", "sen-ov"])
def test_olp_filtering(monkeypatch, s_type):
    """Check if single excitations get filtered for s_types o, v, ov
    Note: this does not check if the filtering logic is working properly
    """

    wfn_olp = APset1roGSD(
        nelec=4,
        nspin=8,
        s_type=s_type,
    )
    wfn_indices_multi = APset1roGSD(
        nelec=4,
        nspin=8,
        s_type=s_type,
    )
    #
    # reference:
    refsd = slater.ground(4, 8)

    wfn_olp.refwfn = refsd
    wfn_indices_multi.rewfwfn = refsd

    # Build excited determinant
    # annihilate:
    #   pair from spatial orbital 0
    #   alpha electron from spatial orbital 1
    #
    # create:
    #   pair in spatial orbital 2
    #   alpha electron in spatial orbital 3
    #

    excited_sd = slater.excite(refsd, 0, 1, 4, 2, 3, 6)

    # Generate all possible exc combinations
    a_inds, c_inds = slater.diff_orbs(refsd, excited_sd)
    wfn_indices_multi.generate_possible_exops(a_inds, c_inds)

    original_indices = wfn_indices_multi.exop_combinations[tuple(a_inds + c_inds)]

    # Capture filtered indices_multi

    captured = {}

    def fake_product_amplitudes_multi(indices_multi):
        captured["indices_multi"] = indices_multi
        return 1.0

    monkeypatch.setattr(
        wfn_olp,
        "product_amplitudes_multi",
        fake_product_amplitudes_multi,
    )

    # Run overlap
    wfn_olp._olp(excited_sd)

    olp_indices = captured["indices_multi"]

    # all sen types need to remove the case, where the excited SD is generated
    # from single excitations only

    removed_any = False

    for exc_order in original_indices:
        if len(olp_indices[exc_order]) < len(original_indices[exc_order]):
            removed_any = True

    assert removed_any

    # make sure some excitations survived

    surviving = any(
        len(olp_indices[exc_order]) > 0
        for exc_order in olp_indices
    )

    assert surviving

def test_olp_no_filtering_for_free(monkeypatch):
    """
    Make sure that seniority type free does not filter single excitations. 
    """

    wfn_olp = APset1roGSD(
        nelec=4,
        nspin=8,
        s_type="free",
    )
    wfn_indices_multi = APset1roGSD(
        nelec=4,
        nspin=8,
        s_type="free",
    )
    refsd = slater.create(0, 0, 1, 4, 5)

    wfn_olp.refwfn = refsd
    wfn_indices_multi.rewfwfn = refsd

    # Build excited determinant
    # annihilate:
    #   pair from spatial orbital 0
    #   alpha electron from spatial orbital 1
    #
    # create:
    #   pair in spatial orbital 2
    #   alpha electron in spatial orbital 3
    #

    excited_sd = refsd

    excited_sd = slater.excite(refsd, 0, 1, 4, 2, 3, 6)

    # Generate all excitation combinations

    a_inds, c_inds = slater.diff_orbs(refsd, excited_sd)
    wfn_indices_multi.generate_possible_exops(a_inds, c_inds)

    original_indices = wfn_indices_multi.exop_combinations[tuple(a_inds + c_inds)]

    # Capture filtered indices_multi

    captured = {}

    def fake_product_amplitudes_multi(indices_multi):
        captured["indices_multi"] = indices_multi
        return 1.0

    monkeypatch.setattr(
        wfn_olp,
        "product_amplitudes_multi",
        fake_product_amplitudes_multi,
    )

    # Run overlap

    wfn_olp._olp(excited_sd)

    olp_indices = captured["indices_multi"]

    removed_any = False

    for exc_order in original_indices:
        if len(olp_indices[exc_order]) < len(original_indices[exc_order]):
            removed_any = True

    assert not removed_any # we should not be removing exc with s_type free

    # make sure there are exc operators present
    # this is a bit redundant, but keeping it here just in case

    surviving = any(
        len(olp_indices[exc_order]) > 0
        for exc_order in olp_indices
    )

    assert surviving

######### DERIV ##########

@pytest.mark.parametrize("sd,s_type", [[0b10100101, "sen-o"], 
                                       [0b11000101, "sen-v"], 
                                       [0b10010101, "sen-ov"], 
                                       [0b11000011, "free"], 
                                       [0b00110011, "sen-o"]])
def test_olp_derivative_finite_difference(sd, s_type):
    """Test overlap Hessian with finite differences."""

    test = AP1roGSDGeneralized(4, 8, s_type = s_type)
    test.assign_params(np.random.rand(test.nparams))

    h = 1e-7

    analytic = test._olp_deriv(sd)

    numerical = np.zeros_like(analytic)

    orig = test.params.copy()

    for j in range(test.nparams):

        test.params = orig.copy()
        test.params[j] += h
        plus = test._olp(sd)

        test.params = orig.copy()
        test.params[j] -= h
        minus = test._olp(sd)

        numerical[j] = (plus - minus) / (2 * h)

    test.params = orig

    assert np.allclose(analytic, numerical, atol=1e-5)

######### DOUBLE DERIV #########

def test_olp_double_derivative_shape():
    """Test overlap Hessian shape."""

    test = PCCD(4, 8)

    hess = test._olp_double_derivative(test.refwfn)

    assert hess.shape == (test.nparams, test.nparams)


def test_olp_double_derivative_symmetric():
    """Test overlap Hessian symmetry."""

    test = PCCD(4, 8)
    test.assign_params(np.random.rand(test.nparams))
    sd = 0b11001100

    hess = test._olp_double_derivative(sd)

    assert np.allclose(hess, hess.T)


def test_olp_double_derivative_zero_diagonal():
    """Test Hessian diagonal vanishes."""

    test = PCCD(4, 8)
    test.assign_params(np.random.rand(test.nparams))
    sd = 0b11001100

    hess = test._olp_double_derivative(sd)

    assert np.allclose(np.diag(hess), 0)

# Test double deriv with multiple s types

@pytest.mark.parametrize("sd,s_type", [[0b10100101, "sen-o"], 
                                       [0b11000101, "sen-v"], 
                                       [0b10010101, "sen-ov"], 
                                       [0b11000011, "free"], 
                                       [0b00110011, "sen-o"]])
def test_olp_double_derivative_finite_difference(sd, s_type):
    """Test overlap Hessian with finite differences."""

    test = AP1roGSDGeneralized(4, 8, s_type = s_type)
    test.assign_params(np.random.rand(test.nparams))

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
