"""Test fanpy.wavefunction.cc.seniority_cc."""
import numpy as np
import pytest
from fanpy.tools import slater
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.cc.seniority_cc import SeniorityCC


class TempSeniorityCC(SeniorityCC):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_nelec():
    """Test SeniorityCC.assign_nelec."""
    test = TempSeniorityCC()
    test.assign_nelec(4)
    assert test.nelec == 4
    with pytest.raises(TypeError):
        test.assign_nelec(4.0)
    with pytest.raises(ValueError):
        test.assign_nelec(-4)
    with pytest.raises(ValueError):
        test.assign_nelec(5)


def test_assign_refwfn():
    """Test SeniorityCC.assign_refwfn."""
    # check method
    test = TempSeniorityCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_refwfn()
    assert test.refwfn == slater.ground(nocc=test.nelec, norbs=test.nspin)
    ci_test = CIWavefunction(nelec=2, nspin=4, spin=0, seniority=0)
    test.assign_refwfn(ci_test)
    # assert test.refwfn == CIWavefunction(nelec=2, nspin=4, spin=0,
    #                                                      seniority=0)
    assert test.refwfn.nelec == 2
    assert test.refwfn.nspin == 4
    assert test.refwfn.spin == 0
    assert test.refwfn.seniority == 0
    # FIXME: check if sds is correct
    slater_test = 0b0101
    test.assign_refwfn(slater_test)
    assert test.refwfn == 0b0101
    # check errors
    # FIXME: bad tests
    with pytest.raises(TypeError):
        test.assign_refwfn("This is not a gmpy2 or CIwfn object")
    # with pytest.raises(AttributeError):
    #     test.assign_refwfn("This doesn't have a sd_vec attribute")
    with pytest.raises(ValueError):
        test.assign_refwfn(0b1111)
    # with pytest.raises(ValueError):
    #     test.assign_refwfn(0b001001)


def test_get_overlap():
    """Test SeniorityCC.get_overlap."""
    test = TempSeniorityCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_ranks([1])
    test.assign_refwfn()
    test.refresh_exops = None
    test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_params(np.array(range(17)[1:]))
    test.assign_memory("1gb")
    test.load_cache()
    sd = 0b10100011
    np.allclose([test.get_overlap(sd)], [16])

# --- Extra coverage for SeniorityCC ---
from fanpy.wfn.ci.base import CIWavefunction


def test_assign_refwfn_rejects_ci_ref_with_nonzero_seniority():
    w = SeniorityCC(nelec=2, nspin=4, ranks=[1])

    bad_ci = object.__new__(CIWavefunction)  # skip __init__
    bad_ci.nelec = w.nelec
    bad_ci.nspin = w.nspin
    bad_ci.sds = ()             # attribute must exist (assign_refwfn checks it)
    bad_ci._seniority = 2       # backing field for the read-only property

    with pytest.raises(ValueError, match="seniority-0"):
        w.assign_refwfn(bad_ci)

def test_olp_identity_is_one_and_deriv_zero():
    w = SeniorityCC(nelec=2, nspin=4, ranks=[1])
    sd = w.refwfn
    assert pytest.approx(w._olp(sd)) == 1.0
    d = w._olp_deriv(sd)
    if isinstance(d, np.ndarray):
        assert d.shape == (w.nparams,)
        assert np.allclose(d, 0.0)
    else:
        assert d == 0.0

def test_generate_possible_exops_creates_key_and_filters_by_existing_exops(capsys):
    """generate_possible_exops stores combinations only if ops exist in w.exops."""
    w = SeniorityCC(nelec=2, nspin=4, ranks=[1])
    # Construct a single valid rank-1 operator: annihilate 0, create 1 (same spin block)
    # Ensure it's in exops so the combs will be kept
    op = (0, 1)  # BaseCC stores exops as tuples of [a_inds..., c_inds...]; rank-1 => length 2
    # Some versions encode as [a, c]; SeniorityCC.generate_possible_exops checks membership directly.
    # Make sure the op is present:
    if tuple(op) not in w.exops:
        # Replace exops with a minimal set containing exactly this op
        w.exops = {tuple(op)}
        w._exop_to_ind = {tuple(op): 0}
        w._ind_to_exop = {0: tuple(op)}
        w.params = np.ones(1)

    a_inds = [0]
    c_inds = [1]
    w.generate_possible_exops(a_inds, c_inds)

    key = tuple(a_inds + c_inds)
    assert key in w.exop_combinations
    # Every stored sub-list must contain only ops that exist in w.exops
    for op_list in w.exop_combinations[key]:
        assert all(tuple(o) in w.exops for o in op_list)


def test_generate_possible_exops_resets_when_refresh_threshold_exceeded(capsys):
    """refresh_exops > 0 triggers reset of exop_combinations when length exceeds threshold."""
    w = SeniorityCC(nelec=2, nspin=4, ranks=[1], refresh_exops=1)
    # Pre-fill with two keys so len > refresh_exops
    w.exop_combinations = {
        (0, 2): [((0, 2),)],
        (1, 3): [((1, 3),)],
    }
    # Ensure at least one valid exop exists
    op = (0, 1)
    w.exops = {op}
    w._exop_to_ind = {op: 0}
    w._ind_to_exop = {0: op}
    w.params = np.ones(1)

    # This call should print "Resetting..." and replace the dict with only the new key
    w.generate_possible_exops([0], [1])
    out, _ = capsys.readouterr()
    assert "Resetting exop_combinations at size 1" in out

    assert list(w.exop_combinations.keys()) == [(0, 1)]

# -------- Extra coverage: CI reference paths & edge branches --------
import numpy as np
import pytest
from fanpy.tools import slater
from fanpy.wfn.cc.seniority_cc import SeniorityCC
from fanpy.wfn.ci.base import CIWavefunction

# A tiny CI stub that *is* a CIWavefunction, but avoids Base init;
# we just provide the attributes SeniorityCC uses.
class _FakeCI(CIWavefunction):
    def __init__(self, nelec, nspin, sd_vec, coeff=1.0):
        # don't call super().__init__
        self.nelec = nelec
        self.nspin = nspin
        self._seniority = 0   # backing field for read-only property
        self.sds = tuple(sd_vec)  # assign_refwfn checks presence of 'sds'
        self.sd_vec = list(sd_vec)
        self._coeff = coeff
    def get_overlap(self, sd):
        return self._coeff

def test_olp_with_ci_reference_accumulates():
    """Exercise the CIWavefunction branch in _olp (sum over sd_vec)."""
    w = SeniorityCC(nelec=2, nspin=4, ranks=[1])
    gs = slater.ground(nocc=w.nelec, norbs=w.nspin)
    ci = _FakeCI(w.nelec, w.nspin, [gs], coeff=1.0)
    w.assign_refwfn(ci)
    # sd == only vector in sd_vec -> temp_olp==1.0, get_overlap==1.0, sum -> 1.0
    assert pytest.approx(w._olp(gs)) == 1.0

def test_olp_deriv_with_ci_reference_zero_on_identity():
    """CI branch in _olp_deriv returns a zero vector when sd == reference vector."""
    w = SeniorityCC(nelec=2, nspin=4, ranks=[1])
    gs = slater.ground(nocc=w.nelec, norbs=w.nspin)
    ci = _FakeCI(w.nelec, w.nspin, [gs], coeff=1.0)
    w.assign_refwfn(ci)
    d = w._olp_deriv(gs)
    assert isinstance(d, np.ndarray)
    assert d.shape == (w.nparams,)
    assert np.allclose(d, 0.0)

def test_temp_olp_sign_excite_error_branch_is_caught():
    """
    Force the inner try/except ValueError path in _olp by injecting an invalid excitation:
    attempt to annihilate unoccupied orbitals and create into already-occupied ones.
    """
    w = SeniorityCC(nelec=2, nspin=4, ranks=[1, 2])

    # Reference occ is e.g. {0,2} for 2e/4spin; target sd has occ {1,3}
    refsd = w.refwfn
    sd = int('1010', 2)  # occ at 1 and 3

    # temp_olp will compute this key from diff_orbs(sd, refsd): a=[0,2], c=[1,3]
    key = (0, 2, 1, 3)

    # INVALID op: try to remove from {1,3} (unoccupied in ref) and create into {0,2} (already occupied)
    # This should make sign_excite(...) raise ValueError when applied to refsd.
    bad_exop = (1, 3, 0, 2)

    # Wire the bad op into the wavefunction
    w.exops = {bad_exop: 0}
    w._exop_to_ind = {bad_exop: 0}
    w._ind_to_exop = {0: bad_exop}
    w.params = np.ones(1)

    # Inject the bad op into combinations under the key _olp will use
    w.exop_combinations[key] = [(bad_exop,)]

    # _olp should catch the ValueError from sign_excite and skip it -> zero contribution
    assert w._olp(sd) == pytest.approx(0.0)


def test_generate_possible_exops_no_valid_ops_gives_empty_list():
    """If no ops in w.exops match, the generated combinations list should exist but be empty."""
    w = SeniorityCC(nelec=2, nspin=4, ranks=[1])
    w.exops = {}               # ensure nothing can match
    w._exop_to_ind = {}
    w._ind_to_exop = {}
    w.params = np.zeros(0)
    a_inds, c_inds = [0], [1]
    w.generate_possible_exops(a_inds, c_inds)
    assert tuple(a_inds + c_inds) in w.exop_combinations
    assert w.exop_combinations[tuple(a_inds + c_inds)] == []

def _reset_to_single_exop(w, exop_tuple):
    """Helper to force a single allowed excitation operator & 1 param."""
    w.exops = {tuple(exop_tuple): 0}
    w._exop_to_ind = {tuple(exop_tuple): 0}
    w._ind_to_exop = {0: tuple(exop_tuple)}
    w.params = np.array([1.0])
    w.exop_combinations = {}

def test_olp_nonidentity_alpha_rank1_path():
    """Hit complement_occ/complement_empty (alpha branch) and add to val."""
    w = SeniorityCC(nelec=2, nspin=4, ranks=[1])
    ref = w.refwfn
    # rank-1 alpha excitation: 0 -> 1
    _reset_to_single_exop(w, (0, 1))
    sd = slater.excite(ref, 0, 1)
    # Should pass complement checks, sign_excite succeeds, and contribute |t|=1
    assert abs(w._olp(sd)) == pytest.approx(1.0)

def test_olp_nonidentity_beta_rank1_path():
    """Hit complement_occ/complement_empty (beta branch) and add to val."""
    w = SeniorityCC(nelec=2, nspin=4, ranks=[1])
    ref = w.refwfn
    # rank-1 beta excitation: 2 -> 3 (nspatial=2, so beta indices are 2,3)
    _reset_to_single_exop(w, (2, 3))
    sd = slater.excite(ref, 2, 3)
    assert abs(w._olp(sd)) == pytest.approx(1.0)

def test_olp_deriv_nonidentity_rank1_calls_product_deriv():
    """Derivative path for non-identity: val += sign * product_amplitudes(inds, deriv=True)."""
    w = SeniorityCC(nelec=2, nspin=4, ranks=[1])
    ref = w.refwfn
    _reset_to_single_exop(w, (0, 1))
    sd = slater.excite(ref, 0, 1)
    d = w._olp_deriv(sd)
    # Single parameter -> derivative should be ±1.0 at that index
    assert isinstance(d, np.ndarray) and d.shape == (1,)
    assert abs(d[0]) == pytest.approx(1.0)

def test_generate_possible_exops_rank2_two_paths():
    """
    Hit the rank-2 generation with both decompositions:
    - two disjoint rank-1 ops: (0,1) & (2,3)
    - one rank-2 op:          (0,2, 1,3)
    Exercises int_partition_recursive, unordered partitions, disjointness, append.
    """
    w = SeniorityCC(nelec=2, nspin=4, ranks=[1, 2])
    # Allow both the pair of rank-1 ops and the single rank-2 op:
    w.exops = {(0, 1): 0, (2, 3): 1, (0, 2, 1, 3): 2}
    w._exop_to_ind = {(0, 1): 0, (2, 3): 1, (0, 2, 1, 3): 2}
    w._ind_to_exop = {0: (0, 1), 1: (2, 3), 2: (0, 2, 1, 3)}
    w.params = np.ones(3)
    w.exop_combinations = {}

    a_inds, c_inds = [0, 2], [1, 3]   # excitation rank = 2
    w.generate_possible_exops(a_inds, c_inds)
    key = (0, 2, 1, 3)
    assert key in w.exop_combinations
    combos = w.exop_combinations[key]
    # Expect at least one combo that is the two rank-1 ops, and one that is the single rank-2 op
    assert any(len(lst) == 2 and set(map(tuple, lst)) == {(0, 1), (2, 3)} for lst in combos)
    assert any(len(lst) == 1 and tuple(lst[0]) == (0, 2, 1, 3) for lst in combos)
