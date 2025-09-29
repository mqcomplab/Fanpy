from fanpy.wfn.ci.hci import hCI as hCI
import numpy as np
from fanpy.tools import slater
import pytest
import re


def make_minimal(nelec=2, nspin=6, *, version="new", pattern="pos_diag", hierarchy=1.0):
    """Construct and return a minimal hCI wavefunction instance with safe default settings.

    Parameters
    ----------
    nelec : int, optional
        Number of electrons.
    nspin : int, optional
        Number of spin orbitals.
    version : {"new", "old"}, optional
        hCI version to use.
    pattern : str, optional
        hCI pattern for partitioning (e.g. "pos_diag", "neg_diag", "hch").
    hierarchy : float, optional
        Hierarchy value controlling truncation.

    Returns
    -------
    hCI
        An initialized hCI wavefunction object configured with the given parameters
        and default settings (no custom determinants, memory limits, or reference
        wavefunction).
    """
    w = hCI(
        nelec=nelec,
        nspin=nspin,
        hci_version=version,
        hci_pattern=pattern,
        hierarchy=hierarchy,
        sds=None,
        memory=None,
        refwfn=None,
    )
    return w


# ---------- assign_hci_pattern ----------
def test_assign_hci_pattern_rejects_none():
    """assign_hci_pattern(None) should raise TypeError."""
    w = make_minimal()
    with pytest.raises(TypeError):
        w.assign_hci_pattern(None)


def test_assign_hci_pattern_rejects_unknown():
    """assign_hci_pattern() should reject unknown pattern strings."""
    w = object.__new__(hCI)
    with pytest.raises(TypeError, match="Unknown hci_pattern"):
        w.assign_hci_pattern("diagonal_but_weird")


# ---------- assign_hci_version ----------
def test_assign_hci_version_defaults_to_new():
    """If hci_version is None, default should be 'new'."""
    w = object.__new__(hCI)
    w.assign_hci_version(None)
    assert w.hci_version == "new"


def test_assign_hci_version_typeerror_for_bad_value():
    """assign_hci_version() should raise for invalid strings."""
    w = object.__new__(hCI)
    with pytest.raises(TypeError, match="must be 'old', 'new', or None"):
        w.assign_hci_version("ancient")


# ---------- assign_alphas ----------
def test_assign_alphas_requires_pattern_first():
    """assign_alphas() should fail if hci_pattern not assigned first."""
    w = object.__new__(hCI)
    with pytest.raises(AttributeError, match="hci_pattern must be assigned"):
        w.assign_alphas()


def test_assign_alphas_from_known_pattern_and_bounds():
    """assign_alphas() sets alpha1, alpha2 with bounded values [-1,1]."""
    w = object.__new__(hCI)
    w.assign_hci_pattern("pos_diag")
    w.assign_alphas()
    #check attributes exist
    assert hasattr(w, "alpha1")
    assert hasattr(w, "alpha2")

    #check value bounds
    assert -1 <= w.alpha1 <= 1
    assert -1 <= w.alpha2 <= 1


# ---------- assign_hierarchy ----------
def test_assign_hierarchy_type_errors():
    """assign_hierarchy() should raise TypeError for non-numeric inputs."""
    w = object.__new__(hCI)
    with pytest.raises(TypeError, match="integer, float or `None`"):
        w.assign_hierarchy(hierarchy="hi")


def test_assign_hierarchy_accepts_int_float_none():
    """assign_hierarchy() should accept int, float, or None."""
    w = object.__new__(hCI)
    w.assign_hierarchy(2)
    assert w.hierarchy == 2
    w.assign_hierarchy(2.5)
    assert w.hierarchy == 2.5
    w.assign_hierarchy(None)
    assert w.hierarchy is None


# ---------- assign_pos_hierarchies ----------
@pytest.mark.parametrize(
    "pattern, hierarchy, expect",
    [
        ("pos_diag", 2.0, np.array([0.0, 0.5, 1.0, 1.5, 2.0])),
        ("hch",      1.5, np.array([0.0, 0.5, 1.0, 1.5])),
        ("neg_diag", 3.0, np.array([0, 1, 2, 3])),
    ]
)
def test_pos_hierarchies_new_non_vch(pattern, hierarchy, expect):
    """assign_pos_hierarchies() should compute correct sequence for new-version non-VCH patterns.

    Parameters
    ----------
    pattern : str
        Partitioning scheme of the Hilbert space (e.g., "pos_diag", "neg_diag", "hch").
    hierarchy : float
        Numerical value controlling the truncation level of excitations/seniority sectors.
    expect : list of float
        possible hierarchies computed positive hierarchy values consistent with the chosen version and pattern.
    """
    w = object.__new__(hCI)
    w.assign_hci_version("new")
    w.assign_hci_pattern(pattern)
    w.assign_hierarchy(hierarchy)
    w.assign_pos_hierarchies()
    np.testing.assert_allclose(np.asarray(w.pos_hierarchies), expect)


def test_pos_hierarchies_new_vch_sym_range():
    """VCH pattern should produce symmetric range [-h, ..., h]."""
    w = object.__new__(hCI)
    w.assign_hci_version("new")
    w.assign_hci_pattern("vch")
    w.assign_hierarchy(2)
    w.assign_pos_hierarchies()
    assert np.array_equal(np.asarray(w.pos_hierarchies), np.array([-2, -1, 0, 1, 2]))


def test_pos_hierarchies_old_uses_single_value():
    """Old version should set pos_hierarchies as a single-element list."""
    w = object.__new__(hCI)
    w.assign_hci_version("old")
    w.assign_hci_pattern("pos_diag")
    w.assign_hierarchy(1.25)
    w.assign_pos_hierarchies()
    assert w.pos_hierarchies == [1.25]


# ---------- assign_refwfn ----------
def test_assign_refwfn_default_is_ground_state():
    """By default, refwfn should be HF ground-state determinant (integer)."""
    w = make_minimal()
    assert isinstance(w.refwfn, int)


def test_assign_refwfn_int_wrong_electron_count_raises():
    """assign_refwfn() should raise if determinant has wrong electron count."""
    w = make_minimal(nelec=2, nspin=6)
    bad_ref = 0b000111  # 3 electrons
    with pytest.raises(ValueError, match=r"refwfn must have 2 electrons"):
        w.assign_refwfn(bad_ref)


def test_assign_refwfn_bad_type_raises_typeerror():
    """assign_refwfn() should raise TypeError for invalid type inputs."""
    w = make_minimal()
    with pytest.raises(TypeError, match="CIWavefunction or a int"):
        w.assign_refwfn(refwfn="HF")


# ---------- assign_sds ----------
def test_assign_sds_rejects_custom_list():
    """Custom SDS lists should be rejected."""
    w = make_minimal()
    with pytest.raises(ValueError, match=re.escape("Only the default list of Slater determinants is allowed")):
        w.assign_sds(sds=[1, 2, 3])


# ---------- calculate_e_s_pairs ----------
def test_calculate_e_s_pairs_closed_shell_pos_diag_expected():
    """
    Verify calculate_e_s_pairs() returns expected excitation/seniority pairs
    for a closed-shell system with pos_diag pattern and hierarchy=2.0.
    """
    w = make_minimal(nelec=2, nspin=6, version="new", pattern="pos_diag", hierarchy=2.0)
    pairs = w.calculate_e_s_pairs()
    assert set(pairs) == {(0, 0), (1, 2), (2, 0), (2, 2)}

def test_assign_sds_raises_warning_when_pairs_empty(monkeypatch):
    """assign_sds() should raise Warning if calculate_e_s_pairs() returns empty list."""
    w = make_minimal()

    def _empty_pairs():
        return []

    monkeypatch.setattr(w, "calculate_e_s_pairs", _empty_pairs, raising=True)
    with pytest.raises(Warning, match="No compatible \\(e,s\\) pairs"):
        w.assign_sds()


def test_assign_sds_builds_space_includes_ground_and_unique(capsys):
    """assign_sds() should include ground-state determinant and ensure uniqueness."""
    w = make_minimal(nelec=2, nspin=6, version="new", pattern="pos_diag", hierarchy=2.0)
    w.assign_sds()

    assert isinstance(w.sds, tuple)
    assert len(w.sds) >= 1

    gs = slater.ground(w.nelec, w.nspin)
    assert isinstance(gs, int)
    assert gs in w.sds

    assert len(set(w.sds)) == len(w.sds)  # uniqueness

    out, _ = capsys.readouterr()
    assert f"version={w.hci_version}" in out and f"pattern={w.hci_pattern}" in out


def test_assign_sds_builds_correct_sds():
    """assign_sds() should build the correct default list of determinants."""
    w = make_minimal(nelec=2, nspin=6, version="new", pattern="pos_diag", hierarchy=1.5)
    w.assign_sds()
    assert w.sds == (
        0b100001,  # 33
        0b100010,  # 34
        0b000011,  # 3
        0b100100,  # 36
        0b000101,  # 5
        0b000110,  # 6
        0b101000,  # 40
        0b001001,  # 9
        0b001010,  # 10
        0b001100,  # 12
        0b110000,  # 48
        0b010001,  # 17
        0b010010,  # 18
        0b010100,  # 20
        0b011000  # 24
    )


# ---------- end-to-end construction smoke ----------
def test_full_init_smoke():
    """Smoke test: full hCI initialization should execute without errors and set key attributes."""
    w = make_minimal(nelec=2, nspin=6, version="new", pattern="hch", hierarchy=1.5)
    assert w.nelec == 2 and w.nspin == 6
    assert w.hci_version == "new"
    assert w.hci_pattern == "hch"
    assert hasattr(w, "alpha1") and hasattr(w, "alpha2")
    assert hasattr(w, "pos_hierarchies")
    assert isinstance(w.sds, tuple)
    assert isinstance(w.refwfn, int)
