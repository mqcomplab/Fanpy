import numpy as np
import types
import pytest

from fanpy.fanpt.containers.base import FANPTContainer


# -----------------------------
# Minimal concrete subclass so we can instantiate the abstract base
# -----------------------------
class SimpleContainer(FANPTContainer):
    """Concrete shell: just set simple outputs so the base can run."""
    def der_g_lambda(self):
        self.d_g_lambda = np.zeros(self.nequation)

    def der2_g_lambda_wfnparams(self):
        # keep shape simple: (nequation, max(1, nactive-1))
        self.d2_g_lambda_wfnparams = np.zeros((self.nequation, max(1, self.nactive - 1)), order="F")

    def gen_coeff_matrix(self):
        self.c_matrix = np.ones((self.nequation, self.nactive))


# -----------------------------
# Simple fake objective & interface
# -----------------------------
class MiniObjective:
    """Only the attributes/methods the base class touches."""
    def __init__(self, *, nactive=3, nequation=5, nproj=4, mask_last=False, constraints=None):
        self.nactive = nactive
        self.nequation = nequation
        self.nproj = nproj
        self.mask = np.zeros(nactive, dtype=bool)
        self.mask[-1] = bool(mask_last)  # last is energy
        self.constraints = list(constraints or [])
        self.wfn = "WAVEFUNCTION"        # required by pyci.sparse_op
        # provide simple overlap methods if not supplied to __init__
        self._calls = {"ovlp": 0, "d_ovlp": 0, "dd_ovlp": 0}

    def compute_overlap(self, wfn_params, space):
        self._calls["ovlp"] += 1
        return np.arange(1, self.nproj + 1, dtype=float)

    def compute_overlap_deriv(self, wfn_params, space):
        self._calls["d_ovlp"] += 1
        return np.ones((self.nproj, self.nactive), order="F")

    def compute_overlap_double_deriv(self, wfn_params, space):
        self._calls["dd_ovlp"] += 1
        return np.zeros((self.nproj, self.nactive, self.nactive), order="F")


class MiniInterface:
    """Holds the objective and receives .pyci_ham"""
    def __init__(self, objective):
        self.objective = objective
        self.pyci_ham = None


# -----------------------------
# Helpers to build inputs quickly
# -----------------------------
def make_inputs(*, mask_last=False, constraints=None):
    obj = MiniObjective(mask_last=mask_last, constraints=constraints)
    iface = MiniInterface(obj)
    params = np.array([0.1, 0.2, -1.5])  # last is energy
    ham0, ham1 = "H0", "H1"
    return iface, params, ham0, ham1


# -----------------------------
# Tests
# -----------------------------

def test_builds_ops_and_overlaps_when_missing(monkeypatch):
    """If ops/overlaps are not provided, the base builds them and updates interface ham."""
    # Replace linear_comb_ham with a tiny function and record calls
    calls = {"linear": 0, "sparse": 0}
    def fake_linear_comb_ham(h1, h0, a1, a0):
        calls["linear"] += 1
        return {"ham1": h1, "ham0": h0, "a1": a1, "a0": a0}
    monkeypatch.setattr("fanpy.fanpt.containers.base.linear_comb_ham", fake_linear_comb_ham)

    # Replace pyci.sparse_op with a small constructor
    class FakePyci:
        @staticmethod
        def sparse_op(ham, wfn, nproj, symmetric=False):
            calls["sparse"] += 1
            return types.SimpleNamespace(kind="sparse", ham=ham, wfn=wfn, nproj=nproj, symmetric=symmetric)
    monkeypatch.setattr("fanpy.fanpt.containers.base.pyci", FakePyci)

    iface, params, ham0, ham1 = make_inputs(mask_last=True)

    cont = SimpleContainer(
        fanci_interface=iface,
        params=params,
        ham0=ham0,
        ham1=ham1,
        l=0.3,
        ref_sd=0,
        inorm=False,
        norm_det=None,
        ham_ci_op=None,
        f_pot_ci_op=None,
        ovlp_s=None,
        d_ovlp_s=None,
        dd_ovlp_s=None,
    )

    # Built once for ham (l,1-l) and once for f_pot (1,-1)
    assert calls["linear"] == 2
    # Sparse ops created for ham and f_pot
    assert calls["sparse"] == 2
    # Interface ham updated
    assert iface.pyci_ham == cont.ham
    # Overlaps computed (once each)
    assert iface.objective._calls == {"ovlp": 1, "d_ovlp": 1, "dd_ovlp": 1}
    # Proxy properties & flags
    assert cont.nactive == iface.objective.nactive
    assert cont.nequation == iface.objective.nequation
    assert cont.nproj == iface.objective.nproj
    assert cont.active_energy  # mask[-1] was True


def test_uses_provided_ops_and_overlaps(monkeypatch):
    """If ops/overlaps are provided, no sparse_op is called and overlaps aren't computed."""
    # Count linear_comb_ham calls (should be 1: for ham with l and 1-l)
    calls = {"linear": 0}
    def fake_linear_comb_ham(h1, h0, a1, a0):
        calls["linear"] += 1
        return {"ham1": h1, "ham0": h0, "a1": a1, "a0": a0}
    monkeypatch.setattr("fanpy.fanpt.containers.base.linear_comb_ham", fake_linear_comb_ham)

    # Ensure sparse_op is not called
    class FakePyci:
        @staticmethod
        def sparse_op(*args, **kwargs):
            raise AssertionError("sparse_op should not be called when ops are provided")
    monkeypatch.setattr("fanpy.fanpt.containers.base.pyci", FakePyci)

    iface, params, ham0, ham1 = make_inputs(mask_last=False)

    provided_ham_op = types.SimpleNamespace(kind="ham_op")
    provided_fpot_op = types.SimpleNamespace(kind="fpot_op")
    # Use lists so the base's `if ovlp_s:` checks don't hit NumPy truth-value ambiguity
    ovlp_s = [10.0, 20.0, 30.0, 40.0]
    d_ovlp_s = [[1.0] * iface.objective.nactive for _ in range(iface.objective.nproj)]
    dd_ovlp_s = [
        [[0.0] * iface.objective.nactive for _ in range(iface.objective.nactive)]
        for _ in range(iface.objective.nproj)
    ]

    cont = SimpleContainer(
        fanci_interface=iface,
        params=params,
        ham0=ham0,
        ham1=ham1,
        l=0.25,
        ref_sd=0,
        inorm=False,
        norm_det=None,
        ham_ci_op=provided_ham_op,
        f_pot_ci_op=provided_fpot_op,
        ovlp_s=ovlp_s,
        d_ovlp_s=d_ovlp_s,
        dd_ovlp_s=dd_ovlp_s,
    )

    # linear_comb_ham called once to build ham
    assert calls["linear"] == 1
    # operators used as given
    assert cont.ham_ci_op is provided_ham_op
    assert cont.f_pot_ci_op is provided_fpot_op
    # overlaps taken as provided (compare after NumPy conversion in assert)
    np.testing.assert_allclose(cont.ovlp_s, ovlp_s)
    np.testing.assert_allclose(cont.d_ovlp_s, d_ovlp_s)
    np.testing.assert_allclose(cont.dd_ovlp_s, dd_ovlp_s)


def test_inorm_constraint_required(monkeypatch):
    """If inorm=True and the exact constraint is not present, __init__ should raise KeyError."""
    # Minimal stubs so init can run
    monkeypatch.setattr("fanpy.fanpt.containers.base.linear_comb_ham", lambda *a, **k: {"ok": True})
    class FakePyci:
        @staticmethod
        def sparse_op(*args, **kwargs):
            return types.SimpleNamespace(kind="sparse")
    monkeypatch.setattr("fanpy.fanpt.containers.base.pyci", FakePyci)

    iface, params, ham0, ham1 = make_inputs(mask_last=False, constraints=[])
    with pytest.raises(KeyError):
        SimpleContainer(
            fanci_interface=iface,
            params=params,
            ham0=ham0,
            ham1=ham1,
            l=0.5,
            ref_sd=2,
            inorm=True,   # requires the exact normalization constraint string to exist
            norm_det=None,
            ham_ci_op=None,
            f_pot_ci_op=None,
            ovlp_s=None,
            d_ovlp_s=None,
            dd_ovlp_s=None,
        )


def test_property_proxies(monkeypatch):
    """Light sanity: properties proxy and energy flag reflects mask[-1]."""
    monkeypatch.setattr("fanpy.fanpt.containers.base.linear_comb_ham", lambda *a, **k: {"ok": True})
    class FakePyci:
        @staticmethod
        def sparse_op(*args, **kwargs):
            return types.SimpleNamespace(kind="sparse")
    monkeypatch.setattr("fanpy.fanpt.containers.base.pyci", FakePyci)

    iface, params, ham0, ham1 = make_inputs(mask_last=True)  # energy active
    cont = SimpleContainer(
        fanci_interface=iface,
        params=params,
        ham0=ham0,
        ham1=ham1,
        l=0.0,
        ref_sd=0,
        inorm=False,
        norm_det=None,
        ham_ci_op=None,
        f_pot_ci_op=None,
        ovlp_s=None,
        d_ovlp_s=None,
        dd_ovlp_s=None,
    )
    assert cont.nactive == iface.objective.nactive
    assert cont.nequation == iface.objective.nequation
    assert cont.nproj == iface.objective.nproj
    assert cont.active_energy  # True because mask[-1] was True
