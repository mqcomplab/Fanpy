"""
Unit tests for `fanpy.fanpt.containers.base.FANPTContainer` that exercise its
constructor logic, property passthroughs, and normalization checks.

Strategy
--------
- Import the base class from your package.
- Provide a tiny concrete subclass implementing abstract hooks.
- Stub only `fanpy.fanpt.utils.linear_comb_ham` (so we don't require true pyci hamiltonians).
- **Inject** `ham_ci_op` and `f_pot_ci_op` so the real `pyci.sparse_op` is never called.
- Provide a minimal "objective" + "interface" with the attributes/methods the base needs.

These tests are fast and avoid bringing in PySCF/FanPy heavy machinery.
"""

from __future__ import annotations

import numpy as np
import pytest


class _HamSentinel:
    """Lightweight sentinel to represent a (combined) Hamiltonian in assertions."""
    def __init__(self, tag: str):
        self.tag = tag
    def __repr__(self):
        return f"<Ham:{self.tag}>"


class _OpSentinel:
    """Injected operator sentinel for ham_ci_op / f_pot_ci_op."""
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"<Op:{self.name}>"


class _Objective:
    """
    Minimal 'objective' stand-in exposing attributes/methods used by FANPTContainer.

    Attributes
    ----------
    nproj : int
    nequation : int
    nactive : int
    mask : np.ndarray
        Last entry denotes energy activity.
    constraints : set[str]
    wfn : object
        Placeholder wavefunction; only identity storage needed.
    calls : dict[str, int]
        Counters to ensure compute_* methods are called when arrays not injected.
    """
    def __init__(self, nproj=4, nequation=6, nactive=5, energy_active=True, constraints=None):
        self.nproj = nproj
        self.nequation = nequation
        self.nactive = nactive
        self.mask = np.array([1]*(nactive-1) + [1 if energy_active else 0], dtype=int)
        self.constraints = constraints or set()
        self.wfn = object()
        self.calls = dict(ovlp=0, dovlp=0, ddovlp=0)

    def compute_overlap(self, wfn_params, space):
        self.calls["ovlp"] += 1
        return np.full(self.nproj, fill_value=len(wfn_params), dtype=float)

    def compute_overlap_deriv(self, wfn_params, space):
        self.calls["dovlp"] += 1
        return np.full((self.nproj, len(wfn_params)), 1.0, dtype=float)

    def compute_overlap_double_deriv(self, wfn_params, space):
        self.calls["ddovlp"] += 1
        return np.zeros((self.nproj, len(wfn_params), len(wfn_params)), dtype=float)


class _Interface:
    """Holds `.objective` and a writable `.pyci_ham` attribute, like the real interface."""
    def __init__(self, objective: _Objective):
        self.objective = objective
        self.pyci_ham = None


class _HooksMixin:
    """Concrete implementations of abstract hooks with predictable outputs."""
    def der_g_lambda(self):
        self.d_g_lambda = np.arange(self.nequation, dtype=float)
    def der2_g_lambda_wfnparams(self):
        self.d2_g_lambda_wfnparams = np.ones((self.nequation, self.fanci_objective.nactive - 1))
    def gen_coeff_matrix(self):
        self.c_matrix = np.eye(self.nequation, self.fanci_objective.nactive)


def _Concrete():
    """Return a concrete subclass of FANPTContainer for testing."""
    from fanpy.fanpt.containers.base import FANPTContainer
    class Concrete(_HooksMixin, FANPTContainer):
        """Concrete test subclass of FANPTContainer."""
        pass
    return Concrete


def test_builds_combined_hams_and_updates_interface(monkeypatch):
    """`ham` uses (l, 1-l) and `f_pot` uses (1, -1); `pyci_ham` must mirror `ham`."""
    import fanpy.fanpt.containers.base as base_mod
    monkeypatch.setattr(base_mod, "linear_comb_ham",
                        lambda h1, h0, a1, a0: _HamSentinel(f"a1={a1},a0={a0}"),
                        raising=True)

    Concrete = _Concrete()
    obj = _Objective(nproj=3, nequation=5, nactive=4, energy_active=True)
    interface = _Interface(obj)

    cc = Concrete(
        fanci_interface=interface,
        params=np.array([0.1, 0.2, 0.3, -1.0]),
        ham0="H0", ham1="H1",
        l=0.25,
        # Inject ops to avoid touching pyci.sparse_op
        ham_ci_op=_OpSentinel("ham"),
        f_pot_ci_op=_OpSentinel("fpot"),
    )

    assert isinstance(cc.ham, _HamSentinel) and "a1=0.25,a0=0.75" in cc.ham.tag
    assert isinstance(cc.f_pot, _HamSentinel) and "a1=1.0,a0=-1.0" in cc.f_pot.tag
    assert interface.pyci_ham is cc.ham


def test_overlap_calls_and_bypass(monkeypatch):
    """If arrays omitted, compute_* must run once; if provided, they must *not* run."""
    import fanpy.fanpt.containers.base as base_mod
    monkeypatch.setattr(base_mod, "linear_comb_ham",
                        lambda *a, **k: _HamSentinel("x"),
                        raising=True)

    Concrete = _Concrete()
    obj = _Objective(nproj=2, nequation=4, nactive=3)
    # Case 1: no arrays provided -> calls should increment
    cc1 = Concrete(
        fanci_interface=_Interface(obj),
        params=np.array([0.5, -2.0, -0.1]),
        ham0="H0", ham1="H1",
        ham_ci_op=_OpSentinel("ham"),
        f_pot_ci_op=_OpSentinel("fpot"),
    )
    assert obj.calls == dict(ovlp=1, dovlp=1, ddovlp=1)
    assert cc1.ovlp_s.shape == (obj.nproj,)

    # Case 2: arrays provided -> no further calls
    ovlp = np.array([9.0, 8.0])
    d_ovlp = np.array([[1.0, 2.0], [3.0, 4.0]])
    dd_ovlp = np.zeros((2, 2, 2))
    cc2 = Concrete(
        fanci_interface=_Interface(_Objective(nproj=2, nequation=4, nactive=3)),
        params=np.array([0.5, -2.0, -0.1]),
        ham0="H0", ham1="H1",
        ham_ci_op=_OpSentinel("ham"),
        f_pot_ci_op=_OpSentinel("fpot"),
        ovlp_s=ovlp, d_ovlp_s=d_ovlp, dd_ovlp_s=dd_ovlp,
    )
    np.testing.assert_allclose(cc2.ovlp_s, ovlp)


def test_property_passthroughs_and_param_split(monkeypatch):
    """`nactive`, `nequation`, `nproj` passthrough and `[wfn_params | energy]` split."""
    import fanpy.fanpt.containers.base as base_mod
    monkeypatch.setattr(base_mod, "linear_comb_ham",
                        lambda *a, **k: _HamSentinel("x"),
                        raising=True)

    Concrete = _Concrete()
    obj = _Objective(nproj=7, nequation=9, nactive=6, energy_active=False)

    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, -7.])
