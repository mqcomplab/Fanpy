"""Tests for fanpy.fanpt.containers.base.FANPTContainer."""

import types

import numpy as np
import pytest

from fanpy.fanpt.containers.base import FANPTContainer


N_ACTIVE = 3
N_EQUATION = 5
N_PROJ = 4
PARAMS = np.array([0.1, 0.2, -1.5])


class SimpleContainer(FANPTContainer):
    """Minimal concrete FANPTContainer used to test the base initializer."""

    def der_g_lambda(self):
        self.d_g_lambda = np.zeros(self.nequation)

    def der2_g_lambda_wfnparams(self):
        n_wfn_params = max(1, self.nactive - 1)
        self.d2_g_lambda_wfnparams = np.zeros((self.nequation, n_wfn_params), order="F")

    def gen_coeff_matrix(self):
        self.c_matrix = np.ones((self.nequation, self.nactive))


class MiniObjective:
    """Minimal objective exposing the attributes and methods used by FANPTContainer."""

    def __init__(
        self,
        *,
        nactive=N_ACTIVE,
        nequation=N_EQUATION,
        nproj=N_PROJ,
        mask_last=False,
        constraints=None,
    ):
        self.nactive = nactive
        self.nequation = nequation
        self.nproj = nproj
        self.constraints = list(constraints or [])
        self.wfn = "WAVEFUNCTION"

        self.mask = np.zeros(nactive, dtype=bool)
        self.mask[-1] = bool(mask_last)

        self.calls = {"ovlp": 0, "d_ovlp": 0, "dd_ovlp": 0}

    def compute_overlap(self, wfn_params, space):
        del wfn_params, space
        self.calls["ovlp"] += 1
        return np.arange(1, self.nproj + 1, dtype=float)

    def compute_overlap_deriv(self, wfn_params, space):
        del wfn_params, space
        self.calls["d_ovlp"] += 1
        return np.ones((self.nproj, self.nactive), order="F")

    def compute_overlap_double_deriv(self, wfn_params, space):
        del wfn_params, space
        self.calls["dd_ovlp"] += 1
        return np.zeros((self.nproj, self.nactive, self.nactive), order="F")


class MiniInterface:
    """Minimal interface object wrapping the objective."""

    def __init__(self, objective):
        self.objective = objective
        self.pyci_ham = None


def make_norm_constraint(ref_sd):
    """Return the normalization constraint string used by FANPT."""
    return f"<\\psi_{{{ref_sd}}}|\\Psi> - v_{{{ref_sd}}}"


def make_inputs(*, mask_last=False, constraints=None):
    """Build reusable FANPTContainer inputs."""
    objective = MiniObjective(mask_last=mask_last, constraints=constraints)
    interface = MiniInterface(objective)
    return interface, PARAMS.copy(), "H0", "H1"


def patch_linear_comb_ham(monkeypatch, calls=None):
    """Patch linear_comb_ham with a deterministic fake."""

    def fake_linear_comb_ham(ham1, ham0, coeff1, coeff0):
        if calls is not None:
            calls["linear"] += 1

        return {
            "ham1": ham1,
            "ham0": ham0,
            "coeff1": coeff1,
            "coeff0": coeff0,
        }

    monkeypatch.setattr(
        "fanpy.fanpt.containers.base.linear_comb_ham",
        fake_linear_comb_ham,
    )


def patch_sparse_op(monkeypatch, calls=None, *, should_raise=False):
    """Patch pyci.sparse_op with a deterministic fake."""

    class FakePyci:
        @staticmethod
        def sparse_op(ham, wfn, nproj, symmetric=False):
            if should_raise:
                raise AssertionError("sparse_op should not be called")

            if calls is not None:
                calls["sparse"] += 1

            return types.SimpleNamespace(
                kind="sparse",
                ham=ham,
                wfn=wfn,
                nproj=nproj,
                symmetric=symmetric,
            )

    monkeypatch.setattr("fanpy.fanpt.containers.base.pyci", FakePyci)


def build_container(
    interface,
    params,
    ham0,
    ham1,
    *,
    l=0.3,
    ref_sd=0,
    inorm=False,
    norm_det=None,
    ham_ci_op=None,
    f_pot_ci_op=None,
    ovlp_s=None,
    d_ovlp_s=None,
    dd_ovlp_s=None,
    quasi_approximation_order=2,
):
    """Construct the SimpleContainer with explicit defaults."""
    return SimpleContainer(
        fanci_interface=interface,
        params=params,
        ham0=ham0,
        ham1=ham1,
        l=l,
        ref_sd=ref_sd,
        inorm=inorm,
        norm_det=norm_det,
        ham_ci_op=ham_ci_op,
        f_pot_ci_op=f_pot_ci_op,
        ovlp_s=ovlp_s,
        d_ovlp_s=d_ovlp_s,
        dd_ovlp_s=dd_ovlp_s,
        quasi_approximation_order=quasi_approximation_order,
    )


def test_builds_ops_and_overlaps_when_missing_qao2(monkeypatch):
    """QAO2 builds operators and first-derivative overlap data, but not double derivatives."""
    calls = {"linear": 0, "sparse": 0}
    patch_linear_comb_ham(monkeypatch, calls)
    patch_sparse_op(monkeypatch, calls)

    interface, params, ham0, ham1 = make_inputs(mask_last=True)

    container = build_container(
        interface,
        params,
        ham0,
        ham1,
        ovlp_s=None,
        d_ovlp_s=None,
        dd_ovlp_s=None,
        quasi_approximation_order=2,
    )

    assert calls == {"linear": 2, "sparse": 2}
    assert interface.pyci_ham == container.ham
    assert interface.objective.calls == {"ovlp": 1, "d_ovlp": 1, "dd_ovlp": 0}

    assert container.nactive == interface.objective.nactive
    assert container.nequation == interface.objective.nequation
    assert container.nproj == interface.objective.nproj
    assert container.active_energy


def test_builds_double_overlap_derivatives_for_qao3(monkeypatch):
    """QAO3 computes double-overlap derivatives when they are not provided."""
    calls = {"linear": 0, "sparse": 0}
    patch_linear_comb_ham(monkeypatch, calls)
    patch_sparse_op(monkeypatch, calls)

    interface, params, ham0, ham1 = make_inputs(mask_last=True)

    build_container(
        interface,
        params,
        ham0,
        ham1,
        ovlp_s=None,
        d_ovlp_s=None,
        dd_ovlp_s=None,
        quasi_approximation_order=3,
    )

    assert calls == {"linear": 2, "sparse": 2}
    assert interface.objective.calls == {"ovlp": 1, "d_ovlp": 1, "dd_ovlp": 1}


def test_uses_provided_ops_and_overlaps(monkeypatch):
    """Provided operators and overlaps are reused without recomputing sparse operators."""
    calls = {"linear": 0}
    patch_linear_comb_ham(monkeypatch, calls)
    patch_sparse_op(monkeypatch, should_raise=True)

    interface, params, ham0, ham1 = make_inputs(mask_last=False)

    provided_ham_op = types.SimpleNamespace(kind="ham_op")
    provided_fpot_op = types.SimpleNamespace(kind="fpot_op")

    ovlp_s = [10.0, 20.0, 30.0, 40.0]
    d_ovlp_s = [[1.0] * interface.objective.nactive for _ in range(interface.objective.nproj)]
    dd_ovlp_s = [
        [[0.0] * interface.objective.nactive for _ in range(interface.objective.nactive)]
        for _ in range(interface.objective.nproj)
    ]

    container = build_container(
        interface,
        params,
        ham0,
        ham1,
        l=0.25,
        ham_ci_op=provided_ham_op,
        f_pot_ci_op=provided_fpot_op,
        ovlp_s=ovlp_s,
        d_ovlp_s=d_ovlp_s,
        dd_ovlp_s=dd_ovlp_s,
    )

    assert calls["linear"] == 1
    assert container.ham_ci_op is provided_ham_op
    assert container.f_pot_ci_op is provided_fpot_op

    np.testing.assert_allclose(container.ovlp_s, ovlp_s)
    np.testing.assert_allclose(container.d_ovlp_s, d_ovlp_s)
    np.testing.assert_allclose(container.dd_ovlp_s, dd_ovlp_s)

    assert interface.objective.calls == {"ovlp": 0, "d_ovlp": 0, "dd_ovlp": 0}


def test_inorm_constraint_required(monkeypatch):
    """inorm=True requires the exact normalization constraint to exist."""
    patch_linear_comb_ham(monkeypatch)
    patch_sparse_op(monkeypatch)

    interface, params, ham0, ham1 = make_inputs(mask_last=False, constraints=[])

    with pytest.raises(KeyError):
        build_container(
            interface,
            params,
            ham0,
            ham1,
            l=0.5,
            ref_sd=2,
            inorm=True,
        )


def test_property_proxies(monkeypatch):
    """Container properties proxy objective attributes."""
    patch_linear_comb_ham(monkeypatch)
    patch_sparse_op(monkeypatch)

    interface, params, ham0, ham1 = make_inputs(mask_last=True)

    container = build_container(
        interface,
        params,
        ham0,
        ham1,
        l=0.0,
    )

    assert container.nactive == interface.objective.nactive
    assert container.nequation == interface.objective.nequation
    assert container.nproj == interface.objective.nproj
    assert container.active_energy
