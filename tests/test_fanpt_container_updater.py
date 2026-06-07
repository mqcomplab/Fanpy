"""Tests for fanpy.fanpt.containers.updater.FANPTUpdater."""

from types import SimpleNamespace

import numpy as np
import pytest

from fanpy.fanpt.containers.base import FANPTContainer


class FakeObjective:
    """Minimal FanCI objective used by the updater tests."""

    def __init__(self, nproj=3, energy_active=True):
        self.mask = np.array([True, True, bool(energy_active)])
        self.nproj = nproj
        self.wfn = object()

    def compute_overlap(self, wfn_params, space):
        """Return deterministic overlaps."""
        del wfn_params, space
        return np.arange(1, self.nproj + 1, dtype=float)


class DummyContainer(FANPTContainer):
    """Minimal FANPTContainer-like object for updater tests."""

    def __init__(
        self,
        nactive=4,
        nequation=3,
        nproj=3,
        energy_active=True,
        l=0.2,
        inorm=False,
        ref_sd=0,
        energy=-1.0,
    ):
        self.fanci_objective = FakeObjective(nproj=nproj, energy_active=energy_active)

        self._nactive = nactive
        self._nequation = nequation
        self._nproj = nproj

        self.l = l
        self.inorm = inorm
        self.ref_sd = ref_sd
        self.energy = energy
        self.active_energy = energy_active

        self.wfn_params = np.zeros(nactive - 1)
        self.c_matrix = np.eye(nequation, nactive)
        self.constant_terms = np.zeros((nequation, 1))
        self.d2_g_lambda_wfnparams = np.ones((nequation, nactive - 1))
        self.d_g_lambda = np.arange(nequation, dtype=float)

        self.ham0 = "H0"
        self.ham1 = "H1"

    @property
    def nactive(self):
        return self._nactive

    @property
    def nequation(self):
        return self._nequation

    @property
    def nproj(self):
        return self._nproj

    def der_g_lambda(self):
        pass

    def der2_g_lambda_wfnparams(self):
        pass

    def gen_coeff_matrix(self):
        pass


class FakeSparseOp:
    """Minimal pyci sparse operator replacement."""

    def __init__(self, ham, wfn, nproj, symmetric=False):
        del ham, wfn, symmetric
        self.nproj = nproj

    def __call__(self, vector, out):
        out[:] = 2.0 * np.asarray(vector, dtype=float)


def make_pyci():
    """Return a minimal pyci-like namespace."""
    return SimpleNamespace(sparse_op=FakeSparseOp, c_double=float)


def import_updater():
    """Import the updater class and module."""
    import fanpy.fanpt.containers.updater as updater_module
    from fanpy.fanpt.containers.updater import FANPTUpdater

    return updater_module, FANPTUpdater


def patch_updater_dependencies(monkeypatch, updater_module, constant_terms_factory=None):
    """Patch external updater dependencies with deterministic fakes."""
    monkeypatch.setattr(updater_module, "pyci", make_pyci())
    monkeypatch.setattr(updater_module, "linear_comb_ham", lambda *args, **kwargs: {})

    if constant_terms_factory is None:

        def constant_terms_factory(*args, **kwargs):
            fanpt_container = kwargs["fanpt_container"]
            return SimpleNamespace(
                constant_terms=np.zeros(fanpt_container.c_matrix.shape[0])
            )

    monkeypatch.setattr(updater_module, "FANPTConstantTerms", constant_terms_factory)


def make_constant_terms(value=0.0):
    """Return a fake FANPTConstantTerms constructor."""

    def fake_constant_terms(*args, **kwargs):
        del args

        fanpt_container = kwargs["fanpt_container"]
        return SimpleNamespace(
            constant_terms=np.full(fanpt_container.c_matrix.shape[0], value)
        )

    return fake_constant_terms


def test_basic_init_and_final_l_validation(monkeypatch):
    """Updater validates final_order and final_l before valid initialization."""
    updater_module, Updater = import_updater()
    container = DummyContainer(l=0.2)

    patch_updater_dependencies(monkeypatch, updater_module)

    with pytest.raises(TypeError):
        Updater(container, final_order=1, final_l=1, solver=None)

    with pytest.raises(ValueError):
        Updater(container, final_order=1, final_l=0.1, solver=None)

    with pytest.raises(ValueError):
        Updater(container, final_order=1, final_l=1.1, solver=None)

    updater = Updater(container, final_order=1, final_l=0.8, solver=None)
    assert isinstance(updater, Updater)


def test_resum_path(monkeypatch):
    """Updater builds resummation correction when resum=True."""
    updater_module, Updater = import_updater()

    container = DummyContainer(
        nactive=4,
        nequation=4,
        energy_active=False,
        l=0.1,
    )
    container.c_matrix = np.eye(4)
    container.d2_g_lambda_wfnparams = np.eye(4)
    container.d_g_lambda = np.arange(1, 5, dtype=float)

    patch_updater_dependencies(monkeypatch, updater_module)

    updater = Updater(
        container,
        final_order=2,
        final_l=0.6,
        solver=None,
        resum=True,
    )

    assert updater.resum
    assert hasattr(updater, "resum_correction")


def test_non_resum_path(monkeypatch):
    """Updater computes finite response vectors in the non-resummation path."""
    updater_module, Updater = import_updater()

    container = DummyContainer(nactive=5, nequation=3, l=0.2)
    rng = np.random.default_rng(0)
    container.c_matrix = rng.normal(size=(3, 5))

    def fake_constant_terms(*args, **kwargs):
        del args

        fanpt_container = kwargs["fanpt_container"]
        order = kwargs["order"]
        return SimpleNamespace(
            constant_terms=np.full(fanpt_container.c_matrix.shape[0], order)
        )

    patch_updater_dependencies(
        monkeypatch,
        updater_module,
        constant_terms_factory=fake_constant_terms,
    )

    updater = Updater(
        container,
        final_order=2,
        final_l=0.8,
        solver=None,
        resum=False,
    )

    assert updater.responses.shape == (2, container.nactive)
    assert np.all(np.isfinite(updater.responses))


def test_fanpt_e_formula(monkeypatch):
    """fanpt_e_response uses the Taylor correction from active-energy responses."""
    updater_module, Updater = import_updater()

    container = DummyContainer(
        nactive=5,
        nequation=4,
        l=0.1,
        energy=-5.0,
    )

    patch_updater_dependencies(monkeypatch, updater_module)

    updater = Updater(container, final_order=2, final_l=0.6, solver=None)
    updater.responses = np.zeros((2, container.nactive))
    updater.responses[0, -1] = 1.5
    updater.responses[1, -1] = 0.5

    updater.fanpt_e_response()

    dlambda = 0.5
    expected = -5.0 + dlambda * 1.5 + (dlambda**2 / 2.0) * 0.5
    assert updater.fanpt_e == pytest.approx(expected)
