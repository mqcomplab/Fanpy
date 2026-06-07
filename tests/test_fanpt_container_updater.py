import numpy as np
import pytest
from types import SimpleNamespace
from fanpy.fanpt.containers.base import FANPTContainer

# ---- Basic fakes ----

class FakeObjective:
    def __init__(self, nproj=3, energy_active=True):
        self.mask = np.array([1, 1, 1 if energy_active else 0])
        self.nproj = nproj
        self.wfn = object()

    def compute_overlap(self, wfn_params, space):
        # Return predictable overlap: [1,2,3,...,nproj]
        return np.arange(1, self.nproj + 1, dtype=float)


from fanpy.fanpt.containers.base import FANPTContainer

class DummyContainer(FANPTContainer):
    """Simpler mock container for tests."""
    def __init__(self, nactive=4, nequation=3, nproj=3, energy_active=True, l=0.2, inorm=False, ref_sd=0, energy=-1.0):
        self.fanci_objective = FakeObjective(nproj, energy_active)

        # Store privately since FANPTContainer defines read-only properties
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

    # ---- Override property getters ----
    @property
    def nactive(self):
        return self._nactive

    @property
    def nequation(self):
        return self._nequation

    @property
    def nproj(self):
        return self._nproj

    # ---- Required abstract methods ----
    def der_g_lambda(self):
        pass

    def der2_g_lambda_wfnparams(self):
        pass

    def gen_coeff_matrix(self):
        pass



class FakeSparseOp:
    def __init__(self, ham, wfn, nproj, symmetric=False):
        self.nproj = nproj

    def __call__(self, vec, out):
        out[:] = 2 * np.asarray(vec, dtype=float)


def make_pyci():
    return SimpleNamespace(sparse_op=FakeSparseOp, c_double=float)


# ---- Import target classes ----

def import_updater():
    import fanpy.fanpt.containers.base as base_mod
    from fanpy.fanpt.containers.updater import FANPTUpdater as Updater
    return base_mod, Updater


# ---- Tests ----

def test_basic_init_and_final_l(monkeypatch):
    base_mod, Updater = import_updater()
    cont = DummyContainer(l=0.2)
    import fanpy.fanpt.containers.updater as upd

    monkeypatch.setattr(upd, "pyci", make_pyci())
    monkeypatch.setattr(upd, "linear_comb_ham", lambda *a, **k: {})
    monkeypatch.setattr(upd, "FANPTConstantTerms", lambda *a, **k: SimpleNamespace(constant_terms=np.zeros(4)))
    monkeypatch.setattr(upd,"FANPTConstantTerms",lambda *a, **k: SimpleNamespace(constant_terms=np.zeros(cont.c_matrix.shape[0])),)

    # Bad inputs
    with pytest.raises(TypeError):
        Updater(cont, final_order=1, final_l=1, solver=None)
    with pytest.raises(ValueError):
        Updater(cont, final_order=1, final_l=0.1, solver=None)
    with pytest.raises(ValueError):
        Updater(cont, final_order=1, final_l=1.1, solver=None)

    # Valid
    ok = Updater(cont, final_order=1, final_l=0.8, solver=None)
    assert isinstance(ok, Updater)


def test_resum_path(monkeypatch):
    base_mod, Updater = import_updater()
    cont = DummyContainer(nactive=4, nequation=4, energy_active=False, l=0.1)
    cont.c_matrix = np.eye(4)
    cont.d2_g_lambda_wfnparams = np.eye(4)
    cont.d_g_lambda = np.arange(1, 5)

    import fanpy.fanpt.containers.updater as upd
    monkeypatch.setattr(upd, "pyci", make_pyci())
    monkeypatch.setattr(upd, "linear_comb_ham", lambda *a, **k: {})
    monkeypatch.setattr(upd, "FANPTConstantTerms", lambda **kw: SimpleNamespace(constant_terms=np.zeros(4)))
    monkeypatch.setattr(upd, "FANPTConstantTerms", lambda *a, **k: SimpleNamespace(constant_terms=np.zeros(4)))

    up = Updater(cont, final_order=2, final_l=0.6, solver=None, resum=True)
    assert up.resum
    assert hasattr(up, "resum_correction")


def test_non_resum_path(monkeypatch):
    base_mod, Updater = import_updater()
    cont = DummyContainer(nactive=5, nequation=3, l=0.2)

    rng = np.random.default_rng(0)
    cont.c_matrix = rng.normal(size=(3, 5))

    def fake_ct(*, fanpt_container, order, previous_responses):
        # length must match rows of c_matrix (nequation)
        return SimpleNamespace(constant_terms=np.ones(fanpt_container.c_matrix.shape[0]) * order)


    import fanpy.fanpt.containers.updater as upd
    monkeypatch.setattr(upd, "pyci", make_pyci())
    monkeypatch.setattr(upd, "linear_comb_ham", lambda *a, **k: {})
    monkeypatch.setattr(upd, "FANPTConstantTerms", fake_ct)
    monkeypatch.setattr(upd, "FANPTConstantTerms", fake_ct)
    up = Updater(cont, final_order=2, final_l=0.8, solver=None, resum=False)
    assert up.responses.shape[0] == 2
    assert np.all(np.isfinite(up.responses))


def test_fanpt_e_formula(monkeypatch):
    base_mod, Updater = import_updater()
    cont = DummyContainer(nactive=5, nequation=4, l=0.1, energy=-5.0)

    import fanpy.fanpt.containers.updater as upd
    monkeypatch.setattr(upd, "pyci", make_pyci())
    monkeypatch.setattr(upd, "linear_comb_ham", lambda *a, **k: {})
    monkeypatch.setattr(upd, "FANPTConstantTerms", lambda *a, **k: SimpleNamespace(constant_terms=np.zeros(5)))
    monkeypatch.setattr(upd,"FANPTConstantTerms",lambda *a, **k: SimpleNamespace(constant_terms=np.zeros(cont.c_matrix.shape[0])),)

    up = Updater(cont, final_order=2, final_l=0.6, solver=None)
    up.responses = np.zeros((2, cont.nactive))
    up.responses[0, -1] = 1.5
    up.responses[1, -1] = 0.5

    up.fanpt_e_response()
    expected = -5.0 + (0.5 * 1.5) + (0.5**2 / 2 * 0.5)
    assert up.fanpt_e == pytest.approx(expected)






































































































































































































































































































































































































