# test_fanpt_fanpt.py  — Refactored to use a context manager (Option B)
import numpy as np
import types
import pytest
from contextlib import ExitStack
from unittest.mock import patch

from fanpy.fanpt.fanpt import FANPT
import fanpy.fanpt.fanpt as mod


#
# ---------- Tiny fakes ----------
#

class FakeHam:
    """Mock representation of a molecular Hamiltonian."""
    def __init__(self, ecore=0.0, one_mo=None, two_mo=None):
        self.ecore = ecore
        self.one_mo = np.array(one_mo) if one_mo is not None else np.zeros((2, 2))
        self.two_mo = np.array(two_mo) if two_mo is not None else np.zeros((2, 2, 2, 2))


class FakeObjective:
    """Mock objective class simulating FanCI's projected Schrödinger objective."""
    def __init__(self, *, nequation=3, nactive=3, constraints=None, last_mask=False, ham=None, fill="full"):
        self.nequation = nequation
        self.nactive = nactive
        # mask: True = active; last index corresponds to energy
        self.mask = np.zeros(nactive, dtype=bool)
        self.mask[-1] = last_mask
        self.constraints = list(constraints or [])
        self.ham = ham or FakeHam(0.0, np.eye(2), np.zeros((2, 2, 2, 2)))
        self.fill = fill
        # required by FANPT: .fanpy_objective.refwfn used when ref_sd is None
        self.fanpy_objective = types.SimpleNamespace(refwfn=0)

        # counters for freeze/unfreeze/constraint removal
        self.frozen_calls = 0
        self.unfrozen_calls = 0
        self.removed_constraints = []

    def freeze_parameter(self, idx):
        assert idx == -1
        self.frozen_calls += 1
        self.mask[idx] = False

    def unfreeze_parameter(self, idx):
        assert idx == -1
        self.unfrozen_calls += 1
        self.mask[idx] = True

    def remove_constraint(self, s):
        self.removed_constraints.append(s)
        try:
            self.constraints.remove(s)
        except ValueError:
            pass

    def optimize(self, params, **kwargs):
        # deterministic fake: return a result with `x` of length = number of active params
        active_count = int(self.mask.sum())
        x = np.arange(active_count, dtype=float)
        class Result(dict):
            def __init__(self, x): super().__init__(); self.x = x
        return Result(x)


class FakePYCI:
    """Mock PyCI interface class for FANPT testing."""
    def __init__(self, objective, energy_nuc, legacy_fanci=True):
        self.objective = objective
        self.energy_nuc = energy_nuc
        self.legacy_fanci = legacy_fanci
        self.update_calls = []

    def update_objective(self, ham):
        self.update_calls.append(ham)


class DummyContainer:
    """Mock FANPT container (EParam/EFree) used in tests."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs  # record inputs for assertions


class DummyUpdater:
    """Mock FANPTUpdater for unit tests."""
    def __init__(self, fanpt_container, final_order, final_l, solver, resum):
        nactive = fanpt_container.kwargs["params"].size - 1
        self.new_wfn_params = np.ones(nactive) * 7.0
        self.new_energy = 11.5
        self.new_ham = FakeHam(ecore=99.0, one_mo=np.eye(2), two_mo=np.zeros((2, 2, 2, 2)))


#
# ---------- Context-manager patching environment (Option B) ----------
#

def _fake_reduce_to_fock(two_mo):
    """Return a sentinel so tests can assert this was used."""
    return np.full_like(two_mo, 123.0)

class CtxEnv:
    """Context manager that patches FANPT's dependencies to fakes."""
    def __init__(self, fake_objective=None):
        self.stack = ExitStack()
        self.ham_calls = []
        self.fake_objective = fake_objective or FakeObjective()
        self.fake_projected = None  # set in __enter__

    def __enter__(self):
        # Patch reduce_to_fock in fanpt module
        self.stack.enter_context(patch.object(mod, "reduce_to_fock", _fake_reduce_to_fock))

        # Patch pyci.hamiltonian inside fanpt module and record constructor calls
        def _ham_ctor(ecore, one_mo, two_mo):
            self.ham_calls.append((ecore, np.array(one_mo), np.array(two_mo)))
            return FakeHam(ecore, one_mo, two_mo)
        self.stack.enter_context(patch.object(mod.pyci, "hamiltonian", _ham_ctor))

        # Patch PYCI constructor to return our FakePYCI holding self.fake_objective
        def _fake_PYCI(fanpy_objective, energy_nuc, legacy_fanci=True):
            return FakePYCI(self.fake_objective, energy_nuc, legacy_fanci=legacy_fanci)
        self.stack.enter_context(patch.object(mod.fanpy.interface.pyci, "PYCI", _fake_PYCI))

        # Patch container classes + updater
        self.stack.enter_context(patch.object(mod, "FANPTContainerEParam", DummyContainer))
        self.stack.enter_context(patch.object(mod, "FANPTContainerEFree",  DummyContainer))
        self.stack.enter_context(patch.object(mod, "FANPTUpdater",         DummyUpdater))

        # Patch ProjectedSchrodinger and create a fake instance
        class _FakeProjectedSchrodinger: pass
        self.stack.enter_context(patch.object(mod, "ProjectedSchrodinger", _FakeProjectedSchrodinger))
        self.fake_projected = _FakeProjectedSchrodinger()

        return self

    def __exit__(self, exc_type, exc, tb):
        self.stack.close()


#
# ---------- Tests ----------
#

def test_init_selects_eparam_and_unfreezes_energy():
    """energy_active=True → use EParam, unfreeze energy if last mask is False, build ham0 via reduce_to_fock."""
    fake_obj = FakeObjective(nequation=4, nactive=3, constraints=[], last_mask=False, ham=FakeHam(
        ecore=0.5, one_mo=np.eye(2), two_mo=np.zeros((2, 2, 2, 2))
    ))

    with CtxEnv(fake_objective=fake_obj) as env:
        fanpt = FANPT(
            fanpy_objective=env.fake_projected,
            energy_nuc=1.234,
            legacy_fanci=False,
            energy_active=True,
            ref_sd=0,
            final_order=1,
            steps=1,
        )

        # container class chosen
        assert fanpt.fanpt_container_class is DummyContainer
        # energy unfreezing happened (last mask was False initially)
        assert fake_obj.unfrozen_calls == 1

        # ham0 built via pyci.hamiltonian with reduced two-electron integrals
        assert len(env.ham_calls) == 1
        ecore, one_mo, two_mo = env.ham_calls[0]
        assert np.all(two_mo == 123.0)  # reduce_to_fock sentinel
        assert fanpt.ham1 is fake_obj.ham
        assert isinstance(fanpt.ham0, FakeHam)


def test_init_selects_efree_and_freezes_energy_when_active():
    """energy_active=False → use EFree, freeze energy if mask[-1] is True."""
    # last_mask=True triggers freeze in EFree branch
    fake_obj = FakeObjective(nequation=4, nactive=3, constraints=[], last_mask=True, ham=FakeHam())

    with CtxEnv(fake_objective=fake_obj) as env:
        fanpt = FANPT(
            fanpy_objective=env.fake_projected,
            energy_nuc=2.0,
            energy_active=False,
            ref_sd=0,
        )

        assert fanpt.fanpt_container_class is DummyContainer
        assert fake_obj.frozen_calls == 1


def test_init_inorm_detection_and_norm_det_assignment():
    """Detects normalization constraint and sets inorm/norm_det accordingly."""
    ref_sd = 2
    norm_str = f"<\\psi_{{{ref_sd}}}|\\Psi> - v_{{{ref_sd}}}"
    fake_obj = FakeObjective(
        nequation=4, nactive=3, constraints=[norm_str], last_mask=False,
        ham=FakeHam(0.0, np.eye(2), np.zeros((2, 2, 2, 2)))
    )

    with CtxEnv(fake_objective=fake_obj) as env:
        fanpt = FANPT(
            fanpy_objective=env.fake_projected,
            energy_nuc=0.0,
            energy_active=True,
            ref_sd=ref_sd,
        )
        assert fanpt.inorm is True
        assert fanpt.norm_det == [(ref_sd, 1.0)]


def test_init_resum_requires_inactive_energy():
    """resum=True with energy_active=True should error."""
    fake_obj = FakeObjective(nequation=4, nactive=3, constraints=[], last_mask=False, ham=FakeHam())

    with CtxEnv(fake_objective=fake_obj) as env:
        with pytest.raises(ValueError, match="energy parameter must be inactive"):
            FANPT(
                fanpy_objective=env.fake_projected,
                energy_nuc=0.0,
                energy_active=True,
                resum=True,
            )


def test_init_resum_branch_normalizes_or_removes_constraint():
    """
    resum=True & energy_active=False:
      - if not inorm and nequation == nactive → norm_det is set
      - if inorm and (nequation - 1) == nactive → constraint removed, inorm=False
    """
    # Case 1: not inorm, nequation == nactive
    ham1 = FakeHam(0.0, np.eye(2), np.zeros((2, 2, 2, 2)))
    ref_sd = 0
    obj1 = FakeObjective(nequation=4, nactive=4, constraints=[], last_mask=False, ham=ham1)

    with CtxEnv(fake_objective=obj1) as env1:
        fanpt1 = FANPT(
            fanpy_objective=env1.fake_projected,
            energy_nuc=0.0,
            energy_active=False,
            resum=True,
            ref_sd=ref_sd,
        )
        assert fanpt1.norm_det == [(ref_sd, 1.0)]
        assert fanpt1.inorm is False

    # Case 2: inorm, nequation - 1 == nactive, and constraint present for ref_sd=0
    norm_str = f"<\\psi_{{{ref_sd}}}|\\Psi> - v_{{{ref_sd}}}"
    obj2 = FakeObjective(nequation=5, nactive=4, constraints=[norm_str], last_mask=False, ham=ham1)

    with CtxEnv(fake_objective=obj2) as env2:
        fanpt2 = FANPT(
            fanpy_objective=env2.fake_projected,
            energy_nuc=0.0,
            energy_active=False,
            resum=True,
            ref_sd=ref_sd,
        )
        assert norm_str in obj2.removed_constraints
        assert fanpt2.inorm is False


def test_optimize_toggles_freeze_when_energy_inactive():
    """
    energy_active=False:
    - unfreeze before each FanCI solve, then freeze after,
    - both at the initial solve and inside the lambda loop.
    """
    obj = FakeObjective(nequation=4, nactive=3, constraints=[], last_mask=False, ham=FakeHam())

    with CtxEnv(fake_objective=obj) as env:
        fanpt = FANPT(
            fanpy_objective=env.fake_projected,
            energy_nuc=0.0,
            energy_active=False,
            steps=1,
        )

        fanpt.optimize(guess_params=np.array([0.0, 0.0]), guess_energy=0.0)

        # Initial solve: unfreeze → freeze; Loop solve: unfreeze → freeze
        assert obj.unfrozen_calls >= 2
        assert obj.frozen_calls   >= 2
