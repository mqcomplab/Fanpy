"""Tests for fanpy.fanpt.fanpt.FANPT."""

from contextlib import ExitStack
from unittest.mock import patch
import types

import numpy as np
import pytest

from fanpy.fanpt.fanpt import FANPT
import fanpy.fanpt.fanpt as fanpt_module


N_SPINORB = 2
FOCK_SENTINEL = 123.0
DUMMY_WFN_VALUE = 7.0
DUMMY_ENERGY = 11.5
DUMMY_ECORE = 99.0


def make_two_mo():
    """Return a zero two-electron integral tensor."""
    return np.zeros((N_SPINORB, N_SPINORB, N_SPINORB, N_SPINORB))


def make_one_mo():
    """Return a small identity one-electron integral matrix."""
    return np.eye(N_SPINORB)


def make_norm_constraint(ref_sd):
    """Return the normalization constraint string used by FanCI."""
    return f"<\\psi_{{{ref_sd}}}|\\Psi> - v_{{{ref_sd}}}"


class FakeHam:
    """Minimal Hamiltonian object used by FANPT tests."""

    def __init__(self, ecore=0.0, one_mo=None, two_mo=None):
        self.ecore = ecore
        self.one_mo = np.array(one_mo) if one_mo is not None else np.zeros((N_SPINORB, N_SPINORB))
        self.two_mo = np.array(two_mo) if two_mo is not None else make_two_mo()


class FakeOptimizeResult(dict):
    """Minimal optimization result with scipy-like `.x` attribute."""

    def __init__(self, x):
        super().__init__()
        self.x = x


class FakeObjective:
    """Minimal projected Schrödinger objective used by FANPT."""

    def __init__(
        self,
        *,
        nequation=3,
        nactive=3,
        constraints=None,
        last_mask=False,
        ham=None,
        fill="full",
    ):
        self.nequation = nequation
        self.nactive = nactive
        self.constraints = list(constraints or [])
        self.ham = ham or FakeHam(0.0, make_one_mo(), make_two_mo())
        self.fill = fill

        # FanCI convention: True means active. Last parameter is the energy.
        self.mask = np.zeros(nactive, dtype=bool)
        self.mask[-1] = last_mask

        # Required by FANPT when ref_sd is not explicitly provided.
        self.fanpy_objective = types.SimpleNamespace(refwfn=0)

        self.freeze_calls = 0
        self.unfreeze_calls = 0
        self.removed_constraints = []

    def freeze_parameter(self, index):
        """Freeze the energy parameter."""
        assert index == -1
        self.freeze_calls += 1
        self.mask[index] = False

    def unfreeze_parameter(self, index):
        """Unfreeze the energy parameter."""
        assert index == -1
        self.unfreeze_calls += 1
        self.mask[index] = True

    def remove_constraint(self, constraint):
        """Remove a constraint if present."""
        self.removed_constraints.append(constraint)

        if constraint in self.constraints:
            self.constraints.remove(constraint)

    def optimize(self, params, **kwargs):
        """Return deterministic active-parameter values."""
        del params, kwargs

        active_count = int(self.mask.sum())
        return FakeOptimizeResult(np.arange(active_count, dtype=float))


class FakePYCI:
    """Minimal PYCI interface wrapper."""

    def __init__(self, objective, energy_nuc, legacy_fanci=True):
        self.objective = objective
        self.energy_nuc = energy_nuc
        self.legacy_fanci = legacy_fanci
        self.update_calls = []

    def update_objective(self, ham):
        """Record Hamiltonian updates."""
        self.update_calls.append(ham)


class DummyContainer:
    """Minimal FANPT container replacement."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyUpdater:
    """Minimal FANPTUpdater replacement."""

    def __init__(
        self,
        fanpt_container,
        final_order,
        final_l,
        solver,
        resum,
        quasi_approximation_order=None,
        **kwargs,
    ):
        del final_order, final_l, solver, resum, quasi_approximation_order, kwargs

        nactive_wfn_params = fanpt_container.kwargs["params"].size - 1
        self.new_wfn_params = np.full(nactive_wfn_params, DUMMY_WFN_VALUE)
        self.new_energy = DUMMY_ENERGY
        self.new_ham = FakeHam(DUMMY_ECORE, make_one_mo(), make_two_mo())


def fake_reduce_to_fock(two_mo):
    """Return a sentinel tensor so tests can verify reduce_to_fock was used."""
    return np.full_like(two_mo, FOCK_SENTINEL)


class PatchedFANPTEnvironment:
    """Patch FANPT dependencies with lightweight fakes."""

    def __init__(self, objective=None):
        self.objective = objective or FakeObjective()
        self.ham_calls = []
        self.fake_projected = None
        self._stack = ExitStack()

    def __enter__(self):
        self._patch_reduce_to_fock()
        self._patch_hamiltonian_constructor()
        self._patch_pyci_interface()
        self._patch_fanpt_components()
        self._patch_projected_schrodinger()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stack.close()

    def _patch_reduce_to_fock(self):
        self._stack.enter_context(
            patch.object(fanpt_module, "reduce_to_fock", fake_reduce_to_fock)
        )

    def _patch_hamiltonian_constructor(self):
        def fake_hamiltonian(ecore, one_mo, two_mo):
            self.ham_calls.append((ecore, np.array(one_mo), np.array(two_mo)))
            return FakeHam(ecore, one_mo, two_mo)

        self._stack.enter_context(
            patch.object(fanpt_module.pyci, "hamiltonian", fake_hamiltonian)
        )

    def _patch_pyci_interface(self):
        def fake_pyci(fanpy_objective, energy_nuc, legacy_fanci=True):
            del fanpy_objective
            return FakePYCI(self.objective, energy_nuc, legacy_fanci=legacy_fanci)

        self._stack.enter_context(
            patch.object(fanpt_module.fanpy.interface.pyci, "PYCI", fake_pyci)
        )

    def _patch_fanpt_components(self):
        patches = [
            ("FANPTContainerEParam", DummyContainer),
            ("FANPTContainerEFree", DummyContainer),
            ("FANPTUpdater", DummyUpdater),
        ]

        for name, replacement in patches:
            self._stack.enter_context(patch.object(fanpt_module, name, replacement))

    def _patch_projected_schrodinger(self):
        class FakeProjectedSchrodinger:
            pass

        self._stack.enter_context(
            patch.object(fanpt_module, "ProjectedSchrodinger", FakeProjectedSchrodinger)
        )
        self.fake_projected = FakeProjectedSchrodinger()


def test_init_selects_eparam_and_unfreezes_energy():
    """energy_active=True uses EParam and unfreezes inactive energy."""
    objective = FakeObjective(
        nequation=4,
        nactive=3,
        constraints=[],
        last_mask=False,
        ham=FakeHam(0.5, make_one_mo(), make_two_mo()),
    )

    with PatchedFANPTEnvironment(objective) as env:
        fanpt = FANPT(
            fanpy_objective=env.fake_projected,
            energy_nuc=1.234,
            legacy_fanci=False,
            energy_active=True,
            ref_sd=0,
            final_order=1,
            steps=1,
        )

    assert fanpt.fanpt_container_class is DummyContainer
    assert objective.unfreeze_calls == 1

    assert len(env.ham_calls) == 1
    _, _, two_mo = env.ham_calls[0]
    assert np.all(two_mo == FOCK_SENTINEL)

    assert fanpt.ham1 is objective.ham
    assert isinstance(fanpt.ham0, FakeHam)


def test_init_selects_efree_and_freezes_energy_when_active():
    """energy_active=False uses EFree and freezes active energy."""
    objective = FakeObjective(
        nequation=4,
        nactive=3,
        constraints=[],
        last_mask=True,
        ham=FakeHam(),
    )

    with PatchedFANPTEnvironment(objective) as env:
        fanpt = FANPT(
            fanpy_objective=env.fake_projected,
            energy_nuc=2.0,
            energy_active=False,
            ref_sd=0,
        )

    assert fanpt.fanpt_container_class is DummyContainer
    assert objective.freeze_calls == 1


def test_init_inorm_detection_and_norm_det_assignment():
    """FANPT detects the normalization constraint and sets norm_det."""
    ref_sd = 2
    norm_constraint = make_norm_constraint(ref_sd)

    objective = FakeObjective(
        nequation=4,
        nactive=3,
        constraints=[norm_constraint],
        last_mask=False,
        ham=FakeHam(0.0, make_one_mo(), make_two_mo()),
    )

    with PatchedFANPTEnvironment(objective) as env:
        fanpt = FANPT(
            fanpy_objective=env.fake_projected,
            energy_nuc=0.0,
            energy_active=True,
            ref_sd=ref_sd,
        )

    assert fanpt.inorm is True
    assert fanpt.norm_det == [(ref_sd, 1.0)]


def test_init_resum_requires_inactive_energy():
    """resum=True requires energy_active=False."""
    objective = FakeObjective(
        nequation=4,
        nactive=3,
        constraints=[],
        last_mask=False,
        ham=FakeHam(),
    )

    with PatchedFANPTEnvironment(objective) as env:
        with pytest.raises(ValueError, match="energy parameter must be inactive"):
            FANPT(
                fanpy_objective=env.fake_projected,
                energy_nuc=0.0,
                energy_active=True,
                resum=True,
            )


def test_init_resum_sets_norm_det_when_no_constraint_and_square_system():
    """resum=True sets norm_det when no explicit normalization constraint exists."""
    ref_sd = 0
    objective = FakeObjective(
        nequation=4,
        nactive=4,
        constraints=[],
        last_mask=False,
        ham=FakeHam(0.0, make_one_mo(), make_two_mo()),
    )

    with PatchedFANPTEnvironment(objective) as env:
        fanpt = FANPT(
            fanpy_objective=env.fake_projected,
            energy_nuc=0.0,
            energy_active=False,
            resum=True,
            ref_sd=ref_sd,
        )

    assert fanpt.norm_det == [(ref_sd, 1.0)]
    assert fanpt.inorm is False


def test_init_resum_removes_norm_constraint_when_overdetermined_by_one():
    """resum=True removes explicit normalization when nequation - 1 == nactive."""
    ref_sd = 0
    norm_constraint = make_norm_constraint(ref_sd)

    objective = FakeObjective(
        nequation=5,
        nactive=4,
        constraints=[norm_constraint],
        last_mask=False,
        ham=FakeHam(0.0, make_one_mo(), make_two_mo()),
    )

    with PatchedFANPTEnvironment(objective) as env:
        fanpt = FANPT(
            fanpy_objective=env.fake_projected,
            energy_nuc=0.0,
            energy_active=False,
            resum=True,
            ref_sd=ref_sd,
        )

    assert norm_constraint in objective.removed_constraints
    assert fanpt.inorm is False


def test_optimize_toggles_freeze_when_energy_inactive():
    """energy_active=False temporarily unfreezes energy during FanCI solves."""
    objective = FakeObjective(
        nequation=4,
        nactive=3,
        constraints=[],
        last_mask=False,
        ham=FakeHam(),
    )

    with PatchedFANPTEnvironment(objective) as env:
        fanpt = FANPT(
            fanpy_objective=env.fake_projected,
            energy_nuc=0.0,
            energy_active=False,
            steps=1,
        )

        fanpt.optimize(
            guess_params=np.array([0.0, 0.0]),
            guess_energy=0.0,
        )

    assert objective.unfreeze_calls >= 2
    assert objective.freeze_calls >= 2
