"""Test fanpy.wfn.utils."""
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.tools import slater
from fanpy.wfn.geminal.ap1rog import AP1roG
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.utils import wfn_factory, convert_to_fanci

import numpy as np

import pytest

from utils import find_datafile, skip_init


def test_wfn_utils():
    """Test wfn.utils.wfn_factory."""

    def olp(sd, params):
        """Test overlap."""
        return np.sum(params)

    def olp_deriv(sd, params):
        """Test overlap deriv."""
        return params

    def assign_params(self, params):
        """Test assign_params."""
        self.params = np.array(params)

    params = np.random.rand(100)

    wfn = wfn_factory(olp, olp_deriv, 3, 6, params)
    assert wfn.nspin == 6
    assert np.allclose(wfn.params, params)
    assert np.allclose(wfn.get_overlap(0b000111), np.sum(params))
    assert np.allclose(wfn.get_overlap(0b000111, np.arange(50)), params[:50])

    wfn = wfn_factory(olp, olp_deriv, 3, 6, params.tolist(), assign_params=assign_params)
    assert wfn.nspin == 6
    assert np.allclose(wfn.params, params)
    assert np.allclose(wfn.get_overlap(0b000111), np.sum(params))
    assert np.allclose(wfn.get_overlap(0b000111, np.arange(50)), params[:50])


def test_convert_to_fanci():
    """Test fanpy.utils.convert_to_fanci."""
    fanci = pytest.importorskip("fanci")
    pyci = pytest.importorskip("pyci")

    one_int = np.load(find_datafile("data/data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data/data_lih_hf_sto6g_twoint.npy"))

    ham = RestrictedMolecularHamiltonian(one_int, two_int, update_prev_params=True)
    pyci_ham = pyci.hamiltonian(0, ham.one_int, ham.two_int)

    wfn = AP1roG(4, one_int.shape[0] * 2, params=None, memory=None, ref_sd=None, ngem=None)
    fanci_wfn = convert_to_fanci(wfn, pyci_ham, nproj=189, step_print=False)

    # convert slater determinants
    sds = []
    for i, occs in enumerate(fanci_wfn._sspace):
        # convert occupation vector to sd
        sd = slater.create(0, *occs[0])
        sd = slater.create(sd, *(occs[1] + wfn.nspatial))
        sds.append(sd)

    wfn = CIWavefunction(4, one_int.shape[0] * 2, params=None, memory=None, sds=sds)
    wfn.assign_params(wfn.params + 0.5 * 2 * (np.random.rand(*wfn.params.shape) - 0.5))
    fanci_wfn = convert_to_fanci(wfn, pyci_ham, nproj=len(sds), step_print=False)
    energy = np.random.rand(1)
    params = np.hstack([wfn.params.flatten(), energy])

    objective = ProjectedSchrodinger(wfn, ham, refwfn=sds[0], pspace=sds[:len(sds)],
                                     energy_type="variable", energy=energy, step_print=False)
    assert np.allclose(
        objective.objective(objective.active_params)[:-1], fanci_wfn.compute_objective(params)[:-1]
    )
    # FIXME: no clue why, but the following np.allclose gets segmentation fault if the array has
    # 2**15 or more elements
    assert np.allclose(
        objective.jacobian(objective.active_params).flatten()[:32767],
        fanci_wfn.compute_jacobian(params).flatten()[:32767]
    )
    assert np.allclose(
        objective.jacobian(objective.active_params)[:-1].flatten()[32767:],
        fanci_wfn.compute_jacobian(params)[:-1].flatten()[32767:]
    )

    objective = ProjectedSchrodinger(
        wfn, ham, refwfn=sds, pspace=sds, energy_type="variable", energy=energy, step_print=False,
        eqn_weights=np.ones(len(sds) + 1),
    )
    assert np.allclose(
        objective.objective(objective.active_params)[-1], fanci_wfn.compute_objective(params)[-1]
    )
    assert np.allclose(
        objective.jacobian(objective.active_params)[-1], fanci_wfn.compute_jacobian(params)[-1]
    )

    fanci_wfn = convert_to_fanci(wfn, pyci_ham, nproj=len(sds), step_print=False, objective_type="energy")
    objective = EnergyOneSideProjection(wfn, ham, refwfn=sds, step_print=False)
    objective.gradient(objective.active_params), fanci_wfn.compute_jacobian(params)
    assert np.allclose(
        objective.objective(objective.active_params), fanci_wfn.compute_objective(params)
    )
    assert np.allclose(
        objective.gradient(objective.active_params), fanci_wfn.compute_jacobian(params)
    )


def test_convert_to_fanci_ap1rog():
    """Test fanpy.utils.convert_to_fanci."""
    fanci = pytest.importorskip("fanci")
    pyci = pytest.importorskip("pyci")

    one_int = np.load(find_datafile("data/data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data/data_lih_hf_sto6g_twoint.npy"))

    ham = RestrictedMolecularHamiltonian(one_int, two_int, update_prev_params=True)
    pyci_ham = pyci.hamiltonian(0, ham.one_int, ham.two_int)

    wfn = AP1roG(4, one_int.shape[0] * 2, params=None, memory=None, ref_sd=None, ngem=None)
    wfn.assign_params(wfn.params + 0.5 * 2 * (np.random.rand(*wfn.params.shape) - 0.5))
    fanci_wfn = convert_to_fanci(wfn, pyci_ham, nproj=225, step_print=False)

    energy = np.random.rand(1)
    params = np.hstack([wfn.params.flatten(), energy])

    # convert slater determinants
    sds = []
    for i, occs in enumerate(fanci_wfn._sspace):
        # convert occupation vector to sd
        sd = slater.create(0, *occs[0])
        sd = slater.create(sd, *(occs[1] + wfn.nspatial))
        sds.append(sd)

    objective = ProjectedSchrodinger(wfn, ham, refwfn=sds[0], pspace=sds[:225],
                                     energy_type="variable", energy=energy, step_print=False)
    assert np.allclose(
        objective.objective(objective.active_params)[:-1], fanci_wfn.compute_objective(params)[:-1]
    )
    assert np.allclose(
        objective.jacobian(objective.active_params)[:-1], fanci_wfn.compute_jacobian(params)[:-1]
    )

    objective = ProjectedSchrodinger(
        wfn, ham, refwfn=sds, pspace=sds, energy_type="variable", energy=energy, step_print=False,
        eqn_weights=np.ones(len(sds) + 1),
    )
    assert np.allclose(
        objective.objective(objective.active_params)[-1], fanci_wfn.compute_objective(params)[-1]
    )
    assert np.allclose(
        objective.jacobian(objective.active_params)[-1], fanci_wfn.compute_jacobian(params)[-1]
    )

    fanci_wfn = convert_to_fanci(wfn, pyci_ham, nproj=225, step_print=False, objective_type="energy")
    objective = EnergyOneSideProjection(wfn, ham, refwfn=sds[:225], step_print=False)
    objective.gradient(objective.active_params), fanci_wfn.compute_jacobian(params)
    assert np.allclose(
        objective.objective(objective.active_params), fanci_wfn.compute_objective(params)
    )
    assert np.allclose(
        objective.gradient(objective.active_params, normalize=False),
        fanci_wfn.compute_jacobian(params),
    )
