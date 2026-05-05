""" Test for fanpy.interface.fanci.pyci """

import numpy as np

from fanpy.interface.fanci.pyci import ProjectedSchrodingerPyCI
import pyci

from test_interface_pyci import FakeHamiltonian, FakeWavefunction

from fanpy.eqn.projected import BaseSchrodinger

class FakeSchrodinger(BaseSchrodinger):
    """fake fanpy objective for testing purposes"""
    def __init__(self, wfn, ham):
        super().__init__(wfn, ham)
    def objective(self, params):
        return 3.08

def make_test_instance(**overrides):
    """make test instance of ProjectedSchrodingerPyCI with fake fanpy objective and fake pyci hamiltonian and wavefunction
    """
    # build Fake fanpy objective
    wfn = FakeWavefunction(2, 4, np.ones(4))
    nocc = wfn.nelec // 2
    ham = FakeHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    obj = FakeSchrodinger(wfn, ham)

    # build fake pyci hamiltonian
    energy_nuc = 0.0
    pyci_ham = pyci.hamiltonian(energy_nuc, ham.one_int, ham.two_int)

    # build pyci wavefunction 
    # use FCI pspace wavefunction
    pyci_wfn = pyci.fullci_wfn(pyci_ham.nbasis, wfn.nelec - nocc, nocc)

    defaults = {
        "fanpy_objective" : obj,
        "ham" : pyci_ham,
        "wfn" : pyci_wfn,
        "nocc" : 2,
        "seniority": wfn.seniority,
        "nproj": 1,
        "fill": "excitation",
        "mask": np.ones(wfn.params.shape),
        "constraints": {},
        "param_selection": obj.indices_component_params,
        "norm_param": None,
        "norm_det": None,
        "objective_type": "projected",
        "max_memory": 8000,
        "step_print": False,
        "step_save": False,
        "tmpfile": ""
    }
    defaults.update(overrides)
    return ProjectedSchrodingerPyCI(**defaults)


def test_compute_overlap():
    pyci_obj = make_test_instance()

    # compute overlap between the pyci wavefunction and a random vector
    # occ indices are the p-space:
    overlap = pyci_obj.compute_overlap(np.random.rand(4), "P")
    olp_size = len(pyci_obj.pspace) # note p-space contains occ indices
    assert len(overlap) == olp_size
    assert np.allclose(overlap, np.ones(olp_size))

    # compute overlap between the pyci wavefunction and a random vector
    # occ indices are the s-space:
    overlap = pyci_obj.compute_overlap(np.random.rand(4), "S")
    olp_size = len(pyci_obj.sspace) # note s-space contains occ indices
    assert len(overlap) == len(pyci_obj.sspace)
    assert np.allclose(overlap, np.ones(olp_size))

    # compute overlap between the pyci wavefunction and a random vector
    occ_indices = np.asarray([[0, 1]]) # use DOCI occs representation
    overlap = pyci_obj.compute_overlap(np.random.rand(4), occ_indices)
    olp_size = len(occ_indices) 
    assert overlap.shape == (olp_size,)
    assert np.allclose(overlap, np.ones(olp_size))

    # compute overlap between the pyci wavefunction and a random vector
    occ_indices = np.asarray([[[0, 1], [1, 0]], [[1, 1], [0, 0]]]) # use FCI occs representation, with two sd dets
    overlap = pyci_obj.compute_overlap(np.random.rand(4), occ_indices)
    olp_size = len(occ_indices) 
    assert overlap.shape == (olp_size,)
    assert np.allclose(overlap, np.ones(olp_size))