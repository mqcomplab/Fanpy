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
    overlap = pyci_obj.compute_overlap(np.random.rand(4), "P")
    assert overlap[0] == 1.0

