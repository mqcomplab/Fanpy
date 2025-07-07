"""Test fanpy.wfn.geminal.apsetg."""
import types

from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.solver.equation import cma, minimize
from fanpy.solver.system import least_squares
from fanpy.wfn.geminal.apsetg import BasicAPsetG

import numpy as np

import pytest

from utils import find_datafile

# FIXME: answer should be brute force or external (should not depend on the code)
def answer_apsetg_lih_sto6g():
    """Find the BasicAPsetG/STO-6G wavefunction variationally for LiH system."""
    one_int = np.load(find_datafile("../data/data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_lih_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.995317634356
    ham = RestrictedMolecularHamiltonian(one_int, two_int)
    apsetg = BasicAPsetG(4, 12)
    full_sds = [
        1 << i | 1 << j | 1 << k | 1 << l
        for i in range(6)
        for j in range(i + 1, 6)
        for k in range(6, 12)
        for l in range(k + 1, 12)  # noqa: E741
    ]

    objective = EnergyOneSideProjection(apsetg, ham, refwfn=full_sds)
    results = minimize(objective)
    print(results)
    print(apsetg.params)


def test_apsetg_lih_sto6g_slow():
    """Test BasicAPsetG with LiH using HF/STO-6G orbitals.

    HF Value :       -8.9472891719
    Old Code Value : -8.96353105152
    FCI Value :      -8.96741814557

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("../data/data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_lih_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.995317634356
    ham = RestrictedMolecularHamiltonian(one_int, two_int)
    apsetg = BasicAPsetG(4, 12)
    full_sds = [
        1 << i | 1 << j | 1 << k | 1 << l
        for i in range(6)
        for j in range(i + 1, 6)
        for k in range(6, 12)
        for l in range(k + 1, 12)  # noqa: E741
    ]

    # Solve system of equations
    objective = ProjectedSchrodinger(apsetg, ham, refwfn=full_sds)
    results = least_squares(objective)
    assert np.allclose(results["energy"], 0.0)
