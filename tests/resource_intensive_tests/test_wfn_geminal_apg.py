"""Test fanpy.wavefunction.geminals.apg."""

from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.solver.equation import cma, minimize
from fanpy.solver.system import least_squares
from fanpy.wfn.geminal.apg import APG

import numpy as np

from utils import find_datafile


# FIXME: answer should be brute force or external (should not depend on the code)
def answer_apg_lih_sto6g():
    """Find the APG/STO-6G wavefunction variationally for LiH system."""
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("../data/data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_lih_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.995317634356
    ham = RestrictedMolecularHamiltonian(one_int, two_int)
    apg = APG(4, 12)
    full_sds = [
        1 << i | 1 << j | 1 << k | 1 << l
        for i in range(12)
        for j in range(i + 1, 12)
        for k in range(j + 1, 12)
        for l in range(k + 1, 12)  # noqa: E741
    ]

    objective = EnergyOneSideProjection(apg, ham, refwfn=full_sds)
    results = minimize(objective)
    print(results)
    print(apg.params)


def test_apg_lih_sto6g_slow():
    """Test APG wavefunction using H2 with LiH/STO-6G orbital.

    Answers obtained from answer_apg_lih_sto6g

    HF (Electronic) Energy : -8.9472891719
    APG Energy :

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
    apg = APG(4, 12)
    full_sds = [
        1 << i | 1 << j | 1 << k | 1 << l
        for i in range(12)
        for j in range(i + 1, 12)
        for k in range(j + 1, 12)
        for l in range(k + 1, 12)  # noqa: E741
    ]

    # Solve system of equations
    objective = ProjectedSchrodinger(apg, ham, refwfn=full_sds)
    results = least_squares(objective)
    assert np.allclose(results["energy"], 0.0)
