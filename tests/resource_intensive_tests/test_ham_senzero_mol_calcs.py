"""Test fanpy.ham.senzero."""

from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.ham.senzero import SeniorityZeroHamiltonian
from fanpy.tools.slater import get_seniority

import numpy as np

from utils import find_datafile

def test_integrate_sd_sd_h2_631gdp():
    """Test SeniorityZeroHamiltonian.integrate_sd_sd using H2 HF/6-31G** orbitals.

    Compare CI matrix with the PySCF result.

    """
    one_int = np.load(find_datafile("../data/data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_h2_hf_631gdp_twoint.npy"))
    full_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    test_ham = SeniorityZeroHamiltonian(one_int, two_int)

    ref_pspace = np.load(find_datafile("../data/data_h2_hf_631gdp_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        sd1 = int(sd1)
        if get_seniority(sd1, one_int.shape[0]) != 0:
            continue
        for j, sd2 in enumerate(ref_pspace):
            sd2 = int(sd2)
            if get_seniority(sd2, one_int.shape[0]) != 0:
                continue
            assert np.allclose(full_ham.integrate_sd_sd(sd1, sd2), test_ham.integrate_sd_sd(sd1, sd2))


def test_integrate_sd_sd_lih_631g_full():
    """Test SeniorityZeroHamiltonian.integrate_sd_sd using LiH HF/6-31G orbitals.

    Compared to all of the CI matrix.

    """
    one_int = np.load(find_datafile("../data/data_lih_hf_631g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_lih_hf_631g_twoint.npy"))
    full_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    test_ham = SeniorityZeroHamiltonian(one_int, two_int)

    ref_pspace = np.load(find_datafile("../data/data_lih_hf_631g_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        sd1 = int(sd1)
        if get_seniority(sd1, one_int.shape[0]) != 0:
            continue
        for j, sd2 in enumerate(ref_pspace):
            sd2 = int(sd2)
            if get_seniority(sd2, one_int.shape[0]) != 0:
                continue
            assert np.allclose(full_ham.integrate_sd_sd(sd1, sd2), test_ham.integrate_sd_sd(sd1, sd2))