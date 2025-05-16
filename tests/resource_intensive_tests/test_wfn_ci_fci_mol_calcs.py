"""Test fanpy.wavefunction.fci."""
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.solver.ci import brute
from fanpy.wfn.ci.fci import FCI

import numpy as np

import pytest

from utils import find_datafile, skip_init

def test_fci_lih_sto6g():
    """Test FCI wavefunction for LiH STO-6G.

    HF energy: -7.95197153880
    FCI energy: -7.9723355823
    """
    nelec = 4
    nspin = 12
    fci = FCI(nelec, nspin)

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_lih_hf_sto6g_twoint.npy"))
    nuc_nuc = 0.995317634356
    ham = RestrictedMolecularHamiltonian(one_int, two_int)

    # optimize
    results = brute(fci, ham)
    energy = results["energy"]
    # compare with number from Gaussian
    assert abs(energy + nuc_nuc - (-7.9723355823)) < 1e-7


def test_fci_lih_631g_slow():
    """Test FCI wavefunction for LiH 6-31G.

    HF energy: -7.97926894940
    FCI energy: -7.9982761
    """
    nelec = 4
    nspin = 22
    fci = FCI(nelec, nspin)

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_631g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data_lih_hf_631g_oneint.npy"))
    two_int = np.load(find_datafile("data_lih_hf_631g_twoint.npy"))
    nuc_nuc = 0.995317634356
    ham = RestrictedMolecularHamiltonian(one_int, two_int)

    # optimize
    results = brute(fci, ham)
    energy = results["energy"]
    # compare with number from Gaussian
    assert abs(energy + nuc_nuc - (-7.9982761)) < 1e-7
