"""Test fanpy.wavefunction.fci."""
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.solver.ci import brute
from fanpy.wfn.ci.fci import FCI

import numpy as np

import pytest

from utils import find_datafile, skip_init


def test_fci_assign_seniority():
    """Test FCI.assign_seniority."""
    test = skip_init(FCI)
    with pytest.raises(ValueError):
        test.assign_seniority(0)
    with pytest.raises(ValueError):
        test.assign_seniority(1)
    test.assign_seniority(None)
    assert test.seniority is None


def test_fci_assign_sds():
    """Test FCI.assign_sds."""
    test = FCI(2, 4)
    with pytest.raises(ValueError):
        test.assign_sds(1)
    with pytest.raises(ValueError):
        test.assign_sds([0b0101])
    test.assign_sds(None)
    assert test.sds == (0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010)


def test_fci_h2_631gdp():
    """Test FCI wavefunction for H2 (6-31g**).

    HF energy: -1.13126983927
    FCI energy: -1.1651487496
    """
    nelec = 2
    nspin = 20
    fci = FCI(nelec, nspin)

    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data/data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("data/data_h2_hf_631gdp_twoint.npy"))
    nuc_nuc = 0.71317683129
    ham = RestrictedMolecularHamiltonian(one_int, two_int)

    # optimize
    results = brute(fci, ham)
    energy = results["energy"]
    # compare with number from Gaussian
    assert abs(energy + nuc_nuc - (-1.1651486697)) < 1e-7
