"""Test fanpy.tools.wrapper.psi4."""
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.solver.ci import brute
from fanpy.wfn.ci.fci import FCI

import pytest

from utils import find_datafile


def test_generate_fci_cimatrix_h2_631gdp():
    """Test PySCF's FCI calculation against H2 FCI 6-31G** data from Gaussian.

    HF energy: -1.13126983927
    FCI energy: -1.1651487496

    """
    pytest.importorskip("psi4")
    from fanpy.tools.wrapper.psi4 import hartreefock

    hf_data = hartreefock(find_datafile("data/data_h2.xyz"), "6-31gss")

    nelec = 2
    nuc_nuc = hf_data["nuc_nuc"]
    one_int = hf_data["one_int"]
    two_int = hf_data["two_int"]

    wfn = FCI(nelec, one_int.shape[0] * 2)
    ham = RestrictedMolecularHamiltonian(one_int, two_int)
    results = brute(wfn, ham)
    energy = results["energy"]
    assert abs(energy + nuc_nuc - (-1.1651486697)) < 1e-7
