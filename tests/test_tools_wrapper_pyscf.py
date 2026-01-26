"""Test fanpy.tools.wrapper.pyscf."""
import os

import numpy as np

import pytest

from test_wrapper_python_wrapper import check_data_h2_rhf_sto6g, check_data_lih_rhf_sto6g

from utils import find_datafile


def test_pyscf_hartreefock_h2_rhf_sto6g():
    """Test PySCF HF against H2 RHF STO-6G data from Gaussian."""
    pytest.importorskip("pyscf")
    from fanpy.tools.wrapper.pyscf_tools import hartreefock

    hf_data = hartreefock(find_datafile("data/data_h2.xyz"), "sto-6g")
    check_data_h2_rhf_sto6g(
        hf_data["hf_energy"], hf_data["nuc_nuc"], hf_data["one_int"], hf_data["two_int"]
    )


def test_pyscf_hartreefock_lih_rhf_sto6g():
    """Test PySCF HF against LiH RHF STO6G data from Gaussian."""
    pytest.importorskip("pyscf")
    from fanpy.tools.wrapper.pyscf_tools import hartreefock, __file__ as cwd

    hf_data = hartreefock(find_datafile("data/data_lih.xyz"), "sto-6g")
    check_data_lih_rhf_sto6g(
        hf_data["hf_energy"], hf_data["nuc_nuc"], hf_data["one_int"], hf_data["two_int"]
    )

    path = find_datafile("data/data_lih.xyz")
    cwd = os.path.dirname(cwd)
    hf_data = hartreefock(os.path.relpath(path, cwd), "sto-6g")
    check_data_lih_rhf_sto6g(
        hf_data["hf_energy"], hf_data["nuc_nuc"], hf_data["one_int"], hf_data["two_int"]
    )

    with pytest.raises(ValueError):
        hartreefock("afasfasd", "sto-6g")
    with pytest.raises(NotImplementedError):
        hartreefock(find_datafile("data/data_lih.xyz"), "sto-6g", is_unrestricted=True)


def test_generate_fci_cimatrix_h2_631gdp():
    """Test PySCF's FCI calculation against H2 FCI 6-31G** data from Gaussian.

    HF energy: -1.13126983927
    FCI energy: -1.1651487496

    """
    pytest.importorskip("pyscf")
    from fanpy.tools.wrapper.pyscf_tools import hartreefock
    from fanpy.tools.wrapper.pyscf_tools import fci_cimatrix

    hf_data = hartreefock(find_datafile("data/data_h2.xyz"), "6-31gss")

    nelec = 2
    nuc_nuc = hf_data["nuc_nuc"]
    one_int = hf_data["one_int"]
    two_int = hf_data["two_int"]

    # physicist notation
    ci_matrix, pspace = fci_cimatrix(one_int, two_int, nelec, is_chemist_notation=False)
    ground_energy = np.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-1.1651486697)) < 1e-7

    ci_matrix, pspace = fci_cimatrix(one_int, two_int, (1, 1))
    ground_energy = np.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-1.1651486697)) < 1e-7

    # chemist notation
    ci_matrix, pspace = fci_cimatrix(
        one_int, np.einsum("ikjl->ijkl", two_int), nelec, is_chemist_notation=True
    )
    ground_energy = np.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-1.1651486697)) < 1e-7

    with pytest.raises(ValueError):
        fci_cimatrix(one_int, two_int, "3")


def test_generate_fci_cimatrix_lih_sto6g():
    """Test python_wrapper.generate_fci_cimatrix with LiH STO-6G.

    HF energy: -7.95197153880
    FCI energy: -7.9723355823.

    """
    pytest.importorskip("pyscf")
    from fanpy.tools.wrapper.pyscf_tools import hartreefock
    from fanpy.tools.wrapper.pyscf_tools import fci_cimatrix

    hf_data = hartreefock(find_datafile("data/data_lih.xyz"), "sto-6g")

    nelec = 4
    nuc_nuc = hf_data["nuc_nuc"]
    one_int = hf_data["one_int"]
    two_int = hf_data["two_int"]

    # physicist notation
    ci_matrix, pspace = fci_cimatrix(one_int, two_int, nelec, is_chemist_notation=False)
    ground_energy = np.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-7.9723355823)) < 1e-7
    # chemist notation
    ci_matrix, pspace = fci_cimatrix(
        one_int, np.einsum("ikjl->ijkl", two_int), nelec, is_chemist_notation=True
    )
    ground_energy = np.linalg.eigh(ci_matrix)[0][0] + nuc_nuc
    assert abs(ground_energy - (-7.9723355823)) < 1e-7
