"""Test fanpy.interface.pyscf."""

import pytest
from pyscf import gto, scf

from fanpy.interface.pyscf import PYSCF
from test_wrapper_python_wrapper import check_data_h2_rhf_sto6g

def test_hf_data_h2_sto6g():
    """Make sure computed integrals, and extracted HF energies are the same as Gaussian."""
    geom = [["H", [0.000000,     0.000000,    -0.371000000]], 
            ["H", [0.000000,     0.000000,    0.371000000]]]
    mol = gto.M(atom=geom, basis="sto-6g", parse_arg=False, unit="angstrom")
    hf = scf.RHF(mol)
    # run hf
    hf.scf()
    pyscf_interface = PYSCF(hf)
    check_data_h2_rhf_sto6g(pyscf_interface.energy_elec, pyscf_interface.energy_nuc, pyscf_interface.one_int, pyscf_interface.two_int)

def test_error():
    """ Make sure error gets raised"""
    # type error should be raised if input not instance of pyscf.scf.hf.SCF
    with pytest.raises(TypeError):
        PYSCF("Not HF")