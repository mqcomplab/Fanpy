"""Test fanpy.wavefunction.geminals.apg."""
import types

from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.solver.equation import minimize
from fanpy.solver.system import least_squares
from fanpy.tools.graphs import generate_complete_pmatch
from fanpy.wfn.geminal.apg import APG

import numpy as np

import pytest

from utils import find_datafile, skip_init


def test_apg_assign_orbpairs():
    """Test APG.assign_orbpairs."""
    test = skip_init(APG)
    test.nspin = 10
    with pytest.raises(ValueError):
        test.assign_orbpairs((0, 1))
    with pytest.raises(ValueError):
        test.assign_orbpairs([(0, 1)])
    test.assign_orbpairs(None)
    dict_orbpair_ind = {}
    dict_ind_orbpair = {}
    counter = 0
    for i in range(10):
        for j in range(i + 1, 10):
            dict_orbpair_ind[(i, j)] = counter
            dict_ind_orbpair[counter] = (i, j)
            counter += 1


def test_apg_get_col_ind():
    """Test APG.get_col_ind."""
    test = skip_init(APG)
    test.nspin = 10

    orbpairs = [(i, j) for i in range(10) for j in range(i + 1, 10)]

    for i, orbpair in enumerate(orbpairs):
        assert test.get_col_ind(orbpair) == i

    assert test.get_col_ind((1, 0)) == 0
    with pytest.raises(ValueError):
        test.get_col_ind((0, 10))
    with pytest.raises(ValueError):
        test.get_col_ind((0, 0))
    with pytest.raises(ValueError):
        test.get_col_ind((9, 10))
    with pytest.raises(TypeError):
        test.get_col_ind([1, 4])


def test_apg_get_orbpair():
    """Test APG.get_orbpair."""
    test = skip_init(APG)
    test.nspin = 10

    orbpairs = [(i, j) for i in range(10) for j in range(i + 1, 10)]

    for i, orbpair in enumerate(orbpairs):
        assert test.get_orbpair(i) == orbpair

    with pytest.raises(ValueError):
        test.get_orbpair(-1)
    with pytest.raises(ValueError):
        test.get_orbpair(len(orbpairs))


def test_assign_pmatch_generator():
    """Test APG.generate_possible_orbpairs."""
    test = skip_init(APG)
    sd = (0, 1, 2, 3, 4, 5)
    assert isinstance(test.generate_possible_orbpairs(sd), types.GeneratorType)
    for i, j in zip(test.generate_possible_orbpairs(sd), generate_complete_pmatch(sd)):
        assert i == j


# FIXME: answer should be brute force or external (should not depend on the code)
def answer_apg_h2_sto6g():
    """Find the APG/STO-6G wavefunction variationally for H2 system."""
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.71317683129
    ham = RestrictedMolecularHamiltonian(one_int, two_int)
    apg = APG(2, 4)
    full_sds = (0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100)

    objective = EnergyOneSideProjection(apg, ham, refwfn=full_sds)
    results = minimize(objective)
    print(results)
    print(apg.params)


def test_apg_h2_sto6g():
    """Test APG wavefunction using H2 with HF/STO-6G orbitals.

    Answers obtained from answer_apg_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    APG Energy : -1.8590898441488932
    APG Coeffs : [0.00000000e+00, 1.00061615e+00, 4.43043808e-16, 4.43043808e-16, -1.13806005e-01,
                  0.00000000e+00]

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.71317683129
    ham = RestrictedMolecularHamiltonian(one_int, two_int)
    apg = APG(2, 4)
    full_sds = (0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100)

    # Solve system of equations
    objective = ProjectedSchrodinger(apg, ham, refwfn=full_sds)
    results = least_squares(objective)
    assert np.allclose(results["energy"], -1.8590898441488932)


