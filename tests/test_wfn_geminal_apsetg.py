"""Test fanpy.wfn.geminal.apsetg."""
import types

from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.solver.equation import minimize
from fanpy.solver.system import least_squares
from fanpy.tools import graphs
from fanpy.wfn.geminal.apsetg import BasicAPsetG

import numpy as np

import pytest

from utils import find_datafile, skip_init


def test_apsetg_assign_orbpairs():
    """Test BasicAPsetG.assign_orbpairs."""
    test = skip_init(BasicAPsetG)
    test.nspin = 10
    with pytest.raises(ValueError):
        test.assign_orbpairs((0, 5))
    with pytest.raises(ValueError):
        test.assign_orbpairs([(0, 5)])
    test.assign_orbpairs(None)
    dict_orbpair_ind = {}
    dict_ind_orbpair = {}
    counter = 0
    for i in range(5):
        for j in range(5, 10):
            dict_orbpair_ind[(i, j)] = counter
            dict_ind_orbpair[counter] = (i, j)
            counter += 1


def test_apsetg_get_col_ind():
    """Test BasicAPsetG.get_col_ind."""
    test = skip_init(BasicAPsetG)
    test.nspin = 10

    orbpairs = [(i, j) for i in range(5) for j in range(5, 10)]

    for i, orbpair in enumerate(orbpairs):
        assert test.get_col_ind(orbpair) == i

    with pytest.raises(ValueError):
        test.get_col_ind((0, 4))
    with pytest.raises(ValueError):
        test.get_col_ind((0, 0))
    with pytest.raises(ValueError):
        test.get_col_ind((0, 10))


def test_apsetg_get_orbpair():
    """Test BasicAPsetG.get_orbpair."""
    test = skip_init(BasicAPsetG)
    test.nspin = 10

    orbpairs = [(i, j) for i in range(5) for j in range(5, 10)]

    for i, orbpair in enumerate(orbpairs):
        assert test.get_orbpair(i) == orbpair

    with pytest.raises(ValueError):
        test.get_orbpair(-1)
    with pytest.raises(ValueError):
        test.get_orbpair(len(orbpairs))


def test_assign_pmatch_generator():
    """Test BasicAPsetG.generate_possible_orbpairs."""
    test = skip_init(BasicAPsetG)
    test.assign_nspin(6)
    sd = (0, 1, 2, 3, 4, 5)
    assert isinstance(test.generate_possible_orbpairs(sd), types.GeneratorType)
    for i, j in zip(
        test.generate_possible_orbpairs(sd), graphs.generate_biclique_pmatch((0, 1, 2), (3, 4, 5))
    ):
        assert i == j


# FIXME: answer should be brute force or external (should not depend on the code)
def answer_apsetg_h2_sto6g():
    """Find the APsetG/STO-6G wavefunction variationally for H2 system."""
    one_int = np.load(find_datafile("data/data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data/data_h2_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.71317683129
    ham = RestrictedMolecularHamiltonian(one_int, two_int)
    apsetg = BasicAPsetG(2, 4)
    full_sds = (0b0101, 0b1001, 0b0110, 0b1010)

    objective = EnergyOneSideProjection(apsetg, ham, refwfn=full_sds)
    results = minimize(objective)
    print(results)
    print(apsetg.params)


def test_apsetg_h2_sto6g():
    """Test BasicAPsetG wavefunction using H2 with HF/STO-6G orbitals.

    Answers obtained from answer_apsetg_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    APsetG Energy : -1.859089844148893
    APsetG Coeffs : [0.00000000e+00, 9.99041662e-01, 4.39588902e-16, 4.39588902e-16,
                     -1.13626930e-01, 0.00000000e+00])

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data/data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data/data_h2_hf_sto6g_twoint.npy"))
    # nuc_nuc = 0.71317683129
    ham = RestrictedMolecularHamiltonian(one_int, two_int)
    apsetg = BasicAPsetG(2, 4)
    full_sds = (0b0101, 0b1001, 0b0110, 0b1010)

    # Solve system of equations
    objective = ProjectedSchrodinger(apsetg, ham, refwfn=full_sds)
    results = least_squares(objective)
    assert np.allclose(results["energy"], -1.8590898441488932)
