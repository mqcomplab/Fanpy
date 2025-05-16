"""Test fanpy.ham.generalized_chemical."""

from fanpy.ham.generalized_chemical import GeneralizedMolecularHamiltonian
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np

from utils import find_datafile

def test_integrate_sd_sd_h2_631gdp():
    """Test GenrealizedMolecularHamiltonian.integrate_sd_sd using H2 HF/6-31G** orbitals.

    Compare CI matrix with the PySCF result.
    Integrals that correspond to restricted orbitals were used.

    """
    restricted_one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    restricted_two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    one_int = np.zeros((20, 20))
    one_int[:10, :10] = restricted_one_int
    one_int[10:, 10:] = restricted_one_int
    two_int = np.zeros((20, 20, 20, 20))
    two_int[:10, :10, :10, :10] = restricted_two_int
    two_int[:10, 10:, :10, 10:] = restricted_two_int
    two_int[10:, :10, 10:, :10] = restricted_two_int
    two_int[10:, 10:, 10:, 10:] = restricted_two_int

    ham = GeneralizedMolecularHamiltonian(one_int, two_int)

    ref_ci_matrix = np.load(find_datafile("data_h2_hf_631gdp_cimatrix.npy"))
    ref_pspace = np.load(find_datafile("data_h2_hf_631gdp_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose(ham.integrate_sd_sd(sd1, sd2), ref_ci_matrix[i, j])


def test_integrate_sd_wfn_h2_631gdp():
    """Test GeneralizedMolecularHamiltonian.integrate_sd_wfn using H2 HF/6-31G** orbitals.

    Compare projected energy with the transformed CI matrix from PySCF.
    Compare projected energy with the transformed integrate_sd_sd.
    Integrals that correspond to restricted orbitals were used.

    """
    restricted_one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    restricted_two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    one_int = np.zeros((20, 20))
    one_int[:10, :10] = restricted_one_int
    one_int[10:, 10:] = restricted_one_int
    two_int = np.zeros((20, 20, 20, 20))
    two_int[:10, :10, :10, :10] = restricted_two_int
    two_int[:10, 10:, :10, 10:] = restricted_two_int
    two_int[10:, :10, 10:, :10] = restricted_two_int
    two_int[10:, 10:, 10:, 10:] = restricted_two_int

    ham = GeneralizedMolecularHamiltonian(one_int, two_int)

    ref_ci_matrix = np.load(find_datafile("data_h2_hf_631gdp_cimatrix.npy"))
    ref_pspace = np.load(find_datafile("data_h2_hf_631gdp_civec.npy")).tolist()

    params = np.random.rand(len(ref_pspace))
    wfn = CIWavefunction(2, 10, sds=ref_pspace, params=params)
    for i, sd in enumerate(ref_pspace):
        assert np.allclose(ham.integrate_sd_wfn(sd, wfn), ref_ci_matrix[i, :].dot(params))
        assert np.allclose(
            ham.integrate_sd_wfn(sd, wfn),
            sum(ham.integrate_sd_sd(sd, sd1) * wfn.get_overlap(sd1) for sd1 in ref_pspace),
        )


def test_integrate_sd_wfn_h4_sto6g():
    """Test GeneralizedMolecularHamiltonian.integrate_sd_wfn using H4 HF/STO6G orbitals.

    Compare projected energy with the transformed integrate_sd_sd.
    Integrals that correspond to restricted orbitals were used.

    """
    nelec = 4
    nspin = 8
    sds = sd_list(4, 8, num_limit=None, exc_orders=None)
    wfn = CIWavefunction(nelec, nspin, sds=sds)
    np.random.seed(1000)
    wfn.assign_params(np.random.rand(len(sds)))

    restricted_one_int = np.load(find_datafile("data_h4_square_hf_sto6g_oneint.npy"))
    restricted_two_int = np.load(find_datafile("data_h4_square_hf_sto6g_twoint.npy"))
    one_int = np.zeros((8, 8))
    one_int[:4, :4] = restricted_one_int
    one_int[4:, 4:] = restricted_one_int
    two_int = np.zeros((8, 8, 8, 8))
    two_int[:4, :4, :4, :4] = restricted_two_int
    two_int[:4, 4:, :4, 4:] = restricted_two_int
    two_int[4:, :4, 4:, :4] = restricted_two_int
    two_int[4:, 4:, 4:, 4:] = restricted_two_int

    ham = GeneralizedMolecularHamiltonian(one_int, two_int)

    for sd in sds:
        assert np.allclose(
            ham.integrate_sd_wfn(sd, wfn),
            sum(ham.integrate_sd_sd(sd, sd1) * wfn.get_overlap(sd1) for sd1 in sds),
        )


def test_integrate_sd_sd_lih_631g_trial_slow():
    """Test GeneralizedMolecularHamiltonian.integrate_sd_sd using LiH HF/6-31G orbitals.

    Integrals that correspond to restricted orbitals were used.

    """
    restricted_one_int = np.load(find_datafile("data_lih_hf_631g_oneint.npy"))
    restricted_two_int = np.load(find_datafile("data_lih_hf_631g_twoint.npy"))
    one_int = np.zeros((22, 22))
    one_int[:11, :11] = restricted_one_int
    one_int[11:, 11:] = restricted_one_int
    two_int = np.zeros((22, 22, 22, 22))
    two_int[:11, :11, :11, :11] = restricted_two_int
    two_int[:11, 11:, :11, 11:] = restricted_two_int
    two_int[11:, :11, 11:, :11] = restricted_two_int
    two_int[11:, 11:, 11:, 11:] = restricted_two_int

    ham = GeneralizedMolecularHamiltonian(one_int, two_int)

    ref_ci_matrix = np.load(find_datafile("data_lih_hf_631g_cimatrix.npy"))
    ref_pspace = np.load(find_datafile("data_lih_hf_631g_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose(ham.integrate_sd_sd(sd1, sd2), ref_ci_matrix[i, j])


def test_integrate_sd_sd_deriv_fdiff_h2_sto6g():
    """Test GeneralizedMolecularHamiltonian._integrate_sd_sd_deriv using H2/STO6G.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    restricted_one_int = np.load(find_datafile("data_h4_square_hf_sto6g_oneint.npy"))
    restricted_two_int = np.load(find_datafile("data_h4_square_hf_sto6g_twoint.npy"))
    one_int = np.zeros((8, 8))
    one_int[:4, :4] = restricted_one_int
    one_int[4:, 4:] = restricted_one_int
    two_int = np.zeros((8, 8, 8, 8))
    two_int[:4, :4, :4, :4] = restricted_two_int
    two_int[:4, 4:, :4, 4:] = restricted_two_int
    two_int[4:, :4, 4:, :4] = restricted_two_int
    two_int[4:, 4:, 4:, 4:] = restricted_two_int

    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)
    epsilon = 1e-8

    for sd1 in [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]:
        for sd2 in [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = GeneralizedMolecularHamiltonian(one_int, two_int, params=addition)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd_decomposed(sd1, sd2))
                    - np.array(test_ham.integrate_sd_sd_decomposed(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv_decomposed(sd1, sd2, np.array([i])).ravel()
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)


def test_integrate_sd_sd_deriv_fdiff_h4_sto6g_trial_slow():
    """Test GeneralizedMolecularHamiltonian._integrate_sd_sd_deriv using H4-STO6G integrals.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    restricted_one_int = np.load(find_datafile("data_h4_square_hf_sto6g_oneint.npy"))
    restricted_two_int = np.load(find_datafile("data_h4_square_hf_sto6g_twoint.npy"))
    one_int = np.zeros((8, 8))
    one_int[:4, :4] = restricted_one_int
    one_int[4:, 4:] = restricted_one_int
    two_int = np.zeros((8, 8, 8, 8))
    two_int[:4, :4, :4, :4] = restricted_two_int
    two_int[:4, 4:, :4, 4:] = restricted_two_int
    two_int[4:, :4, 4:, :4] = restricted_two_int
    two_int[4:, 4:, 4:, 4:] = restricted_two_int

    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)
    epsilon = 1e-8

    sds = sd_list(4, 8, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = GeneralizedMolecularHamiltonian(one_int, two_int, params=addition)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd_decomposed(sd1, sd2))
                    - np.array(test_ham.integrate_sd_sd_decomposed(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv_decomposed(sd1, sd2, np.array([i])).ravel()
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)

