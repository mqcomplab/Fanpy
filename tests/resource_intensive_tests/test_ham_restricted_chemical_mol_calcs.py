"""Test fanpy.ham.restricted_chemical."""

from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.tools.sd_list import sd_list

import numpy as np

from utils import find_datafile

def test_integrate_sd_sd_h2_631gdp():
    """Test RestrictedMolecularHamiltonian.integrate_sd_sd using H2 HF/6-31G** orbitals.

    Compare CI matrix with the PySCF result.

    """
    one_int = np.load(find_datafile("../data/data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_h2_hf_631gdp_twoint.npy"))
    ham = RestrictedMolecularHamiltonian(one_int, two_int)

    ref_ci_matrix = np.load(find_datafile("../data/data_h2_hf_631gdp_cimatrix.npy"))
    ref_pspace = np.load(find_datafile("../data/data_h2_hf_631gdp_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose((ham.integrate_sd_sd(sd1, sd2)), ref_ci_matrix[i, j])


def test_integrate_sd_sd_decomposed_lih_631g_case():
    """Test RestrictedMolecularHamiltonian.integrate_sd_sd using sd's of LiH HF/6-31G orbitals."""
    one_int = np.load(find_datafile("../data/data_lih_hf_631g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_lih_hf_631g_twoint.npy"))
    ham = RestrictedMolecularHamiltonian(one_int, two_int)

    sd1 = 0b0000000001100000000111
    sd2 = 0b0000000001100100001001
    assert np.allclose(
        (0, two_int[1, 2, 3, 8], -two_int[1, 2, 8, 3]),
        ham.integrate_sd_sd_decomposed(sd1, sd2),
    )


def test_integrate_sd_sd_deriv_fdiff_h2_sto6g():
    """Test RestrictedMolecularHamiltonian._integrate_sd_sd_deriv using H2/STO6G.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.load(find_datafile("../data/data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_h2_hf_sto6g_twoint.npy"))
    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    epsilon = 1e-8

    for sd1 in [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]:
        for sd2 in [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = RestrictedMolecularHamiltonian(one_int, two_int, params=addition)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2)) - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, np.array([i]))
                derivative = np.sum(derivative)
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)


# TODO: add test for comparing Unrestricted with Generalized
def test_integrate_sd_sd_deriv_fdiff_h4_sto6g_slow():
    """Test RestrictedMolecularHamiltonian._integrate_sd_sd_deriv using H4/STO6G.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.load(find_datafile("../data/data_h4_square_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_h4_square_hf_sto6g_twoint.npy"))

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    epsilon = 1e-8

    sds = sd_list(4, 8, num_limit=None, exc_orders=None)

    assert np.allclose(one_int, one_int.T)
    assert np.allclose(np.einsum("ijkl->jilk", two_int), two_int)
    assert np.allclose(np.einsum("ijkl->klij", two_int), two_int)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = RestrictedMolecularHamiltonian(one_int, two_int, params=addition)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2)) - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, np.array([i]))
                derivative = np.sum(derivative)
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)
