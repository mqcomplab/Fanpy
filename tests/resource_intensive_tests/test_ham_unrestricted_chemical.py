"""Test fanpy.ham.unrestricted_chemical."""

from fanpy.ham.unrestricted_chemical import UnrestrictedMolecularHamiltonian
from fanpy.tools.sd_list import sd_list

import numpy as np

from utils import find_datafile


def test_integrate_sd_sd_h2_631gdp():
    """Test UnrestrictedMolecularHamiltonian.integrate_sd_sd using H2 HF/6-31G** orbitals.

    Compare CI matrix with the PySCF result.
    Integrals that correspond to restricted orbitals were used.

    """
    one_int = np.load(find_datafile("../data/data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_h2_hf_631gdp_twoint.npy"))
    ham = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3)

    ref_ci_matrix = np.load(find_datafile("../data/data_h2_hf_631gdp_cimatrix.npy"))
    ref_pspace = np.load(find_datafile("../data/data_h2_hf_631gdp_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose((ham.integrate_sd_sd(sd1, sd2)), ref_ci_matrix[i, j])


def test_integrate_sd_sd_decomposed_lih_631g_case():
    """Test UnrestrictedMolecularHamiltonian.integrate_sd_sd using sd's of LiH HF/6-31G orbitals."""
    one_int = np.load(find_datafile("../data/data_lih_hf_631g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_lih_hf_631g_twoint.npy"))
    ham = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3)

    sd1 = 0b0000000001100000000111
    sd2 = 0b0000000001100100001001
    assert np.allclose(
        (0, two_int[1, 2, 3, 8], -two_int[1, 2, 8, 3]),
        ham.integrate_sd_sd_decomposed(sd1, sd2),
    )
    sd1 = 0b0000000000000000000011
    sd2 = 0b0000000000000000000101
    assert np.allclose(
        (one_int[1, 2], two_int[0, 1, 0, 2], -two_int[0, 1, 2, 0]),
        ham.integrate_sd_sd_decomposed(sd1, sd2),
    )
    sd1 = 0b0000000001100000000000
    sd2 = 0b0000000010100000000000
    assert np.allclose(
        (one_int[1, 2], two_int[0, 1, 0, 2], -two_int[0, 1, 2, 0]),
        ham.integrate_sd_sd_decomposed(sd1, sd2),
    )


def test_integrate_sd_sd_deriv_fdiff_h2_sto6g():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sd_deriv using H2/STO6G.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.load(find_datafile("../data/data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_h2_hf_sto6g_twoint.npy"))
    test_ham = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3)
    epsilon = 1e-8

    for sd1 in [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]:
        for sd2 in [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]:
            for i in range(2):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3, params=addition)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2)) - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, np.array([i]))
                derivative = np.sum(derivative)
                assert np.allclose(finite_diff, derivative, atol=1e-5)


def test_integrate_sd_sd_deriv_fdiff_h4_sto6g_slow():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sd_deriv with using H4/STO6G.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.load(find_datafile("../data/data_h4_square_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("../data/data_h4_square_hf_sto6g_twoint.npy"))

    test_ham = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3)
    epsilon = 1e-8
    sds = sd_list(4, 8, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3, params=addition)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2)) - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, np.array([i]))
                derivative = np.sum(derivative)
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)
