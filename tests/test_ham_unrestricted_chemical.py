"""Test fanpy.ham.unrestricted_chemical."""

import itertools as it

from fanpy.ham.base import BaseHamiltonian
from fanpy.ham.unrestricted_chemical import UnrestrictedMolecularHamiltonian
from fanpy.tools import slater
from fanpy.tools.math_tools import unitary_matrix
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.composite.lincomb import LinearCombinationWavefunction

import numpy as np

import pytest

from utils import disable_abstract, find_datafile


def test_set_ref_ints():
    """Test UnrestrictedMolecularHamiltonian.set_ref_ints."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3)
    assert np.allclose(test._ref_one_int, [one_int] * 2)
    assert np.allclose(test._ref_two_int, [two_int] * 3)

    new_one_int = np.random.rand(2, 2)
    new_two_int = np.random.rand(2, 2, 2, 2)
    test.assign_integrals([new_one_int] * 2, [new_two_int] * 3)
    assert np.allclose(test._ref_one_int, [one_int] * 2)
    assert np.allclose(test._ref_two_int, [two_int] * 3)

    test.set_ref_ints()
    assert np.allclose(test._ref_one_int, [new_one_int] * 2)
    assert np.allclose(test._ref_two_int, [new_two_int] * 3)


def test_cache_two_ints():
    """Test UnrestrictedMolecularHamiltonian.cache_two_ints."""
    one_int = [np.arange(1, 5, dtype=float).reshape(2, 2)] * 2
    two_int = [np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)] * 3
    two_int_ijij = np.array([[5, 10], [15, 20]])
    two_int_ijji = np.array([[5, 11], [14, 20]])

    test = UnrestrictedMolecularHamiltonian(one_int, two_int)
    assert np.allclose(test._cached_two_int_0_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_1_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_2_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_0_ijji, two_int_ijji)
    assert np.allclose(test._cached_two_int_2_ijji, two_int_ijji)

    test.two_int = [np.arange(21, 37).reshape(2, 2, 2, 2)] * 3
    new_two_int_ijij = np.array([[21, 26], [31, 36]])
    new_two_int_ijji = np.array([[21, 27], [30, 36]])
    assert np.allclose(test._cached_two_int_0_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_1_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_2_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_0_ijji, two_int_ijji)
    assert np.allclose(test._cached_two_int_2_ijji, two_int_ijji)

    test.cache_two_ints()
    assert np.allclose(test._cached_two_int_0_ijij, new_two_int_ijij)
    assert np.allclose(test._cached_two_int_1_ijij, new_two_int_ijij)
    assert np.allclose(test._cached_two_int_2_ijij, new_two_int_ijij)
    assert np.allclose(test._cached_two_int_0_ijji, new_two_int_ijji)
    assert np.allclose(test._cached_two_int_2_ijji, new_two_int_ijji)


def test_assign_params():
    """Test UnrestrictedMolecularHamiltonian.assign_params."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)

    test = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3)
    with pytest.raises(ValueError):
        test.assign_params([0, 0])
    with pytest.raises(ValueError):
        test.assign_params(np.array([[0], [0]]))
    with pytest.raises(ValueError):
        test.assign_params(np.array([0]))

    test.assign_params(np.array([0, 0]))
    assert np.allclose(test.params, np.zeros(1))
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    test.assign_params(np.array([10, 0]))
    assert np.allclose(test.params, np.array([10, 0]))
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    test.assign_params(np.array([5, 5]))
    assert np.allclose(test.params, np.array([5, 5]))
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    # make sure that transformation is independent of the transformations that came before it
    test.assign_params(np.array([10, 0]))
    assert np.allclose(test.params, np.array([10, 0]))
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)


def test_integrate_sd_sd_decomposed_trivial():
    """Test UnrestrictedMolecularHamiltonian.integrate_sd_sd for trivial cases."""
    one_int = np.random.rand(3, 3)
    two_int = np.random.rand(3, 3, 3, 3)
    test = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3)

    assert np.allclose((0, 0, 0), test.integrate_sd_sd_decomposed(0b000111, 0b001001))
    assert np.allclose((0, 0, 0), test.integrate_sd_sd_decomposed(0b000111, 0b111000))
    assert np.allclose((0, two_int[0, 1, 1, 0], 0), test.integrate_sd_sd_decomposed(0b110001, 0b101010))
    assert np.allclose(
        (0, -two_int[1, 1, 1, 0] + two_int[0, 1, 0, 0], 0),
        test.integrate_sd_sd_decomposed(0b110001, 0b101010, deriv=np.array([0])).ravel(),
    )

    with pytest.raises(TypeError):
        test.integrate_sd_sd(0b110001, "1")
    with pytest.raises(TypeError):
        test.integrate_sd_sd("1", 0b101010)


def test_integrate_sd_sd_h2_631gdp():
    """Test UnrestrictedMolecularHamiltonian.integrate_sd_sd using H2 HF/6-31G** orbitals.

    Compare CI matrix with the PySCF result.
    Integrals that correspond to restricted orbitals were used.

    """
    one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    ham = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3)

    ref_ci_matrix = np.load(find_datafile("data_h2_hf_631gdp_cimatrix.npy"))
    ref_pspace = np.load(find_datafile("data_h2_hf_631gdp_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose((ham.integrate_sd_sd(sd1, sd2)), ref_ci_matrix[i, j])


def test_integrate_sd_sd_decomposed_lih_631g_case():
    """Test UnrestrictedMolecularHamiltonian.integrate_sd_sd using sd's of LiH HF/6-31G orbitals."""
    one_int = np.load(find_datafile("data_lih_hf_631g_oneint.npy"))
    two_int = np.load(find_datafile("data_lih_hf_631g_twoint.npy"))
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


def test_integrate_sd_sd_lih_631g_slow():
    """Test UnrestrictedMolecularHamiltonian.integrate_sd_sd using LiH HF/6-31G orbitals.

    Integrals that correspond to restricted orbitals were used.

    """
    one_int = np.load(find_datafile("data_lih_hf_631g_oneint.npy"))
    two_int = np.load(find_datafile("data_lih_hf_631g_twoint.npy"))
    ham = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3)

    ref_ci_matrix = np.load(find_datafile("data_lih_hf_631g_cimatrix.npy"))
    ref_pspace = np.load(find_datafile("data_lih_hf_631g_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        for j, sd2 in enumerate(ref_pspace):
            sd1, sd2 = int(sd1), int(sd2)
            assert np.allclose((ham.integrate_sd_sd(sd1, sd2)), ref_ci_matrix[i, j])


def test_integrate_sd_sd_particlenum():
    """Test UnrestrictedMolecularHamiltonian.integrate_sd_sd and break particle number symmetery."""
    one_int = np.arange(1, 17, dtype=float).reshape(4, 4)
    two_int = np.arange(1, 257, dtype=float).reshape(4, 4, 4, 4)
    ham = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3)
    civec = [0b01, 0b11]

    # \braket{1 | h_{11} | 1}
    assert np.allclose((ham.integrate_sd_sd(civec[0], civec[0])), 1)
    # \braket{12 | H | 1} = 0
    assert np.allclose((ham.integrate_sd_sd(civec[1], civec[0])), 0)
    assert np.allclose((ham.integrate_sd_sd(civec[0], civec[1])), 0)
    # \braket{12 | h_{11} + h_{22} + g_{1212} - g_{1221} | 12}
    assert np.allclose((ham.integrate_sd_sd(civec[1], civec[1])), 4)

    assert np.allclose(ham.integrate_sd_sd_decomposed(civec[0], civec[1]), 0)


def test_integrate_sd_wfn():
    """Test UnrestrictedMolecularHamiltonian.integrate_sd_wfn."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test_ham = disable_abstract(
        UnrestrictedMolecularHamiltonian, {"integrate_sd_wfn": BaseHamiltonian.integrate_sd_wfn}
    )([one_int] * 2, [two_int] * 3)
    test_wfn = type(
        "Temporary wavefunction.",
        (object,),
        {"get_overlap": lambda sd, deriv=None: 1 if sd == 0b0101 else 2 if sd == 0b1010 else 3 if sd == 0b1100 else 0},
    )

    one_energy, coulomb, exchange = test_ham.integrate_sd_wfn(0b0101, test_wfn, components=True)
    assert one_energy == 1 * 1 + 1 * 1
    assert coulomb == 1 * 5 + 2 * 8
    assert exchange == 0

    one_energy, coulomb, exchange = test_ham.integrate_sd_wfn(0b1010, test_wfn, components=True)
    assert one_energy == 2 * 4 + 2 * 4
    assert coulomb == 1 * 17 + 2 * 20
    assert exchange == 0

    one_energy, coulomb, exchange = test_ham.integrate_sd_wfn(0b0110, test_wfn, components=True)
    assert one_energy == 1 * 3 + 2 * 2
    # NOTE: results are different from the restricted results b/c integrals are not symmetric
    assert coulomb == 1 * 13 + 2 * 16
    assert exchange == 0

    one_energy, coulomb, exchange = test_ham.integrate_sd_wfn(0b1100, test_wfn, components=True)
    assert one_energy == 1 * 3 + 3 * 4
    assert coulomb == 3 * 10
    assert exchange == -3 * 11

    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn("1", test_wfn)
    with pytest.raises(ValueError):
        test_ham.integrate_sd_wfn(0b0101, test_wfn, wfn_deriv=np.array([0]), ham_deriv=np.array([0]))


def test_param_ind_to_rowcol_ind():
    """Test UnrestrictedMolecularHamiltonian.param_ind_to_rowcol_ind."""
    for n in range(1, 20):
        ham = UnrestrictedMolecularHamiltonian([np.random.rand(n, n)] * 2, [np.random.rand(n, n, n, n)] * 3)
        for row_ind in range(n):
            for col_ind in range(row_ind + 1, n):
                param_ind = row_ind * n - row_ind * (row_ind + 1) / 2 + col_ind - row_ind - 1
                assert ham.param_ind_to_rowcol_ind(param_ind) == (0, row_ind, col_ind)
                assert ham.param_ind_to_rowcol_ind(param_ind + ham.nparams // 2) == (
                    1,
                    row_ind,
                    col_ind,
                )


def test_integrate_sd_sd_deriv():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sd_deriv."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test_ham = UnrestrictedMolecularHamiltonian([one_int] * 2, [two_int] * 3)

    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, 0.0)
    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, -1)
    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, 2)
    assert np.allclose(test_ham._integrate_sd_sd_deriv_decomposed(0b0101, 0b0001, np.array([0])), 0)
    assert np.allclose(test_ham._integrate_sd_sd_deriv(0b0101, 0b0001, np.array([0])), 0)
    assert np.allclose(test_ham._integrate_sd_sd_deriv_decomposed(0b000111, 0b111000, np.array([0])), 0)

    with pytest.raises(TypeError):
        test_ham._integrate_sd_sd_deriv(0b110001, "1", np.array([0]))
    with pytest.raises(TypeError):
        test_ham._integrate_sd_sd_deriv("1", 0b101010, np.array([0]))


def test_integrate_sd_sd_deriv_fdiff_h2_sto6g():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sd_deriv using H2/STO6G.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.load(find_datafile("data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_sto6g_twoint.npy"))
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
    one_int = np.load(find_datafile("data_h4_square_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_h4_square_hf_sto6g_twoint.npy"))

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


def test_integrate_sd_sd_deriv_fdiff_random():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sd_deriv using random integrals.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int_a = np.random.rand(3, 3)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(3, 3)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(3, 3, 3, 3)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(3, 3, 3, 3)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(3, 3, 3, 3)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    # check that the in tegrals have the appropriate symmetry
    assert np.allclose(one_int_a, one_int_a.T)
    assert np.allclose(one_int_b, one_int_b.T)
    assert np.allclose(two_int_aaaa, np.einsum("ijkl->jilk", two_int_aaaa))
    assert np.allclose(two_int_aaaa, np.einsum("ijkl->klij", two_int_aaaa))
    assert np.allclose(two_int_abab, np.einsum("ijkl->klij", two_int_abab))
    assert np.allclose(two_int_bbbb, np.einsum("ijkl->jilk", two_int_bbbb))
    assert np.allclose(two_int_bbbb, np.einsum("ijkl->klij", two_int_bbbb))

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])
    epsilon = 1e-7
    sds = sd_list(3, 6, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = UnrestrictedMolecularHamiltonian(
                    [one_int_a, one_int_b],
                    [two_int_aaaa, two_int_abab, two_int_bbbb],
                    params=addition,
                )

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2)) - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, np.array([i]))
                derivative = np.sum(derivative)
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2)) - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, np.array([i]))
                derivative = np.sum(derivative)
                assert np.allclose(finite_diff, derivative, atol=60 * epsilon)


def test_integrate_sd_sd_deriv_fdiff_random_small():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sd_deriv using random 1e system.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int_a = np.random.rand(2, 2)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(2, 2)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(2, 2, 2, 2)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(2, 2, 2, 2)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(2, 2, 2, 2)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    # check that the in tegrals have the appropriate symmetry
    assert np.allclose(one_int_a, one_int_a.T)
    assert np.allclose(one_int_b, one_int_b.T)
    assert np.allclose(two_int_aaaa, np.einsum("ijkl->jilk", two_int_aaaa))
    assert np.allclose(two_int_aaaa, np.einsum("ijkl->klij", two_int_aaaa))
    assert np.allclose(two_int_abab, np.einsum("ijkl->klij", two_int_abab))
    assert np.allclose(two_int_bbbb, np.einsum("ijkl->jilk", two_int_bbbb))
    assert np.allclose(two_int_bbbb, np.einsum("ijkl->klij", two_int_bbbb))

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])
    epsilon = 1e-8
    sds = sd_list(1, 4, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = UnrestrictedMolecularHamiltonian(
                    [one_int_a, one_int_b],
                    [two_int_aaaa, two_int_abab, two_int_bbbb],
                    params=addition,
                )

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2)) - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, np.array([i]))
                derivative = np.sum(derivative)
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)


def test_integrate_sd_sds_zero():
    """Test UnrestrictedHam._integrate_sd_sds_zero against _integrate_sd_sd_zero."""
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3])
    assert np.allclose(
        test_ham._integrate_sd_sds_zero(occ_alpha, occ_beta),
        np.array(test_ham._integrate_sd_sd_zero(occ_alpha, occ_beta)).reshape(3, 1),
    )


def test_integrate_sd_sds_one_alpha():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_one_alpha.

    Compared against UnrestrictedMolecularHamiltonian._integrate_sd_sd_one.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])

    assert np.allclose(
        test_ham._integrate_sd_sds_one_alpha(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                np.array(test_ham._integrate_sd_sd_one((i,), (j,), occ_alpha[occ_alpha != i], occ_beta))
                * slater.sign_excite(0b101101011001, [i], [j])
                for i in occ_alpha.tolist()
                for j in vir_alpha.tolist()
            ]
        ).T,
    )


def test_integrate_sd_sds_one_beta():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_one_beta.

    Compared against UnrestrictedMolecularHamiltonian._integrate_sd_sd_one.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])

    assert np.allclose(
        test_ham._integrate_sd_sds_one_beta(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                np.array(test_ham._integrate_sd_sd_one((i + 6,), (j + 6,), occ_alpha, occ_beta[occ_beta != i]))
                * slater.sign_excite(0b101101011001, [i + 6], [j + 6])
                for i in occ_beta.tolist()
                for j in vir_beta.tolist()
            ]
        ).T,
    )


def test_integrate_sd_sds_two_aa():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_two_aa.

    Compared against UnrestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    assert np.allclose(
        test_ham._integrate_sd_sds_two_aa(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                np.array(test_ham._integrate_sd_sd_two(diff1, diff2))
                * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                for diff1 in it.combinations(occ_alpha.tolist(), 2)
                for diff2 in it.combinations(vir_alpha.tolist(), 2)
            ]
        ).T[[1, 2]],
    )


def test_integrate_sd_sds_two_ab():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_two_ab.

    Compared against UnrestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    vir_beta = np.array([1, 4])

    assert np.allclose(
        test_ham._integrate_sd_sds_two_ab(occ_alpha, occ_beta, vir_alpha, vir_beta),
        np.array(
            [
                np.array(test_ham._integrate_sd_sd_two(diff1, diff2))
                * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                for diff1 in it.product(occ_alpha.tolist(), (occ_beta + 6).tolist())
                for diff2 in it.product(vir_alpha.tolist(), (vir_beta + 6).tolist())
            ]
        ).T[1],
    )


def test_integrate_sd_sds_two_bb():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_two_bb.

    Compared against UnrestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])

    assert np.allclose(
        test_ham._integrate_sd_sds_two_bb(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                np.array(test_ham._integrate_sd_sd_two(diff1, diff2))
                * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                for diff1 in it.combinations((occ_beta + 6).tolist(), 2)
                for diff2 in it.combinations((vir_beta + 6).tolist(), 2)
            ]
        ).T[[1, 2]],
    )


def test_integrate_sd_sds_deriv_zero_alpha():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_deriv_zero_alpha.

    Compared with UnrestrictedMolecularHamiltonian._integrate_sd_sd_zero.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_zero_alpha(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                [
                    test_ham._integrate_sd_sd_deriv_zero(0, x, y, occ_alpha, occ_beta)
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
            ]
        ).T,
    )


def test_integrate_sd_sds_deriv_zero_beta():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_deriv_zero_beta.

    Compared with UnrestrictedMolecularHamiltonian._integrate_sd_sd_zero.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_zero_beta(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                [
                    test_ham._integrate_sd_sd_deriv_zero(1, x, y, occ_alpha, occ_beta)
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
            ]
        ).T,
    )


def test_integrate_sd_sds_deriv_one_aa():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_deriv_one_aa.

    Compared with UnrestrictedMolecularHamiltonian._integrate_sd_sd_one.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])

    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one_aa(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one([i], [j], 0, x, y, occ_alpha[occ_alpha != i], occ_beta)
                    )
                    * slater.sign_excite(0b101101011001, [i], [j])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_alpha.tolist()
                for j in vir_alpha.tolist()
            ]
        ).T,
    )


def test_integrate_sd_sds_deriv_one_ab():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_deriv_one_ab.

    Compared with UnrestrictedMolecularHamiltonian._integrate_sd_sd_one.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one_ab(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one(
                            [i + 6], [j + 6], 0, x, y, occ_alpha, occ_beta[occ_beta != i]
                        )
                    )
                    * slater.sign_excite(0b101101011001, [i + 6], [j + 6])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_beta.tolist()
                for j in vir_beta.tolist()
            ]
        ).T[1],
    )

    occ_alpha = np.array([0])
    occ_beta = np.array([0])
    vir_beta = np.array([1, 2, 3, 4, 5])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one_ab(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one(
                            [i + 6], [j + 6], 0, x, y, occ_alpha, occ_beta[occ_beta != i]
                        )
                    )
                    * slater.sign_excite(0b000001000001, [i + 6], [j + 6])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_beta.tolist()
                for j in vir_beta.tolist()
            ]
        ).T[1],
    )


def test_integrate_sd_sds_deriv_one_ba():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_deriv_one_ba.

    Compared with UnrestrictedMolecularHamiltonian._integrate_sd_sd_one.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one_ba(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one([i], [j], 1, x, y, occ_alpha[occ_alpha != i], occ_beta)
                    )
                    * slater.sign_excite(0b101101011001, [i], [j])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_alpha.tolist()
                for j in vir_alpha.tolist()
            ]
        ).T[1],
    )

    occ_alpha = np.array([0])
    occ_beta = np.array([0])
    vir_alpha = np.array([1, 2, 3, 4, 5])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one_ba(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one([i], [j], 1, x, y, occ_alpha[occ_alpha != i], occ_beta)
                    )
                    * slater.sign_excite(0b000001000001, [i], [j])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_alpha.tolist()
                for j in vir_alpha.tolist()
            ]
        ).T[1],
    )


def test_integrate_sd_sds_deriv_one_bb():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_deriv_one_bb.

    Compared with UnrestrictedMolecularHamiltonian._integrate_sd_sd_one.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one_bb(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one(
                            [i + 6], [j + 6], 1, x, y, occ_alpha, occ_beta[occ_beta != i]
                        )
                    )
                    * slater.sign_excite(0b101101011001, [i + 6], [j + 6])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_beta.tolist()
                for j in vir_beta.tolist()
            ]
        ).T,
    )


def test_integrate_sd_sds_deriv_two_aaa():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_deriv_two_aaa.

    Compared with UnrestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two_aaa(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_two(diff1, diff2, 0, x, y))
                    * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for diff1 in it.combinations(occ_alpha.tolist(), 2)
                for diff2 in it.combinations(vir_alpha.tolist(), 2)
            ]
        ).T[[1, 2]],
    )


def test_integrate_sd_sds_deriv_two_aab():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_deriv_two_aab.

    Compared with UnrestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two_aab(occ_alpha, occ_beta, vir_alpha, vir_beta),
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_two(diff1, diff2, 0, x, y))
                    * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for diff1 in it.product(occ_alpha.tolist(), (occ_beta + 6).tolist())
                for diff2 in it.product(vir_alpha.tolist(), (vir_beta + 6).tolist())
            ]
        ).T[1],
    )


def test_integrate_sd_sds_deriv_two_bab():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_deriv_two_bab.

    Compared with UnrestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two_bab(occ_alpha, occ_beta, vir_alpha, vir_beta),
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_two(diff1, diff2, 1, x, y))
                    * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for diff1 in it.product(occ_alpha.tolist(), (occ_beta + 6).tolist())
                for diff2 in it.product(vir_alpha.tolist(), (vir_beta + 6).tolist())
            ]
        ).T[1],
    )


def test_integrate_sd_sds_deriv_two_bbb():
    """Test UnrestrictedMolecularHamiltonian._integrate_sd_sds_deriv_two_bbb.

    Compared with UnrestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int_a = np.random.rand(6, 6)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(6, 6)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(6, 6, 6, 6)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(6, 6, 6, 6)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(6, 6, 6, 6)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two_bbb(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_two(diff1, diff2, 1, x, y))
                    * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for diff1 in it.combinations((occ_beta + 6).tolist(), 2)
                for diff2 in it.combinations((vir_beta + 6).tolist(), 2)
            ]
        ).T[[1, 2]],
    )


def test_integrate_sd_wfn_compare_basehamiltonian():
    """Test UnrestrictedMolecularHamiltonian.integrate_sd_wfn with integrate_sd_wfn."""
    one_int_a = np.random.rand(4, 4)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(4, 4)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(4, 4, 4, 4)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(4, 4, 4, 4)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(4, 4, 4, 4)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    test_ham = UnrestrictedMolecularHamiltonian([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])
    test_ham2 = disable_abstract(
        UnrestrictedMolecularHamiltonian, {"integrate_sd_wfn": BaseHamiltonian.integrate_sd_wfn}
    )([one_int_a, one_int_b], [two_int_aaaa, two_int_abab, two_int_bbbb])

    for i in range(1, 4):
        wfn = CIWavefunction(i, 8)
        wfn.assign_params(np.random.rand(*wfn.params.shape))
        for occ_indices in it.combinations(range(8), i):
            assert np.allclose(
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn),
                test_ham2.integrate_sd_wfn(slater.create(0, *occ_indices), wfn),
            )
            assert np.allclose(
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, wfn_deriv=np.arange(wfn.nparams)),
                test_ham2.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, wfn_deriv=np.arange(wfn.nparams)),
            )
            assert np.allclose(
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, ham_deriv=np.arange(test_ham.nparams)),
                test_ham2.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, ham_deriv=np.arange(test_ham2.nparams)),
            )
            assert np.allclose(
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, components=True),
                test_ham2.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, components=True),
            )

    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn(0b01101010, wfn, ham_deriv=np.arange(20).tolist())
    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn(0b01101010, wfn, ham_deriv=np.arange(20).astype(float))
    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn(0b01101010, wfn, ham_deriv=np.arange(20).reshape(2, 10))
    with pytest.raises(ValueError):
        bad_indices = np.arange(20)
        bad_indices[0] = -1
        test_ham.integrate_sd_wfn(0b01101010, wfn, ham_deriv=bad_indices)
    with pytest.raises(ValueError):
        bad_indices = np.arange(20)
        bad_indices[0] = 20
        test_ham.integrate_sd_wfn(0b01101010, wfn, ham_deriv=bad_indices)
    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn("1", wfn)
    with pytest.raises(ValueError):
        test_ham.integrate_sd_wfn(0b00110011, wfn, wfn_deriv=np.array([0]), ham_deriv=np.array([0]))


def test_integrate_sd_wfn_deriv_fdiff():
    """Test UnrestrictedMolecularHamiltonian.integrate_sd_wfn with finite difference."""
    nd = pytest.importorskip("numdifftools")
    wfn = CIWavefunction(3, 6)
    wfn.assign_params(np.random.rand(*wfn.params.shape))

    one_int_a = np.random.rand(3, 3)
    one_int_a = one_int_a + one_int_a.T
    one_int_b = np.random.rand(3, 3)
    one_int_b = one_int_b + one_int_b.T

    two_int_aaaa = np.random.rand(3, 3, 3, 3)
    two_int_aaaa = np.einsum("ijkl->jilk", two_int_aaaa) + two_int_aaaa
    two_int_aaaa = np.einsum("ijkl->klij", two_int_aaaa) + two_int_aaaa
    two_int_abab = np.random.rand(3, 3, 3, 3)
    two_int_abab = np.einsum("ijkl->klij", two_int_abab) + two_int_abab
    two_int_bbbb = np.random.rand(3, 3, 3, 3)
    two_int_bbbb = np.einsum("ijkl->jilk", two_int_bbbb) + two_int_bbbb
    two_int_bbbb = np.einsum("ijkl->klij", two_int_bbbb) + two_int_bbbb

    ham = UnrestrictedMolecularHamiltonian(
        (one_int_a, one_int_b), (two_int_aaaa, two_int_abab, two_int_bbbb), update_prev_params=True
    )
    original = np.random.rand(ham.params.size)
    step1 = np.random.rand(ham.params.size)
    step2 = np.random.rand(ham.params.size)
    ham.assign_params(original.copy())
    ham.assign_params(original + step1)
    ham.assign_params(original + step1 + step2)

    nhalf = ham.nparams // 2
    um_orig = [unitary_matrix(original[:nhalf]), unitary_matrix(original[nhalf:])]
    um_step1 = [unitary_matrix(step1[:nhalf]), unitary_matrix(step1[nhalf:])]
    um_step2 = [unitary_matrix(step2[:nhalf]), unitary_matrix(step2[nhalf:])]

    temp_ham = UnrestrictedMolecularHamiltonian(
        (one_int_a, one_int_b), (two_int_aaaa, two_int_abab, two_int_bbbb), update_prev_params=True
    )
    temp_ham.orb_rotate_matrix(
        [um_orig[0].dot(um_step1[0]).dot(um_step2[0]), um_orig[1].dot(um_step1[1]).dot(um_step2[1])]
    )
    assert np.allclose(ham.one_int, temp_ham.one_int)
    assert np.allclose(ham.two_int, temp_ham.two_int)

    def objective(params):
        temp_ham = UnrestrictedMolecularHamiltonian(
            (one_int_a, one_int_b),
            (two_int_aaaa, two_int_abab, two_int_bbbb),
            update_prev_params=True,
        )
        temp_ham.orb_rotate_matrix(
            [
                um_orig[0].dot(um_step1[0]).dot(um_step2[0]),
                um_orig[1].dot(um_step1[1]).dot(um_step2[1]),
            ]
        )
        temp_ham.set_ref_ints()
        temp_ham._prev_params = ham.params.copy()
        temp_ham.assign_params(params.copy())
        return temp_ham.integrate_sd_wfn(wfn.sds[0], wfn)

    assert np.allclose(
        nd.Gradient(objective)(ham.params),
        ham.integrate_sd_wfn(wfn.sds[0], wfn, ham_deriv=np.arange(ham.nparams)),
    )

    wfn = LinearCombinationWavefunction(3, 6, [CIWavefunction(3, 6), CIWavefunction(3, 6)])
    wfn.assign_params(np.random.rand(wfn.nparams))
    wfn.wfns[0].assign_params(np.random.rand(wfn.wfns[0].nparams))
    wfn.wfns[1].assign_params(np.random.rand(wfn.wfns[1].nparams))

    def objective(params):
        temp_wfn = LinearCombinationWavefunction(3, 6, [CIWavefunction(3, 6), CIWavefunction(3, 6)])
        temp_wfn.assign_params(wfn.params.copy())
        temp_wfn.wfns[0].assign_params(params.copy())
        temp_wfn.wfns[1].assign_params(wfn.wfns[1].params.copy())
        return ham.integrate_sd_wfn(0b001011, temp_wfn)

    assert np.allclose(
        nd.Gradient(objective)(wfn.wfns[0].params),
        ham.integrate_sd_wfn(0b001011, wfn, wfn_deriv=(wfn.wfns[0], np.arange(wfn.wfns[0].nparams))),
    )


def test_unrestrictedmolecularhamiltonian_save_params(tmp_path):
    """Test UnrestrictedMolecularHamiltonian.sav_params."""
    ham = UnrestrictedMolecularHamiltonian(
        [np.arange(4, dtype=float).reshape(2, 2)] * 2,
        [np.arange(16, dtype=float).reshape(2, 2, 2, 2)] * 3,
        update_prev_params=True,
    )
    ham.assign_params(np.random.rand(ham.nparams))
    ham.assign_params(np.random.rand(ham.nparams))
    ham.save_params(str(tmp_path / "temp.npy"))
    assert np.allclose(np.load(str(tmp_path / "temp.npy")), ham.params)
    assert np.allclose(np.load(str(tmp_path / "temp_um.npy")), [ham._prev_unitary_alpha, ham._prev_unitary_beta])
