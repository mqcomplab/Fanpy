"""Test fanpy.ham.restricted_chemical."""

import itertools as it

from fanpy.ham.base import BaseHamiltonian
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.tools import slater
from fanpy.tools.math_tools import unitary_matrix
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.composite.lincomb import LinearCombinationWavefunction

import numpy as np

import pytest

from utils import disable_abstract, find_datafile


def test_nspin():
    """Test BaseGeneralizedHamiltonian.nspin."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = RestrictedMolecularHamiltonian(one_int, two_int)
    assert test.nspin == 4


def test_integrate_sd_sd_decomposed_trivial():
    """Test RestrictedMolecularHamiltonian.integrate_sd_sd for trivial cases."""
    one_int = np.random.rand(3, 3)
    two_int = np.random.rand(3, 3, 3, 3)
    test = RestrictedMolecularHamiltonian(one_int, two_int)

    assert np.allclose((0, 0, 0), test.integrate_sd_sd_decomposed(0b000111, 0b001001))
    assert np.allclose((0, 0, 0), test.integrate_sd_sd_decomposed(0b000111, 0b111000))
    assert np.allclose((0, two_int[0, 1, 1, 0], 0), test.integrate_sd_sd_decomposed(0b110001, 0b101010))
    assert np.allclose(
        (
            0,
            -two_int[1, 1, 1, 0] - two_int[0, 1, 1, 1] + two_int[0, 0, 1, 0] + two_int[0, 1, 0, 0],
            0,
        ),
        test.integrate_sd_sd_decomposed(0b110001, 0b101010, deriv=np.array([0])).ravel(),
    )

    with pytest.raises(TypeError):
        test.integrate_sd_sd_decomposed(0b110001, "1")
    with pytest.raises(TypeError):
        test.integrate_sd_sd_decomposed("1", 0b101010)


def test_integrate_sd_sd_particlenum():
    """Test RestrictedMolecularHamiltonian.integrate_sd_sd and break particle number symmetery."""
    one_int = np.arange(1, 17, dtype=float).reshape(4, 4)
    two_int = np.arange(1, 257, dtype=float).reshape(4, 4, 4, 4)
    ham = RestrictedMolecularHamiltonian(one_int, two_int)
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
    """Test RestrictedMolecularHamiltonian.integrate_sd_wfn."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test_ham = disable_abstract(RestrictedMolecularHamiltonian, {"integrate_sd_wfn": BaseHamiltonian.integrate_sd_wfn})(
        one_int, two_int
    )
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
    assert coulomb == 1 * 9 + 2 * 16
    assert exchange == 0

    one_energy, coulomb, exchange = test_ham.integrate_sd_wfn(0b1100, test_wfn, components=True)
    assert one_energy == 1 * 3 + 3 * 4
    assert coulomb == 3 * 10
    assert exchange == -3 * 11


def test_param_ind_to_rowcol_ind():
    """Test RestrictedMolecularHamiltonian.param_ind_to_rowcol_ind."""
    for n in range(1, 20):
        ham = RestrictedMolecularHamiltonian(np.random.rand(n, n), np.random.rand(n, n, n, n))
        for row_ind in range(n):
            for col_ind in range(row_ind + 1, n):
                param_ind = row_ind * n - row_ind * (row_ind + 1) / 2 + col_ind - row_ind - 1
                assert ham.param_ind_to_rowcol_ind(param_ind) == (row_ind, col_ind)


def test_integrate_sd_sd_deriv():
    """Test RestrictedMolecularHamiltonian._integrate_sd_sd_deriv."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, 0.0)
    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, -1)
    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, 2)
    assert np.allclose(test_ham._integrate_sd_sd_deriv_decomposed(0b0101, 0b0001, np.array([0])), 0)
    assert np.allclose(test_ham._integrate_sd_sd_deriv(0b0101, 0b0001, np.array([0])), 0)

    with pytest.raises(TypeError):
        test_ham._integrate_sd_sd_deriv(0b110001, "1", np.array([0]))
    with pytest.raises(TypeError):
        test_ham._integrate_sd_sd_deriv("1", 0b101010, np.array([0]))


def test_integrate_sd_sd_deriv_fdiff_random():
    """Test RestrictedMolecularHamiltonian._integrate_sd_sd_deriv using random integrals.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.random.rand(3, 3)
    one_int = one_int + one_int.T

    two_int = np.random.rand(3, 3, 3, 3)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    # check that the in tegrals have the appropriate symmetry
    assert np.allclose(one_int, one_int.T)
    assert np.allclose(two_int, np.einsum("ijkl->jilk", two_int))
    assert np.allclose(two_int, np.einsum("ijkl->klij", two_int))

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    epsilon = 1e-8
    sds = sd_list(3, 6, num_limit=None, exc_orders=None)

    for sd1 in sds:
        for sd2 in sds:
            for i in range(test_ham.nparams):
                addition = np.zeros(test_ham.nparams)
                addition[i] = epsilon
                test_ham2 = RestrictedMolecularHamiltonian(one_int, two_int, params=addition)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd_decomposed(sd1, sd2))
                    - np.array(test_ham.integrate_sd_sd_decomposed(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv_decomposed(sd1, sd2, np.array([i])).ravel()
                assert np.allclose(finite_diff, derivative, atol=20 * epsilon)

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2)) - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, np.array([i]))
                derivative = np.sum(derivative)
                assert np.allclose(finite_diff, derivative, atol=60 * epsilon)


def test_integrate_sd_sd_deriv_fdiff_random_small():
    """Test GeneralizedMolecularHamiltonian._integrate_sd_sd_deriv using random 1e system.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.random.rand(2, 2)
    one_int = one_int + one_int.T
    two_int = np.random.rand(2, 2, 2, 2)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    # check that the integrals have the appropriate symmetry
    assert np.allclose(one_int, one_int.T)
    assert np.allclose(two_int, np.einsum("ijkl->jilk", two_int))
    assert np.allclose(two_int, np.einsum("ijkl->klij", two_int))

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    epsilon = 1e-8
    sds = sd_list(1, 4, num_limit=None, exc_orders=None)

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


def test_integrate_sd_sds_zero():
    """Test RestrictedMolecularHamiltonian._integrate_sd_sds_zero against _integrate_sd_sd_zero."""
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3])
    assert np.allclose(
        test_ham._integrate_sd_sds_zero(occ_alpha, occ_beta),
        np.array(test_ham._integrate_sd_sd_zero(occ_alpha, occ_beta)).reshape(3, 1),
    )


def test_integrate_sd_sds_one_alpha():
    """Test RestrictedMolecularHamiltonian._integrate_sd_sds_one_alpha.

    Compared against RestrictedMolecularHamiltonian._integrate_sd_sd_one.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

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
    """Test RestrictedMolecularHamiltonian._integrate_sd_sds_one_beta.

    Compared against RestrictedMolecularHamiltonian._integrate_sd_sd_one.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

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
    """Test RestrictedMolecularHamiltonian._integrate_sd_sds_two_aa.

    Compared against RestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

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
    """Test RestrictedMolecularHamiltonian._integrate_sd_sds_two_ab.

    Compared against RestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

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
    """Test RestrictedMolecularHamiltonian._integrate_sd_sds_two_bb.

    Compared against RestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

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


def test_integrate_sd_sds_deriv_zero():
    """Test RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_zero_alpha and _beta.

    Compared with RestrictedMolecularHamiltonian._integrate_sd_sd_zero.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_zero_alpha(occ_alpha, occ_beta, vir_alpha)
        + test_ham._integrate_sd_sds_deriv_zero_beta(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                [
                    test_ham._integrate_sd_sd_deriv_zero(x, y, occ_alpha, occ_beta)
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
            ]
        ).T,
    )


def test_integrate_sd_sds_deriv_one_a():
    """Test RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_one_aa and _ba.

    Compared with RestrictedMolecularHamiltonian._integrate_sd_sd_one.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])

    results = test_ham._integrate_sd_sds_deriv_one_aa(occ_alpha, occ_beta, vir_alpha)
    results[1] += test_ham._integrate_sd_sds_deriv_one_ba(occ_alpha, occ_beta, vir_alpha)
    assert np.allclose(
        results,
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_one([i], [j], x, y, occ_alpha[occ_alpha != i], occ_beta))
                    * slater.sign_excite(0b101101011001, [i], [j])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_alpha.tolist()
                for j in vir_alpha.tolist()
            ]
        ).T,
    )

    occ_alpha = np.array([0])
    occ_beta = np.array([0])
    vir_alpha = np.array([1, 2, 3, 4, 5])
    results = test_ham._integrate_sd_sds_deriv_one_aa(occ_alpha, occ_beta, vir_alpha)[1]
    results += test_ham._integrate_sd_sds_deriv_one_ba(occ_alpha, occ_beta, vir_alpha)
    assert np.allclose(
        results,
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_one([i], [j], x, y, occ_alpha[occ_alpha != i], occ_beta))
                    * slater.sign_excite(0b000001000001, [i], [j])
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for i in occ_alpha.tolist()
                for j in vir_alpha.tolist()
            ]
        ).T[1],
    )


def test_integrate_sd_sds_deriv_one_b():
    """Test RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_one_bb and _ab.

    Compared with RestrictedMolecularHamiltonian._integrate_sd_sd_one.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])

    results = test_ham._integrate_sd_sds_deriv_one_bb(occ_alpha, occ_beta, vir_beta)
    results[1] += test_ham._integrate_sd_sds_deriv_one_ab(occ_alpha, occ_beta, vir_beta)
    assert np.allclose(
        results,
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one([i + 6], [j + 6], x, y, occ_alpha, occ_beta[occ_beta != i])
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

    occ_alpha = np.array([0])
    occ_beta = np.array([0])
    vir_beta = np.array([1, 2, 3, 4, 5])
    results = test_ham._integrate_sd_sds_deriv_one_aa(occ_alpha, occ_beta, vir_beta)[1]
    results += test_ham._integrate_sd_sds_deriv_one_ba(occ_alpha, occ_beta, vir_beta)
    assert np.allclose(
        results,
        np.array(
            [
                [
                    np.array(
                        test_ham._integrate_sd_sd_deriv_one([i + 6], [j + 6], x, y, occ_alpha[occ_alpha != i], occ_beta)
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


def test_integrate_sd_sds_deriv_two_aaa():
    """Test RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_two_aa.

    Compared with RestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])

    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two_aaa(occ_alpha, occ_beta, vir_alpha),
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_two(diff1, diff2, x, y))
                    * slater.sign_excite(0b101101011001, diff1, reversed(diff2))
                    for x in range(5)
                    for y in range(x + 1, 6)
                ]
                for diff1 in it.combinations(occ_alpha.tolist(), 2)
                for diff2 in it.combinations(vir_alpha.tolist(), 2)
            ]
        ).T[[1, 2]],
    )


def test_integrate_sd_sds_deriv_two_ab():
    """Test RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_two_aab and _bab.

    Compared with RestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_alpha = np.array([1, 2, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two_aab(occ_alpha, occ_beta, vir_alpha, vir_beta)
        + test_ham._integrate_sd_sds_deriv_two_bab(occ_alpha, occ_beta, vir_alpha, vir_beta),
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_two(diff1, diff2, x, y))
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
    """Test RestrictedMolecularHamiltonian._integrate_sd_sds_deriv_two_bbb.

    Compared with RestrictedMolecularHamiltonian._integrate_sd_sd_two.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)

    occ_alpha = np.array([0, 3, 4])
    occ_beta = np.array([0, 2, 3, 5])
    vir_beta = np.array([1, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two_bbb(occ_alpha, occ_beta, vir_beta),
        np.array(
            [
                [
                    np.array(test_ham._integrate_sd_sd_deriv_two(diff1, diff2, x, y))
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
    """Test RestrictedMolecularHamiltonian.integrate_sd_wfn by comparing with BaseHamiltonian."""
    one_int = np.random.rand(4, 4)
    one_int = one_int + one_int.T

    two_int = np.random.rand(4, 4, 4, 4)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    test_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    test_ham2 = disable_abstract(
        RestrictedMolecularHamiltonian, {"integrate_sd_wfn": BaseHamiltonian.integrate_sd_wfn}
    )(one_int, two_int)

    for i in range(1, 4):
        wfn = CIWavefunction(i, 8)
        wfn.assign_params(np.random.rand(*wfn.params.shape))
        for occ_indices in it.combinations(range(8), i):
            assert np.allclose(
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, wfn_deriv=None),
                test_ham2.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, wfn_deriv=None),
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
        test_ham.integrate_sd_wfn(0b01101010, wfn, ham_deriv=np.arange(10).tolist())
    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn(0b01101010, wfn, ham_deriv=np.arange(10).astype(float))
    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn(0b01101010, wfn, ham_deriv=np.arange(10).reshape(2, 5))
    with pytest.raises(ValueError):
        bad_indices = np.arange(10)
        bad_indices[0] = -1
        test_ham.integrate_sd_wfn(0b01101010, wfn, ham_deriv=bad_indices)
    with pytest.raises(ValueError):
        bad_indices = np.arange(10)
        bad_indices[0] = 10
        test_ham.integrate_sd_wfn(0b01101010, wfn, ham_deriv=bad_indices)

    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn("1", wfn)
    with pytest.raises(ValueError):
        test_ham.integrate_sd_wfn(0b00110011, wfn, wfn_deriv=np.array([0]), ham_deriv=np.array([0]))


def test_integrate_sd_wfn_deriv_fdiff():
    """Test RestrictedMolecularHamiltonian.integrate_sd_wfn_deriv with finite difference."""
    nd = pytest.importorskip("numdifftools")
    wfn = CIWavefunction(3, 6)
    wfn.assign_params(np.random.rand(*wfn.params.shape))

    one_int = np.random.rand(3, 3)
    one_int = one_int + one_int.T

    two_int = np.random.rand(3, 3, 3, 3)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    ham = RestrictedMolecularHamiltonian(one_int, two_int, update_prev_params=True)
    original = np.random.rand(ham.params.size)
    step1 = np.random.rand(ham.params.size)
    step2 = np.random.rand(ham.params.size)
    ham.assign_params(original.copy())
    ham.assign_params(original + step1)
    ham.assign_params(original + step1 + step2)

    temp_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    temp_ham.orb_rotate_matrix(unitary_matrix(original).dot(unitary_matrix(step1)).dot(unitary_matrix(step2)))
    assert np.allclose(ham.one_int, temp_ham.one_int)
    assert np.allclose(ham.two_int, temp_ham.two_int)

    def objective(params):
        temp_ham = RestrictedMolecularHamiltonian(one_int, two_int)
        temp_ham.orb_rotate_matrix(unitary_matrix(original).dot(unitary_matrix(step1)).dot(unitary_matrix(step2)))
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


def test_restrictedmolecularhamiltonian_save_params(tmp_path):
    """Test RestrictedMolecularHamiltonian.sav_params."""
    ham = RestrictedMolecularHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2),
        np.arange(16, dtype=float).reshape(2, 2, 2, 2),
        update_prev_params=True,
    )
    ham.assign_params(np.random.rand(ham.nparams))
    ham.assign_params(np.random.rand(ham.nparams))
    ham.save_params(str(tmp_path / "temp.npy"))
    assert np.allclose(np.load(str(tmp_path / "temp.npy")), ham.params)
    assert np.allclose(np.load(str(tmp_path / "temp_um.npy")), ham._prev_unitary)
