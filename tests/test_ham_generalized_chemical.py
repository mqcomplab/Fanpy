"""Test fanpy.ham.generalized_chemical."""

import itertools as it

from fanpy.ham.base import BaseHamiltonian
from fanpy.ham.generalized_chemical import GeneralizedMolecularHamiltonian
from fanpy.tools import slater
from fanpy.tools.math_tools import unitary_matrix
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.composite.lincomb import LinearCombinationWavefunction

import numpy as np

import pytest

from utils import disable_abstract, find_datafile


def test_init():
    """Test GeneralizedMolecularHamiltonian.__init__."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = GeneralizedMolecularHamiltonian(one_int, two_int)
    assert np.allclose(test.params, np.zeros(6))
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)


def test_set_ref_ints():
    """Test GeneralizedMolecularHamiltonian.set_ref_ints."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = GeneralizedMolecularHamiltonian(one_int, two_int)
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    new_one_int = np.random.rand(2, 2)
    new_two_int = np.random.rand(2, 2, 2, 2)
    test.assign_integrals(new_one_int, new_two_int)
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    test.set_ref_ints()
    assert np.allclose(test._ref_one_int, new_one_int)
    assert np.allclose(test._ref_two_int, new_two_int)


def test_cache_two_ints():
    """Test GeneralizedMolecularHamiltonian.cache_two_ints."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    two_int_ijij = np.array([[5, 10], [15, 20]])
    two_int_ijji = np.array([[5, 11], [14, 20]])

    test = GeneralizedMolecularHamiltonian(one_int, two_int)
    assert np.allclose(test._cached_two_int_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_ijji, two_int_ijji)

    test.two_int = np.arange(21, 37).reshape(2, 2, 2, 2)
    new_two_int_ijij = np.array([[21, 26], [31, 36]])
    new_two_int_ijji = np.array([[21, 27], [30, 36]])
    assert np.allclose(test._cached_two_int_ijij, two_int_ijij)
    assert np.allclose(test._cached_two_int_ijji, two_int_ijji)

    test.cache_two_ints()
    assert np.allclose(test._cached_two_int_ijij, new_two_int_ijij)
    assert np.allclose(test._cached_two_int_ijji, new_two_int_ijji)


def test_assign_params():
    """Test GeneralizedMolecularHamiltonian.assign_params."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)

    test = GeneralizedMolecularHamiltonian(one_int, two_int)
    with pytest.raises(ValueError):
        test.assign_params([0])
    with pytest.raises(ValueError):
        test.assign_params(np.array([[0]]))
    with pytest.raises(ValueError):
        test.assign_params(np.array([0, 1]))

    test.assign_params(np.array([0]))
    assert np.allclose(test.params, np.zeros(1))
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    test.assign_params(np.array([10]))
    assert np.allclose(test.params, 10)
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    test.assign_params(np.array([5]))
    assert np.allclose(test.params, 5)
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)

    # make sure that transformation is independent of the transformations that came before it
    test.assign_params(np.array([10]))
    assert np.allclose(test.params, 10)
    assert np.allclose(test._ref_one_int, one_int)
    assert np.allclose(test._ref_two_int, two_int)


def test_integrate_sd_wfn():
    """Test GeneralizedMolecularHamiltonian.integrate_sd_wfn.

    Integrals that correspond to restricted orbitals were used.

    """
    restricted_one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    restricted_two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    one_int = np.zeros((4, 4))
    one_int[:2, :2] = restricted_one_int
    one_int[2:, 2:] = restricted_one_int
    two_int = np.zeros((4, 4, 4, 4))
    two_int[:2, :2, :2, :2] = restricted_two_int
    two_int[:2, 2:, :2, 2:] = restricted_two_int
    two_int[2:, :2, 2:, :2] = restricted_two_int
    two_int[2:, 2:, 2:, 2:] = restricted_two_int

    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)
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

    with pytest.raises(ValueError):
        test_ham.integrate_sd_wfn(0b0101, test_wfn, wfn_deriv=0, ham_deriv=0)
    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn(0b01101010, test_wfn, ham_deriv=np.arange(28).tolist())
    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn(0b01101010, test_wfn, ham_deriv=np.arange(28).astype(float))
    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn(0b01101010, test_wfn, ham_deriv=np.arange(28).reshape(2, 14))
    with pytest.raises(ValueError):
        bad_indices = np.arange(28)
        bad_indices[0] = -1
        test_ham.integrate_sd_wfn(0b01101010, test_wfn, ham_deriv=bad_indices)
    with pytest.raises(ValueError):
        bad_indices = np.arange(28)
        bad_indices[0] = 28
        test_ham.integrate_sd_wfn(0b01101010, test_wfn, ham_deriv=bad_indices)

    with pytest.raises(TypeError):
        test_ham.integrate_sd_wfn("1", test_wfn)


def test_integrate_sd_sd_decomposed_trivial():
    """Test GeneralizedMolecularHamiltonian.integrate_sd_sd for trivial cases."""
    one_int = np.random.rand(6, 6)
    two_int = np.random.rand(6, 6, 6, 6)
    test = GeneralizedMolecularHamiltonian(one_int, two_int)

    assert np.allclose((0, 0, 0), test.integrate_sd_sd_decomposed(0b000111, 0b001001))
    assert np.allclose((0, 0, 0), test.integrate_sd_sd_decomposed(0b000111, 0b111000))
    assert np.allclose(
        (0, two_int[0, 1, 2, 3], -two_int[0, 1, 3, 2]),
        test.integrate_sd_sd_decomposed(0b100011, 0b101100),
    )
    assert two_int[0, 1, 2, 3] - two_int[0, 1, 3, 2] == test.integrate_sd_sd(0b100011, 0b101100)
    assert np.allclose((one_int[0, 0], 0, 0), test.integrate_sd_sd_decomposed(0b1, 0b1))
    assert np.allclose((one_int[0, 1], 0, 0), test.integrate_sd_sd_decomposed(0b1, 0b10))
    assert np.allclose(
        (0, -two_int[1, 4, 1, 3] + two_int[0, 4, 0, 3], two_int[1, 4, 3, 1] - two_int[0, 4, 3, 0]),
        test.integrate_sd_sd_decomposed(0b110001, 0b101010, deriv=np.array([0])).ravel(),
    )

    with pytest.raises(TypeError):
        test.integrate_sd_sd(0b110001, "1")
    with pytest.raises(TypeError):
        test.integrate_sd_sd("1", 0b101010)




def test_integrate_sd_sd_particlenum():
    """Test GeneralizedMolecularHamiltonian.integrate_sd_sd and break particle number symmetery."""
    restricted_one_int = np.arange(1, 17, dtype=float).reshape(4, 4)
    restricted_two_int = np.arange(1, 257, dtype=float).reshape(4, 4, 4, 4)
    one_int = np.zeros((8, 8))
    one_int[:4, :4] = restricted_one_int
    one_int[4:, 4:] = restricted_one_int
    two_int = np.zeros((8, 8, 8, 8))
    two_int[:4, :4, :4, :4] = restricted_two_int
    two_int[:4, 4:, :4, 4:] = restricted_two_int
    two_int[4:, :4, 4:, :4] = restricted_two_int
    two_int[4:, 4:, 4:, 4:] = restricted_two_int

    ham = GeneralizedMolecularHamiltonian(one_int, two_int)
    civec = [0b01, 0b11]

    # \braket{1 | h_{11} | 1}
    assert np.allclose(ham.integrate_sd_sd(civec[0], civec[0]), 1)
    # \braket{12 | H | 1} = 0
    assert np.allclose(ham.integrate_sd_sd(civec[1], civec[0]), 0)
    assert np.allclose(ham.integrate_sd_sd(civec[0], civec[1]), 0)
    # \braket{12 | h_{11} + h_{22} + g_{1212} - g_{1221} | 12}
    assert np.allclose(ham.integrate_sd_sd(civec[1], civec[1]), 4)

    assert np.allclose(ham.integrate_sd_sd_decomposed(civec[0], civec[1]), 0)


def test_param_ind_to_rowcol_ind():
    """Test GeneralizedMolecularHamiltonian.param_ind_to_rowcol_ind."""
    for n in range(1, 40):
        ham = GeneralizedMolecularHamiltonian(np.random.rand(n, n), np.random.rand(n, n, n, n))
        for row_ind in range(n):
            for col_ind in range(row_ind + 1, n):
                param_ind = row_ind * n - row_ind * (row_ind + 1) / 2 + col_ind - row_ind - 1
                assert ham.param_ind_to_rowcol_ind(param_ind) == (row_ind, col_ind)


def test_integrate_sd_sd_deriv():
    """Test GeneralizedMolecularHamiltonian._integrate_sd_sd_deriv for trivial cases."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)

    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, 0.0)
    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, -1)
    with pytest.raises(ValueError):
        test_ham._integrate_sd_sd_deriv(0b0101, 0b0101, 2)
    assert test_ham._integrate_sd_sd_deriv(0b0101, 0b0001, np.array([0])) == 0
    assert np.allclose(test_ham._integrate_sd_sd_deriv_decomposed(0b0101, 0b0001, np.array([0])), 0)
    assert test_ham._integrate_sd_sd_deriv(0b000111, 0b111000, np.array([0])) == 0

    with pytest.raises(TypeError):
        test_ham._integrate_sd_sd_deriv(0b110001, "1", np.array([0]))
    with pytest.raises(TypeError):
        test_ham._integrate_sd_sd_deriv("1", 0b101010, np.array([0]))
        

def test_integrate_sd_sd_deriv_fdiff_random():
    """Test GeneralizedMolecularHamiltonian._integrate_sd_sd_deriv using random integrals.

    Computed derivatives are compared against finite difference of the `integrate_sd_sd`.

    """
    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T
    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    # check that the integrals have the appropriate symmetry
    assert np.allclose(one_int, one_int.T)
    assert np.allclose(two_int, np.einsum("ijkl->jilk", two_int))
    assert np.allclose(two_int, np.einsum("ijkl->klij", two_int))

    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)
    epsilon = 1e-8
    sds = sd_list(3, 6, num_limit=None, exc_orders=None)

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

                finite_diff = (
                    np.array(test_ham2.integrate_sd_sd(sd1, sd2)) - np.array(test_ham.integrate_sd_sd(sd1, sd2))
                ) / epsilon
                derivative = test_ham._integrate_sd_sd_deriv(sd1, sd2, np.array([i])).ravel()
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

    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)
    epsilon = 1e-8
    sds = sd_list(1, 2, num_limit=None, exc_orders=None)

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


def test_integrate_sd_sds_zero():
    """Test GeneralizedMolecularHamiltonian._integrate_sd_sds_zero against _integrate_sd_sd_zero."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)

    occ_indices = np.array([0, 1, 3, 4])
    assert np.allclose(
        test_ham._integrate_sd_sds_zero(occ_indices),
        np.array(test_ham._integrate_sd_sd_zero(occ_indices)).reshape(3, 1),
    )


def test_integrate_sd_sds_one():
    """Test GeneralizedMolecularHamiltonian._integrate_sd_sds_one against _integrate_sd_sd_one."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)

    occ_indices = np.array([0, 1, 3, 4])
    vir_indices = np.array([2, 5, 6, 7])
    assert np.allclose(
        test_ham._integrate_sd_sds_one(occ_indices, vir_indices),
        np.array(
            [
                np.array(test_ham._integrate_sd_sd_one((i,), (j,), occ_indices[occ_indices != i]))
                * slater.sign_excite(0b11011, [i], [j])
                for i in occ_indices.tolist()
                for j in vir_indices.tolist()
            ]
        ).T,
    )


def test_integrate_sd_sds_two():
    """Test GeneralizedMolecularHamiltonian._integrate_sd_sds_two against _integrate_sd_sd_two."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)

    occ_indices = np.array([0, 1, 3, 4])
    vir_indices = np.array([2, 5, 6, 7])
    assert np.allclose(
        test_ham._integrate_sd_sds_two(occ_indices, vir_indices),
        np.array(
            [
                np.array(test_ham._integrate_sd_sd_two(diff1, diff2))
                * slater.sign_excite(0b11011, diff1, reversed(diff2))
                for diff1 in it.combinations(occ_indices.tolist(), 2)
                for diff2 in it.combinations(vir_indices.tolist(), 2)
            ]
        ).T,
    )


def test_integrate_sd_sds_deriv_zero():
    """Test GeneralizedMolecularHamiltonian._integrate_sd_sds_deriv_zero."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)

    occ_indices = np.array([0, 1, 3, 4])
    vir_indices = np.array([2, 5, 6, 7])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_zero(occ_indices, vir_indices),
        np.array(
            [[test_ham._integrate_sd_sd_deriv_zero(i, j, occ_indices) for i in range(7) for j in range(i + 1, 8)]]
        ).T,
    )


def test_integrate_sd_sds_deriv_one():
    """Test GeneralizedMolecularHamiltonian._integrate_sd_sds_deriv_one."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)

    occ_indices = np.array([0, 1, 3, 4])
    vir_indices = np.array([2, 5, 6, 7])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_one(occ_indices, vir_indices),
        np.transpose(
            np.array(
                [
                    [
                        np.array(test_ham._integrate_sd_sd_deriv_one((i,), (j,), x, y, occ_indices[occ_indices != i]))
                        * slater.sign_excite(0b11011, [i], [j])
                        for x in range(8)
                        for y in range(x + 1, 8)
                    ]
                    for i in occ_indices.tolist()
                    for j in vir_indices.tolist()
                ]
            )
        ),
    )


def test_integrate_sd_sds_deriv_two():
    """Test GeneralizedMolecularHamiltonian._integrate_sd_sds_deriv_two."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)

    occ_indices = np.array([0, 1, 3, 4])
    vir_indices = np.array([2, 5, 6, 7])
    assert np.allclose(
        test_ham._integrate_sd_sds_deriv_two(occ_indices, vir_indices),
        np.transpose(
            np.array(
                [
                    [
                        np.array(test_ham._integrate_sd_sd_deriv_two(occ, vir, x, y))
                        * slater.sign_excite(0b11011, occ, reversed(vir))
                        for x in (range(8))
                        for y in range(x + 1, 8)
                    ]
                    for occ in it.combinations(occ_indices.tolist(), 2)
                    for vir in it.combinations(vir_indices.tolist(), 2)
                ]
            )
        ),
    )


def test_integrate_sd_wfn_compare_basehamiltonian():
    """Test GeneralizedMolecularHamiltonian.integrate_sd_wfn by comparing with BaseHamiltonian."""
    one_int = np.random.rand(8, 8)
    one_int = one_int + one_int.T
    two_int = np.random.rand(8, 8, 8, 8)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    test_ham = GeneralizedMolecularHamiltonian(one_int, two_int)
    test_ham2 = disable_abstract(
        GeneralizedMolecularHamiltonian, {"integrate_sd_wfn": BaseHamiltonian.integrate_sd_wfn}
    )(one_int, two_int)

    for n in range(1, 8):
        wfn = CIWavefunction(n, 8)
        wfn.assign_params(np.random.rand(*wfn.params.shape))
        for occ_indices in it.combinations(range(8), n):
            assert np.allclose(
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn),
                test_ham2.integrate_sd_wfn(slater.create(0, *occ_indices), wfn),
            )
            assert np.allclose(
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, wfn_deriv=np.arange(wfn.nparams)),
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, wfn_deriv=np.arange(wfn.nparams)),
            )
            assert np.allclose(
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, ham_deriv=np.arange(test_ham.nparams)),
                test_ham.integrate_sd_wfn(slater.create(0, *occ_indices), wfn, ham_deriv=np.arange(test_ham2.nparams)),
            )


def test_integrate_sd_wfn_deriv_fdiff():
    """Test GeneralizedMolecularHamiltonian.integrate_sd_wfn_deriv with finite difference."""
    nd = pytest.importorskip("numdifftools")
    wfn = CIWavefunction(5, 6)
    wfn.assign_params(np.random.rand(*wfn.params.shape))

    one_int = np.random.rand(6, 6)
    one_int = one_int + one_int.T

    two_int = np.random.rand(6, 6, 6, 6)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    ham = GeneralizedMolecularHamiltonian(one_int, two_int, update_prev_params=True)
    original = np.random.rand(ham.params.size)
    step1 = np.random.rand(ham.params.size)
    step2 = np.random.rand(ham.params.size)
    ham.assign_params(original.copy())
    ham.assign_params(original + step1)
    ham.assign_params(original + step1 + step2)

    temp_ham = GeneralizedMolecularHamiltonian(one_int, two_int)
    temp_ham.orb_rotate_matrix(unitary_matrix(original).dot(unitary_matrix(step1)).dot(unitary_matrix(step2)))
    assert np.allclose(ham.one_int, temp_ham.one_int)
    assert np.allclose(ham.two_int, temp_ham.two_int)

    def objective(params):
        temp_ham = GeneralizedMolecularHamiltonian(one_int, two_int)
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


def test_generalizedmolecularhamiltonian_save_params(tmp_path):
    """Test GeneralizedMolecularHamiltonian.sav_params."""
    ham = GeneralizedMolecularHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2),
        np.arange(16, dtype=float).reshape(2, 2, 2, 2),
        update_prev_params=True,
    )
    ham.assign_params(np.random.rand(ham.nparams))
    ham.assign_params(np.random.rand(ham.nparams))
    ham.save_params(str(tmp_path / "temp.npy"))
    assert np.allclose(np.load(str(tmp_path / "temp.npy")), ham.params)
    assert np.allclose(np.load(str(tmp_path / "temp_um.npy")), ham._prev_unitary)
