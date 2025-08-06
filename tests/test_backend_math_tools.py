"""Test for fanpy.tools.math_tools."""
from fanpy.tools.math_tools import (
    adjugate,
    binomial,
    permanent_borchardt,
    permanent_combinatoric,
    permanent_ryser,
    unitary_matrix,
)

import numpy as np

import pytest

def test_binomial():
    """Test binomial."""
    assert binomial(4, 0) == 1
    assert binomial(4, -1) == 0
    assert binomial(-1, -1) == 0
    assert binomial(-1, 0) == 0
    assert binomial(4, 4) == 1
    assert binomial(4, 5) == 0
    assert binomial(1000, 10) == 263409560461970212832400
    assert binomial(1000.99, 10) == 263409560461970212832400
    assert binomial(1001, 10) == 266067578226470416796400


def test_adjugate():
    """Test adjugate."""
    # 0 by 1
    with pytest.raises(ValueError):
        adjugate(np.ndarray(shape=(0, 0)))
    with pytest.raises(ValueError):
        adjugate(np.ndarray(shape=(0, 1)))
    with pytest.raises(ValueError):
        adjugate(np.ndarray(shape=(1, 0)))
    # 1 by 1
    assert np.allclose(adjugate(np.array([[1]])), 1)
    assert np.allclose(adjugate(np.array([[3]])), 1)
    assert np.allclose(adjugate(np.array([[-3]])), 1)
    # 2 by 2
    matrix = np.random.rand(4).reshape(2, 2)
    assert np.allclose(
        adjugate(matrix), np.array([[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]])
    )
    # 3 by 3
    matrix = np.random.rand(9).reshape(3, 3)
    row_mask = np.ones(3, dtype=bool)
    col_mask = np.ones(3, dtype=bool)
    cofactor = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            row_mask[i] = False
            col_mask[j] = False
            cofactor[i, j] = (-1) ** ((i + j) % 2) * np.linalg.det(matrix[row_mask][:, col_mask])
            row_mask[i] = True
            col_mask[j] = True
    assert np.allclose(adjugate(matrix), cofactor.T)
    # non invertible matrix
    matrix[0] = matrix[1] + matrix[2]
    with pytest.raises(np.linalg.LinAlgError):
        adjugate(matrix)


def test_permanent_combinatoric():
    """Test permanent_combinatoric."""
    with pytest.raises(ValueError):
        permanent_combinatoric(np.ndarray(shape=(0, 0)))
    with pytest.raises(ValueError):
        permanent_combinatoric(np.ndarray(shape=(1, 0)))
    with pytest.raises(ValueError):
        permanent_combinatoric(np.ndarray(shape=(0, 1)))

    assert np.allclose(permanent_combinatoric(np.arange(1, 10).reshape(3, 3)), 450)
    assert np.allclose(permanent_combinatoric(np.arange(1, 13).reshape(3, 4)), 3900)


def test_permanent_ryser():
    """Test permanent_ryser."""
    with pytest.raises(ValueError):
        permanent_ryser(np.ndarray(shape=(0, 0)))
    with pytest.raises(ValueError):
        permanent_ryser(np.ndarray(shape=(1, 0)))
    with pytest.raises(ValueError):
        permanent_ryser(np.ndarray(shape=(0, 1)))

    for i in range(1, 6):
        for j in range(1, 6):
            assert np.allclose(
                permanent_ryser(np.arange(1, i * j + 1).reshape(i, j)),
                permanent_combinatoric(np.arange(1, i * j + 1).reshape(i, j)),
            )


def test_permanent_borchardt_square():
    """Test permanent_borchardt on square matrices."""
    p = 6
    lambdas = np.random.rand(p)
    epsilons = np.random.rand(p)
    zetas = np.random.rand(p)
    etas = np.random.rand(p)

    with pytest.raises(ValueError):
        permanent_borchardt(lambdas, epsilons[: p - 1], zetas, etas=None)
    with pytest.raises(ValueError):
        permanent_borchardt(lambdas, epsilons, zetas[: p - 1], etas=None)
    with pytest.raises(ValueError):
        permanent_borchardt(lambdas, epsilons, zetas, etas[: p - 1])
    with pytest.raises(ValueError):
        permanent_borchardt(lambdas[: p - 1], epsilons, zetas, etas)

    # without etas
    gem_coeffs = zetas / (epsilons - lambdas[:, np.newaxis])

    assert np.allclose(
        permanent_combinatoric(gem_coeffs), permanent_borchardt(lambdas, epsilons, zetas)
    )

    # with etas
    gem_coeffs = zetas * etas[:, np.newaxis] / (epsilons - lambdas[:, np.newaxis])

    assert np.allclose(
        permanent_combinatoric(gem_coeffs), permanent_borchardt(lambdas, epsilons, zetas, etas)
    )


def test_permanent_borchardt_rect():
    """Test permanent_borchardt on rectangular matrices."""
    p = 5
    k = 8
    # let nrows < ncol
    lambdas = np.random.rand(p)
    epsilons = np.random.rand(k)
    zetas = np.random.rand(k)
    etas = np.random.rand(p)

    # without etas
    gem_coeffs = zetas / (lambdas[:, np.newaxis] - epsilons)
    perm_comb = permanent_combinatoric(gem_coeffs)
    perm_borch = permanent_borchardt(lambdas, epsilons, zetas)
    perm_rys = permanent_ryser(gem_coeffs)
    assert np.allclose(perm_comb, perm_borch)
    assert np.allclose(perm_rys, perm_borch)

    # with etas (ncol > nrow)
    gem_coeffs = zetas * etas[:, np.newaxis] / (lambdas[:, np.newaxis] - epsilons)
    perm_comb = permanent_combinatoric(gem_coeffs)
    perm_borch = permanent_borchardt(lambdas, epsilons, zetas, etas)
    perm_rys = permanent_ryser(gem_coeffs)
    assert np.allclose(perm_comb, perm_borch)
    assert np.allclose(perm_rys, perm_borch)

    # let nrows < ncol
    lambdas = np.random.rand(k)
    epsilons = np.random.rand(p)
    zetas = np.random.rand(p)
    etas = np.random.rand(k)

    # without etas
    gem_coeffs = zetas / (lambdas[:, np.newaxis] - epsilons)
    perm_comb = permanent_combinatoric(gem_coeffs)
    perm_borch = permanent_borchardt(lambdas, epsilons, zetas)
    perm_rys = permanent_ryser(gem_coeffs)
    assert np.allclose(perm_comb, perm_borch)
    assert np.allclose(perm_rys, perm_borch)

    # with etas
    gem_coeffs = zetas * etas[:, np.newaxis] / (lambdas[:, np.newaxis] - epsilons)
    perm_comb = permanent_combinatoric(gem_coeffs)
    perm_borch = permanent_borchardt(lambdas, epsilons, zetas, etas)
    perm_rys = permanent_ryser(gem_coeffs)
    assert np.allclose(perm_comb, perm_borch)
    assert np.allclose(perm_rys, perm_borch)


def test_unitary_matrix():
    """Test math_tools.unitary_matrix."""
    with pytest.raises(TypeError):
        unitary_matrix([1, 2, 3])
    with pytest.raises(TypeError):
        unitary_matrix(np.array([[1, 2], [3]], dtype=object))
    with pytest.raises(TypeError):
        unitary_matrix(np.array([[1, 2], [0, 3]]))
    with pytest.raises(TypeError):
        unitary_matrix(np.array([[1, 2], [-2, 3]]))
    with pytest.raises(ValueError):
        unitary_matrix(np.array([1, 2, 3, 4]))
    with pytest.raises(ValueError):
        unitary_matrix(np.array([1, 2, 3, 4, 5]))

    assert np.allclose(unitary_matrix(np.zeros(1)), np.identity(2))
    assert np.allclose(unitary_matrix(np.zeros(3)), np.identity(3))
    assert np.allclose(unitary_matrix(np.zeros(6)), np.identity(4))

    # FIXME: tests fail for some random numbers
    np.random.seed(424242)

    antiherm_elements = np.random.rand(1)
    matrix = unitary_matrix(antiherm_elements, norm_threshold=1e-8)
    assert np.allclose(matrix.dot(matrix.T), np.identity(2))
    assert np.allclose(matrix.T.dot(matrix), np.identity(2))

    antiherm_elements = np.random.rand(6)
    matrix = unitary_matrix(antiherm_elements, norm_threshold=1e-8)
    assert np.allclose(matrix.dot(matrix.T), np.identity(4))
    assert np.allclose(matrix.T.dot(matrix), np.identity(4))

    antiherm_elements = np.random.rand(6) * 10
    matrix = unitary_matrix(antiherm_elements, norm_threshold=1e-8)
    assert np.allclose(matrix.dot(matrix.T), np.identity(4))
    assert np.allclose(matrix.T.dot(matrix), np.identity(4))

    antiherm_elements = np.random.rand(55) * 10
    matrix = unitary_matrix(antiherm_elements, norm_threshold=1e-8)
    assert np.allclose(matrix.dot(matrix.T), np.identity(11))
    assert np.allclose(matrix.T.dot(matrix), np.identity(11))
