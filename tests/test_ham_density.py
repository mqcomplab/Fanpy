"""Test fanpy.ham.density."""
from fanpy.ham.density import add_one_density, add_two_density, density_matrix

import numpy as np

import pytest

from utils import find_datafile


def test_add_one_density():
    """Test density.add_one_density."""
    # check error
    #  tuple of numpy array
    with pytest.raises(TypeError):
        add_one_density((np.zeros((2, 2)),), 1, 1, 0.5, "restricted")
    #  list of nonnumpy array
    with pytest.raises(TypeError):
        add_one_density([np.zeros((2, 2)).tolist()], 1, 1, 0.5, "restricted")
    #  bad matrix shape
    with pytest.raises(TypeError):
        add_one_density([np.zeros((2, 2, 2))], 1, 1, 0.5, "restricted")
    with pytest.raises(TypeError):
        add_one_density([np.zeros((2, 3))], 1, 1, 0.5, "generalized")
    #  bad orbital type
    with pytest.raises(ValueError):
        add_one_density([np.zeros((2, 2))], 1, 1, 0.5, "dsfsdfdsf")
    with pytest.raises(ValueError):
        add_one_density([np.zeros((2, 2))], 1, 1, 0.5, "Restricted")
    #  mismatching orbital type and matrices
    with pytest.raises(ValueError):
        add_one_density([np.zeros((2, 2))] * 2, 1, 1, 0.5, "restricted")
    with pytest.raises(ValueError):
        add_one_density([np.zeros((2, 2))] * 2, 1, 1, 0.5, "generalized")
    with pytest.raises(ValueError):
        add_one_density([np.zeros((2, 2))], 1, 1, 0.5, "unrestricted")

    # restricted (3 spatial orbitals, 6 spin orbitals)
    matrices = [np.zeros((3, 3))]
    add_one_density(matrices, 0, 1, 1, "restricted")
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    add_one_density(matrices, 3, 1, 1, "restricted")
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    add_one_density(matrices, 0, 4, 1, "restricted")
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    add_one_density(matrices, 3, 4, 1, "restricted")
    assert np.allclose(matrices[0], np.array([[0, 2, 0], [0, 0, 0], [0, 0, 0]]))

    # unrestricted (3 spatial orbitals, 6 spin orbitals)
    matrices = [np.zeros((3, 3)), np.zeros((3, 3))]
    add_one_density(matrices, 0, 1, 1, "unrestricted")
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    assert np.allclose(matrices[1], np.zeros((3, 3)))
    add_one_density(matrices, 3, 1, 1, "unrestricted")
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    assert np.allclose(matrices[1], np.zeros((3, 3)))
    add_one_density(matrices, 0, 4, 1, "unrestricted")
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    assert np.allclose(matrices[1], np.zeros((3, 3)))
    add_one_density(matrices, 3, 4, 1, "unrestricted")
    assert np.allclose(matrices[0], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    assert np.allclose(matrices[1], np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))

    # generalized (6 orbitals)
    matrices = [np.zeros((6, 6))]
    add_one_density(matrices, 0, 1, 1, "generalized")
    assert np.allclose(
        matrices[0],
        np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    )
    add_one_density(matrices, 3, 1, 1, "generalized")
    assert np.allclose(
        matrices[0],
        np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    )
    add_one_density(matrices, 0, 4, 1, "generalized")
    assert np.allclose(
        matrices[0],
        np.array(
            [
                [0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    )
    add_one_density(matrices, 3, 4, 1, "generalized")
    assert np.allclose(
        matrices[0],
        np.array(
            [
                [0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    )


def test_add_two_density():
    """Test density.add_two_density."""
    # check error
    #  tuple of numpy array
    with pytest.raises(TypeError):
        add_two_density((np.zeros((2, 2, 2, 2)),), 1, 1, 1, 1, 0.5, "restricted")
    #  list of nonnumpy array
    with pytest.raises(TypeError):
        add_two_density([np.zeros((2, 2, 2, 2)).tolist()], 1, 1, 1, 1, 0.5, "restricted")
    #  bad matrix shape
    with pytest.raises(TypeError):
        add_two_density([np.zeros((2, 2, 2))], 1, 1, 1, 1, 0.5, "restricted")
    with pytest.raises(TypeError):
        add_two_density([np.zeros((2, 2, 2, 3))], 1, 1, 1, 1, 0.5, "unrestricted")
    #  bad orbital type
    with pytest.raises(ValueError):
        add_two_density([np.zeros((2, 2, 2, 2))], 1, 1, 1, 1, 0.5, "dsfsdfdsf")
    with pytest.raises(ValueError):
        add_two_density([np.zeros((2, 2, 2, 2))], 1, 1, 1, 1, 0.5, "Restricted")
    #  mismatching orbital type and matrices
    with pytest.raises(ValueError):
        add_two_density([np.zeros((2, 2, 2, 2))] * 3, 1, 1, 1, 1, 0.5, "restricted")
    with pytest.raises(ValueError):
        add_two_density([np.zeros((2, 2, 2, 2))] * 3, 1, 1, 1, 1, 0.5, "generalized")
    with pytest.raises(ValueError):
        add_two_density([np.zeros((2, 2, 2, 2))], 1, 1, 1, 1, 0.5, "unrestricted")

    # restricted (3 spatial orbitals, 6 spin orbitals)
    matrices = [np.zeros((4, 4, 4, 4))]
    answer = np.zeros((4, 4, 4, 4))
    add_two_density(matrices, 0, 1, 2, 3, 1, "restricted")
    answer[0, 1, 2, 3] += 1
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 1, 2, 3, 1, "restricted")
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 2, 3, 1, "restricted")
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 1, 6, 3, 1, "restricted")
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 1, 2, 7, 1, "restricted")
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 5, 2, 3, 1, "restricted")
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 1, 6, 3, 1, "restricted")
    answer[0, 1, 2, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 1, 2, 7, 1, "restricted")
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 6, 3, 1, "restricted")
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 2, 7, 1, "restricted")
    answer[0, 1, 2, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 1, 6, 7, 1, "restricted")
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 5, 6, 3, 1, "restricted")
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 5, 2, 7, 1, "restricted")
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 1, 6, 7, 1, "restricted")
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 6, 7, 1, "restricted")
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 5, 6, 7, 1, "restricted")
    answer[0, 1, 2, 3] += 1
    assert np.allclose(matrices[0], answer)

    # unrestricted (3 spatial orbitals, 6 spin orbitals)
    # NOTE: [np.zeros((4, 4, 4, 4))]*3 results in three matrices with the same reference
    matrices = [np.zeros((4, 4, 4, 4)), np.zeros((4, 4, 4, 4)), np.zeros((4, 4, 4, 4))]
    answer_alpha = np.zeros((4, 4, 4, 4))
    answer_alpha_beta = np.zeros((4, 4, 4, 4))
    answer_beta = np.zeros((4, 4, 4, 4))

    def check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta):
        """Check unrestricted answers."""
        assert np.allclose(matrices[0], answer_alpha)
        assert np.allclose(matrices[1], answer_alpha_beta)
        assert np.allclose(matrices[2], answer_beta)

    add_two_density(matrices, 0, 1, 2, 3, 1, "unrestricted")
    answer_alpha[0, 1, 2, 3] += 1
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)

    add_two_density(matrices, 4, 1, 2, 3, 1, "unrestricted")
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 5, 2, 3, 1, "unrestricted")
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 1, 6, 3, 1, "unrestricted")
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 1, 2, 7, 1, "unrestricted")
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)

    add_two_density(matrices, 4, 5, 2, 3, 1, "unrestricted")
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 4, 1, 6, 3, 1, "unrestricted")
    # this is beta-alpha-beta-alpha, and only the alpha-beta-alpha-beta matrix is stored
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 4, 1, 2, 7, 1, "unrestricted")
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 5, 6, 3, 1, "unrestricted")
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 5, 2, 7, 1, "unrestricted")
    answer_alpha_beta[0, 1, 2, 3] += 1
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 1, 6, 7, 1, "unrestricted")
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)

    add_two_density(matrices, 4, 5, 6, 3, 1, "unrestricted")
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 4, 5, 2, 7, 1, "unrestricted")
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 4, 1, 6, 7, 1, "unrestricted")
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)
    add_two_density(matrices, 0, 5, 6, 7, 1, "unrestricted")
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)

    add_two_density(matrices, 4, 5, 6, 7, 1, "unrestricted")
    answer_beta[0, 1, 2, 3] += 1
    check_answer_unrestricted(matrices, answer_alpha, answer_alpha_beta, answer_beta)

    # generalized (6 orbitals)
    matrices = [np.zeros((8, 8, 8, 8))]
    answer = np.zeros((8, 8, 8, 8))

    add_two_density(matrices, 0, 1, 2, 3, 1, "generalized")
    answer[0, 1, 2, 3] += 1
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 1, 2, 3, 1, "generalized")
    answer[4, 1, 2, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 2, 3, 1, "generalized")
    answer[0, 5, 2, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 1, 6, 3, 1, "generalized")
    answer[0, 1, 6, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 1, 2, 7, 1, "generalized")
    answer[0, 1, 2, 7] += 1
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 5, 2, 3, 1, "generalized")
    answer[4, 5, 2, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 1, 6, 3, 1, "generalized")
    answer[4, 1, 6, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 1, 2, 7, 1, "generalized")
    answer[4, 1, 2, 7] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 6, 3, 1, "generalized")
    answer[0, 5, 6, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 2, 7, 1, "generalized")
    answer[0, 5, 2, 7] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 1, 6, 7, 1, "generalized")
    answer[0, 1, 6, 7] += 1
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 5, 6, 3, 1, "generalized")
    answer[4, 5, 6, 3] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 5, 2, 7, 1, "generalized")
    answer[4, 5, 2, 7] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 4, 1, 6, 7, 1, "generalized")
    answer[4, 1, 6, 7] += 1
    assert np.allclose(matrices[0], answer)
    add_two_density(matrices, 0, 5, 6, 7, 1, "generalized")
    answer[0, 5, 6, 7] += 1
    assert np.allclose(matrices[0], answer)

    add_two_density(matrices, 4, 5, 6, 7, 1, "generalized")
    answer[4, 5, 6, 7] += 1
    assert np.allclose(matrices[0], answer)


def test_density_matrix():
    """Test density.density_matrix."""
    # check type
    with pytest.raises(ValueError):
        density_matrix(
            np.arange(1, 5),
            [0b0101, 0b1001, 0b0110, 0b1010],
            2,
            is_chemist_notation=False,
            val_threshold=0,
            orbtype="daslkfjaslkdf",
        )
    # restricted
    one_density_r, two_density_r = density_matrix(
        np.arange(1, 4),
        [0b0101, 0b1010, 0b0110],
        2,
        is_chemist_notation=False,
        val_threshold=0,
        orbtype="restricted",
    )
    assert np.allclose(
        one_density_r,
        np.array([[1 * 1 + 1 * 1 + 3 * 3, 1 * 3 + 2 * 3], [1 * 3 + 2 * 3, 2 * 2 + 2 * 2 + 3 * 3]]),
    )
    assert np.allclose(
        two_density_r,
        np.array(
            [
                [
                    [[1 * 1 + 1 * 1, 1 * 3], [1 * 3, 1 * 2 + 1 * 2]],
                    [[1 * 3, 3 * 3], [0, 1 * 3 + 1 * 3]],
                ],
                [
                    [[1 * 3, 0], [3 * 3, 1 * 3 + 1 * 3]],
                    [[2 * 1 + 2 * 1, 2 * 3], [2 * 3, 2 * 2 + 2 * 2]],
                ],
            ]
        ),
    )
    # unrestricted
    one_density_u, two_density_u = density_matrix(
        np.arange(1, 4),
        [0b0101, 0b1010, 0b0110],
        2,
        is_chemist_notation=False,
        val_threshold=0,
        orbtype="unrestricted",
    )
    assert np.allclose(one_density_u[0], np.array([[1 * 1, 1 * 3], [1 * 3, 2 * 2 + 3 * 3]]))
    assert np.allclose(one_density_u[1], np.array([[1 * 1 + 3 * 3, 2 * 3], [2 * 3, 2 * 2]]))
    assert np.allclose(
        two_density_u[0],
        np.array([[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]),
    )
    assert np.allclose(
        two_density_u[1],
        np.array(
            [
                [[[1 * 1, 0], [1 * 3, 1 * 2]], [[0, 0], [0, 0]]],
                [[[1 * 3, 0], [3 * 3, 1 * 3 + 1 * 3]], [[2 * 1, 0], [2 * 3, 2 * 2]]],
            ]
        ),
    )
    assert np.allclose(
        two_density_u[2],
        np.array([[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]),
    )
    # generalized
    one_density_g, two_density_g = density_matrix(
        np.arange(1, 4),
        [0b0101, 0b1010, 0b0110],
        2,
        is_chemist_notation=False,
        val_threshold=0,
        orbtype="generalized",
    )
    assert np.allclose(
        one_density_g[0],
        np.array(
            [
                [1 * 1, 1 * 3, 0, 0],
                [1 * 3, 2 * 2 + 3 * 3, 0, 0],
                [0, 0, 1 * 1 + 3 * 3, 2 * 3],
                [0, 0, 2 * 3, 2 * 2],
            ]
        ),
    )
    assert np.allclose(
        two_density_g[0],
        np.array(
            [
                [
                    [[0, 0, 0, 0]] * 4,
                    [[0, 0, 0, 0]] * 4,
                    [[0, 0, 1, 0], [0, 0, 3, 2.0], [-1, -3, 0, 0], [0, -2, 0, 0]],
                    [[0, 0, 0, 0]] * 4,
                ],
                [
                    [[0, 0, 0, 0]] * 4,
                    [[0, 0, 0, 0]] * 4,
                    [[0, 0, 3, 0], [0, 0, 9, 6], [-3, -9, 0, 0], [0, -6, 0, 0]],
                    [[0, 0, 2, 0], [0, 0, 6, 4], [-2, -6, 0, 0], [0, -4, 0, 0]],
                ],
                [
                    [[0, 0, -1, 0], [0, 0, -3, -2.0], [1, 3, 0, 0], [0, 2, 0, 0]],
                    [[0, 0, -3, 0], [0, 0, -9, -6.0], [3, 9, 0, 0], [0, 6, 0, 0]],
                    [[0, 0, 0, 0]] * 4,
                    [[0, 0, 0, 0]] * 4,
                ],
                [
                    [[0, 0, 0, 0]] * 4,
                    [[0, 0, -2, 0], [0, 0, -6, -4], [2, 6, 0, 0], [0, 4, 0, 0]],
                    [[0, 0, 0, 0]] * 4,
                    [[0, 0, 0, 0]] * 4,
                ],
            ]
        ),
    )
    # compare restricted with unrestricted
    assert np.allclose(one_density_r[0], one_density_u[0] + one_density_u[1])
    assert np.allclose(
        two_density_r[0],
        (
            two_density_u[0]
            + two_density_u[2]
            + two_density_u[1]
            + two_density_u[1].transpose((1, 0, 3, 2))
        ),
    )

    # truncate
    one_density, two_density = density_matrix(
        np.arange(1, 4),
        [0b0101, 0b1010, 0b0110],
        2,
        is_chemist_notation=False,
        val_threshold=2,
        orbtype="restricted",
    )
    assert np.allclose(
        one_density, np.array([[3 * 3, 1 * 3 + 2 * 3], [1 * 3 + 2 * 3, 2 * 2 + 2 * 2 + 3 * 3]])
    )
    assert np.allclose(
        two_density,
        np.array(
            [
                [[[0, 1 * 3], [1 * 3, 1 * 2 + 1 * 2]], [[1 * 3, 3 * 3], [0, 1 * 3 + 1 * 3]]],
                [
                    [[1 * 3, 0], [3 * 3, 1 * 3 + 1 * 3]],
                    [[2 * 1 + 2 * 1, 2 * 3], [2 * 3, 2 * 2 + 2 * 2]],
                ],
            ]
        ),
    )

    one_density, two_density = density_matrix(
        np.arange(1, 4),
        [0b0101, 0b1010, 0b0110],
        2,
        is_chemist_notation=False,
        val_threshold=17,
        orbtype="restricted",
    )
    assert np.allclose(one_density, np.array([[3 * 3, 1 * 3 + 2 * 3], [1 * 3 + 2 * 3, 3 * 3]]))
    assert np.allclose(
        two_density,
        np.array(
            [
                [[[0, 1 * 3], [1 * 3, 0]], [[1 * 3, 3 * 3], [0, 1 * 3 + 1 * 3]]],
                [[[1 * 3, 0], [3 * 3, 1 * 3 + 1 * 3]], [[0, 2 * 3], [2 * 3, 0]]],
            ]
        ),
    )

    one_density, two_density = density_matrix(
        np.arange(1, 4),
        [0b0101, 0b1010, 0b0110],
        2,
        is_chemist_notation=False,
        val_threshold=28,
        orbtype="restricted",
    )
    assert np.allclose(one_density, np.array([[3 * 3, 2 * 3], [2 * 3, 3 * 3]]))
    assert np.allclose(
        two_density,
        np.array(
            [
                [[[0, 0], [0, 0]], [[0, 3 * 3], [0, 1 * 3 + 1 * 3]]],
                [[[0, 0], [3 * 3, 1 * 3 + 1 * 3]], [[0, 2 * 3], [2 * 3, 0]]],
            ]
        ),
    )

    # break particle number symmetry
    one_density, two_density = density_matrix(
        np.arange(1, 3),
        [0b1111, 0b0001],
        2,
        is_chemist_notation=False,
        val_threshold=0,
        orbtype="restricted",
    )
    assert np.allclose(one_density, np.array([[1 * 1 + 1 * 1 + 2 * 2, 0], [0, 1 * 1 + 1 * 1]]))
    assert np.allclose(
        two_density,
        np.array(
            [
                [
                    [[1 * 1 + 1 * 1, 0], [0, 0]],
                    [[0, 1 * 1 + 1 * 1 + 1 * 1 + 1 * 1], [-1 * 1 - 1 * 1, 0]],
                ],
                [
                    [[0, -1 * 1 - 1 * 1], [1 * 1 + 1 * 1 + 1 * 1 + 1 * 1, 0]],
                    [[0, 0], [0, 1 * 1 + 1 * 1]],
                ],
            ]
        ),
    )

