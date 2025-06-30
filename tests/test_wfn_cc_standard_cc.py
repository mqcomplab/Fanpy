"""Test fanpy.wavefunction.cc.standard_cc."""
import pytest
from fanpy.tools import slater
from fanpy.wfn.cc.standard_cc import StandardCC
import numpy as np


class TempStandardCC(StandardCC):
    """CC wavefunction that skips initialization."""
    def __init__(self):
        self._cache_fns = {}
        self.exop_combinations = {}


def test_assign_exops():
    """Test StandardCC.assign_exops."""
    test = TempStandardCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn()
    with pytest.raises(TypeError):
        test.assign_exops([[0, 1, 4, 5], [2, 3, 6, 7]])
    test.assign_ranks([2])
    test.assign_exops()
    assert test.exops == {(0, 1, 2, 3): 0, (0, 1, 2, 6): 1, (0, 1, 2, 7): 2, (0, 1, 3, 6): 3,
                          (0, 1, 3, 7): 4, (0, 1, 6, 7): 5, (0, 4, 2, 3): 6, (0, 4, 2, 6): 7,
                          (0, 4, 2, 7): 8, (0, 4, 3, 6): 9, (0, 4, 3, 7): 10, (0, 4, 6, 7): 11,
                          (0, 5, 2, 3): 12, (0, 5, 2, 6): 13, (0, 5, 2, 7): 14, (0, 5, 3, 6): 15,
                          (0, 5, 3, 7): 16, (0, 5, 6, 7): 17, (1, 4, 2, 3): 18, (1, 4, 2, 6): 19,
                          (1, 4, 2, 7): 20, (1, 4, 3, 6): 21, (1, 4, 3, 7): 22, (1, 4, 6, 7): 23,
                          (1, 5, 2, 3): 24, (1, 5, 2, 6): 25, (1, 5, 2, 7): 26, (1, 5, 3, 6): 27,
                          (1, 5, 3, 7): 28, (1, 5, 6, 7): 29, (4, 5, 2, 3): 30, (4, 5, 2, 6): 31,
                          (4, 5, 2, 7): 32, (4, 5, 3, 6): 33, (4, 5, 3, 7): 34, (4, 5, 6, 7): 35} 


def test_assign_refwfn():
    """Test StandardCC.assign_refwfn."""
    test = TempStandardCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    with pytest.raises(TypeError):
        test.assign_refwfn("This is not a gmpy2 instance")
    with pytest.raises(ValueError):
        test.assign_refwfn(0b00010001)
    test.assign_refwfn()
    assert test.refwfn == 0b00110011


def check_sign(occ_indices, exops):
    sd = slater.create(0, *occ_indices)
    sign = 1
    for exop in exops:
        sign *= slater.sign_excite(sd, exop[:len(exop) // 2], exop[len(exop) // 2:])
        sd = slater.excite(sd, *exop)
    return sign

def scale_sign(sgn):
    """Convert sign to 1 or -1.
    
    Args:
        sgn (uint): Sign of the excitation in exop_comb (-1 is stored as 0).
    
    Returns:
        int: 1 or -1.
    """
    if sgn == 0:
        return -1
    elif sgn == 1:
        return 1
    else:
        raise ValueError(f"Invalid sign value: {sgn}. Expected 1 or 0, where 0 represents -1.")

def test_generate_possible_exops():
    """Test StandardCC.generate_possible_exops."""
    test = TempStandardCC()
    test.assign_nelec(2)
    test.assign_nspin(4)
    test.assign_refwfn()
    test.assign_ranks([1])
    test.assign_exops()
    test.refresh_exops = None
    test.params = np.random.rand(4)
    test.generate_possible_exops([0, 2], [1, 3])
    assert np.allclose(
        test.exop_combinations[(0, 2, 1, 3)][2][0][0:2], [test.get_ind((0, 1)), test.get_ind((2, 3))]
    )
    assert np.allclose(
        test.exop_combinations[(0, 2, 1, 3)][2][1][0:2], [test.get_ind((0, 3)), test.get_ind((2, 1))]
    )
    base_sign = slater.sign_excite(0b0101, [0, 2], [1, 3]) # base sign of excitation
    assert  base_sign * scale_sign(test.exop_combinations[(0, 2, 1, 3)][2][0][-1] ) == check_sign([0, 2], [[0, 1], [2, 3]])
    assert base_sign * scale_sign(test.exop_combinations[(0, 2, 1, 3)][2][1][-1] ) == check_sign([0, 2], [[0, 3], [2, 1]])

    test.assign_ranks([1, 2])
    test.assign_exops()
    test.generate_possible_exops([0, 2], [1, 3])
    assert np.allclose(
        test.exop_combinations[(0, 2, 1, 3)][2][0][0:2], [test.get_ind((0, 1)), test.get_ind((2, 3))]
    )
    assert np.allclose(
        test.exop_combinations[(0, 2, 1, 3)][2][1][0:2], [test.get_ind((0, 3)), test.get_ind((2, 1))]
    )
    assert np.allclose(
        test.exop_combinations[(0, 2, 1, 3)][1][0][0:1], [test.get_ind((0, 2, 1, 3))]
    )
    base_sign = slater.sign_excite(0b0101, [0, 2], [1, 3])
    assert base_sign * scale_sign(test.exop_combinations[(0, 2, 1, 3)][2][0][-1]) == check_sign([0, 2], [[0, 1], [2, 3]])
    assert base_sign * scale_sign(test.exop_combinations[(0, 2, 1, 3)][2][1][-1]) == check_sign([0, 2], [[0, 3], [2, 1]])
    assert base_sign * scale_sign(test.exop_combinations[(0, 2, 1, 3)][1][0][-1]) == check_sign([0, 2], [[0, 2, 1, 3]])

    test = TempStandardCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn(0b00110011)
    test.assign_ranks([1])
    test.assign_exops()
    test.refresh_exops = None
    test.params = np.random.rand(8)
    test.generate_possible_exops([0, 1], [2, 3])
    assert np.allclose(
        test.exop_combinations[(0, 1, 2, 3)][2][0][0:2], [test.get_ind((0, 2)), test.get_ind((1, 3))]
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 2, 3)][2][1][0:2], [test.get_ind((0, 3)), test.get_ind((1, 2))]
    )
    base_sign = slater.sign_excite(0b0011, [0, 1], [2, 3])
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 2, 3)][2][0][-1]) == check_sign([0, 1], [[0, 2], [1, 3]])
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 2, 3)][2][1][-1]) == check_sign([0, 1], [[0, 3], [1, 2]])

    test.generate_possible_exops([0, 1, 4], [2, 3, 6])
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][0][0:3],
        (test.get_ind((0, 2)), test.get_ind((1, 3)), test.get_ind((4, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][1][0:3],
            (test.get_ind((0, 2)), test.get_ind((1, 6)), test.get_ind((4, 3))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][2][0:3],
            (test.get_ind((0, 3)), test.get_ind((1, 2)), test.get_ind((4, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][3][0:3],
            (test.get_ind((0, 3)), test.get_ind((1, 6)), test.get_ind((4, 2))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][4][0:3],
            (test.get_ind((0, 6)), test.get_ind((1, 2)), test.get_ind((4, 3))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][5][0:3],
            (test.get_ind((0, 6)), test.get_ind((1, 3)), test.get_ind((4, 2))),
    )
    base_sign = slater.sign_excite(0b10011, [0, 1, 4], [2, 3, 6])
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][0][-1]) == check_sign([0, 1, 4],
        ((0, 2), (1, 3), (4, 6)))
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][1][-1]) == check_sign([0, 1, 4],
            ((0, 2), (1, 6), (4, 3)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][2][-1]) == check_sign([0, 1, 4],
            ((0, 3), (1, 2), (4, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][3][-1]) == check_sign([0, 1, 4],
            ((0, 3), (1, 6), (4, 2)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][4][-1]) == check_sign([0, 1, 4],
            ((0, 6), (1, 2), (4, 3)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][5][-1]) == check_sign([0, 1, 4],
            ((0, 6), (1, 3), (4, 2)),)

    test = TempStandardCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn(0b00110011)
    test.assign_ranks([1, 2])
    test.assign_exops()
    test.refresh_exops = None
    test.params = np.random.rand(8)
    test.generate_possible_exops([0, 1], [2, 3])
    assert np.allclose(
        test.exop_combinations[(0, 1, 2, 3)][2][0][0:2], [test.get_ind((0, 2)), test.get_ind((1, 3))]
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 2, 3)][2][1][0:2], [test.get_ind((0, 3)), test.get_ind((1, 2))]
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 2, 3)][1][0][0:1], [test.get_ind((0, 1, 2, 3))]
    )
    base_sign = slater.sign_excite(0b0011, [0, 1], [2, 3])
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 2, 3)][2][0][-1]) == check_sign([0, 1], [[0, 2], [1, 3]])
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 2, 3)][2][1][-1]) == check_sign([0, 1], [[0, 3], [1, 2]])
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 2, 3)][1][0][-1]) == check_sign([0, 1], [[0, 1, 2, 3]])

    test.generate_possible_exops([0, 1, 4], [2, 3, 6])
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][0][0:3],
        (test.get_ind((0, 2)), test.get_ind((1, 3)), test.get_ind((4, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][1][0:3],
            (test.get_ind((0, 2)), test.get_ind((1, 6)), test.get_ind((4, 3))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][2][0:3],
            (test.get_ind((0, 3)), test.get_ind((1, 2)), test.get_ind((4, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][3][0:3],
            (test.get_ind((0, 3)), test.get_ind((1, 6)), test.get_ind((4, 2))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][4][0:3],
            (test.get_ind((0, 6)), test.get_ind((1, 2)), test.get_ind((4, 3))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][5][0:3],
            (test.get_ind((0, 6)), test.get_ind((1, 3)), test.get_ind((4, 2))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][0][0:2],
            (test.get_ind((0, 2)), test.get_ind((1, 4, 3, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][1][0:2],
        (test.get_ind((0, 3)), test.get_ind((1, 4, 2, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][2][0:2],
        (test.get_ind((0, 6)), test.get_ind((1, 4, 2, 3))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][3][0:2],
        (test.get_ind((1, 2)), test.get_ind((0, 4, 3, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][4][0:2],
        (test.get_ind((1, 3)), test.get_ind((0, 4, 2, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][5][0:2],
        (test.get_ind((1, 6)), test.get_ind((0, 4, 2, 3))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][6][0:2],
        (test.get_ind((4, 2)), test.get_ind((0, 1, 3, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][7][0:2],
        (test.get_ind((4, 3)), test.get_ind((0, 1, 2, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][8][0:2],
        (test.get_ind((4, 6)), test.get_ind((0, 1, 2, 3))),
    )
    base_sign = slater.sign_excite(0b10011, [0, 1, 4], [2, 3, 6])
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][0][-1]) == check_sign([0, 1, 4],
        ((0, 2), (1, 3), (4, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][1][-1]) == check_sign([0, 1, 4],
            ((0, 2), (1, 6), (4, 3)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][2][-1]) == check_sign([0, 1, 4],
            ((0, 3), (1, 2), (4, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][3][-1]) == check_sign([0, 1, 4],
            ((0, 3), (1, 6), (4, 2)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][4][-1]) == check_sign([0, 1, 4],
            ((0, 6), (1, 2), (4, 3)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][5][-1]) == check_sign([0, 1, 4],
            ((0, 6), (1, 3), (4, 2)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][0][-1]) == check_sign([0, 1, 4],
            ((0, 2), (1, 4, 3, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][1][-1]) == check_sign([0, 1, 4],
            ((0, 3), (1, 4, 2, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][2][-1]) == check_sign([0, 1, 4],
            ((0, 6), (1, 4, 2, 3)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][3][-1]) == check_sign([0, 1, 4],
            ((1, 2), (0, 4, 3, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][4][-1]) == check_sign([0, 1, 4],
            ((1, 3), (0, 4, 2, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][5][-1]) == check_sign([0, 1, 4],
            ((1, 6), (0, 4, 2, 3)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][6][-1]) == check_sign([0, 1, 4],
            ((4, 2), (0, 1, 3, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][7][-1]) == check_sign([0, 1, 4],
            ((4, 3), (0, 1, 2, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][8][-1]) == check_sign([0, 1, 4],
            ((4, 6), (0, 1, 2, 3)),)

    test = TempStandardCC()
    test.assign_nelec(4)
    test.assign_nspin(8)
    test.assign_refwfn(0b00110011)
    test.assign_ranks([1, 2, 3])
    test.assign_exops()
    test.assign_params(np.random.rand(68))
    test.refresh_exops = None
    test.generate_possible_exops([0, 1], [2, 3])
    assert np.allclose(
        test.exop_combinations[(0, 1, 2, 3)][2][0][0:2], [test.get_ind((0, 2)), test.get_ind((1, 3))]
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 2, 3)][2][1][0:2], [test.get_ind((0, 3)), test.get_ind((1, 2))]
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 2, 3)][1][0][0:1], [test.get_ind((0, 1, 2, 3))]
    )
    base_sign = slater.sign_excite(0b0011, [0, 1], [2, 3])
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 2, 3)][2][0][-1]) == check_sign([0, 1], [[0, 2], [1, 3]])
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 2, 3)][2][1][-1]) == check_sign([0, 1], [[0, 3], [1, 2]])
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 2, 3)][1][0][-1]) == check_sign([0, 1], [[0, 1, 2, 3]])

    test.generate_possible_exops([0, 1, 4], [2, 3, 6])
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][0][0:3],
        (test.get_ind((0, 2)), test.get_ind((1, 3)), test.get_ind((4, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][1][0:3],
            (test.get_ind((0, 2)), test.get_ind((1, 6)), test.get_ind((4, 3))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][2][0:3],
            (test.get_ind((0, 3)), test.get_ind((1, 2)), test.get_ind((4, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][3][0:3],
            (test.get_ind((0, 3)), test.get_ind((1, 6)), test.get_ind((4, 2))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][4][0:3],
            (test.get_ind((0, 6)), test.get_ind((1, 2)), test.get_ind((4, 3))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][5][0:3],
            (test.get_ind((0, 6)), test.get_ind((1, 3)), test.get_ind((4, 2))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][0][0:2],
            (test.get_ind((0, 2)), test.get_ind((1, 4, 3, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][1][0:2],
        (test.get_ind((0, 3)), test.get_ind((1, 4, 2, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][2][0:2],
        (test.get_ind((0, 6)), test.get_ind((1, 4, 2, 3))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][3][0:2],
        (test.get_ind((1, 2)), test.get_ind((0, 4, 3, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][4][0:2],
        (test.get_ind((1, 3)), test.get_ind((0, 4, 2, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][5][0:2],
        (test.get_ind((1, 6)), test.get_ind((0, 4, 2, 3))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][6][0:2],
        (test.get_ind((4, 2)), test.get_ind((0, 1, 3, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][7][0:2],
        (test.get_ind((4, 3)), test.get_ind((0, 1, 2, 6))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][8][0:2],
        (test.get_ind((4, 6)), test.get_ind((0, 1, 2, 3))),
    )
    assert np.allclose(
        test.exop_combinations[(0, 1, 4, 2, 3, 6)][1][0][0],
        (test.get_ind((0, 1, 4, 2, 3, 6)), ),
    )
    base_sign = slater.sign_excite(0b10011, [0, 1, 4], [2, 3, 6])
    print("DEBUG: base_sign", base_sign)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][0][-1]) == check_sign([0, 1, 4],
        ((0, 2), (1, 3), (4, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][1][-1]) == check_sign([0, 1, 4],
            ((0, 2), (1, 6), (4, 3)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][2][-1]) == check_sign([0, 1, 4],
            ((0, 3), (1, 2), (4, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][3][-1]) == check_sign([0, 1, 4],
            ((0, 3), (1, 6), (4, 2)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][4][-1]) == check_sign([0, 1, 4],
            ((0, 6), (1, 2), (4, 3)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][3][5][-1]) == check_sign([0, 1, 4],
            ((0, 6), (1, 3), (4, 2)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][0][-1]) == check_sign([0, 1, 4],
            ((0, 2), (1, 4, 3, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][1][-1]) == check_sign([0, 1, 4],
            ((0, 3), (1, 4, 2, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][2][-1]) == check_sign([0, 1, 4],
            ((0, 6), (1, 4, 2, 3)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][3][-1]) == check_sign([0, 1, 4],
            ((1, 2), (0, 4, 3, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][4][-1]) == check_sign([0, 1, 4],
            ((1, 3), (0, 4, 2, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][5][-1]) == check_sign([0, 1, 4],
            ((1, 6), (0, 4, 2, 3)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][6][-1]) == check_sign([0, 1, 4],
            ((4, 2), (0, 1, 3, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][7][-1]) == check_sign([0, 1, 4],
            ((4, 3), (0, 1, 2, 6)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][2][8][-1]) == check_sign([0, 1, 4],
            ((4, 6), (0, 1, 2, 3)),)
    assert base_sign * scale_sign(test.exop_combinations[(0, 1, 4, 2, 3, 6)][1][0][1]) == check_sign([0, 1, 4],
            ((0, 1, 4, 2, 3, 6), ),)
