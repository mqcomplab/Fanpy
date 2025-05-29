"""Test AntisymmeterizedProductTetrets."""
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.solver.system import least_squares
from fanpy.wfn.quasiparticle.tetret import AntisymmeterizedProductTetrets

import numpy as np
from numpy.testing import assert_raises

from utils import find_datafile


class TempAntisymmeterizedProductTetrets(AntisymmeterizedProductTetrets):
    """AntisymmeterizedProductTetrets that skips initialization."""

    def __init__(self):
        """Initialize."""
        pass


def test_assign_nelec():
    """Test AntisymeterizedProductTetrets.assign_nelec."""
    test = TempAntisymmeterizedProductTetrets()
    test.assign_nelec(4)
    assert test.nelec == 4
    test.assign_nelec(8)
    assert test.nelec == 8
    assert_raises(ValueError, test.assign_nelec, 5)


def test_assign_nquasiparticle():
    """Test AntisymeterizedProductTetrets.assign_nquasiparticle."""
    test = TempAntisymmeterizedProductTetrets()
    test.nelec = 4
    test.assign_nquasiparticle()
    assert test.nquasiparticle == 1
    test.assign_nquasiparticle(1)
    assert test.nquasiparticle == 1
    assert_raises(ValueError, test.assign_nquasiparticle, 2)


def test_asign_orbsubsets():
    """Test AntisymeterizedProductTetrets.assign_orbsubsets."""
    test = TempAntisymmeterizedProductTetrets()
    test.nspin = 4
    test.assign_orbsubsets()
    assert test.dict_orbsubset_ind == {(0, 1, 2, 3): 0}
    assert test.dict_ind_orbsubset == {0: (0, 1, 2, 3)}
    assert test.orbsubset_sizes == (4,)

    test.nspin = 6
    test.assign_orbsubsets()
    assert test.dict_orbsubset_ind == {
        (0, 1, 2, 3): 0,
        (0, 1, 2, 4): 1,
        (0, 1, 2, 5): 2,
        (0, 1, 3, 4): 3,
        (0, 1, 3, 5): 4,
        (0, 1, 4, 5): 5,
        (0, 2, 3, 4): 6,
        (0, 2, 3, 5): 7,
        (0, 2, 4, 5): 8,
        (0, 3, 4, 5): 9,
        (1, 2, 3, 4): 10,
        (1, 2, 3, 5): 11,
        (1, 2, 4, 5): 12,
        (1, 3, 4, 5): 13,
        (2, 3, 4, 5): 14,
    }
    assert test.dict_ind_orbsubset == {
        0: (0, 1, 2, 3),
        1: (0, 1, 2, 4),
        2: (0, 1, 2, 5),
        3: (0, 1, 3, 4),
        4: (0, 1, 3, 5),
        5: (0, 1, 4, 5),
        6: (0, 2, 3, 4),
        7: (0, 2, 3, 5),
        8: (0, 2, 4, 5),
        9: (0, 3, 4, 5),
        10: (1, 2, 3, 4),
        11: (1, 2, 3, 5),
        12: (1, 2, 4, 5),
        13: (1, 3, 4, 5),
        14: (2, 3, 4, 5),
    }
    assert test.orbsubset_sizes == (4,)

    assert_raises(
        NotImplementedError, test.assign_orbsubsets, [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]
    )
    # test.assign_orbsubsets([(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)])
    # assert test.dict_orbsubset_ind == {(0, 1, 2, 3): 0, (1, 2, 3, 4): 1, (2, 3, 4, 5): 2}
    # assert test.dict_orbsubset_ind == {0: (0, 1, 2, 3), 1: (1, 2, 3, 4), 2: (2, 3, 4, 5)}
    # assert test.orbsubset_sizes == (4, )

    # assert_raises(ValueError,  test.assign_orbsubsets, [(0, 1, 2, 3), (4, 5)])


def test_generate_possible_orbsubsets():
    """Test AntisymeterizedProductTetrets.generate_possible_orbsubsets."""
    test = TempAntisymmeterizedProductTetrets()
    assert_raises(ValueError, next, test.generate_possible_orbsubsets([0, 1, 2, 3, 4]))

    answer = [[(0, 1, 2, 3)]]
    for i, orbsubset in enumerate(test.generate_possible_orbsubsets([0, 1, 2, 3])):
        assert orbsubset == answer[i]

    answer = [[(0, 1, 2, 3, 4, 5, 6, 7, 8)]]
    answer = [
        # swap nothing
        [(0, 1, 2, 3), (4, 5, 6, 7)],
        # swap 3
        [(0, 1, 2, 4), (3, 5, 6, 7)],
        [(0, 1, 2, 5), (3, 4, 6, 7)],
        [(0, 1, 2, 6), (3, 4, 5, 7)],
        [(0, 1, 2, 7), (3, 4, 5, 6)],
        # swap 2
        [(0, 1, 3, 4), (2, 5, 6, 7)],
        [(0, 1, 3, 5), (2, 4, 6, 7)],
        [(0, 1, 3, 6), (2, 4, 5, 7)],
        [(0, 1, 3, 7), (2, 4, 5, 6)],
        # swap 2 and 3
        [(0, 1, 4, 5), (2, 3, 6, 7)],
        [(0, 1, 4, 6), (2, 3, 5, 7)],
        [(0, 1, 4, 7), (2, 3, 5, 6)],
        [(0, 1, 5, 6), (2, 3, 4, 7)],
        [(0, 1, 5, 7), (2, 3, 4, 6)],
        [(0, 1, 6, 7), (2, 3, 4, 5)],
        # swap 1
        [(0, 2, 3, 4), (1, 5, 6, 7)],
        [(0, 2, 3, 5), (1, 4, 6, 7)],
        [(0, 2, 3, 6), (1, 4, 5, 7)],
        [(0, 2, 3, 7), (1, 4, 5, 6)],
        # swap 1 and 3
        [(0, 2, 4, 5), (1, 3, 6, 7)],
        [(0, 2, 4, 6), (1, 3, 5, 7)],
        [(0, 2, 4, 7), (1, 3, 5, 6)],
        [(0, 2, 5, 6), (1, 3, 4, 7)],
        [(0, 2, 5, 7), (1, 3, 4, 6)],
        [(0, 2, 6, 7), (1, 3, 4, 5)],
        # swap 1 and 2
        [(0, 3, 4, 5), (1, 2, 6, 7)],
        [(0, 3, 4, 6), (1, 2, 5, 7)],
        [(0, 3, 4, 7), (1, 2, 5, 6)],
        [(0, 3, 5, 6), (1, 2, 4, 7)],
        [(0, 3, 5, 7), (1, 2, 4, 6)],
        [(0, 3, 6, 7), (1, 2, 4, 5)],
        # swap 1, 2, and 3
        [(0, 4, 5, 6), (1, 2, 3, 7)],
        [(0, 4, 5, 7), (1, 2, 3, 6)],
        [(0, 4, 6, 7), (1, 2, 3, 5)],
        # swap 1, 2, 3, and 4
        [(0, 5, 6, 7), (1, 2, 3, 4)],
    ]
    for orbsubset in test.generate_possible_orbsubsets([0, 1, 2, 3, 4, 5, 6, 7]):
        assert orbsubset in answer


# FIXME: check properly
def test_run():
    """Check if calculation runs with APT."""
    one_int = np.load(find_datafile("data/data_h4_square_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data/data_h4_square_hf_sto6g_twoint.npy"))
    ham = RestrictedMolecularHamiltonian(one_int, two_int)
    wfn = AntisymmeterizedProductTetrets(4, 8, nquasiparticle=1)

    # Solve system of equations
    objective = ProjectedSchrodinger(wfn, ham)
    results = least_squares(objective)
    print(results)
