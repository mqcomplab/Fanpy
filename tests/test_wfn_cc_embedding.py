"""Test fanpy.wfn.cc.embedding."""
from fanpy.wfn.cc.embedding import EmbeddedCC
from fanpy.wfn.cc.standard_cc import StandardCC

import numpy as np


def test_embeddingcc_init():
    """Test EmbeddedCC.__init__."""
    test = EmbeddedCC([2, 2], [4, 4], [[0, 1, 2, 3], [4, 5, 6, 7]], [StandardCC, StandardCC])
    assert test.nelec == 4
    assert test.nspin == 8
    assert test.indices_list == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert test.dict_system_sub == {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3),
                                    4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (1, 3)}
    assert test.dict_sub_system == {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3,
                                    (1, 0): 4, (1, 1): 5, (1, 2): 6, (1, 3): 7}
    assert test.ranks == [1, 2]
    assert test.refwfn == 0b00110011
    assert test.refwfn_list == [0b0101, 0b0101]
    assert test.exops == {(0, 1): 0, (0, 3): 1, (2, 1): 2, (2, 3): 3, (0, 2, 1, 3): 4,
                          (4, 5): 5, (4, 7): 6, (6, 5): 7, (6, 7): 8, (4, 6, 5, 7): 9}
    assert np.allclose(test.params, np.zeros(test.nparams))

    test = EmbeddedCC([2, 2], [4, 4], [[0, 2, 4, 6], [1, 3, 5, 7]], [StandardCC, StandardCC])
    assert test.nelec == 4
    assert test.nspin == 8
    assert test.indices_list == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert test.dict_system_sub == {0: (0, 0), 2: (0, 1), 4: (0, 2), 6: (0, 3),
                                    1: (1, 0), 3: (1, 1), 5: (1, 2), 7: (1, 3)}
    assert test.dict_sub_system == {(0, 0): 0, (0, 1): 2, (0, 2): 4, (0, 3): 6,
                                    (1, 0): 1, (1, 1): 3, (1, 2): 5, (1, 3): 7}
    assert test.ranks == [1, 2]
    assert test.refwfn == 0b00110011
    assert test.refwfn_list == [0b0101, 0b0101]
    assert test.exops == {(0, 2): 0, (0, 6): 1, (4, 2): 2, (4, 6): 3, (0, 4, 2, 6): 4,
                          (1, 3): 5, (1, 7): 6, (5, 3): 7, (5, 7): 8, (1, 5, 3, 7): 9}
    assert np.allclose(test.params, np.zeros(test.nparams))

    params1 = np.random.rand(5)
    params2 = np.random.rand(4)
    params3 = np.random.rand(8)
    test = EmbeddedCC([2, 2], [4, 4], [[0, 2, 4, 6], [1, 3, 5, 7]], [StandardCC, StandardCC],
                      ranks_list=[2, 1], params_list=[params1, params2], inter_params=params3,
                      inter_exops=[(0, 3), (0, 7), (4, 3), (4, 7), (1, 2), (1, 6), (5, 2), (5, 6)])
    assert test.nelec == 4
    assert test.nspin == 8
    assert test.indices_list == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert test.dict_system_sub == {0: (0, 0), 2: (0, 1), 4: (0, 2), 6: (0, 3),
                                    1: (1, 0), 3: (1, 1), 5: (1, 2), 7: (1, 3)}
    assert test.dict_sub_system == {(0, 0): 0, (0, 1): 2, (0, 2): 4, (0, 3): 6,
                                    (1, 0): 1, (1, 1): 3, (1, 2): 5, (1, 3): 7}
    assert test.ranks == [1, 2]
    assert test.refwfn == 0b00110011
    assert test.refwfn_list == [0b0101, 0b0101]
    assert test.exops == {(0, 2): 0, (0, 6): 1, (4, 2): 2, (4, 6): 3, (0, 4, 2, 6): 4,
                          (1, 3): 5, (1, 7): 6, (5, 3): 7, (5, 7): 8,
                          (0, 3): 9, (0, 7): 10, (4, 3): 11, (4, 7): 12,
                          (1, 2): 13, (1, 6): 14, (5, 2): 15, (5, 6): 16}
    assert np.allclose(test.params, np.hstack([params1, params2, params3]))
