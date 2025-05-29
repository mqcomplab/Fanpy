"""Test fanpy.wavefunction.composite.embedding."""
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.composite.embedding import EmbeddedWavefunction
from fanpy.tools.sd_list import sd_list

import numpy as np

import pytest


def test_init():
    """Test EmbeddedWavefunction.assign_wfns."""
    wfn1 = CIWavefunction(2, 6)
    wfn2 = CIWavefunction(2, 6)
    wfn = EmbeddedWavefunction(4, 12, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2])
    assert wfn.nelec == 4
    assert wfn.nspin == 12
    assert wfn.wfns == [wfn1, wfn2]
    assert wfn.indices_list == [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]]
    assert wfn.dict_system_sub == {0: [(0, 0)], 2: [(0, 1)], 4: [(0, 2)],
                                   6: [(0, 3)], 8: [(0, 4)], 10: [(0, 5)],
                                   1: [(1, 0)], 3: [(1, 1)], 5: [(1, 2)],
                                   7: [(1, 3)], 9: [(1, 4)], 11: [(1, 5)]}
    assert wfn.dict_sub_system == {(0, 0): 0, (0, 1): 2, (0, 2): 4,
                                   (0, 3): 6, (0, 4): 8, (0, 5): 10,
                                   (1, 0): 1, (1, 1): 3, (1, 2): 5,
                                   (1, 3): 7, (1, 4): 9, (1, 5): 11}

    with pytest.raises(ValueError):
        wfn = EmbeddedWavefunction(4, 12, [[0, 0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, 2])
    with pytest.raises(TypeError):
        wfn = EmbeddedWavefunction(4, 12, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, 2])
    with pytest.raises(ValueError):
        wfn = EmbeddedWavefunction(4, 12, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1])
    with pytest.raises(ValueError):
        wfn1.nspin = 7
        wfn = EmbeddedWavefunction(4, 12, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2])
    with pytest.raises(ValueError):
        wfn1.nelec = 1
        wfn1.nspin = 6
        wfn = EmbeddedWavefunction(
            4, 12, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2], disjoint=True
        )
    with pytest.raises(ValueError):
        wfn1.nelec = 2
        wfn1.nspin = 5
        wfn = EmbeddedWavefunction(
            4, 12, [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2], disjoint=True
        )
    with pytest.raises(ValueError):
        wfn1.nelec = 2
        wfn1.nspin = 7
        wfn = EmbeddedWavefunction(
            4, 12, [[0, 1, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2], disjoint=True
        )

    wfn1 = CIWavefunction(2, 6)
    wfn = EmbeddedWavefunction(4, 12, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn1])
    assert wfn.nelec == 4
    assert wfn.nspin == 12
    assert wfn.wfns == [wfn1, wfn1]
    assert wfn.indices_list == [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]]
    assert wfn.dict_system_sub == {0: [(0, 0)], 2: [(0, 1)], 4: [(0, 2)],
                                   6: [(0, 3)], 8: [(0, 4)], 10: [(0, 5)],
                                   1: [(1, 0)], 3: [(1, 1)], 5: [(1, 2)],
                                   7: [(1, 3)], 9: [(1, 4)], 11: [(1, 5)]}
    assert wfn.dict_sub_system == {(0, 0): 0, (0, 1): 2, (0, 2): 4,
                                   (0, 3): 6, (0, 4): 8, (0, 5): 10,
                                   (1, 0): 1, (1, 1): 3, (1, 2): 5,
                                   (1, 3): 7, (1, 4): 9, (1, 5): 11}

    wfn1 = CIWavefunction(2, 8)
    wfn2 = CIWavefunction(2, 6)
    with pytest.raises(ValueError):
        wfn = EmbeddedWavefunction(4, 12, [[0, 1, 2, 3, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2],
                                   disjoint=False)
    with pytest.raises(ValueError):
        wfn = EmbeddedWavefunction(4, 12, [[0, 1, 2, 4, 5, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2],
                                   disjoint=False)
    wfn = EmbeddedWavefunction(4, 12, [[0, 1, 2, 4, 6, 7, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2],
                               disjoint=False)
    assert wfn.nelec == 4
    assert wfn.nspin == 12
    assert wfn.wfns == [wfn1, wfn2]
    assert wfn.indices_list == [[0, 1, 2, 4, 6, 7, 8, 10], [1, 3, 5, 7, 9, 11]]
    assert wfn.dict_system_sub == {0: [(0, 0)], 1: [(0, 1), (1, 0)], 2: [(0, 2)], 4: [(0, 3)],
                                   6: [(0, 4)], 7: [(0, 5), (1, 3)], 8: [(0, 6)], 10: [(0, 7)],
                                   3: [(1, 1)], 5: [(1, 2)],
                                   9: [(1, 4)], 11: [(1, 5)]}
    assert wfn.dict_sub_system == {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 4,
                                   (0, 4): 6, (0, 5): 7, (0, 6): 8, (0, 7): 10,
                                   (1, 0): 1, (1, 1): 3, (1, 2): 5,
                                   (1, 3): 7, (1, 4): 9, (1, 5): 11}


def test_num_systems():
    """Test EmbeddedWavefunction.num_systems."""
    wfn1 = CIWavefunction(2, 6)
    wfn2 = CIWavefunction(2, 6)
    wfn = EmbeddedWavefunction(4, 12, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2])
    assert wfn.num_systems == 2
    wfn = EmbeddedWavefunction(4, 12, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]] * 2,
                               [wfn1, wfn1, wfn1, wfn1], disjoint=False)
    assert wfn.num_systems == 4


def test_split_sd():
    """Test EmbeddedWavefunction.split_sd."""
    wfn1 = CIWavefunction(2, 6)
    wfn2 = CIWavefunction(2, 6)
    wfn = EmbeddedWavefunction(4, 12, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2])
    assert wfn.split_sd(0b000011000011) == [0b001001, 0b001001]
    wfn1 = CIWavefunction(3, 8)
    wfn = EmbeddedWavefunction(4, 12, [[0, 1, 2, 4, 6, 7, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2],
                               disjoint=False)
    assert wfn.split_sd(0b000011000011) == [0b00110011, 0b001001]


def test_get_overlap():
    """Test EmbeddedWavefunction.get_overlap."""
    wfn1 = CIWavefunction(2, 6)
    wfn1.assign_params(np.random.rand(wfn1.nparams))
    wfn2 = CIWavefunction(2, 6)
    wfn2.assign_params(np.random.rand(wfn2.nparams))

    test = EmbeddedWavefunction(4, 12, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]],
                                [wfn1, wfn2], disjoint=False)
    for sd in sd_list(4, 12):
        sd1, sd2 = test.split_sd(sd)
        olp1 = wfn1.params[wfn1.dict_sd_index[sd1]] if sd1 in wfn1.dict_sd_index else 0
        olp2 = wfn2.params[wfn2.dict_sd_index[sd2]] if sd2 in wfn2.dict_sd_index else 0
        assert test.get_overlap(sd) == olp1 * olp2

    assert np.allclose(
        test.get_overlap(0b000011000011, deriv=(test.wfns[0], np.array([0, 1]))),
        np.array([wfn2.get_overlap(0b001001), 0]) * np.array([1]),
    )
    assert np.allclose(
        test.get_overlap(0b000011000011, deriv=(test.wfns[1], np.array([0, 1]))),
        np.array([wfn1.get_overlap(0b001001), 0]) * np.array([1]),
    )

    wfn1 = CIWavefunction(4, 8)
    wfn1.assign_params(np.random.rand(wfn1.nparams))
    wfn2 = CIWavefunction(2, 6)
    wfn2.assign_params(np.random.rand(wfn2.nparams))

    test = EmbeddedWavefunction(4, 12, [[0, 1, 2, 4, 6, 7, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2],
                                disjoint=False)
    for sd in sd_list(4, 12):
        sd1, sd2 = test.split_sd(sd)
        olp1 = wfn1.params[wfn1.dict_sd_index[sd1]] if sd1 in wfn1.dict_sd_index else 0
        olp2 = wfn2.params[wfn2.dict_sd_index[sd2]] if sd2 in wfn2.dict_sd_index else 0
        assert test.get_overlap(sd) == olp1 * olp2

    with pytest.raises(TypeError):
        test.get_overlap(0b000110001010, deriv=np.array([0]))
    with pytest.raises(ValueError):
        test.get_overlap(0b000110001010, deriv=(CIWavefunction(2, 4), np.array([0])))

    sd1, sd2 = test.split_sd(0b000011000011)
    assert np.allclose(
        test.get_overlap(0b000011000011, deriv=(test.wfns[0], np.array([0, 1]))),
        np.array([wfn2.get_overlap(0b001001), 0]) * np.array([1]),
    )
    assert np.allclose(
        test.get_overlap(0b000011000011, deriv=(test.wfns[1], np.array([0, 1]))),
        np.array([wfn1.get_overlap(0b00110011), 0]) * np.array([1]),
    )


def test_get_overlaps():
    """Test EmbeddedWavefunction.get_overlaps."""
    wfn1 = CIWavefunction(2, 6)
    wfn1.assign_params(np.random.rand(wfn1.nparams))
    wfn2 = CIWavefunction(2, 6)
    wfn2.assign_params(np.random.rand(wfn2.nparams))

    test = EmbeddedWavefunction(4, 12, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]],
                                [wfn1, wfn2], disjoint=False)
    for answer, sd in zip(test.get_overlaps(sd_list(4, 12)), sd_list(4, 12)):
        sd1, sd2 = test.split_sd(sd)
        olp1 = wfn1.params[wfn1.dict_sd_index[sd1]] if sd1 in wfn1.dict_sd_index else 0
        olp2 = wfn2.params[wfn2.dict_sd_index[sd2]] if sd2 in wfn2.dict_sd_index else 0
        assert np.allclose(answer, olp1 * olp2)

    assert np.allclose(
        test.get_overlaps(sd_list(4, 12), deriv=(test.wfns[0], np.arange(test.wfns[0].nparams))),
        [[test.get_overlap(sd, deriv=(test.wfns[0], i)) for i in range(test.wfns[0].nparams)]
         for sd in sd_list(4, 12)],
    )
    assert np.allclose(
        test.get_overlaps(sd_list(4, 12), deriv=(test.wfns[1], np.arange(test.wfns[1].nparams))),
        [[test.get_overlap(sd, deriv=(test.wfns[1], i)) for i in range(test.wfns[1].nparams)]
         for sd in sd_list(4, 12)],
    )

    wfn1 = CIWavefunction(4, 8)
    wfn1.assign_params(np.random.rand(wfn1.nparams))
    wfn2 = CIWavefunction(2, 6)
    wfn2.assign_params(np.random.rand(wfn2.nparams))
    test = EmbeddedWavefunction(4, 12, [[0, 1, 2, 4, 6, 7, 8, 10], [1, 3, 5, 7, 9, 11]], [wfn1, wfn2],
                                disjoint=False)
    for answer, sd in zip(test.get_overlaps(sd_list(4, 12)), sd_list(4, 12)):
        sd1, sd2 = test.split_sd(sd)
        olp1 = wfn1.params[wfn1.dict_sd_index[sd1]] if sd1 in wfn1.dict_sd_index else 0
        olp2 = wfn2.params[wfn2.dict_sd_index[sd2]] if sd2 in wfn2.dict_sd_index else 0
        assert np.allclose(answer, olp1 * olp2)

    assert np.allclose(
        test.get_overlaps(sd_list(4, 12), deriv=(test.wfns[0], np.arange(test.wfns[0].nparams))),
        [[test.get_overlap(sd, deriv=(test.wfns[0], i)) for i in range(test.wfns[0].nparams)]
         for sd in sd_list(4, 12)],
    )
    assert np.allclose(
        test.get_overlaps(sd_list(4, 12), deriv=(test.wfns[1], np.arange(test.wfns[1].nparams))),
        [[test.get_overlap(sd, deriv=(test.wfns[1], i)) for i in range(test.wfns[1].nparams)]
         for sd in sd_list(4, 12)],
    )

@pytest.mark.skip(reason="This test fails and is being worked on (Issue 30).")
def test_get_overlaps_rbm_ap1rog():
    """Test EmbeddedWavefunction.get_overlaps using RBM and AP1roG."""
    from fanpy.wfn.geminal.ap1rog import AP1roG
    from fanpy.wfn.network.rbm import RestrictedBoltzmannMachine

    ap1rog = AP1roG(2, 6, params=None, memory=None, ref_sd=None, ngem=None)
    ap1rog.assign_params(ap1rog.params + 2 * (np.random.rand(*ap1rog.params.shape) - 0.5))
    rbm = RestrictedBoltzmannMachine(2, 6, nbath=6, num_layers=1, orders=(1, 2))
    rbm.assign_params(rbm.params + 2 * (np.random.rand(*rbm.params.shape) - 0.5))

    test = EmbeddedWavefunction(4, 12, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]],
                                [ap1rog, rbm], disjoint=False)
    sds = sd_list(4, 12)
    assert np.allclose(
        test.get_overlaps(sds),
        [ap1rog.get_overlap(test.split_sd(sd)[0]) * rbm.get_overlap(test.split_sd(sd)[1])
         for sd in sds]
    )
    assert np.allclose(
        test.get_overlaps(sds, deriv=(ap1rog, np.arange(ap1rog.nparams))),
        [ap1rog.get_overlap(test.split_sd(sd)[0], deriv=np.arange(ap1rog.nparams)) *
         rbm.get_overlap(test.split_sd(sd)[1]) for sd in sds]
    )
    assert np.allclose(
        test.get_overlaps(sds, deriv=(rbm, np.arange(rbm.nparams))),
        [ap1rog.get_overlap(test.split_sd(sd)[0]) *
         rbm.get_overlap(test.split_sd(sd)[1], deriv=np.arange(rbm.nparams)) for sd in sds]
    )
    deriv_ap1rog = np.random.choice(np.arange(ap1rog.nparams), ap1rog.nparams // 2, replace=False)
    assert np.allclose(
        test.get_overlaps(sds, deriv=(ap1rog, deriv_ap1rog)),
        np.array([ap1rog.get_overlap(test.split_sd(sd)[0], deriv=deriv_ap1rog) * rbm.get_overlap(test.split_sd(sd)[1]) for sd in sds])
    )
    deriv_rbm = np.random.choice(np.arange(rbm.nparams), rbm.nparams // 2)
    assert np.allclose(
        test.get_overlaps(sds, deriv=(rbm, deriv_rbm)),
        [rbm.get_overlap(test.split_sd(sd)[1], deriv=deriv_rbm) * ap1rog.get_overlap(test.split_sd(sd)[0]) for sd in sds]
    )
