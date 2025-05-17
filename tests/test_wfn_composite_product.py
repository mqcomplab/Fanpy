"""Test fanpy.wavefunction.composite.product."""


from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.composite.product import ProductWavefunction
from fanpy.tools.sd_list import sd_list

import numpy as np

import pytest


def test_init():
    """Test ProductWavefunction.assign_wfns."""
    wfn1 = CIWavefunction(4, 10)
    wfn2 = CIWavefunction(4, 10)
    wfn = ProductWavefunction([wfn1, wfn2])

    assert wfn.nelec == 4
    assert wfn.nspin == 10
    assert wfn.wfns == [wfn1, wfn2]

    with pytest.raises(TypeError):
        wfn = ProductWavefunction([wfn1, 2])

    wfn1.nelec = 3
    with pytest.raises(ValueError):
        wfn = ProductWavefunction([wfn1, wfn2])

    with pytest.raises(ValueError):
        wfn = ProductWavefunction([wfn1, wfn1])

@pytest.mark.skip(reason="This test fails and is being worked on (Issue 55).")
def test_get_overlap():
    """Test ProductWavefunction.get_overlap."""
    wfn1 = CIWavefunction(4, 10)
    wfn1.assign_params(np.random.rand(wfn1.nparams))
    wfn2 = CIWavefunction(4, 10)
    wfn2.assign_params(np.random.rand(wfn2.nparams))

    test = ProductWavefunction((wfn1, wfn2))
    for sd in wfn1.sds:
        assert test.get_overlap(sd) == (wfn1.params * wfn2.params)[wfn1.dict_sd_index[sd]]

    with pytest.raises(TypeError):
        test.get_overlap(0b0011001010, deriv=np.array([0]))
    with pytest.raises(TypeError):
        test.get_overlap(0b0011001010, deriv=(test.wfns[0], np.array([0.0])))
    with pytest.raises(TypeError):
        test.get_overlap(0b0011001010, deriv=(test.wfns[0], np.array([[0]])))
    with pytest.raises(TypeError):
        test.get_overlap(0b0011001010, deriv=(test.wfns[0], np.array([[0]])))
    with pytest.raises(ValueError):
        test.get_overlap(0b0011001010, deriv=(CIWavefunction(2, 4), np.array([0])))

    assert np.allclose(
        test.get_overlap(0b0001100011, deriv=(test.wfns[0], np.array([0, 1]))),
        np.array([wfn2.get_overlap(0b0001100011), 0]) * np.array([1]),
    )
    assert np.allclose(
        test.get_overlap(0b0001100011, deriv=(test.wfns[1], np.array([0, 1]))),
        np.array([wfn1.get_overlap(0b0001100011), 0]) * np.array([1]),
    )


def test_get_overlaps():
    """Test ProductWavefunction.get_overlaps."""
    wfn1 = CIWavefunction(4, 10)
    wfn1.assign_params(np.random.rand(wfn1.nparams))
    wfn2 = CIWavefunction(4, 10)
    wfn2.assign_params(np.random.rand(wfn2.nparams))

    test = ProductWavefunction((wfn1, wfn2))
    assert np.allclose(test.get_overlaps(wfn1.sds), wfn1.params * wfn2.params)

    assert np.allclose(
        test.get_overlaps([wfn1.sds[0], wfn1.sds[1]], deriv=(test.wfns[0], np.array([0, 1]))),
        np.array([[wfn2.get_overlap(wfn1.sds[0]), 0], [0, wfn2.get_overlap(wfn1.sds[1])]]),
    )
    assert np.allclose(
        test.get_overlaps([wfn1.sds[0], wfn1.sds[1]], deriv=(test.wfns[1], np.array([0, 1]))),
        np.array([[wfn1.get_overlap(wfn2.sds[0]), 0], [0, wfn1.get_overlap(wfn2.sds[1])]]),
    )

@pytest.mark.skip(reason="This test fails and is being worked on (Issue 30).")
def test_get_overlaps_rbm_ap1rog():
    """Test ProductWavefunction.get_overlaps using RBM and AP1roG."""
    from fanpy.wfn.geminal.ap1rog import AP1roG
    from fanpy.wfn.network.rbm import RestrictedBoltzmannMachine

    ap1rog = AP1roG(4, 10, params=None, memory=None, ref_sd=None, ngem=None)
    ap1rog.assign_params(ap1rog.params + 2 * (np.random.rand(*ap1rog.params.shape) - 0.5))
    rbm = RestrictedBoltzmannMachine(4, 10, nbath=10, num_layers=1, orders=(1, 2))
    rbm.assign_params(rbm.params + 2 * (np.random.rand(*rbm.params.shape) - 0.5))

    test = ProductWavefunction((ap1rog, rbm))
    sds = sd_list(4, 10)
    assert np.allclose(
        test.get_overlaps(sds), [ap1rog.get_overlap(sd) * rbm.get_overlap(sd) for sd in sds]
    )
    assert np.allclose(
        test.get_overlaps(sds, deriv=(ap1rog, np.arange(ap1rog.nparams))),
        [ap1rog.get_overlap(sd, deriv=np.arange(ap1rog.nparams)) * rbm.get_overlap(sd) for sd in sds]
    )
    assert np.allclose(
        test.get_overlaps(sds, deriv=(rbm, np.arange(rbm.nparams))),
        [ap1rog.get_overlap(sd) * rbm.get_overlap(sd, deriv=np.arange(rbm.nparams)) for sd in sds]
    )
    deriv_ap1rog = np.random.choice(np.arange(ap1rog.nparams), ap1rog.nparams // 2, replace=False)
    assert np.allclose(
        test.get_overlaps(sds, deriv=(ap1rog, deriv_ap1rog)),
        np.array([ap1rog.get_overlap(sd, deriv=deriv_ap1rog) * rbm.get_overlap(sd) for sd in sds])
    )
    deriv_rbm = np.random.choice(np.arange(rbm.nparams), rbm.nparams // 2)
    assert np.allclose(
        test.get_overlaps(sds, deriv=(rbm, deriv_rbm)),
        [rbm.get_overlap(sd, deriv=deriv_rbm) * ap1rog.get_overlap(sd) for sd in sds]
    )
