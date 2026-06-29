""" Test for AP1roGSD """

import numpy as np
import pytest
from fanpy.tools.sd_list import sd_list
from fanpy.tools.slater import excite, ground

from fanpy.wfn.cc.ap1rog_generalized import AP1roGSDGeneralized

def test_assign_ranks():
    # only default rank allowed
    with pytest.raises(TypeError):
        AP1roGSDGeneralized(2, 4, ranks=2)

    with pytest.raises(TypeError):
        AP1roGSDGeneralized(2, 4, ranks=[1, 2, 3])

    # AP1roGSD only has single and double excitations
    wfn = AP1roGSDGeneralized(2, 4)
    assert wfn.ranks == [1, 2]

def test_assign_expos():
    # only default exops allowed
    with pytest.raises(TypeError):
        AP1roGSDGeneralized(2, 4, indices=[[0, 2]])
    # build double excitations for PCCD:
    n_elec = 6
    n_spin = 14
    wfn = AP1roGSDGeneralized(n_elec, n_spin)

    # build single and double excitations manually
    doubles = sd_list(n_elec, n_spin, exc_orders=[2], spin=0, seniority=0)
    singles = sd_list(n_elec, n_spin, exc_orders=[1])
    exc_sds = []
    # sd list adds ground state as well because it satisfies the spin and sen restrictions
    # ground is added as the first element, we jump over it here
    exc_sds.extend(doubles[1:])
    exc_sds.extend(singles[1:])

    # generate excited sds from exc ops in wfn 
    wfn_exc_sds = []
    ground_state = ground(n_elec, n_spin)
    for exc_orders in wfn.exops.keys():
        sd = excite(ground_state, *exc_orders)
        wfn_exc_sds.append(sd)

    # convert arrays to numpy so that comparison easier
    exc_sds_np = np.asarray(exc_sds, dtype=int)
    exc_sds_np = np.sort(exc_sds_np)
    # sort arrays 
    wfn_exc_sds_np = np.asarray(wfn_exc_sds, dtype=int)
    wfn_exc_sds_np = np.sort(wfn_exc_sds_np)

    # compare sds
    np.testing.assert_array_equal(exc_sds_np, wfn_exc_sds_np)


