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
    exc_sds.extend(doubles)
    exc_sds.extend(singles)

    # generate excited sds from exc ops in wfn 
    wfn_exc_sds = []
    ground_state = ground(n_elec, n_spin)
    for exc_orders in wfn.exop_combinations:
        sd = excite(ground_state, *exc_orders)
        wfn_exc_sds.append(sd)

    # compare sds
    assert exc_sds.sort() == wfn_exc_sds.sort()


