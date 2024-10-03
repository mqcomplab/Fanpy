from fanpy.wfn.creation_cc import CreationCC 
import numpy as np
from fanpy.tools import slater

def hardcoded_six_electron_olp(sd, wfn):
    occ_indices = slater.occ_indices(sd)
    num_occ = len(occ_indices)
    param_indices = np.zeros((num_occ, num_occ), dtype=int)
    for i in range(num_occ):
        for j in range(i+1, num_occ):
            orbpair = (occ_indices[i], occ_indices[j])
            param_indices[i, j] = int(wfn.get_col_ind(orbpair))
    wfn_params = wfn.params
    olp = 0
    olp += wfn_params[param_indices[0, 1]] * wfn_params[param_indices[2, 3]]* wfn_params[param_indices[4, 5]]
    olp -= wfn_params[param_indices[0, 1]] * wfn_params[param_indices[2, 4]]* wfn_params[param_indices[3, 5]]
    olp += wfn_params[param_indices[0, 1]] * wfn_params[param_indices[2, 5]]* wfn_params[param_indices[3, 4]]
    olp -= wfn_params[param_indices[0, 2]] * wfn_params[param_indices[1, 3]]* wfn_params[param_indices[4, 5]]
    olp += wfn_params[param_indices[0,2]] * wfn_params[param_indices[1, 4]]* wfn_params[param_indices[3, 5]]
    olp -= wfn_params[param_indices[0,2]] * wfn_params[param_indices[1, 5]]* wfn_params[param_indices[3, 4]]
    olp += wfn_params[param_indices[0,3]] * wfn_params[param_indices[1, 2]]* wfn_params[param_indices[4, 5]]
    olp -= wfn_params[param_indices[0,3]] * wfn_params[param_indices[1, 4]]* wfn_params[param_indices[2, 5]]
    olp += wfn_params[param_indices[0,3]] * wfn_params[param_indices[1, 5]]* wfn_params[param_indices[2, 4]]
    olp -= wfn_params[param_indices[0,4]] * wfn_params[param_indices[1, 2]]* wfn_params[param_indices[3, 5]]
    olp += wfn_params[param_indices[0,4]] * wfn_params[param_indices[1, 3]]* wfn_params[param_indices[2, 5]]
    olp -= wfn_params[param_indices[0,4]] * wfn_params[param_indices[1, 5]]* wfn_params[param_indices[2, 3]]
    olp += wfn_params[param_indices[0,5]] * wfn_params[param_indices[1, 2]]* wfn_params[param_indices[3, 4]]
    olp -= wfn_params[param_indices[0,5]] * wfn_params[param_indices[1, 3]]* wfn_params[param_indices[2, 4]]
    olp += wfn_params[param_indices[0,5]] * wfn_params[param_indices[1, 4]]* wfn_params[param_indices[2, 3]]
    return olp 


def test_creationcc_olp():
    nocc = 6
    norb = 14
    wfn = CreationCC(nelec=nocc, nspin=norb)
    wfn.params = np.random.normal(size=wfn.nparams)
    sd = 851 # 1101010011
    olp = wfn.get_overlap(sd)
    olp_hardcoded = hardcoded_six_electron_olp(sd, wfn)
    assert np.allclose(olp, olp_hardcoded)