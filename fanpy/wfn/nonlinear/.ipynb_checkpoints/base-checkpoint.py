"""Parent class of CI wavefunctions."""
import itertools

from fanpy.tools import slater
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.ci import FCI

import numpy as np

class NonlinearWavefunction(FCI):




    def assign_params(self, params=None, add_noise=False):

        if params is None:
            params = np.zeros(len(self.nspin)) 
            ground_state = slater.ground(self.nelec, self.nspin)
            occs = slater.occ_indices(ground_state)
            params[occs] = 4
        else:
            if params.shape != self.nspin:
                raise ValueError('The number of the parameters must be equal to the number of spin orbitals.')  
        super().assign_params(params=params, add_noise=add_noise)




