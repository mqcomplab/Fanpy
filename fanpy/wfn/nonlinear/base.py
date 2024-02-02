"""Parent class of CI wavefunctions."""
import itertools

from fanpy.tools import slater
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.ci.fci import FCI
from fanpy.wfn.ci.base import CIWavefunction 

import numpy as np

class NonlinearWavefunction(FCI):




    def assign_params(self, params=None, add_noise=False):

        if params is None:
            params = np.zeros(self.nspin) 
            ground_state = slater.ground(self.nelec, self.nspin)
            occs = slater.occ_indices(ground_state)
            params[occs] = 4
        else:
            if params.shape != (self.nspin, ):
                raise ValueError('The number of the parameters must be equal to the number of spin orbitals.')  
        #super().assign_params(params=params, add_noise=add_noise)


    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the nonlinear wavefunction with a Slater determinant.

        The overlap of the nonlinear  wavefunction with a Slater determinant is the product of tanh of the parameters corresponding to the occupied orbitals of that
        Slater determinant in the wavefunction.

        .. math::

            \left< m \middle| \Psi \right> = \prod_i tanh(p_i)

        where

        .. math::

            \left| \Psi \right> = \sum_m (\prod_i tanh(p_i_m)) \left| m \right>
            \left| m \right> = \left \phi_1 \phi_2 ... \phi_N \right>

            Here, each orbital \phi_i of a Slater determinant |m> has one parameter p_i.

        Parameters
        ----------
        sd : int
            Slater Determinant against which the overlap is taken.
        deriv : {np.ndarray, None}
            Indices of the parameters with respect to which the overlap is derivatized.
            Default returns the overlap without derivatization.

        Returns
        -------
        overlap : {float, np.ndarray}
            Overlap (or derivative of the overlap) of the wavefunction with the given Slater
            determinant.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.

        """ 
        occ_indx = slater.occ_indices(sd)
        sd_params = np.array(self.params[occ_indx])
        if sd_params.size == 
        tanh_sd_params = np.tanh(sd_params)
        output = np.prod(tanh_sd_params)
        




