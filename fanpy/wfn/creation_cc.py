import numpy as np

from fanpy.wfn.base import BaseWavefunction
from fanpy.tools import math_tools, slater

class CreationCC(BaseWavefunction):
    def __init__(self, nelec, nspin, memory=None, orbpairs=None, params=None):
        """ Initialize the wavefunction
        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            If number is provided, it is the number of bytes.
            If string is provided, it should end iwth either "mb" or "gb" to specify the units.
            Default does not limit memory usage (i.e. infinite).
        orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct the exponential geminal.
        params : np.ndarray
            Coefficients.
        """
        
        super().__init__(nelec, nspin, memory=memory)
        self.assign_orbpairs(orbpairs=orbpairs)
        self.assign_params(params=params)
        
    def assign_nelec(self, nelec: int):
        """Assign the number of electrons.

        Parameters
        ----------
        nelec : int
            Number of electrons.

        Raises
        ------
        TypeError
            If number of electrons is not an integer.
        ValueError
            If number of electrons is not a positive number.
        NotImplementedError
            If number of electrons is odd.

        """
        
        super().assign_nelec(nelec)
        if self.nelec % 2 != 0:
            raise NotImplementedError("Odd number of electrons is not supported.")
        
    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the creation cc wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of creation cc. Default: None
        add_noise : bool
            Option to add noise to the given parameters. Default: False
        """
        
        elec_pairs = int(self.nelec / 2)
        if params is None:
            n_spin = self.nspatial * 2
            number_params = int(n_spin*(n_spin-1)/2) 
            params = np.zeros(number_params)
        for i in range(elec_pairs):
            col_ind = self.get_col_ind((i*self.nspatial, i*self.nspatial + 1))
            params[col_ind] = 1
        super().assign_params(params=params, add_noise=add_noise)
    
    def assign_orbpairs(self, orbpairs=None):
        """Assign the orbital pairs.

        Parameters
        ----------
        orbpairs : iterable of 2-tuple/list of ints
            Indices of the orbital pairs that will be used to construct the wavefunction.
            Default is all possible orbital pairs.

        Raises
        ------
        TypeError
            If `orbpairs` is not an iterable.
            If an orbital pair is not given as a 2-tuple/list of integers.
        ValueError
            If an orbital pair has the same integer.
            If an orbital pair occurs more than once.

        Notes
        -----
        Must have `nspin` defined for the default option.

        """
        if orbpairs is None:
            orbpairs = tuple((i, j) for i in range(self.nspin) for j in range(i + 1, self.nspin))

        if __debug__ and not hasattr(orbpairs, "__iter__"):
            raise TypeError("`orbpairs` must iterable.")
        dict_orbpair_ind = {}
        for i, orbpair in enumerate(orbpairs):
            if __debug__:
                if not (
                    isinstance(orbpair, (list, tuple))
                    and len(orbpair) == 2
                    and all(isinstance(ind, int) for ind in orbpair)
                ):
                    raise TypeError("Each orbital pair must be a 2-tuple/list of integers.")
                if orbpair[0] == orbpair[1]:
                    raise ValueError("Orbital pair of the same orbital is invalid")

            orbpair = tuple(orbpair)
            # sort orbitals within the pair
            if orbpair[0] > orbpair[1]:
                orbpair = orbpair[::-1]
            if __debug__ and orbpair in dict_orbpair_ind:
                raise ValueError(
                    "The given orbital pairs have multiple entries of {0}.".format(orbpair)
                )
            dict_orbpair_ind[orbpair] = i

        self.dict_orbpair_ind = dict_orbpair_ind
        self.dict_ind_orbpair = {i: orbpair for orbpair, i in dict_orbpair_ind.items()}
        
    def get_col_ind(self, orbpair: tuple[int]):
        """Get the column index that corresponds to the given orbital pair.

        Parameters
        ----------
        orbpair : 2-tuple of int
            Indices of the orbital pair. 

        Returns
        -------
        col_ind : int
            Column index that corresponds to the given orbital pair.

        Raises
        ------
        ValueError
            If given orbital pair is not valid.

        """
        try:
            return self.dict_orbpair_ind[orbpair]
        except (KeyError, TypeError):
            raise ValueError(
                f"Given orbital pair, {orbpair}, is not included in the wavefunction."
            )
    
    def _olp(self, sd: int):
        """ Calculate overlap with Slater determinant. Only 4 e- supported.
        
        Parameters
        ----------
        sd : int 
            Occupation vector of a Slater determinant given as a bitstring.
        
        Returns
        -------
        olp : float
            Overlap of the Slater determinant with the exponential geminal.
            
        """
        
        olp = 0
        occ_indices = slater.occ_indices(sd)
        param_indices = np.zeros((4,4), dtype=int)
        for i in range(4):
            for j in range(i+1, 4):
                orbpair = (occ_indices[i], occ_indices[j])
                param_indices[i, j] = int(self.get_col_ind(orbpair))
        olp += self.params[param_indices[0, 1]]*self.params[param_indices[2, 3]]
        olp -= self.params[param_indices[0, 2]]*self.params[param_indices[1, 3]]
        olp += self.params[param_indices[0, 3]]*self.params[param_indices[1, 2]]
        return olp
    
    def get_possible_pairs(self, indices: list[int]):
        """ Generate possible orbital pairs that can construct a given Slater determinant.
        
        Parameters
        ----------
        indices : tuple/list of int
            Indices for which 
            
        Returns
        -------
        pairs : list of ints 
            List of all possible orbital pairs.
        """
        
        pairs = []
        num_indices = len(indices)
        for i in range(num_indices):
            for j in range(i+1, num_indices):
                pairs.append((indices[i], indices[j]))
        return pairs
    
    def _olp_deriv(self, sd: int):
        """ Calculate the derivative of the overlap. Only 4 e- supported.
        
        Parameters
        ----------
        sd : int 
            Occupation vector of a Slater determinant given as a bitstring.
        
        Returns
        -------
        output : np.ndarray
            Overlap of the Slater determinant with the exponential geminal.
        
        """
        
        occ_indices = slater.occ_indices(sd)
        output = np.zeros(len(self.params))
        param_indices = np.zeros((4,4), dtype=int)
        for i in range(4):
            for j in range(i+1, 4):
                orbpair = (occ_indices[i], occ_indices[j])
                param_indices[i, j] = int(self.get_col_ind(orbpair))
        output[param_indices[0,1]] = self.params[param_indices[2,3]]
        output[param_indices[2,3]] = self.params[param_indices[0,1]]
        output[param_indices[0,2]] = - self.params[param_indices[1,3]]
        output[param_indices[1,3]] = - self.params[param_indices[0,2]]
        output[param_indices[0,3]] = self.params[param_indices[1,2]]
        output[param_indices[1,2]] = self.params[param_indices[0,3]]
        return output
        
    
    def get_overlap(self, sd: int, deriv=None):
        """Return the overlap of the wavefunction with a Slater determinant.
        Inlcude math later. Currently only 4 e- supported. 
        
        Parameters
        ----------
        sd : int 
            Occupation vector of a Slater determinant given as a bitstring.
        deriv : I am confused about this 
            whether to calculate the derivative or not. Default: None 
            currently it can only calculate derivative w.r.t. all params
            
        Returns
        -------
        overlap : {float, np.ndarray}
            Overlap (or derivative of the overlap) of the wavefunction with the given Slater
            determinant.
        """
        
        if deriv is None:
            olp = 0
            occ_indices = slater.occ_indices(sd)
            return self._olp(sd)
        else:
            return self._olp_deriv(sd)[deriv]

