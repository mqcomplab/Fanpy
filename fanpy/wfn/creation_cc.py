import numpy as np
from itertools import combinations

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
            Indices of the orbital pairs that will be used to construct creation cc.
        params : np.ndarray
            Coefficients.
        """
        
        super().__init__(nelec, nspin, memory=memory)
        self.assign_orbpairs(orbpairs=orbpairs)
        self.assign_params(params=params)
        self.permutations, self.signs = self.get_permutations()
        
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
        """Assign the parameters of the creation cc wfn.
        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the creation CC wavefunction. Default: None
        add_noise : bool
            Option to add noise to the given parameters. Default: False
        """
        
        elec_pairs = int(self.nelec / 2)
        if params is None:
            number_params = int(self.nspin*(self.nspin-1)/2) 
            params = np.zeros(number_params)
        orbpairs = []
        for i in range(elec_pairs):
            orbpairs.append((i, self.nspatial+i))
        orbpairs = np.array(orbpairs)
        orbpairs = orbpairs.flatten()
        orbpairs = np.sort(orbpairs)
        orbpairs = orbpairs.reshape((elec_pairs, 2))
        for pair in orbpairs:
            col_ind = self.get_col_ind(tuple(pair.tolist()))
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
            if isinstance(orbpair, np.ndarray):
                orbpair = tuple(orbpair)
            return self.dict_orbpair_ind[orbpair]
        except (KeyError, TypeError):
            raise ValueError(
                f"Given orbital pair, {orbpair}, is not included in the wavefunction."
            )
    
    def get_permutations(self):
        """Get the permutations of the given indices.

        Parameters
        ----------
        indices : list of int
            Indices of the orbitals.

        Returns
        -------
        perms : list of list of int
            Permutations of the given indices.

        """
        indices = np.arange(self.nelec, dtype=int)
        perm_list = list(combinations(indices, r=2))

        olp_list = list(combinations(perm_list, r=int(len(indices)/2)))
        perms = []
        signs = []
        for element in olp_list:
            element_flat = [item for sublist in element for item in sublist]
            no_dup = list(set(element_flat))
            if len(no_dup) == len(indices):
                perms.append(element)
                signs.append(self.get_sign(element))
        return perms, signs
    
    def get_sign(self, indices: list[int]):
        """Get the sign of the permutation of the given indices.

        Parameters
        ----------
        indices : list of int
            Indices of the orbitals.

        Returns
        -------
        sign : int
            Sign of the permutation of the given indices.

        """
        olp = [item for pair in indices for item in pair]
        sign = 1
        for i in range(len(olp)):
            for j in range(i+1, len(olp)):
                if olp[i] > olp[j]:
                    sign *= -1
        return sign

    def _olp(self, sd: int):
        """ Calculate overlap with Slater determinant.
        
        Parameters
        ----------
        sd : int 
            Occupation vector of a Slater determinant given as a bitstring.
        
        Returns
        -------
        olp : float
            Overlap of the Slater determinant with creation CC wavefunction.
            
        """
        occ_indices = [slater.occ_indices(sd)]*len(self.permutations)
        single_prods = np.fromiter(map(self.calculate_product, occ_indices, self.permutations, self.signs), dtype=float)
        olp = np.sum(single_prods)
        return olp
    
    def calculate_product(self, occ_indices, permutation, sign):
        col_inds = list(map(self.get_col_ind, occ_indices.take(permutation)))
        prod = sign*np.prod(self.params[col_inds])
        return prod
    
    def _olp_deriv(self, sd: int):
        """ Calculate the derivative of the overlap
        
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
        mapped_permutations = (tuple((occ_indices[i], occ_indices[j]) for i, j in perm) for perm in self.permutations)

        for perm in mapped_permutations:
            sign = self.get_sign(perm)
            for pair in perm:
                col_ind = self.get_col_ind(pair)
                output[col_ind] += sign*np.prod([self.params[self.get_col_ind(p)] for p in perm if p != pair])
        return output
        
    
    def get_overlap(self, sd: int, deriv=None):
        """Return the overlap of the wavefunction with a Slater determinant.
        Inlcude math later. 
        
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
            return self._olp(sd)
        else:
            return self._olp_deriv(sd)[deriv]


