import numpy as np
import functools
from itertools import combinations, permutations, product, repeat
from collections import Counter
from fanpy.tools import slater
from fanpy.tools import graphs
from fanpy.wfn.base import BaseWavefunction
import itertools


class CreationCC(BaseWavefunction):

    def __init__(self, nelec, nspin, memory=None, ranks=None, indices=None,
                 refwfn=None, params=None, exop_combinations=None, refresh_exops=None):


        super().__init__(nelec, nspin, memory=memory)
        super().assign_ranks(ranks=ranks)
        super().assign_exops(indices=indices)
        super().assign_params(params=params)


    @property
    def order(self):
        """Return the maximum order of excitation operators. 
          
        Returns
        -------
        order : int
              Highest order of the excitation operators.
 
        Operators in this wavefuction are of types:

              c_i a^\dagger_i  + c_j c_k a^\dagger_j a^\dagger_k 

        Therefore, the order here will be 2. 
        """
        return 2

    @property
    def nranks(self):
        """Return the number of ranks.

        Returns
        -------
        nranks : int
            Number of ranks.

        """
        return len(self.ranks)


    @property
    def nexops(self):
        """Return number of excitation operators.

        Returns
        -------
        nexops : int
            Number of excitation operators.

        """
        return len(self.exops)

    
    @property
    def template_params(self):
        """Return the template of the parameters of the given wavefunction.

        Uses the spatial orbitals (alpha-beta spin orbital pairs) of HF ground state as reference.
        In the CC case, the default corresponds to setting all CC amplitudes to zero.

        Returns
        -------
        template_params : np.ndarray(nexops)
            Default parameters of the CC wavefunction.

        """
        params = np.zeros(self.nexops)
        return params

     
    @property
    def params_shape(self):
        """Return the shape of the wavefunction parameters.

        Returns
        -------
        params_shape : tuple of int
            Shape of the parameters.

        """
        return (self.nexops,)



    def assign_ranks(self, ranks=None):
        """Assign the ranks of the excitation operators.

        Parameters
        ----------
        ranks : {int, list, None}
            Ranks of the allowed excitation operators.

        Raises
        ------
        TypeError
            If ranks is not given as an int or list of ints.
        ValueError
            If a rank is less or equal to zero.
            If the maximum rank is greater than the number of electrons.

        Notes
        -----
        Takes care of any repetition, and always provides an ordered list.

        """

        if ranks is None:
            ranks = [1, 2]
        if isinstance (ranks, list):
            if not all(isinstance(rank, int) for rank in ranks):
                raise TypeError('`ranks` must be an int or a list of ints.')
            elif not all(rank > 0 and rank < 3 for rank in ranks):
                raise ValueError('All `ranks` must be positive ints and must be less than 3.')
            elif max(ranks) > self.nelec:
                raise ValueError('`ranks` cannot be greater than {},'
                                 'the number of electrons of the wavefunction.'.format(self.nelec))
            else:
                ranks = list(set(ranks))
                ranks.sort()
                self.ranks = ranks
        elif isinstance(ranks, int):
            if ranks <= 0:
                raise ValueError('All `ranks` must be positive ints.')
            elif ranks > self.order:
                raise ValueError('`ranks` cannot be greater than {},'
                                 'the number of electrons of the wavefunction.'.format(self.order))
            #else:
             #   self.ranks = list(range(ranks+1))[1:]
        else:
            raise TypeError('`ranks` must be an int or a list of ints.')



    def assign_exops(self, indices=None):
        """Assign the excitation operators that will be used to construct the CC operator.

        Parameters
        ----------
        indices : {list of list of ints, None}
            Lists of indices of the spin-orbitals that will be used in the excitation operators.
            There are two sub-lists: the first containing the indices from which the CC operator
            will excite (e.g. indices of the annihilation operators), the second containing the
            indices to which the CC operator will excite (e.g. indices of the creation operators).
            Default is all possible excitation operators consistent with the ranks (where in the
            same excitation operator the creation and annihilation string do not share any
            index). If explicit lists of indices are given, there may be common indices between
            them (i.e. it would be possible for an excitation operator to excite from and to the
            same spin orbital, this is useful in some geminal wavefunctions).

        Raises
        ------
        TypeError
            If `indices` is not a list.
            If an element of `indices` is not a list.
            If `indices` does not have exactly two elements.
            If an element of a sub-list of `indices` is not an int.
        ValueError
            If a spin-orbital index is negative.

        Notes
        -----
        The excitation operators are given as a list of lists of ints.
        Each sub-list corresponds to an excitation operator.
        In each sub-list, the first half of indices corresponds to the indices of the
        spin-orbitals to annihilate, and the second half corresponds to the indices of the
        spin-orbitals to create.
        [a1, a2, ..., aN, c1, c2, ..., cN]
        Takes care of any repetition in the sublists, and sorts them before generating the
        excitation operators.

        """
        if indices is None:
            exops = {}
            counter = 0
            for rank in self.ranks:
                for annihilators in combinations(range(self.nspin), rank):
                    for creators in combinations([i for i in range(self.nspin)
                                                  if i not in annihilators], rank):
                        exops[(*annihilators, *creators)] = counter
                        counter += 1
            self.exops = exops
        elif isinstance(indices, list):
            if len(indices) != 2:
                raise TypeError('`indices` must have exactly 2 elements')
            for inds in indices:
                if not isinstance(inds, list):
                    raise TypeError('The elements of `indices` must be lists of non-negative ints')
                elif not all(isinstance(ind, int) for ind in inds):
                    raise TypeError('The elements of `indices` must be lists of non-negative ints')
                elif not all(ind >= 0 for ind in inds):
                    raise ValueError('All `indices` must be lists of non-negative ints')
            ex_from, ex_to = list(set(indices[0])), list(set(indices[1]))
            ex_from.sort()
            ex_to.sort()
            exops = {}
            counter = 0
            for rank in self.ranks:
                for annihilators in combinations(ex_from, rank):
                    for creators in combinations(ex_to, rank):
                        exops[(*annihilators, *creators)] = counter
                        counter += 1
            self.exops = exops
        else:
            raise TypeError('`indices` must be None or a list of 2 lists of non-negative ints')




    def parity(self, list_):
        '''Parity sign'''
	c = 0
	for i in range(len(list_)):
	    for j in range(i+1,len(list_)):
		if list_[j] > list_[i]:
		    c += 1
        return -2*(c % 2) + 1


######Test variables
#indices = [1, 2, 3, 4, 5]
#coins = [1, 2]
#num_coin_types = len(coins)
#total = len(indices)

    def get_olap_terms(self, ):
        """ 
         Get overlap formula given the list of orbitals present in the SD of projection space. 


        Returns
        -------
        sign_factors : list of tuples 
              List of tuples containing only two elements. 
              First element of tuple = parity of the order of indices corresponding to the partition.
                    Can be either +1 or -1
              Second element of tuple = list of lists corresponding to the partition of the indices based on different 
              ccombinations of the excitation orders of rank 1 and/or 2.   
        
        """

        partitions = list(graphs.int_partition_recursive(inp_indices, self.nranks, len(inp_indices))
        factors = []
        for partition in partitions:
            reduced_partition = Counter(partition)
            bin_size_num = []
            for bin_size in sorted(reduced_partition):
                bin_size_num.append((bin_size, reduced_partition[bin_size]))
            terms = list(graphs.generate_unordered_partition(indices, bin_size_num))
            for term in terms:
                factors.append(term)
        signs = []
        for factor in factors:
            flat_factor = list(itertools.chain.from_iterable(factor))
            signs.append(parity(flat_factor))

        signs_factors = [(signs[i], factors[i]) for i in range(len(signs))]


