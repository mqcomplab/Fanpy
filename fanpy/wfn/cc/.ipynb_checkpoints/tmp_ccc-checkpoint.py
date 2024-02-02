import numpy as np
import functools
from itertools import combinations, permutations, product, repeat
from collections import Counter
from fanpy.tools import slater
from fanpy.tools import graphs
from fanpy.wfn.base import BaseWavefunction
import itertools


class CreationCC(BaseWavefunction):
     
    def __init__(self, nelec, nspin, memory=None, ranks=None, 
        params=None, op_param_comb=None, HF_init=True):
        super().__init__(nelec, nspin, memory=memory)
        self.HF_init = HF_init
        self.assign_ranks(ranks=ranks)
        self.assign_ops(nspin,ranks)
        self.get_sign_partition(sd=None)
        self.get_HF_sign_partition()
        if op_param_comb is None:
            self.op_param_comb = {}
        self.assign_params(params=params)
        #global counter 
        #counter = 1



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
    def nops(self):
        """Return number of operators.

        Returns
        -------
        nexops : int
            Number of operators.

        """
        return len(self.ops)

    
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
        params = np.zeros(self.nops)
        if self.HF_init == True: 
            ground_sd = slater.ground(self.nelec, self.nspin)
            ground_occs = slater.occ_indices(ground_sd)
            #signs_partition = self.get_HF_sign_partition()     
            #print(signs_partition)

        for op in ground_occs:
        #    for op in op_partition[1]:
            #print(op)
            params[op] = 1 
 
        #print("\n I was here, params: \n", params, "\n")
        return params

     
    @property
    def params_shape(self):
        """Return the shape of the wavefunction parameters.

        Returns
        -------
        params_shape : tuple of int
            Shape of the parameters.

        """
        return (self.nops,)



    def assign_ranks(self, ranks=None):
        """Assign the ranks of the excitation operators.

        Parameters
        ----------
        ranks : {int, list, None}
            Ranks of the allowed excitation operators.
            Default: [1, 2]

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
            elif not all(rank > 0  for rank in ranks):
                raise ValueError('All `ranks` must be positive ints.')
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
            elif ranks > self.nelec:
                raise ValueError('`ranks` cannot be greater than {},'
                                 'the number of electrons of the wavefunction.'.format(self.nelec))
            else:
                self.ranks = list(range(ranks+1))[1:]
        else:
            raise TypeError('`ranks` must be an int or a list of ints.')

        return ranks



    def assign_ops(self, nspin, ranks):
        """Assign the excitation operators that will be used to construct the Creation-CC operator.
        Parameters
        ----------
        nspin : int
            Number of spin orbitals.
 
        ranks : list of ints 
            Ranks of allowed excitation operators.          
        
        Returns
        -----
        ops : list of tuples
            Tuples in list are ordered with inreasing rank of Creation-CC operators from left to right.
            Tuples corresponding to the same rank of operators are also ordered with increasing order of indices of spin orbitals. 
            Ex., [(0,), (1,), (2,), ..., (0,1), (0,2),....(1,2), (1,3), ...]
        """
        self.ops = []
        for rank in self.ranks:
            iterator = combinations(range(self.nspin), rank)
            for op in iterator:
                self.ops.append(tuple(list(op)))
        #print("Operators", self.ops)
        return self.ops


    def parity(self, list_):
        '''Parity sign'''
        c = 0
        for i in range(len(list_)):
            for j in range(i+1,len(list_)):
                if list_[j] > list_[i]:
                    c += 1
        return -2*(c % 2) + 1

    def get_sign_partition(self, sd=None):
        """
        Returns
        -------
        sign_factors : list of tuples 
              List of tuples containing only two elements. 
              First element of tuple = parity of the order of indices corresponding to the partition.
                    Can be either +1 or -1
              Second element of tuple = list of lists corresponding to the partition of the indices based on different 
              combinations of the excitation orders of rank 1 and/or 2.   
        """
        #print(self.op_param_comb)#.get(tuple(operator)))
        if sd is None:
           sd = slater.ground(self.nelec, self.nspin)
        if slater.is_sd_compatible(sd):
            indices = slater.occ_indices(sd).tolist()

        self.exrank = len(indices)
        partitions = list(graphs.int_partition_recursive(self.ranks, self.nranks,self.exrank))
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
            signs.append(self.parity(flat_factor))

        signs_factors = [(signs[i], factors[i]) for i in range(len(signs))]
        #print(signs_factors, "\n")
        return signs_factors
        
    
    def get_HF_sign_partition(self):

        ground_sd = slater.ground(self.nelec, self.nspin)
        signs_partition = self.get_sign_partition(sd=ground_sd)
        #print("\nGround: ", ground_sd, "\n", signs_partition, "\n")
        return signs_partition


    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the CC wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the Creation-CC wavefunction.
            Default uses the template parameters.
        add_noise : bool
            Flag to add noise to the given parameters.

        Raises
        ------
        ValueError
            If `params` does not have the same shape as the template_params.

        Notes
        -----
        Depends on dtype, template_params, and nparams.
        """

        if params is None:
            params = self.template_params
        elif params.shape != self.template_params.shape:
            raise ValueError(
                "Given parameters must have the shape, {}".format(self.template_params.shape)
            )
        super().assign_params(params=params, add_noise = add_noise)
        for op, param in zip(self.ops, params):
            self.op_param_comb[op] = param
        #print("op_param_comb: \n", self.op_param_comb) 
        


    def get_overlap(self, sd, deriv=None):
        """ 
         Get overlap formula given the list of orbitals present in the SD of projection space. 


        """
        #if counter == 1 and  self.HF_init:
        #    olp_total = 1
        #    print(olp_total)
        #    counter += 1
        #    return olp_total
        #else:
        signs_factors = self.get_sign_partition(sd=sd)
        #print(signs_factors, "\n")
        
        olp_total = 0
        for sign, partition in signs_factors:
            olp_term = 1
            for operator in partition:
                #print(tuple(operator),self.op_param_comb.get(tuple(operator)), "\n")
                olp_term *= self.op_param_comb.get(tuple(operator))
            olp_total += olp_term*sign
        #print(sd, olp_total, "\n")
        #counter += 1
        return olp_total


#indices = [1,2,3,4,5]
#obj = CreationCC(nelec=4, nspin=8)
#print(obj.ops)
#print(obj.params)

#overlap = obj.get_overlap(15)
#print(overlap)







