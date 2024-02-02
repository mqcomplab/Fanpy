"""Standard Coupled Cluster wavefunctions."""
from itertools import combinations, permutations
from fanpy.wfn.cc.base import BaseCC
from fanpy.tools import slater
import numpy as np


class CCS(BaseCC):
    r"""Coupled Cluster Wavefunction with occupied-virtual separation.

    .. math::

        \left| {{\Psi }_{CC}} \right\rangle ={{\prod }_{\begin{matrix}
        a\in virt  \\i\in occ  \\\end{matrix}}}\left( 1+t_{i}^{a}\hat{\tau }_{i}^{a}
        \right)\left| {{\Phi }_{ref}}\right\rangle

    In this case the reference wavefunction can only be a single Slater determinant.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction.
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.
    ranks : list of ints
        Ranks of the excitation operators.
    exops : list of list of int
        Excitation operators given as lists of ints. The first half of indices correspond to
        indices to be annihilated, the second half correspond to indices to be created.
    refwfn : int
        Reference wavefunction upon which the CC operator will act.
    exop_combinations : dict
        dictionary, the keys are tuples with the indices of annihilation and creation
        operators, and the values are the excitation operators that allow to excite from the
        annihilation to the creation operators.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    param_shape : tuple of int
        Shape of the parameters.
    spin : int
        Spin of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    template_params : np.ndarray
        Default parameters of the wavefunction.
    nexops : int
        Number of excitation operators.
    nranks : int
        Number of allowed ranks.

    Methods
    -------
    __init__(self, nelec, nspin, memory=None, ngem=None, orbpairs=None, params=None)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_dtype(self, dtype)
        Assign the data type of the parameters.
    assign_memory(self, memory=None)
        Assign memory available for the wavefunction.
    assign_ranks(self, ranks=None)
        Assign the allowed excitation ranks.
    assign_exops(self, exops=None)
        Assign the allowed excitation operators.
    assign_refwfn(self, refwfn=None)
        Assign the reference wavefunction.
    assign_params(self, params=None, add_noise=False)
        Assign the parameters of the CC wavefunction.
    get_ind(self, exop) : int
        Return the parameter index that corresponds to a given excitation operator.
    get_exop(self, ind) : list of int
        Return the excitation operator that corresponds to a given parameter index.
    product_amplitudes(self, inds, deriv=None) : float
        Return the product of the CC amplitudes of the coefficients corresponding to
        the given indices.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.
    generate_possible_exops(self, a_inds, c_inds):
        Assign the excitation operators that can excite from the given indices to be annihilated
        to the given indices to be created.

    """
    def __init__(self, nelec, nspin, memory=None, ranks=None, indices=None,
                 refwfn=None, params=None, exop_combinations=None, refresh_exops=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type.
            Default is `np.float64`.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).
        ranks : {int, list of int, None}
            Ranks of the excitation operators (in increasing order).
            If int is provided, it takes all the ranks lower than that.
            Default is None, which is equivalent to taking ranks=self.nelec.
        indices : {list of list of ints, None}
            List of lists containing the indices ot the spin-orbitals to annihilate and create.
            The first sub-list contains indices of orbitals to annihilate.
            The second sub-list contains indices of orbitals to create.
            Default generates all possible indices according to the given ranks.
        refwfn : {int, None}
            Reference wavefunction upon which the CC operator will act.
        params : {np.ndarray, BaseCC, None}
            1-vector of CC amplitudes.
        exop_combinations : dict
            dictionary, the keys are tuples with the indices of annihilation and creation
            operators, and the values are the excitation operators that allow to excite from the
            annihilation to the creation operators.

        """
        super().__init__(nelec, nspin, memory=memory, ranks=[1], params=params,
                         exop_combinations=exop_combinations, refresh_exops=refresh_exops)
        self.assign_exops(indices=indices)
        self.assign_refwfn(refwfn=refwfn)

    def assign_exops(self, indices=None):
        """Assign the excitation operators that will be used to construct the CC operator.

        Parameters
        ----------
        indices : None
            The allowed excitation operators are solely defined by the occupied and virtual
            orbitals of the given reference Slater determinant.

        Raises
        ------
        TypeError
            If `indices` is not None.

        Notes
        -----
        The excitation operators are given as a list of lists of ints.
        Each sub-list corresponds to an excitation operator.
        In each sub-list, the first half of indices corresponds to the indices of the
        spin-orbitals to annihilate, and the second half corresponds to the indices of the
        spin-orbitals to create.
        [a1, a2, ..., aN, c1, c2, ..., cN]

        """
        if indices is not None:
            raise TypeError('Only the occ-virt excitation operators constructed by default from '
                            'the given reference Slater determinant are allowed')
        else:
            exops = {}
            counter = 0
            ex_from = slater.occ_indices(self.refwfn)
            ex_to = [i for i in range(self.nspin) if i not in ex_from]
            for rank in self.ranks:
                for annihilators in combinations(ex_from, rank):
                    for creators in combinations(ex_to, rank):
                        exops[(*annihilators, *creators)] = counter
                        counter += 1
            self.exops = exops

    def assign_refwfn(self, refwfn=None):
        """Assign the reference wavefunction upon which the CC operator will act.

        Parameters
        ----------
        refwfn: {int, None}
            Wavefunction (Slater determinant) that will be modified by the CC operator.
            Default is the ground-state Slater determinant.

        Raises
        ------
        TypeError
            If refwfn is not a int object.
        ValueError
            If refwfn does not have the right number of electrons.
            If refwfn does not have the right number of spin orbitals.

        """
        if refwfn is None:
            self.refwfn = slater.ground(nocc=self.nelec, norbs=self.nspin)
        else:
            if not isinstance(refwfn, int):
                raise TypeError('refwfn must be a int object')
            if slater.total_occ(refwfn) != self.nelec:
                raise ValueError('refwfn must have {} electrons'.format(self.nelec))
            # TODO: check that refwfn has the right number of spin-orbs
            self.refwfn = refwfn

    def generate_possible_exops(self, a_inds, c_inds):
        """Assign possible excitation operators from the given creation and annihilation operators.

        Parameters
        ----------
        a_inds : list of int
            Indices of the orbitals of the annihilation operators.
            Must be strictly increasing.
        c_inds : list of int
            Indices of the orbitals of the creation operators.
            Must be strictly increasing.

        Notes
        -----
        The excitation operators are sored as values of the exop_combinations dictionary.
        Each value is a list of lists of possible excitation operators.
        Each sub-list contains excitation operators such that, multiplied together, they allow
        to excite to and from the given indices.

        """
        if self.refresh_exops and len(self.exop_combinations) > self.refresh_exops:
            print(f"Resetting exop_combinations at size {self.refresh_exops}")
            self.exop_combinations = {}

        self.exop_combinations[tuple(a_inds + c_inds)] = []
        for perm_a_inds in permutations(a_inds):
            for perm_c_inds in permutations(c_inds):
                op_list = list(zip(perm_a_inds, perm_c_inds))
                num_hops = 0
                jumbled_a_inds = []
                jumbled_c_inds = []
                prev_hurdles = 0
                for exop in op_list:
                    num_inds = len(exop) // 2
                    num_hops += prev_hurdles * num_inds
                    prev_hurdles += num_inds
                    jumbled_a_inds.extend(exop[:num_inds])
                    jumbled_c_inds.extend(exop[num_inds:])
                # move all the annihilators to one side and creators to another
                sign = (-1) ** num_hops
                # unjumble the annihilators
                sign *= slater.sign_perm(jumbled_a_inds, a_inds)
                # unjumble the creators
                sign *= slater.sign_perm(jumbled_c_inds, c_inds)

                inds = np.array([self.get_ind(exop) for exop in op_list])
                self.exop_combinations[tuple(a_inds + c_inds)].append((inds, sign))
