"""APG1ro wavefunction with single and double excitations."""
from itertools import combinations
from collections import Counter
from fanpy.tools import slater
from fanpy.tools import graphs
from fanpy.wfn.cc.pccd_ap1rog_NEW import PCCD


class AP1roGSDGeneralized(PCCD):
    r"""AP1roG wavefunction with single and double excitations, broken spin symmetry.

    .. math::

        \left| {{\Psi }_{APG1roSD}} \right\rangle =\prod\limits_{i=1}^{N/2\;}
        {\left( 1+\sum\limits_{a,b\in virt}^{{}}{{{t}_{i;ab}}\hat{\tau }_{i\bar{i}}^{ab}}
        \right)}\prod\limits_{i=1}^{N/2\;}{\left( 1+\sum\limits_{a\in virt}^{{}}{{{t}_{\bar{i};a}}
        \hat{\tau }_{i\bar{i}}^{ia}} \right)\prod\limits_{i=1}^{N/2\;}
        {\left( 1+\sum\limits_{a\in virt}^{{}}{{{t}_{i;a}}\hat{\tau }_{i\bar{i}}^{a\bar{i}}}
        \right)}\left| {{\Phi }_{0}} \right\rangle }

    In this case the reference wavefunction can only be a single Slater determinant with
    seniority 0.

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
    def assign_ranks(self, ranks=None):
        """Assign the ranks of the excitation operators.

        Parameters
        ----------
        ranks : {int, list, None}
            Ranks of the allowed excitation operators.

        Raises
        ------
        TypeError
            If 'ranks' is not None.

        """
        if ranks is not None:
            raise TypeError('Only the default: ranks=[1, 2] is allowed')
        else:
            self.ranks = [1, 2]

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
            raise TypeError('Only the excitation operators constructed by default from '
                            'the given reference Slater determinant are allowed')
        else:
            exops = {}
            counter = 0
            ex_from = slater.occ_indices(self.refwfn)
            ex_to = [i for i in range(self.nspin) if i not in ex_from]
            for occ_alpha in ex_from[:len(ex_from) // 2]:
                for virt in ex_to:
                    exop = [occ_alpha, virt]
                    exops[tuple(exop)] = counter
                    counter += 1
            for occ_alpha in ex_from[:len(ex_from) // 2]:
                for virt in ex_to:
                    exop = [occ_alpha + self.nspatial, virt]
                    exops[tuple(exop)] = counter
                    counter += 1
            super().assign_exops(None)
            shift = len(exops)
            for exop, ind in self.exops.items():
                exops[exop] = ind + shift
            self.exops = exops
