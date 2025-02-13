"""APset1roG wavefunction with single and double excitations."""

from fanpy.tools import slater
from fanpy.wfn.cc.apset1rog_d_NEW import APset1roGD


class APset1roGSD(APset1roGD):
    r"""APset1roG wavefunction with single and double excitations.

    .. math::

        \left| {{\Psi }_{APset1roGSD}} \right\rangle =\prod\limits_{i=1}^{N/2\;}
        {\left( 1+\sum\limits_{\begin{smallmatrix}a\in A \\b\in B\\A\bigcap B=\varnothing
        \end{smallmatrix}}^{{}}{{{t}_{i;ab}}\hat{\tau }_{i\bar{i}}^{ab}}
        \right)}\prod\limits_{i=1}^{N/2\;}{\left( 1+\sum\limits_{b\in B}^{{}}{{{t}_{\bar{i};b}}
        \hat{\tau }_{i\bar{i}}^{ib}} \right)\prod\limits_{i=1}^{N/2\;}
        {\left( 1+\sum\limits_{a\in A}^{{}}{{{t}_{i;a}}\hat{\tau }_{i\bar{i}}^{a\bar{i}}}
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
            raise TypeError("Only the default: ranks=[1, 2] is allowed")
        else:
            self.ranks = [1, 2]

    def assign_exops(self, indices=None):
        """Assign the excitation operators that will be used to construct the CC operator.

        Parameters
        ----------
        indices : {list of list of ints, None}
            List of indices of the two disjoint sets of virtual orbitals to which one will excite
            the occupied orbitals.
            The default uses alpha/beta separation.

        Raises
        ------
        TypeError
            If `indices` is not a list of list of ints.
            If `indices` is not None.
        ValueError
            If one tries to excite to occupied spin-orbitals.
            If the lists of creation operators are not disjoint.

        Notes
        -----
        The excitation operators are given as a list of lists of ints.
        Each sub-list corresponds to an excitation operator.
        In each sub-list, the first half of indices corresponds to the indices of the
        spin-orbitals to annihilate, and the second half corresponds to the indices of the
        spin-orbitals to create.
        In previous assign_exops methods if one provides a non-default option for indices the first
        sublist corresponds to indices of annihilation operators and the second sublist to indices
        of creation operators, and we do not check if there are common elements between these
        sublists. In this case both sublists correspond to indices of creation operators,
        and they must be disjoint. It is assumed that one will excite from occupied alpha
        spin-orbitals to indices given in the first sublist, and from occupied beta
        spin-orbitals to indices given in the second sublist.
        Takes care of any repetition in the sublists, and sorts them before generating the
        excitation operators.

        """
        if indices is None:
            exops = {}
            counter = 0
            ex_from = slater.occ_indices(self.refwfn)
            virt_alphas = [i for i in range(self.nspin) if (i not in ex_from) and slater.is_alpha(i, self.nspatial)]
            virt_betas = [i for i in range(self.nspin) if (i not in ex_from) and not slater.is_alpha(i, self.nspatial)]
            for occ_alpha in ex_from[: len(ex_from) // 2]:
                for virt_alpha in virt_alphas:
                    exop = [occ_alpha, virt_alpha]
                    exops[tuple(exop)] = counter
                    counter += 1
            for occ_alpha in ex_from[: len(ex_from) // 2]:
                for virt_beta in virt_betas:
                    exop = [occ_alpha + self.nspatial, virt_beta]
                    exops[tuple(exop)] = counter
                    counter += 1
            super().assign_exops(None)
            shift = len(exops)
            for exop, ind in self.exops.items():
                exops[exop] = ind + shift
            self.exops = exops

        elif isinstance(indices, list):
            exops = {}
            counter = 0
            ex_from = slater.occ_indices(self.refwfn)
            if len(indices) != 2:
                raise TypeError("`indices` must have exactly 2 elements")
            for inds in indices:
                if not isinstance(inds, list):
                    raise TypeError("The elements of `indices` must be lists of non-negative ints")
                elif not all(isinstance(ind, int) for ind in inds):
                    raise TypeError("The elements of `indices` must be lists of non-negative ints")
                elif not all(ind >= 0 for ind in inds):
                    raise ValueError("All `indices` must be lists of non-negative ints")
                if not set(ex_from).isdisjoint(inds):
                    raise ValueError("`indices` cannot correspond to occupied spin-orbitals")
            if not set(indices[0]).isdisjoint(indices[1]):
                raise ValueError("The sets of annihilation operators must be disjoint")
            indices = [list(set(indices[0])), list(set(indices[1]))]
            indices[0].sort()
            indices[1].sort()
            for occ_alpha in ex_from[: len(ex_from) // 2]:
                for i in indices[0]:
                    exop = [occ_alpha, i]
                    exops[tuple(exop)] = counter
                    counter += 1
            for occ_alpha in ex_from[: len(ex_from) // 2]:
                for j in indices[1]:
                    exop = [occ_alpha + self.nspatial, j]
                    exops[tuple(exop)] = counter
                    counter += 1
            super().assign_exops(None)
            shift = len(exops)
            for exop, ind in self.exops.items():
                exops[exop] = ind + shift
            self.exops = exops
