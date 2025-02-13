"""Coupled Cluster SD with seniority 0 Quadruples."""

from fanpy.wfn.cc.ccsdtq_sen0 import CCSDTQsen0


class CCSDQsen0(CCSDTQsen0):
    r"""Coupled Cluster SD with seniority 0 Quadruples.

    .. math::

        \left| {{\Psi }_{CCSDQsen0}} \right\rangle =\prod\limits_{i}{\left( 1+\sum\nolimits_{a}
        {t_{i}^{a}a_{a}^{\dagger }a_{{\bar{a}}}^{\dagger }a_{b}^{\dagger }a_{{\bar{b}}}^{\dagger }
        {{a}_{{\bar{j}}}}{{a}_{j}}{{a}_{{\bar{i}}}}{{a}_{i}}} \right)}\prod\limits_{i}
        {\left( 1+\sum\nolimits_{a}{t_{i}^{a}a_{a}^{\dagger }a_{b}^{\dagger }{{a}_{j}}{{a}_{i}}}
        \right)}\prod\limits_{i}{\left( 1+\sum\nolimits_{a}{t_{i}^{a}a_{a}^{\dagger }{{a}_{i}}}
        \right)}\left| {{\Phi }_{0}} \right\rangle

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
        ranks : None
            Ranks of the allowed excitation operators. Set by default to [1, 2, 4].

        Raises
        ------
        ValueError
            If a value that is not the default is provided.
            If the maximum rank is greater than the number of electrons.

        """
        if ranks is not None:
            raise ValueError("Only the default, rank = [1, 2, 4], is allowed")
        # FIXME: MOVE TO SOMEWHERE ELSE
        if self.nelec <= 3:
            raise ValueError("Only wavefunctions with more than 3 electrons can be considered")
        self.ranks = [1, 2, 4]
