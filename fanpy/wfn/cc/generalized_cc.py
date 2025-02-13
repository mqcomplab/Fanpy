"""Generalized Coupled Cluster wavefunction."""

from fanpy.wfn.cc.base import BaseCC


class GeneralizedCC(BaseCC):
    r"""Generalized Coupled Cluster Wavefunction.

    We will consider CC wavefunctions in their product form.

    .. math::

        \[\left| {{\Psi }_{generalized-CC}} \right\rangle =
        \prod\limits_{u,v}{\left( 1+t_{u}^{v}\hat{\tau }_{u}^{v}
        \right)}\left| {{\Phi }_{ref}} \right\rangle\]

    where the excitation operators:

    .. math::

        \[\hat{\tau }_{u}^{v}=
        a_{{{v}_{1}}}^{\dagger }...a_{{{v}_{n}}}^{\dagger }{{a}_{{{u}_{n}}}}...{{a}_{{{u}_{1}}}}\]

    act on a give reference wavefunction (i.e. :math:`\[\left| {{\Phi }_{ref}} \right\rangle \]`)

    No occupied/virtual separation is assumed.
    Within a given excitation operator, annihilators appear in increasing order, from right to
    left (correspondingly, creators appear in decreasing order, from right to left). The
    excitation operators are ordered in increasing level of rank (e.g. the semi-sum of the
    number of creation and annihilation operators in a given string of such operators), from right
    to left. Operators with the same rank are ordered following the lexicographical ordering of
    cartesian products.

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
    refwfn : {CIWavefunction, int}
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

    def assign_exops(self, indices=None):
        """Assign the excitation operators that will be used to construct the CC operator.

        Parameters
        ----------
        indices : None
            Only the default (i.e. using all possible indices) is allowed.

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
            raise TypeError("Only the default (i.e. using all possible indices) is allowed")
        super().assign_exops(indices)
