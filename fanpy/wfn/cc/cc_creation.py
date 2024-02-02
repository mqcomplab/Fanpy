"""Base class for excitation Coupled Cluster wavefunctions."""
import numpy as np
import functools
from itertools import combinations, permutations, product, repeat
from collections import Counter
from fanpy.tools import slater
from fanpy.tools import graphs
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.ci.base import CIWavefunction



class BaseCC(BaseWavefunction):
    r"""Base Excitation Coupled Cluster Wavefunction.

    We will consider CC wavefunctions in their product form.

    .. math::

        \[\left| {{\Psi }_{CC}} \right\rangle =
        \prod\limits_{u,v}{\left( 1+t_{u}^{v}\hat{\tau }_{u}^{v}
        \right)}\left| {{\Phi }_{ref}} \right\rangle\]

    where the excitation operators:

    .. math::

        \[\hat{\tau }_{u}^{v}=
        a_{{{v}_{1}}}^{\dagger }...a_{{{v}_{n}}}^{\dagger }{{a}_{{{u}_{n}}}}...{{a}_{{{u}_{1}}}}\]

    act on a give reference wavefunction (i.e. :math:`\[\left| {{\Phi }_{ref}} \right\rangle \]`)

    Within a given excitation operator, annihilators appear in increasing order, from right to
    left (correspondingly, creators appear in decreasing order, from right to left). The
    excitation operators are ordered in increasing level of rank (e.g. the semi-sum of the
    number of creation and annihilation operators in a given string of such operators), from right
    to left (with low-rank operators acting first on the reference wavefunction). Operators with
    the same rank are ordered following the lexicographical ordering of cartesian products.

    Note: Do not mistake with the list representation of excitation operators, where the indices
    of the annihilation and creation operators increase from left to right.
    For example: [a1, a2, ..., aN, c1, c2, ..., cN]


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
    exops : dictionary of tuple to int
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
    __init__(self, nelec, nspin, memory=None, ranks=None, indices=None,
             refwfn=None, params=None)
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
    def __init__(self, nelec, nspin, memory=None, ranks=2, indices=None,
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
        refwfn : {CIWavefunction, int, None}
            Reference wavefunction upon which the CC operator will act.
        params : {np.ndarray, BaseCC, None}
            1-vector of CC amplitudes.
        exop_combinations : dict
            dictionary, the keys are tuples with the indices of annihilation and creation
            operators, and the values are the excitation operators that allow to excite from the
            annihilation to the creation operators.

        """
        super().__init__(nelec, nspin, memory=memory)
        self.assign_ranks(ranks=ranks)
        self.assign_refwfn(refwfn=refwfn)
        self.assign_exops(indices=indices)
        self.assign_params(params=params)
        self._cache_fns = {}
        self.load_cache()
        if exop_combinations is None:
            self.exop_combinations = {}
        self.refresh_exops = refresh_exops

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
    def nranks(self):
        """Return the number of ranks.

        Returns
        -------
        nranks : int
            Number of ranks.

        """
        return len(self.ranks)

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
            ranks = self.nelec
        if isinstance(ranks, list):
            if not all(isinstance(rank, int) for rank in ranks):
                raise TypeError('`ranks` must be an int or a list of ints.')
            elif not all(rank > 0 for rank in ranks):
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

    def assign_refwfn(self, refwfn=None):
        """Assign the reference wavefunction upon which the CC operator will act.

        Parameters
        ----------
        refwfn : {CIWavefunction, int, None}
            Wavefunction that will be modified by the CC operator.
            Default is the ground-state Slater determinant.

        Raises
        ------
        TypeError
            If refwfn is not a CIWavefunction or int instance.
        AttributeError
            If refwfn does not have a sd_vec attribute.
        ValueError
            If refwfn does not have the right number of electrons.
            If refwfn does not have the right number of spin orbitals.

        """
        if refwfn is None:
            self.refwfn = slater.ground(nocc=self.nelec, norbs=self.nspin)
        elif isinstance(refwfn, int):
            if slater.total_occ(refwfn) != self.nelec:
                raise ValueError('refwfn must have {} electrons'.format(self.nelec))
            # TODO: check that refwfn has the right number of spin-orbs
            self.refwfn = refwfn
        else:
            if not isinstance(refwfn, CIWavefunction):
                raise TypeError('refwfn must be a CIWavefunction or a int object')
            if not hasattr(refwfn, 'sds'):  # NOTE: Redundant test.
                raise AttributeError('refwfn must have the sds attribute')
            if refwfn.nelec != self.nelec:
                raise ValueError('refwfn must have {} electrons'.format(self.nelec))
            if refwfn.nspin != self.nspin:
                raise ValueError('refwfn must have {} spin orbitals'.format(self.nspin))
            self.refwfn = refwfn

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the CC wavefunction.

        Parameters
        ----------
        params : {np.ndarray, BaseCC, None}
            Parameters of the CC wavefunction.
            If BaseCC instance is given, then the parameters of this instance are used.
            Default uses the template parameters.
        add_noise : bool
            Flag to add noise to the given parameters.

        Raises
        ------
        ValueError
            If `params` does not have the same shape as the template_params.
            If given BaseCC instance does not have the same number of electrons.
            If given BaseCC instance does not have the same number of spin orbitals.
            If given BaseCC instance does not have the same number of excitation operators.

        Notes
        -----
        Depends on dtype, template_params, and nparams.

        """
        if isinstance(params, BaseCC):
            other = params
            if self.nelec != other.nelec:
                raise ValueError('The number of electrons in the two wavefunctions must be the '
                                 'same.')
            if self.nspin != other.nspin:
                raise ValueError('The number of spin orbitals in the two wavefunctions must be the '
                                 'same.')
            #if self.nexops != other.nexops:
            #    raise ValueError('The number of excitation operators in the two wavefunctions'
            #                     'must be the same.')
            params = np.zeros(self.params_shape)
            for exop, ind in other.exops.items():
                try:
                    params[self.get_ind(exop)] = other.params[ind]
                    # FIXME: Couldn't be just: params[self.get_ind(exop)] = other.params[ind]?
                except ValueError:
                    print('The excitation of the given wavefunction is not possible in the '
                          'current wavefunction. Parameters corresponding to this excitation will'
                          ' be ignored.')
        elif params is None:
            params = self.template_params
        elif params.shape != self.template_params.shape:
            raise ValueError(
                "Given parameters must have the shape, {}".format(self.template_params.shape)
            )
        super().assign_params(params=params, add_noise=add_noise)
        self.clear_cache()

    def get_ind(self, exop):
        """Get index that corresponds to the given excitation operator.

        Parameters
        ----------
        exop : list of int
            Indices of the spin-orbitals that will be used to construct the excitation operator.

        Returns
        -------
        ind : int
            Index that corresponds to the given excitation operator.

        Raises
        ------
        ValueError
            If given excitation operator is not valid.

        """
        try:
            return self.exops[tuple(exop)]
        except (ValueError, TypeError, KeyError):
            raise ValueError('Given excitation operator, {0}, is not included in the '
                             'wavefunction.'.format(exop))

    def get_exop(self, ind):
        """Get the excitation operator that corresponds to the given index.

        Parameters
        ----------
        ind : int
            Index that corresponds to the given excitation operator.

        Returns
        -------
        exop : list of int
            Indices of the given excitation operator.

        Raises
        ------
        ValueError
            If index is not valid.

        """
        for exop, i in self.exops.items():
            if i == ind:
                return exop
        raise ValueError('Given index, {0}, is not used in the wavefunction'.format(ind))

    def product_amplitudes(self, inds, deriv=False):
        """Compute the product of the CC amplitudes that corresponds to the given indices.

        Parameters
        ----------
        inds : {list, np.ndarray}
            Indices of the excitation operators that will be used.
        deriv : bool
            Index of the element with respect to which the product is derivatized.
            Default is no derivatization.

        Returns
        -------
        product : {float, np.ndarray}
            Product of the CC selected amplitudes.

        """
        inds = np.array(inds)
        if not deriv:
            return np.prod(self.params[inds])
        else:
            #for i in inds:
            #    trunc_inds = [ind for ind in inds if ind != i]
            #    if trunc_inds:
            #        return self.product_amplitudes(trunc_inds)
            #    else:
            #        return 1
            output = np.zeros(self.nparams)
            selected_params = self.params[inds]
            for ind in inds:
                output[ind] = np.prod(selected_params[ind != inds])
            return output

    def product_amplitudes_multi(self, indices_multi, deriv=False):
        """Compute the product of the CC amplitudes that corresponds to the given indices.

        Parameters
        ----------
        inds : {list, np.ndarray}
            Indices of the excitation operators that will be used.
        deriv : bool
            Index of the element with with respect to which the product is derivatized.
            Default is no derivatization.

        Returns
        -------
        product : {float, np.ndarray}
            Product of the CC selected amplitudes.

        """
        if not deriv:
            output = 0
            for indices_sign in indices_multi.values():
                indices, signs = indices_sign[:, :-1], indices_sign[:, -1]
                signs = signs.astype(np.int8)
                signs[signs > 1] = -1
                output += np.sum(np.prod(self.params[indices], axis=1) * signs)
            return output

        output = np.zeros(self.nparams)
        for indices_sign in indices_multi.values():
            indices, signs = indices_sign[:, :-1], indices_sign[:, -1]
            signs = signs.astype(np.int8)
            signs[signs > 1] = -1
            for ind in set(indices.ravel()):
                bool_indices = ind == indices
                row_inds = np.sum(bool_indices, axis=1, dtype=bool)
                selected_params = self.params[indices]
                old_params = np.copy(selected_params[bool_indices])
                selected_params[bool_indices] = 1
                output[ind] += np.sum(np.prod(selected_params[row_inds], axis=1) * signs[row_inds])
                selected_params[bool_indices] = old_params
        return output

    def load_cache(self):
        """Load the functions whose values will be cached.

        To minimize the cache size, the input is made as small as possible. Therefore, the cached
        function is not a method of an instance (because the instance is an input) and the smallest
        representation of the Slater determinant (an integer) is used as the only input. However,
        the functions must access other properties/methods of the instance, so they are defined
        within this method so that the instance is available within the namespace w/o use of
        `global` or `local`.

        Since the bitstring is used to represent the Slater determinant, they need to be processed,
        which may result in repeated processing depending on when the cached function is accessed.

        It is assumed that the cached functions will not be used to calculate redundant results. All
        simplifications that can be made is assumed to have already been made. For example, it is
        assumed that the overlap derivatized with respect to a parameter that is not associated with
        the given Slater determinant will never need to be evaluated because these conditions are
        caught before calling the cached functions.

        Notes
        -----
        Needs to access `memory` and `params`.

        """
        # assign memory allocated to cache
        if self.memory == np.inf:
            memory = None
        else:
            memory = max(int((self.memory - 5*8*self.params.size) / (self.params.size + 1)), 0)

        # create function that will be cached
        @functools.lru_cache(maxsize=memory, typed=False)
        def _olp(sd1):
            """Cached _olp method without caching the instance."""
            return self._olp(sd1)

        @functools.lru_cache(maxsize=0, typed=False)
        def _olp_deriv(sd1):
            """Cached _olp_deriv method without caching the instance."""
            return self._olp_deriv(sd1)

        # create cache
        if not hasattr(self, '_cache_fns'):
            self._cache_fns = {}

        # store the cached function
        self._cache_fns['overlap'] = _olp
        self._cache_fns['overlap derivative'] = _olp_deriv

    def _olp(self, sd):
        r"""Calculate the matrix element of the CC operator between the Slater determinants.

        .. math::

        \[\left\langle  {{m}_{1}}
        \right|\prod\limits_{u,v}{\left( 1+t_{u}^{v}\hat{\tau }_{u}^{v} \right)}
        \left| {{m}_{2}} \right\rangle \]

        Parameters
        ----------
        sd1 : int
            Occupation vector of the left Slater determinant given as a bitstring.
        sd2 : int
            Occupation vector of the right Slater determinant given as a bitstring.

        Returns
        -------
        olp : {float, complex}
            Matrix element of the CC operator between the given Slater determinant.

        """
        def temp_olp(sd1, sd2):
            if sd1 == sd2:
                return 1.0
            c_inds, a_inds = slater.diff_orbs(sd1, sd2)
            if isinstance(a_inds, np.ndarray):
                a_inds = a_inds.tolist()
            if isinstance(c_inds, np.ndarray):
                c_inds = c_inds.tolist()
            # NOTE: Indices of the annihilation (a_inds) and creation (c_inds) operators
            # that need to be applied to sd2 to turn it into sd1

            # get sign
            sign = slater.sign_excite(sd2, a_inds, c_inds)

            val = 0.0
            if tuple(a_inds + c_inds) not in self.exop_combinations:
                self.generate_possible_exops(a_inds, c_inds)

            # FIXME: sometimes exop contains virtual orbitals in annihilators may need to explicitly
            # excite
            indices_multi = self.exop_combinations[tuple(a_inds + c_inds)]
            amplitudes = self.product_amplitudes_multi(indices_multi)
            val = sign * amplitudes
            return val

        if isinstance(self.refwfn, CIWavefunction):
            val = 0
            for refsd in self.refwfn.sd_vec:
                val += temp_olp(sd, refsd) * self.refwfn.get_overlap(refsd)
            return val
        else:
            return temp_olp(sd, self.refwfn)

    def _olp_deriv(self, sd):
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd1 : int
            Occupation vector of the left Slater determinant given as a bitstring.
        sd2 : int
            Occupation vector of the right Slater determinant given as a bitstring.

        Returns
        -------
        olp : {float, complex}
            Derivative of the overlap with respect to the given parameter.

        """
        def temp_olp(sd1, sd2):
            if sd1 == sd2:
                return np.zeros(self.nparams)
            # FIXME: this should definitely be vectorized
            c_inds, a_inds = slater.diff_orbs(sd1, sd2)
            if isinstance(a_inds, np.ndarray):
                a_inds = a_inds.tolist()
            if isinstance(c_inds, np.ndarray):
                c_inds = c_inds.tolist()
            # NOTE: Indices of the annihilation (a_inds) and creation (c_inds) operators
            # that need to be applied to sd2 to turn it into sd1

            # get sign
            sign = slater.sign_excite(sd2, a_inds, c_inds)

            val = np.zeros(self.nparams)
            if tuple(a_inds + c_inds) not in self.exop_combinations:
                self.generate_possible_exops(a_inds, c_inds)

            # FIXME: sometimes exop contains virtual orbitals in annihilators may need to explicitly
            # excite
            indices_multi = self.exop_combinations[tuple(a_inds + c_inds)]
            amplitudes = self.product_amplitudes_multi(indices_multi, deriv=True)
            val = sign * amplitudes
            return val

        if isinstance(self.refwfn, CIWavefunction):
            val = np.zeros(self.nparams)
            for refsd in self.refwfn.sd_vec:
                val += temp_olp(sd, refsd) * self.refwfn.get_overlap(refsd)
            return val
        else:
            return temp_olp(sd, self.refwfn)

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \[\begin{align}
            & \left| {{\Psi }_{CC}} \right\rangle =\prod\limits_{u,v}
            {\left( 1+t_{u}^{v}\hat{\tau }_{u}^{v} \right)}\left| {{\Phi }_{ref}} \right\rangle  \\
            & =\sum\limits_{{{m}_{ref}}}{{{f}_{ref}}\left( {{m}_{ref}} \right)\prod\limits_{u,
            v}{\left( 1+t_{u}^{v}\hat{\tau }_{u}^{v} \right)\left| {{m}_{ref}} \right\rangle }} \\
            & =\sum\limits_{m}{\sum\limits_{{{m}_{ref}}}{\sum\limits_{\left\{ \hat{\tau }_{u}^{v}
            \right\}:\prod\limits_{u,v}{\hat{\tau }_{u}^{v}\left| {{m}_{ref}} \right\rangle
            =\left| m \right\rangle }}{{{f}_{ref}}\left( {{m}_{ref}} \right)\sgn \left( \sigma
            \left\{ \hat{\tau }_{u}^{v} \right\} \right)\prod\limits_{u,v}{t_{u}^{v}\left| m
            \right\rangle }}}} \\\end{align}\]

        where:

        .. math::

            \left| {{\Phi }_{ref}} \right\rangle =\sum\limits_{{{m}_{ref}}}{{{f}_{ref}}
            \left( {{m}_{ref}} \right)\left| {{m}_{ref}} \right\rangle }

        Parameters
        ----------
        sd : {int, int}
            Slater Determinant against which the overlap is taken.
        deriv : {np.ndarray, None}
            Indices of the parameters with respect to which the overlap is derivatized.
            Default returns the overlap without derivatization.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.

        Notes
        -----
        Bit of performance is lost in exchange for generalizability. Hopefully it is still readable.

        """
        # if no derivatization
        if deriv is None:
            return self._cache_fns['overlap'](sd)
        # if derivatization
        else:
            val = self._cache_fns['overlap derivative'](sd)
            return val[deriv]

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
        The excitation operators are sorted as values of the exop_combinations dictionary.
        Each value is a list of lists of possible excitation operators.
        Each sub-list contains excitation operators such that, multiplied together, they allow
        to excite to and from the given indices.

        """
        if self.refresh_exops and len(self.exop_combinations) > self.refresh_exops:
            print(f"Resetting exop_combinations at size {self.refresh_exops}")
            self.exop_combinations = {}

        exrank = len(a_inds)
        # NOTE: Is necessary to invert the results of int_partition_recursive
        # to be consistent with the ordering of operators in the CC operator.
        inds_multi = {}
        for partition in list(graphs.int_partition_recursive(self.ranks,
                                                             self.nranks, exrank))[::-1]:
            reduced_partition = Counter(partition)
            bin_size_num = []
            for bin_size in sorted(reduced_partition):
                bin_size_num.append((bin_size, reduced_partition[bin_size]))
            nops = 0
            for b_size in bin_size_num:
                nops += b_size[1]
            for annhs in graphs.generate_unordered_partition(a_inds, bin_size_num):
                # group by sizes
                annhs_grouped = {}
                for annh in annhs:
                    if len(annh) in annhs_grouped:
                        annhs_grouped[len(annh)].append(annh)
                    else:
                        annhs_grouped[len(annh)] = [annh]
                # permutations of each size
                # annhs_perms = {size: permutations(annh) for size, annh in annhs_grouped.items()}
                for creas in graphs.generate_unordered_partition(c_inds, bin_size_num):
                    # group by sizes
                    creas_grouped = {}
                    for crea in creas:
                        if len(crea) in creas_grouped:
                            creas_grouped[len(crea)].append(crea)
                        else:
                            creas_grouped[len(crea)] = [crea]
                    # permutations of each size
                    creas_perms = {size: permutations(crea) for size, crea in creas_grouped.items()}

                    # combine permutations of each annihilation and creation pair (of same size)
                    exc_perms = (zip(repeat(annhs_grouped[size_num[0]]), creas_perms[size_num[0]]) for size_num in bin_size_num)

                    # for each size, pick all the other sizes
                    for excs in product(*exc_perms):
                        combs = []
                        is_continue = False
                        for exc in excs:
                            exc = zip(*exc)
                            for annh, crea in exc:
                                # FIXME: remove
                                #if len(annh) == 1:
                                #    if annh[0] < self.nspatial:
                                #        annh = [annh[0], annh[0] + self.nspatial]
                                #        crea = sorted([crea[0], annh[1]])
                                #    else:
                                #        annh = [annh[0] - self.nspatial, annh[0]]
                                #        crea = sorted([crea[0], annh[0]])

                                op = tuple(annh + crea)
                                if op not in self.exops:
                                    is_continue = True
                                    break
                                combs.append(op)
                            if is_continue:
                                break
                        if is_continue:
                            continue

                        num_hops = 0
                        jumbled_a_inds = []
                        jumbled_c_inds = []
                        prev_hurdles = 0
                        for exop in combs:
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

                        inds = [self.get_ind(exop) for exop in combs] + [sign]
                        if len(inds) - 1 in inds_multi:
                            inds_multi[len(inds) - 1].extend(inds)
                        else:
                            inds_multi[len(inds) - 1] = inds

        two_power = np.ceil(np.log2(self.nparams))
        if two_power <= 8:
            dtype = np.uint8
        elif two_power <= 16:
            dtype = np.uint16
        elif two_power <= 32:
            dtype = np.uint32
        elif two_power <= 64:
            dtype = np.uint64
        else:
            raise ValueError("Can only support 2**63 number of parameters")

        for i, indices in inds_multi.items():
            inds_multi[i] = np.array(indices, dtype=dtype).reshape(-1, i+1)
        self.exop_combinations[tuple(a_inds + c_inds)] = inds_multi
