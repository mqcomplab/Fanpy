"""Seniority Coupled Cluster wavefunctions."""

import numpy as np
from itertools import combinations
from collections import Counter
from fanpy.tools import slater
from fanpy.tools import graphs
from fanpy.wfn.cc.base import BaseCC
from fanpy.wfn.ci.base import CIWavefunction


class SeniorityCC(BaseCC):
    r"""Seniority-raising Coupled Cluster Wavefunction.

    .. math::

        \[\left| {{\Psi }_{seniority-CC}} \right\rangle =\prod\limits_{u,v}
        {\left( 1+t_{u}^{v}\hat{\varsigma }_{u}^{v} \right)}\left| {{\Phi }_{ref}} \right\rangle \]

    where the seniority-raising operators are:

    .. math::

        \hat{\varsigma }_{u}^{v}=a_{{{v}_{1}}}^{\dagger }...a_{{{v}_{n}}}^{\dagger }
        {{a}_{{{u}_{n}}}}...{{a}_{{{u}_{1}}}}\left( 1-{{{\hat{n}}}_{{{{\bar{v}}}_{1}}}}
        \right)...\left( 1-{{{\hat{n}}}_{{{{\bar{v}}}_{n}}}} \right){{\hat{n}}_{{{{\bar{u}}}_{n}}}}
        ...{{\hat{n}}_{{{{\bar{u}}}_{1}}}}

    The reference wavefunction must have seniority 0.

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
        An operator of rank r will increase the seniority by 2*r.
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

    def __init__(
        self,
        nelec,
        nspin,
        memory=None,
        ranks=None,
        indices=None,
        refwfn=None,
        params=None,
        exop_combinations=None,
        refresh_exops=None,
    ):
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
        refwfn: {CIWavefunction, int, None}
            Reference wavefunction upon which the CC operator will act.
        params : {np.ndarray, BaseCC, None}
            1-vector of CC amplitudes.
        exop_combinations : dict
            dictionary, the keys are tuples with the indices of annihilation and creation
            operators, and the values are the excitation operators that allow to excite from the
            annihilation to the creation operators.

        """
        super().__init__(
            nelec,
            nspin,
            memory=memory,
            ranks=ranks,
            indices=indices,
            params=params,
            exop_combinations=exop_combinations,
            refresh_exops=refresh_exops,
        )
        self.assign_refwfn(refwfn=refwfn)
        self.load_cache()

    def assign_nelec(self, nelec):
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
            If number of electrons is odd.

        """
        super().assign_nelec(nelec)
        if self.nelec % 2 != 0:
            raise ValueError("Odd number of electrons is not supported")

    def assign_refwfn(self, refwfn=None):
        """Assign the reference wavefunction upon which the CC operator will act.

        Parameters
        ----------
        refwfn: {CIWavefunction, int, None}
            Wavefunction that will be modified by the CC operator.
            Default is the ground-state Slater determinant.

        Raises
        ------
        TypeError
            If refwfn is not a CIWavefunction instance.
        AttributeError
            If refwfn does not have a sd_vec attribute.
        ValueError
            If refwfn does not have the right number of electrons.
            If refwfn does not have the right number of spin orbitals.
            If refwfn is not a seniority-0 wavefunction.

        """
        if refwfn is None:
            self.refwfn = slater.ground(nocc=self.nelec, norbs=self.nspin)
        elif isinstance(refwfn, int):
            if slater.total_occ(refwfn) != self.nelec:
                raise ValueError("refwfn must have {} electrons".format(self.nelec))
            # TODO: check that refwfn has the right number of spin-orbs
            # FIXME: bad check
            # if not all([i + self.nspatial in slater.occ_indices(refwfn) for i in
            #             slater.occ_indices(refwfn)[:self.nspatial]] +
            #            [i - self.nspatial in slater.occ_indices(refwfn) for i in
            #             slater.occ_indices(refwfn)[self.nspatial:]]):
            #     raise ValueError('refwfn must be a seniority-0 wavefuntion')
            self.refwfn = refwfn
        else:
            if not isinstance(refwfn, CIWavefunction):
                raise TypeError("refwfn must be a CIWavefunction or a int object")
            if not hasattr(refwfn, "sds"):  # NOTE: Redundant test.
                raise AttributeError("refwfn must have the sds attribute")
            if refwfn.nelec != self.nelec:
                raise ValueError("refwfn must have {} electrons".format(self.nelec))
            if refwfn.nspin != self.nspin:
                raise ValueError("refwfn must have {} spin orbitals".format(self.nspin))
            if refwfn.seniority != 0:
                raise ValueError("refwfn must be a seniority-0 wavefuntion")
            self.refwfn = refwfn

    def _olp(self, sd):
        r"""Calculate the matrix element of the CC operator between the Slater determinants.

        .. math::

        \[\left\langle  {{m}_{1}}
        \right|\prod\limits_{u,v}{\left( 1+t_{u}^{v}\hat{\varsigma }_{u}^{v} \right)}
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

        def complement_occ(index, sd):
            """Determine if the spin-orb with opposite spin is occupied"""
            if slater.is_alpha(index, self.nspatial):
                if slater.occ(sd, index + self.nspatial):
                    return True
                else:
                    return False
            else:
                if slater.occ(sd, index - self.nspatial):
                    return True
                else:
                    return False

        def complement_empty(index, sd):
            """Determine if the spin-orb with opposite spin is empty"""
            if slater.is_alpha(index, self.nspatial):
                if not slater.occ(sd, index + self.nspatial):
                    return True
                else:
                    return False
            else:
                if not slater.occ(sd, index - self.nspatial):
                    return True
                else:
                    return False

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

            val = 0.0
            if tuple(a_inds + c_inds) not in self.exop_combinations:
                self.generate_possible_exops(a_inds, c_inds)
            for exop_list in self.exop_combinations[tuple(a_inds + c_inds)]:
                if len(exop_list) == 0:
                    continue
                else:
                    inds = np.array([self.get_ind(exop) for exop in exop_list])
                    sign = 1
                    extra_sd = sd2
                    for exop in exop_list:
                        if not all(
                            [complement_occ(i, extra_sd) for i in exop[: len(exop) // 2]]
                            + [complement_empty(i, extra_sd) for i in exop[len(exop) // 2 :]]
                        ):
                            sign = None
                            break
                        try:
                            sign *= slater.sign_excite(extra_sd, exop[: len(exop) // 2], exop[len(exop) // 2 :])
                        except ValueError:
                            sign = None
                            break
                        extra_sd = slater.excite(extra_sd, *exop)
                    if sign:
                        val += sign * self.product_amplitudes(inds)
                    else:
                        continue
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

        def complement_occ(index, sd):
            """Determine if the spin-orb with opposite spin is occupied"""
            if slater.is_alpha(index, self.nspatial):
                if slater.occ(sd, index + self.nspatial):
                    return True
                else:
                    return False
            else:
                if slater.occ(sd, index - self.nspatial):
                    return True
                else:
                    return False

        def complement_empty(index, sd):
            """Determine if the spin-orb with opposite spin is empty"""
            if slater.is_alpha(index, self.nspatial):
                if not slater.occ(sd, index + self.nspatial):
                    return True
                else:
                    return False
            else:
                if not slater.occ(sd, index - self.nspatial):
                    return True
                else:
                    return False

        def temp_olp(sd1, sd2):
            if sd1 == sd2:
                return 0.0
            c_inds, a_inds = slater.diff_orbs(sd1, sd2)
            if isinstance(a_inds, np.ndarray):
                a_inds = a_inds.tolist()
            if isinstance(c_inds, np.ndarray):
                c_inds = c_inds.tolist()
            # NOTE: Indices of the annihilation (a_inds) and creation (c_inds) operators
            # that need to be applied to sd2 to turn it into sd1

            val = np.zeros(self.nparams)
            if tuple(a_inds + c_inds) not in self.exop_combinations:
                self.generate_possible_exops(a_inds, c_inds)
            for exop_list in self.exop_combinations[tuple(a_inds + c_inds)]:
                if len(exop_list) == 0:
                    continue
                else:
                    inds = np.array([self.get_ind(exop) for exop in exop_list])
                    sign = 1
                    extra_sd = sd2
                    for exop in exop_list:
                        if not all(
                            [complement_occ(i, extra_sd) for i in exop[: len(exop) // 2]]
                            + [complement_empty(i, extra_sd) for i in exop[len(exop) // 2 :]]
                        ):
                            sign = None
                            break
                        try:
                            sign *= slater.sign_excite(extra_sd, exop[: len(exop) // 2], exop[len(exop) // 2 :])
                        except ValueError:
                            sign = None
                            break
                        extra_sd = slater.excite(extra_sd, *exop)
                    if sign:
                        val += sign * self.product_amplitudes(inds, deriv=True)
                    else:
                        continue
            return val

        if isinstance(self.refwfn, CIWavefunction):
            val = np.zeros(self.nparams)
            for refsd in self.refwfn.sd_vec:
                val += temp_olp(sd, refsd) * self.refwfn.get_overlap(refsd)
            return val
        else:
            return temp_olp(sd, self.refwfn)

    # Restore old exop api
    # FIXME: BAD structure
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

        exrank = len(a_inds)
        check_ops = []
        # NOTE: Is necessary to invert the results of int_partition_recursive
        # to be consistent with the ordering of operators in the CC operator.
        for partition in list(graphs.int_partition_recursive(self.ranks, self.nranks, exrank))[::-1]:
            reduced_partition = Counter(partition)
            bin_size_num = []
            for bin_size in sorted(reduced_partition):
                bin_size_num.append((bin_size, reduced_partition[bin_size]))
            nops = 0
            for b_size in bin_size_num:
                nops += b_size[1]
            for annhs in graphs.generate_unordered_partition(a_inds, bin_size_num):
                for creas in graphs.generate_unordered_partition(c_inds, bin_size_num):
                    combs = []
                    for annh in annhs:
                        for crea in creas:
                            if len(annh) == len(crea):
                                combs.append(annh + crea)
                    for match in combinations(combs, nops):
                        matchs = []
                        for op in match:
                            other_ops = [other_op for other_op in match if other_op != op]
                            matchs += [set(other_op).isdisjoint(op) for other_op in other_ops]
                        if all(matchs):
                            check_ops.append(match)
        self.exop_combinations[tuple(a_inds + c_inds)] = []
        for op_list in check_ops:
            if all(tuple(op) in self.exops for op in op_list):
                # if all(tuple(op) in self.exops for op in op_list):
                self.exop_combinations[tuple(a_inds + c_inds)].append(op_list)
