"""Pair Coupled Cluster Doubles-AP1roG wavefunction."""

import numpy as np
from fanpy.tools import slater
from fanpy.wfn.cc.base import BaseCC
from fanpy.wfn.ci.base import CIWavefunction


class PCCD(BaseCC):
    r"""Pair CC doubles-AP1roG wavefunction.

    .. math::

        \begin{align}
        & \left| {{\Psi }_{AP1roG}} \right\rangle =\prod\limits_{i}
        {\left( a_{i}^{\dagger }a_{{\bar{i}}}^{\dagger }+
        \sum\nolimits_{a}{t_{i}^{a}a_{a}^{\dagger }a_{{\bar{a}}}^{\dagger }}
        \right)}\left| \theta  \right\rangle  \\ & =\prod\limits_{i}
        {\left( 1+\sum\nolimits_{a}{t_{i}^{a}a_{a}^{\dagger }a_{{\bar{a}}}^{\dagger }{{a}_
        {{\bar{i}}}}{{a}_{i}}} \right)}\left| {{\Phi }_{0}} \right\rangle  \\ \end{align}

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
    s_type : str
        Option indicating how the single excitations are allowed to affect the seniority
        of the occupied and/or virtual spaces.
        free: no restrictions imposed.
        sen-o: single excitations must break an occupied pair.
               (only excite from spin-orbital i if its complement is occupied)
        sen-v: single excitations cannot form a pair.
                (only excite to a virtual spin-orbital a if its complement is empty)
        sen-ov: single excitations must break an occupied pair and cannot form extra pairs

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
    assign_s_type(self, s_type):
        Assign the s_type option.

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
        s_type="sen-o",
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
        refwfn: {int, None}
            Reference wavefunction upon which the CC operator will act.
        params : {np.ndarray, BaseCC, None}
            1-vector of CC amplitudes.
        exop_combinations : dict
            dictionary, the keys are tuples with the indices of annihilation and creation
            operators, and the values are the excitation operators that allow to excite from the
            annihilation to the creation operators.

        """
        super().__init__(
            nelec, nspin, memory=memory, params=params, exop_combinations=exop_combinations, refresh_exops=refresh_exops
        )
        self.assign_s_type(s_type=s_type)
        self.assign_ranks(ranks=ranks)
        self.assign_exops(indices=indices)
        self.assign_refwfn(refwfn=refwfn)

        # mapping from indices to exops
        self.ind_exops = sorted(self.exops.keys(), key=lambda x: self.exops[x])

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

    def assign_ranks(self, ranks=None):
        """Assign the ranks of the excitation operators.

        Parameters
        ----------
        ranks : None
            Ranks of the allowed excitation operators. Set by default to [2].

        Raises
        ------
        ValueError
            If a value that is not the default is provided.
            If the maximum rank is greater than the number of electrons.

        """
        if ranks is not None:
            raise ValueError("Only the default, rank = 2, is allowed")
        if self.nelec <= 1:
            raise ValueError("Only wavefunctions with more than 1 electron can be considered")
        self.ranks = [2]

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
            raise TypeError(
                "Only the excitation operators constructed by default from "
                "the given reference Slater determinant are allowed"
            )
        else:
            exops = {}
            counter = 0
            ex_from = slater.occ_indices(self.refwfn)
            ex_to = [i for i in range(self.nspin) if i not in ex_from]
            for occ_alpha in ex_from[: len(ex_from) // 2]:
                for virt_alpha in ex_to[: len(ex_to) // 2]:
                    exop = [occ_alpha, occ_alpha + self.nspatial, virt_alpha, virt_alpha + self.nspatial]
                    exops[tuple(exop)] = counter
                    counter += 1
            self.exops = exops

    def assign_refwfn(self, refwfn=None):
        """Assign the reference wavefunction upon which the CC operator will act.

        Parameters
        ----------
        refwfn: {int, None}
            Seniority 0 wavefunction that will be modified by the CC operator.
            Default is the ground-state Slater determinant.

        Raises
        ------
        TypeError
            If refwfn is not a int instance.
        ValueError
            If refwfn does not have the right number of electrons.
            If refwfn does not have the right number of spin orbitals.
            If refwfn is not a seniority-0 wavefunction.

        """
        if refwfn is None:
            self.refwfn = slater.ground(nocc=self.nelec, norbs=self.nspin)
        else:
            if not isinstance(refwfn, int):
                raise TypeError("refwfn must be a int object")
            if slater.total_occ(refwfn) != self.nelec:
                raise ValueError("refwfn must have {} electrons".format(self.nelec))
            if not all(
                [i + self.nspatial in slater.occ_indices(refwfn) for i in slater.occ_indices(refwfn)[: self.nspatial]]
                + [i - self.nspatial in slater.occ_indices(refwfn) for i in slater.occ_indices(refwfn)[self.nspatial :]]
            ):
                raise ValueError("refwfn must be a seniority-0 wavefuntion")
            # TODO: check that refwfn has the right number of spin-orbs
            self.refwfn = refwfn

    def assign_s_type(self, s_type):
        """Assign the seniority option for single excitations.

        Parameters
        ----------
        s_type : str
        Option indicating how the single excitations are allowed to affect the seniority
        of the occupied and/or virtual spaces.
        free: no restrictions imposed.
        sen-o: single excitations must break an occupied pair.
               (only excite from spin-orbital i if its complement is occupied)
        sen-v: single excitations cannot form a pair.
                (only excite to a virtual spin-orbital a if its complement is empty)
        sen-ov: single excitations must break an occupied pair and cannot form extra pairs

        Raises
        ------
        ValueError
            If s_type is none of 'free', 'sen-o', 'sen-v', 'sen-ov'.

        """
        self.s_type = s_type

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
            # print("indices_multi", indices_multi)
            # FIXME: filter out rows whose excitation operators does not have annihilator that is
            # doubly occupied
            occ_indices = set(slater.occ_indices(sd2))
            # virt_indices = set(slater.virt_indices(sd, self.nspin))
            # print(indices_multi)
            # "Break occupied pairs"
            if self.s_type == "sen-o":
                for exc_order in indices_multi:
                    indices_sign = indices_multi[exc_order]
                    selected_rows = []
                    for row_ind, row in enumerate(indices_sign):
                        # occupied orbitals but have its spin composite also occupied
                        # AND its composite CANNOT participate in other excitation operators
                        trash = set([])
                        skip_row = False
                        for exop in (self.ind_exops[i] for i in row[:-1]):
                            if len(exop) == 2:
                                if exop[0] in trash:
                                    # skip
                                    skip_row = True
                                    break
                                if exop[0] < self.nspatial:
                                    if exop[0] + self.nspatial not in occ_indices:
                                        # skip
                                        skip_row = True
                                        break
                                    else:
                                        # add to trash
                                        trash.add(exop[0])
                                        trash.add(exop[0] + self.nspatial)
                                if exop[0] >= self.nspatial:
                                    if exop[0] - self.nspatial not in occ_indices:
                                        # skip
                                        skip_row = True
                                        break
                                    else:
                                        # add to trash
                                        trash.add(exop[0])
                                        trash.add(exop[0] - self.nspatial)
                            # FIXME: not sure
                            else:
                                for j in exop[: len(exop) // 2]:
                                    if j in trash:
                                        # skip
                                        skip_row = True
                                        break
                                    else:
                                        trash.add(j)
                                if skip_row:
                                    break

                        if not skip_row:
                            selected_rows.append(row_ind)

                    indices_multi[exc_order] = indices_sign[selected_rows]
                    # FIXME:
                    # print(selected_rows)
                    # print(indices_multi[exc_order])
            # "Don't form virtual pairs"
            elif self.s_type == "sen-v":
                for exc_order in indices_multi:
                    indices_sign = indices_multi[exc_order]
                    selected_rows = []
                    for row_ind, row in enumerate(indices_sign):
                        # occupied orbitals but have its spin composite also occupied
                        # AND its composite CANNOT participate in other excitation operators
                        trash = set([])
                        skip_row = False
                        for exop in (self.ind_exops[i] for i in row[:-1]):
                            if len(exop) == 2:
                                if exop[1] in trash:
                                    # skip
                                    skip_row = True
                                    break
                                if exop[1] < self.nspatial:
                                    if exop[1] + self.nspatial in occ_indices:
                                        # skip
                                        skip_row = True
                                        break
                                    else:
                                        # add to trash
                                        trash.add(exop[1])
                                        trash.add(exop[1] + self.nspatial)
                                if exop[1] >= self.nspatial:
                                    if exop[1] - self.nspatial in occ_indices:
                                        # skip
                                        skip_row = True
                                        break
                                    else:
                                        # add to trash
                                        trash.add(exop[1])
                                        trash.add(exop[1] - self.nspatial)
                            # FIXME: not sure
                            else:
                                for j in exop[len(exop) // 2 :]:
                                    if j in trash:
                                        # skip
                                        skip_row = True
                                        break
                                    else:
                                        trash.add(j)
                                if skip_row:
                                    break

                        if not skip_row:
                            selected_rows.append(row_ind)

                    indices_multi[exc_order] = indices_sign[selected_rows]
                    # FIXME:
                    # print(selected_rows)
                    # print(indices_multi[exc_order])
            # "Break occupied pairs AND don't form virtual pairs"
            elif self.s_type == "sen-ov":
                for exc_order in indices_multi:
                    indices_sign = indices_multi[exc_order]
                    selected_rows = []
                    for row_ind, row in enumerate(indices_sign):
                        # occupied orbitals but have its spin composite also occupied
                        # AND its composite CANNOT participate in other excitation operators
                        trash = set([])
                        skip_row = False
                        for exop in (self.ind_exops[i] for i in row[:-1]):
                            if len(exop) == 2:
                                if exop[0] in trash or exop[1] in trash:
                                    # skip
                                    skip_row = True
                                    break
                                if exop[0] < self.nspatial:
                                    if exop[0] + self.nspatial not in occ_indices:
                                        # skip
                                        skip_row = True
                                        break
                                    else:
                                        # add to trash
                                        trash.add(exop[0])
                                        trash.add(exop[0] + self.nspatial)
                                if exop[0] >= self.nspatial:
                                    if exop[0] - self.nspatial not in occ_indices:
                                        # skip
                                        skip_row = True
                                        break
                                    else:
                                        # add to trash
                                        trash.add(exop[0])
                                        trash.add(exop[0] - self.nspatial)
                                if exop[1] < self.nspatial:
                                    if exop[1] + self.nspatial in occ_indices:
                                        # skip
                                        skip_row = True
                                        break
                                    else:
                                        # add to trash
                                        trash.add(exop[1])
                                        trash.add(exop[1] + self.nspatial)
                                if exop[1] >= self.nspatial:
                                    if exop[1] - self.nspatial in occ_indices:
                                        # skip
                                        skip_row = True
                                        break
                                    else:
                                        # add to trash
                                        trash.add(exop[1])
                                        trash.add(exop[1] - self.nspatial)
                            # FIXME: not sure
                            else:
                                for j in exop:
                                    if j in trash:
                                        # skip
                                        skip_row = True
                                        break
                                    else:
                                        trash.add(j)
                                if skip_row:
                                    break

                        if not skip_row:
                            selected_rows.append(row_ind)

                    indices_multi[exc_order] = indices_sign[selected_rows]
                    # FIXME:
                    # print(selected_rows)
                    # print(indices_multi[exc_order])

            amplitudes = self.product_amplitudes_multi(indices_multi)
            # print(amplitudes)
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
            # FIXME: filter out rows whose excitation operators do not have annihilator that is
            # doubly occupied
            occ_indices = set(slater.occ_indices(sd))
            for exc_order in indices_multi:
                indices_sign = indices_multi[exc_order]
                selected_rows = []
                for row_ind, row in enumerate(indices_sign):
                    # occupied orbitals but have its spin composite also occupied
                    # AND its composite CANNOT participate in other excitation operators
                    trash = set([])
                    skip_row = False
                    for exop in (self.ind_exops[i] for i in row[:-1]):
                        if len(exop) == 2:
                            # skip because annihilator was used in a single excitation as the
                            # opposite spin
                            if exop[0] in trash:
                                # skip
                                skip_row = True
                                break
                            if exop[0] < self.nspatial:
                                # skip because corresponding beta orbital is not occupied
                                if exop[0] + self.nspatial not in occ_indices:
                                    # skip
                                    skip_row = True
                                    break
                                # this annihilator and its beta component cannot be used again
                                else:
                                    # add to trash
                                    trash.add(exop[0])
                                    trash.add(exop[0] + self.nspatial)
                            if exop[0] >= self.nspatial:
                                # skip because corresponding alpha orbital is not occupied
                                if exop[0] - self.nspatial not in occ_indices:
                                    # skip
                                    skip_row = True
                                    break
                                # this annihilator and its alpha component cannot be used again
                                else:
                                    # add to trash
                                    trash.add(exop[0])
                                    trash.add(exop[0] - self.nspatial)
                        # FIXME: not sure
                        else:
                            for j in exop[: len(exop) // 2]:
                                # skip because annihilator was used before as part of a single
                                # excitation or its opposite spin component
                                if j in trash:
                                    # skip
                                    skip_row = True
                                    break
                                # not necessary because same orbital is not annihilated multiple
                                # times by construction
                                else:
                                    trash.add(j)
                            if skip_row:
                                break

                    if not skip_row:
                        selected_rows.append(row_ind)

                indices_multi[exc_order] = indices_sign[selected_rows]
                # FIXME:
                # print(selected_rows)
                # print(indices_multi[exc_order])
                
            amplitudes = self.product_amplitudes_multi(indices_multi, deriv=True)
            # print(amplitudes)
            val = sign * amplitudes
            return val

        if isinstance(self.refwfn, CIWavefunction):
            val = np.zeros(self.nparams)
            for refsd in self.refwfn.sd_vec:
                val += temp_olp(sd, refsd) * self.refwfn.get_overlap(refsd)
            return val
        else:
            return temp_olp(sd, self.refwfn)


    def _olp_double_derivative(self, sd):
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

        def temp_olp_double_deriv(sd1, sd2):
            if sd1 == sd2:
                return np.zeros((self.nparams, self.nparams))
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
            # FIXME: filter out rows whose excitation operators do not have annihilator that is
            # doubly occupied
            occ_indices = set(slater.occ_indices(sd))
            for exc_order in indices_multi:
                indices_sign = indices_multi[exc_order]
                selected_rows = []
                for row_ind, row in enumerate(indices_sign):
                    # occupied orbitals but have its spin composite also occupied
                    # AND its composite CANNOT participate in other excitation operators
                    trash = set([])
                    skip_row = False
                    for exop in (self.ind_exops[i] for i in row[:-1]):
                        if len(exop) == 2:
                            # skip because annihilator was used in a single excitation as the
                            # opposite spin
                            if exop[0] in trash:
                                # skip
                                skip_row = True
                                break
                            if exop[0] < self.nspatial:
                                # skip because corresponding beta orbital is not occupied
                                if exop[0] + self.nspatial not in occ_indices:
                                    # skip
                                    skip_row = True
                                    break
                                # this annihilator and its beta component cannot be used again
                                else:
                                    # add to trash
                                    trash.add(exop[0])
                                    trash.add(exop[0] + self.nspatial)
                            if exop[0] >= self.nspatial:
                                # skip because corresponding alpha orbital is not occupied
                                if exop[0] - self.nspatial not in occ_indices:
                                    # skip
                                    skip_row = True
                                    break
                                # this annihilator and its alpha component cannot be used again
                                else:
                                    # add to trash
                                    trash.add(exop[0])
                                    trash.add(exop[0] - self.nspatial)
                        # FIXME: not sure
                        else:
                            for j in exop[: len(exop) // 2]:
                                # skip because annihilator was used before as part of a single
                                # excitation or its opposite spin component
                                if j in trash:
                                    # skip
                                    skip_row = True
                                    break
                                # not necessary because same orbital is not annihilated multiple
                                # times by construction
                                else:
                                    trash.add(j)
                            if skip_row:
                                break

                    if not skip_row:
                        selected_rows.append(row_ind)

                indices_multi[exc_order] = indices_sign[selected_rows]
                # FIXME:
                # print(selected_rows)
                # print(indices_multi[exc_order])

            hessian = self.product_amplitudes_multi_double_derivative(indices_multi)
            val = sign * hessian
            return val

        if isinstance(self.refwfn, CIWavefunction):
            val = np.zeros(self.nparams, self.nparams)
            for refsd in self.refwfn.sd_vec:
                val += temp_olp_double_deriv(sd, refsd) * self.refwfn.get_overlap(refsd)
            return val
        else:
            return temp_olp_double_deriv(sd, self.refwfn)

