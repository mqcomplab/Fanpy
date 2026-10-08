"""Pair Coupled Cluster Doubles (pCCD)/AP1roG wavefunction."""

import numpy as np
from fanpy.tools import slater
from fanpy.wfn.cc.base import BaseCC
from fanpy.wfn.ci.base import CIWavefunction


def _apply_seniority_filter(s_type, ind_exops, nspatial, indices_multi, occ_indices):
    """Drop rows of `indices_multi`, which product of elementary excitation operators
    violates the seniority condition implied by `s_type`. Mutates in place.

    exops : list of list of int
        Excitation operators given as lists of ints. The first half of indices correspond to
        indices to be annihilated, the second half correspond to indices to be created.    
    s_type : str
        Option indicating how the single excitations are allowed to affect the seniority
        of the occupied and/or virtual spaces.
        free: no restrictions imposed.
        sen-o: single excitations must break an occupied pair.
               (only excite from spin-orbital i if its complement is occupied)
        sen-v: single excitations cannot form a pair.
                (only excite to a virtual spin-orbital a if its complement is empty)
        sen-ov: single excitations must break an occupied pair and cannot form extra pairs
                (only excite from spin-orbital i if its complement is occupied and only excite to a virtual spin-orbital a if its complement is empty)
    """
    if s_type == "free":
        return

    if s_type == "sen-o":
        for exc_order in indices_multi:
            indices_sign = indices_multi[exc_order]
            selected_rows = []
            for row_ind, row in enumerate(indices_sign):
                trash = set()
                skip_row = False
                for exop in (ind_exops[i] for i in row[:-1]):
                    if len(exop) == 2:
                        if exop[0] in trash:
                            skip_row = True
                            break
                        if exop[0] < nspatial:
                            if exop[0] + nspatial not in occ_indices:
                                skip_row = True
                                break
                            trash.add(exop[0])
                            trash.add(exop[0] + nspatial)
                        if exop[0] >= nspatial:
                            if exop[0] - nspatial not in occ_indices:
                                skip_row = True
                                break
                            trash.add(exop[0])
                            trash.add(exop[0] - nspatial)
                    else:
                        for j in exop[: len(exop) // 2]:
                            if j in trash:
                                skip_row = True
                                break
                            trash.add(j)
                        if skip_row:
                            break
                if not skip_row:
                    selected_rows.append(row_ind)
            indices_multi[exc_order] = indices_sign[selected_rows]

    elif s_type == "sen-v":
        for exc_order in indices_multi:
            indices_sign = indices_multi[exc_order]
            selected_rows = []
            for row_ind, row in enumerate(indices_sign):
                trash = set()
                skip_row = False
                for exop in (ind_exops[i] for i in row[:-1]):
                    if len(exop) == 2:
                        if exop[1] in trash:
                            skip_row = True
                            break
                        if exop[1] < nspatial:
                            if exop[1] + nspatial in occ_indices:
                                skip_row = True
                                break
                            trash.add(exop[1])
                            trash.add(exop[1] + nspatial)
                        if exop[1] >= nspatial:
                            if exop[1] - nspatial in occ_indices:
                                skip_row = True
                                break
                            trash.add(exop[1])
                            trash.add(exop[1] - nspatial)
                    else:
                        for j in exop[len(exop) // 2 :]:
                            if j in trash:
                                skip_row = True
                                break
                            trash.add(j)
                        if skip_row:
                            break
                if not skip_row:
                    selected_rows.append(row_ind)
            indices_multi[exc_order] = indices_sign[selected_rows]

    elif s_type == "sen-ov":
        for exc_order in indices_multi:
            indices_sign = indices_multi[exc_order]
            selected_rows = []
            for row_ind, row in enumerate(indices_sign):
                trash = set()
                skip_row = False
                for exop in (ind_exops[i] for i in row[:-1]):
                    if len(exop) == 2:
                        if exop[0] in trash or exop[1] in trash:
                            skip_row = True
                            break
                        if exop[0] < nspatial:
                            if exop[0] + nspatial not in occ_indices:
                                skip_row = True
                                break
                            trash.add(exop[0])
                            trash.add(exop[0] + nspatial)
                        if exop[0] >= nspatial:
                            if exop[0] - nspatial not in occ_indices:
                                skip_row = True
                                break
                            trash.add(exop[0])
                            trash.add(exop[0] - nspatial)
                        if exop[1] < nspatial:
                            if exop[1] + nspatial in occ_indices:
                                skip_row = True
                                break
                            trash.add(exop[1])
                            trash.add(exop[1] + nspatial)
                        if exop[1] >= nspatial:
                            if exop[1] - nspatial in occ_indices:
                                skip_row = True
                                break
                            trash.add(exop[1])
                            trash.add(exop[1] - nspatial)
                    else:
                        for j in exop:
                            if j in trash:
                                skip_row = True
                                break
                            trash.add(j)
                        if skip_row:
                            break
                if not skip_row:
                    selected_rows.append(row_ind)
            indices_multi[exc_order] = indices_sign[selected_rows]

    else:
        raise ValueError("Invalid s_type: {}".format(s_type))


class fastPCCD(BaseCC):
    r"""Pair coupled-cluster doubles (pCCD/AP1roG) wavefunction.

    The wavefunction is parameterized as

    .. math::

        \left| \Psi_{\mathrm{pCCD}} \right\rangle
        = \prod_{\mu \in \mathcal{E}}
          \left( 1 + t_\mu\, \tau_\mu \right) \left| \Phi_0 \right\rangle,
        \qquad
        \mathcal{E} = \left\{ \tau_{i\bar{i}}^{a\bar{a}} \right\},

    where :math:`i, j, k, \ldots` index occupied spin orbitals and
    :math:`a, b, c, \ldots` index virtual spin orbitals. The excitation
    pool :math:`\mathcal{E}` consists exclusively of pair-preserving double
    excitations that annihilate an occupied pair :math:`(i, \bar{i})` and
    create electrons in a virtual pair :math:`(a, \bar{a})`, equivalent to
    the AP1roG geminal formalism via the one-reference orbital construction
    of Limacher et al.[1_]

    The reference wavefunction (``refwfn``) must be a seniority-0 Slater
    determinant.

    See Ref. [2]_ for the full theoretical background.

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

    References
    ----------
    .. [1] P. A. Limacher, P. W. Ayers, P. A. Johnson, S. De Baerdemacker, D. Van Neck, and
           P. Bultinck, "A new mean-field method suitable for strongly correlated electrons: 
           Computationally facile antisymmetric products of nonorthogonal geminals," 
           *J. Chem. Theory Comput.*, **9**, 1394-1401 (2013).
           https://doi.org/10.1021/ct300902c

    .. [2] P. B. Gaikwad, T. D. Kim, M. Richer, R. A. Lokhande, G. Sánchez-Díaz,
           P. A. Limacher, P. W. Ayers and R. A. Miranda-Quintana, "Coupled-cluster-inspired 
           geminal wavefunctions," *J. Chem. Phys.* **160**, 144108 (2024).
           https://doi.org/10.1063/5.0202035
    """

    #: Reproduce the reference implementation's derivative filtering, which ignores
    #: `s_type` and always applies the "sen-o" rule. Set to False to make the
    #: derivative honour `s_type`. Only matters when s_type != "sen-o".
    legacy_deriv_filter = True

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
        # indices are used for assign_exops, without any modification 
        # we can pass it here to BaseCC as assign_exops gets overwritten by PCCD and its child classes
        self._init_caches()
        super().__init__(
            nelec,
            nspin,
            memory=memory,
            params=params,
            exop_combinations=exop_combinations,
            indices=indices,
            refresh_exops=refresh_exops,
        )
        self.assign_s_type(s_type=s_type)
        self.assign_ranks(ranks=ranks)
        self.assign_refwfn(refwfn=refwfn)

        # mapping from parameter indices to excitation operators
        self.ind_exops = sorted(self.exops.keys(), key=lambda x: self.exops[x])

    # ------------------------------------------------------------------
    # cache management
    # ------------------------------------------------------------------
    def _init_caches(self):
        self._filtered_ref_basis = set()   # keys filtered under occ_indices(sd2)
        self._filtered_sd1_basis = set()   # keys filtered under occ_indices(sd1)
        self._olp_cache = {}
        self._olp_deriv_cache = {}

    def clear_filter_cache(self):
        """Forget which excitation have already been seniority-filtered.

        Called automatically when the reference changes or when `exop_combinations`
        is discarded. Call manually only if you mutate wavefunction state through a
        path this class doesn't intercept.
        """
        self._filtered_ref_basis.clear()
        self._filtered_sd1_basis.clear()

    def clear_overlap_cache(self):
        """Invalidate memoized overlaps. Called automatically whenever the
        parameters or the reference change."""
        self._olp_cache = {}
        self._olp_deriv_cache = {}

    def assign_params(self, *args, **kwargs):
        super().assign_params(*args, **kwargs)
        self.clear_overlap_cache()

    def clear_cache(self, *args, **kwargs):
        # BaseCC's own invalidation hook; some call sites use it instead of
        # assign_params.
        super().clear_cache(*args, **kwargs)
        self.clear_overlap_cache()

    def generate_possible_exops(self, a_inds, c_inds):
        # BaseCC may discard exop_combinations wholesale (refresh_exops). If it
        # does, previously filtered keys come back unfiltered, so the filter cache
        # must be dropped at the same moment.
        n_before = len(self.exop_combinations)
        result = super().generate_possible_exops(a_inds, c_inds)
        if len(self.exop_combinations) < n_before:
            self.clear_filter_cache()
        return result

    # ------------------------------------------------------------------
    # assignment methods (carried over from the reference implementation)
    # ------------------------------------------------------------------
    def assign_nelec(self, nelec):
        """Assign the number of electrons. Must be even."""
        super().assign_nelec(nelec)
        if self.nelec % 2 != 0:
            raise ValueError("Odd number of electrons is not supported")

    def assign_ranks(self, ranks=None):
        """Assign the ranks of the excitation operators. Only rank 2 is allowed."""
        if ranks is not None:
            raise ValueError("Only the default, rank = 2, is allowed")
        if self.nelec <= 1:
            raise ValueError("Only wavefunctions with more than 1 electron can be considered")
        self.ranks = [2]

    def assign_exops(self, indices=None):
        """Assign the pair-double excitation operators [i, i_bar, a, a_bar].

        No single excitations are generated, so the `s_type` filtering is dormant
        for the standard pCCD excitation space.
        """
        if indices is not None:
            raise TypeError(
                "Only the excitation operators constructed by default from "
                "the given reference Slater determinant are allowed"
            )
        exops = {}
        counter = 0
        ex_from = slater.occ_indices(self.refwfn)
        ex_to = [i for i in range(self.nspin) if i not in ex_from]
        for occ_alpha in ex_from[: len(ex_from) // 2]:
            for virt_alpha in ex_to[: len(ex_to) // 2]:
                exop = [
                    occ_alpha,
                    occ_alpha + self.nspatial,
                    virt_alpha,
                    virt_alpha + self.nspatial,
                ]
                exops[tuple(exop)] = counter
                counter += 1
        self.exops = exops

    def assign_refwfn(self, refwfn=None):
        """Assign the seniority-0 reference determinant the CC operator acts on."""
        if refwfn is None:
            self.refwfn = slater.ground(nocc=self.nelec, norbs=self.nspin)
        else:
            if not isinstance(refwfn, int):
                raise TypeError("refwfn must be a int object")
            if slater.total_occ(refwfn) != self.nelec:
                raise ValueError("refwfn must have {} electrons".format(self.nelec))
            if not all(
                [
                    i + self.nspatial in slater.occ_indices(refwfn)
                    for i in slater.occ_indices(refwfn)[: self.nelec // 2]
                ]
                + [
                    i - self.nspatial in slater.occ_indices(refwfn)
                    for i in slater.occ_indices(refwfn)[self.nelec // 2 :]
                ]
            ):
                raise ValueError("refwfn must be a seniority-0 wavefuntion")
            self.refwfn = refwfn

        # a new reference invalidates both caches: the filter keys were derived
        # against the old reference, and every stored overlap was computed from it.
        self.clear_filter_cache()
        self.clear_overlap_cache()

    def assign_s_type(self, s_type):
        """Assign the seniority-breaking condition used for single excitations.

        free   : no restrictions imposed.
        sen-o  : single excitations must break an occupied pair.
        sen-v  : single excitations cannot form a virtual pair.
        sen-ov : both.

        For doubles-only wavefunctions these restrictions do not affect the overlap,
        since the excitation space contains no singles.
        """
        if s_type not in ["free", "sen-o", "sen-v", "sen-ov"]:
            raise ValueError("Invalid s_type")
        self.s_type = s_type
        self.clear_filter_cache()
        self.clear_overlap_cache()

    # ------------------------------------------------------------------
    # overlaps
    # ------------------------------------------------------------------
    def _olp(self, sd):
        r"""Matrix element of the CC operator between `sd` and the reference.

        .. math::

            \left\langle m_1 \right| \prod_{u,v}
            \left( 1 + t_u^v \hat{\tau}_u^v \right) \left| m_2 \right\rangle
        """
        if sd in self._olp_cache:
            return self._olp_cache[sd]

        def temp_olp(sd1, sd2):
            if sd1 == sd2:
                return 1.0
            # indices of the annihilation (a_inds) and creation (c_inds) operators
            # that turn sd2 into sd1
            c_inds, a_inds = slater.diff_orbs(sd1, sd2)
            if isinstance(a_inds, np.ndarray):
                a_inds = a_inds.tolist()
            if isinstance(c_inds, np.ndarray):
                c_inds = c_inds.tolist()

            sign = slater.sign_excite(sd2, a_inds, c_inds)

            key = tuple(a_inds + c_inds)
            if key not in self.exop_combinations:
                self.generate_possible_exops(a_inds, c_inds)
            indices_multi = self.exop_combinations[key]

            if self.s_type != "free" and key not in self._filtered_ref_basis:
                _apply_seniority_filter(
                    self.s_type,
                    self.ind_exops,
                    self.nspatial,
                    indices_multi,
                    set(slater.occ_indices(sd2)),
                )
                self._filtered_ref_basis.add(key)

            return sign * self.product_amplitudes_multi(indices_multi)

        if isinstance(self.refwfn, CIWavefunction):
            val = 0.0
            for refsd in self.refwfn.sd_vec:
                val += temp_olp(sd, refsd) * self.refwfn.get_overlap(refsd)
        else:
            val = temp_olp(sd, self.refwfn)

        self._olp_cache[sd] = val
        return val

    def _olp_deriv(self, sd):
        """Derivative of the overlap with `sd` with respect to each amplitude."""
        if sd in self._olp_deriv_cache:
            # hand out a copy: the caller must never be able to corrupt the cache
            return self._olp_deriv_cache[sd].copy()

        s_type = "sen-o" if self.legacy_deriv_filter else self.s_type

        def temp_olp(sd1, sd2):
            if sd1 == sd2:
                return np.zeros(self.nparams)
            c_inds, a_inds = slater.diff_orbs(sd1, sd2)
            if isinstance(a_inds, np.ndarray):
                a_inds = a_inds.tolist()
            if isinstance(c_inds, np.ndarray):
                c_inds = c_inds.tolist()

            sign = slater.sign_excite(sd2, a_inds, c_inds)

            key = tuple(a_inds + c_inds)
            if key not in self.exop_combinations:
                self.generate_possible_exops(a_inds, c_inds)
            indices_multi = self.exop_combinations[key]

            # NOTE: the occupancy basis here is occ_indices(sd), the outer `sd`,
            # i.e. sd1; not occ_indices(sd2) as in _olp. Carried over from the
            # reference implementation, where which determinant is correct for
            # occupancy filtering is flagged as needing further investigation.
            if s_type != "free" and key not in self._filtered_sd1_basis:
                _apply_seniority_filter(
                    s_type,
                    self.ind_exops,
                    self.nspatial,
                    indices_multi,
                    set(slater.occ_indices(sd)),
                )
                self._filtered_sd1_basis.add(key)

            return sign * self.product_amplitudes_multi(indices_multi, deriv=True)

        if isinstance(self.refwfn, CIWavefunction):
            val = np.zeros(self.nparams)
            for refsd in self.refwfn.sd_vec:
                val += temp_olp(sd, refsd) * self.refwfn.get_overlap(refsd)
        else:
            val = temp_olp(sd, self.refwfn)

        self._olp_deriv_cache[sd] = val
        return val.copy()

    def _olp_double_derivative(self, sd):
        r"""Second derivative of the overlap with `sd`.

        .. math::

            H_{ab} = \frac{\partial^2}{\partial t_a \partial t_b}
                \left\langle \mathrm{SD} \mid \Psi_{\mathrm{CC}} \right\rangle

        Not memoized: the Hessian is evaluated far less often than the value and
        gradient, and caching one (nparams, nparams) array per determinant is
        expensive. The filter cache is still used, keyed on the same sd1 basis as
        `_olp_deriv`.
        """
        s_type = "sen-o" if self.legacy_deriv_filter else self.s_type

        def temp_olp_double_deriv(sd1, sd2):
            if sd1 == sd2:
                return np.zeros((self.nparams, self.nparams))
            c_inds, a_inds = slater.diff_orbs(sd1, sd2)
            if isinstance(a_inds, np.ndarray):
                a_inds = a_inds.tolist()
            if isinstance(c_inds, np.ndarray):
                c_inds = c_inds.tolist()

            sign = slater.sign_excite(sd2, a_inds, c_inds)

            key = tuple(a_inds + c_inds)
            if key not in self.exop_combinations:
                self.generate_possible_exops(a_inds, c_inds)
            indices_multi = self.exop_combinations[key]

            # same occupancy basis as _olp_deriv: occ_indices(sd)
            if s_type != "free" and key not in self._filtered_sd1_basis:
                _apply_seniority_filter(
                    s_type,
                    self.ind_exops,
                    self.nspatial,
                    indices_multi,
                    set(slater.occ_indices(sd)),
                )
                self._filtered_sd1_basis.add(key)

            return sign * self.product_amplitudes_multi_double_derivative(indices_multi)

        if isinstance(self.refwfn, CIWavefunction):
            val = np.zeros((self.nparams, self.nparams))
            for refsd in self.refwfn.sd_vec:
                val += temp_olp_double_deriv(sd, refsd) * self.refwfn.get_overlap(refsd)
            return val
        return temp_olp_double_deriv(sd, self.refwfn)


