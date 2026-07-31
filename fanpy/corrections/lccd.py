# Corrected LCCD implementation for AP1roG, based on Boguslawski & Ayers (2015): 10.1021/acs.jctc.5b00776.
#
# Fixes applied relative to the original "first implementation":
#   (A) A_mu_nu accumulator was being reset inside the 'for m in pspace' loop,
#       so only the last pspace determinant ever contributed. Fixed by
#       accumulating over m before applying the diagonal correction.
#   (B) Parameters were enumerated over ALL ordered (i,j,a,b), even though
#       tau_(i,j,a,b) == tau_(j,i,b,a) (E_ai and E_bj commute). See the conditions
#       set in the paper. This produced exact duplicate rows/columns in A -> singular 
#       matrix -> needed lstsq.
#       Fixed by enumerating only unique, unordered occ-virt pairs (P<Q).
#   (C) The excluded pair-excitation case (i=j, a=b, t_ii^aa=0) was still
#       being created as a parameter with an all-zero row/column (since
#       _generate_sub_exops strips all its sub-excitations). This is now
#       automatically excluded by the P<Q parameterization.
#   (D) Solve with np.linalg.solve instead of scipy.linalg.lstsq now that A
#       should be square and (generically) non-singular.
#   Energy formula: corrected to properly sum both index orderings of each
#   unique amplitude (see note in compute_correction).

import numpy as np

from fanpy.wfn.geminal.ap1rog import AP1roG
from fanpy.tools import slater


class LCCD:
    r"""Linearized Coupled Cluster Doubles correction on top of AP1roG.

    .. math::

        \ket{AP1roG-LCCD} = e^{\hat{T}_2} \ket{AP1roG}

    .. math::

        \hat{T}_2 = \sum_{\nu} t_{\nu} \hat{\tau}_{\nu}
                  = \frac{1}{2} \sum_{i,j}^{occ} \sum_{a,b}^{virt}{}'
                    t_{ij}^{ab} \hat{E}_{ai} \hat{E}_{bj},
        \qquad t_{ij}^{ab} = t_{ji}^{ba}, \quad t_{ii}^{aa} = 0

    .. math::

        \hat{E}_{ai} = a_a^{\dagger}a_i + a_{\bar{a}}^{\dagger} a_{\bar{i}}

    Because :math:`\hat E_{ai}` and :math:`\hat E_{bj}` commute,
    :math:`\hat\tau_{(i,j,a,b)} = \hat\tau_{(j,i,b,a)}`: this is the *same*
    excitation operator, not two different ones. Parameters here are
    therefore indexed by unordered pairs of (occupied, virtual) spatial
    orbital indices, {(i,a), (j,b)} with (i,a) != (j,b), each stored once.

    Attributes
    ----------
    wfn : AP1roG
        Wavefunction used to compute the correction. Currently only AP1roG
    ham : GeneralizedMolecularHamiltonian
        Hamiltonian used to compute the correction
    pspace : list of int
        Projection space (list of Slater determinants)
    ref_sd : int
        Reference Slater determinant of the wfn instance
    nspatial : int
        Number of spatial orbitals
    nspin : int
        Number of spin orbitals (alpha and beta)
    a_matrix : np.ndarray
        Matrix A (nparams x nparams)
    b_vector : np.ndarray
        Vector B (nparams,)
    amplitudes : np.ndarray
        Solved amplitudes t_nu
    dict_exops_ind : dictionary of 4-tuple of int to int
        Unique excitation operators (i,j,a,b) (spatial-orbital indices,
        i < j-equivalence enforced via pair ordering) mapped to parameter
        index
    dict_ind_exops : dictionary of int to 4-tuple of int
        Inverse of dict_exops_ind
    nparams : int
        Number of independent amplitudes        

    Methods
    -------
    __init__(self, wfn, ham, pspace)
    generate_exops(self)
    _generate_sub_exops(self, mu)
    calculate_b(self)
    calculate_a(self)
    compute_correction(self)
    test_biorthogonality(self, atol=1e-8)
        Diagnostic: checks <mu|nu> == delta_mu_nu for the bra actually used.
    """

    def __init__(self, wfn, ham, pspace):
        r"""Initialize the LCCD correction instance.

        Parameters
        ----------
        wfn : AP1roG
            Wavefunction used to compute the correction (only accepted so far).
        ham : GeneralizedMolecularHamiltonian
            Hamiltonian that defines the system under study.
        pspace : list of int
            Projection space (list of Slater determinants).

        Raises
        ------
        TypeError
            If wavefunction is not an instance (or child) of AP1roG.
        """
        if not isinstance(wfn, AP1roG):
            raise TypeError("Current LCCD implementation only supports AP1roG.")

        self.wfn = wfn
        self.ham = ham
        self.pspace = pspace

        self.ref_sd = wfn.ref_sd
        self.nspatial = wfn.nspatial
        self.nspin = wfn.nspin

        self.generate_exops()
        self.calculate_b()
        self.calculate_a()

    def generate_exops(self):
        r"""Generate the unique double-excitation operators.

        Parameters are indexed by unordered pairs of (occ, virt) spatial
        orbital index pairs {(i,a), (j,b)}, excluding (i,a) == (j,b) (that
        would be the forbidden pair excitation t_ii^aa). This enforces
        t_ij^ab = t_ji^ba automatically by only ever storing one of the two.
        """
        alpha_indices, _ = slater.split_spin(self.ref_sd, self.nspatial)
        occ_alpha_indices = slater.occ_indices(alpha_indices)
        vir_alpha_indices = slater.vir_indices(alpha_indices, self.nspatial)

        occ_virt_pairs = [(i, a) for i in occ_alpha_indices for a in vir_alpha_indices]

        dict_exops_ind = {}
        param_ind = 0
        for idx_p, (i, a) in enumerate(occ_virt_pairs):
            for idx_q, (j, b) in enumerate(occ_virt_pairs):
                if idx_q <= idx_p:
                    # idx_q < idx_p: duplicate of an already-stored pair
                    # idx_q == idx_p: (i,a) == (j,b), the forbidden t_ii^aa term
                    continue
                dict_exops_ind[(i, j, a, b)] = param_ind
                param_ind += 1

        self.dict_exops_ind = dict_exops_ind
        self.dict_ind_exops = {ind: exop for exop, ind in dict_exops_ind.items()}
        self.nparams = len(self.dict_exops_ind)

    def _generate_sub_exops(self, mu):
        r"""Generate the spin sub-excitations of a spatial-orbital excitation
        mu = (i,j,a,b), corresponding to

        .. math::

            \hat E_{ai}\hat E_{bj} = a_a^\dagger a_i\, a_b^\dagger a_j
              + a_a^\dagger a_i\, a_{\bar b}^\dagger a_{\bar j}
              + a_{\bar a}^\dagger a_{\bar i}\, a_b^\dagger a_j
              + a_{\bar a}^\dagger a_{\bar i}\, a_{\bar b}^\dagger a_{\bar j}

        Pair-excitation sub-terms (which vanish by orthogonality to
        |AP1roG>) are stripped.

        Parameters
        ----------
        mu : 4-tuple of int
            (i, j, a, b) spatial-orbital indices.

        Returns
        -------
        sub_exops : list of 4-tuple of int
            Sub-excitations in spin-orbital indices.
        """
        i, j, a, b = mu
        candidates = [
            (i, j, a, b),
            (i, j + self.nspatial, a, b + self.nspatial),
            (i + self.nspatial, j, a + self.nspatial, b),
            (i + self.nspatial, j + self.nspatial, a + self.nspatial, b + self.nspatial),
        ]
        sub_exops = []
        for k, l, c, d in candidates:
            if k == l + self.nspatial and c == d + self.nspatial:
                continue
            if l == k + self.nspatial and d == c + self.nspatial:
                continue
            sub_exops.append((k, l, c, d))
        return sub_exops

    def calculate_b(self):
        r"""Calculate B_mu = <mu|H|AP1roG>, implemented as
        B_mu = sum_k <mu_k|H|AP1roG> over spin sub-excitations mu_k of mu.
        """
        b = np.zeros(self.nparams)
        for ind_mu, mu in self.dict_ind_exops.items():
            b_mu = 0.0
            for sub_mu_exop in self._generate_sub_exops(mu):
                sub_exc_refsd = slater.excite(self.ref_sd, *sub_mu_exop)
                if sub_exc_refsd is None:
                    continue
                b_mu += self.ham.integrate_sd_wfn(sub_exc_refsd, self.wfn)
            b[ind_mu] = b_mu
        self.b_vector = b

    def calculate_a(self):
        r"""Calculate A_mu_nu.

        .. math::

            A_{\mu\nu} = \frac{1}{2}\Big[\sum_{m\in S}
                \big(\sum_k \sum_l \langle \mu_k | \hat H | \tau_{\nu_l} m\rangle\big)
                \langle m | AP1roG\rangle - \delta_{\mu\nu}\langle\Phi_0|\hat H|AP1roG\rangle\Big]

        Fix applied: the sum over m in pspace is now correctly accumulated
        (previously only the last m contributed) and the diagonal correction
        is applied once per (mu, nu) pair after that sum, not per m.
        """
        a = np.zeros((self.nparams, self.nparams))
        const = self.ham.integrate_sd_wfn(self.ref_sd, self.wfn)

        sub_exops_cache = {
            ind: self._generate_sub_exops(exop) for ind, exop in self.dict_ind_exops.items()
        }

        for ind_mu, mu in self.dict_ind_exops.items():
            sub_mu_list = sub_exops_cache[ind_mu]
            # Precompute mu-side excited reference determinants once.
            mu_refsds = [slater.excite(self.ref_sd, *s) for s in sub_mu_list]
            mu_refsds = [sd for sd in mu_refsds if sd is not None]
            if not mu_refsds:
                continue

            for ind_nu, nu in self.dict_ind_exops.items():
                sub_nu_list = sub_exops_cache[ind_nu]

                a_mu_nu = 0.0
                for m in self.pspace:
                    overlap_m = self.wfn.get_overlap(m)
                    if overlap_m == 0.0:
                        continue

                    term = 0.0
                    for sub_nu_exop in sub_nu_list:
                        sub_exc_nu_m = slater.excite(m, *sub_nu_exop)
                        if sub_exc_nu_m is None:
                            continue
                        for sub_exc_mu_refsd in mu_refsds:
                            term += self.ham.integrate_sd_sd(sub_exc_mu_refsd, sub_exc_nu_m)

                    a_mu_nu += term * overlap_m

                if ind_nu == ind_mu:
                    a_mu_nu -= const

                a[ind_mu, ind_nu] = a_mu_nu

        self.a_matrix = 0.5 * a

    def compute_correction(self):
        r"""Compute the LCCD correction to the energy.

        The paper's energy formula sums over ALL (unrestricted) i,j,a,b:

        .. math::

            E_{corr} = \sum_{ijab} t_{ij}^{ab}(2\langle ij|ab\rangle - \langle ij|ba\rangle)

        Since parameters here are stored only once per unordered pair
        {(i,a),(j,b)} (using t_ij^ab = t_ji^ba), each stored amplitude must
        contribute BOTH index orderings' integral terms to recover the full
        (unrestricted) sum:

        .. math::

            E_{corr} = \sum_{\text{unique }\nu} t_\nu \big[
                (2\langle ij|ab\rangle - \langle ij|ba\rangle)
              + (2\langle ji|ba\rangle - \langle ji|ab\rangle) \big]

        Returns
        -------
        E_corr : float
            LCCD correction to the energy.
        """
        try:
            amplitudes = np.linalg.solve(self.a_matrix, -self.b_vector)
        except np.linalg.LinAlgError as err:
            raise np.linalg.LinAlgError(
                "A matrix is singular. This should not happen after "
                "deduplicating parameters (Bug B/C fix); check that pspace "
                "is complete and that the accumulation fix (Bug A) was "
                "applied correctly."
            ) from err

        self.amplitudes = amplitudes

        two_int = self.ham.two_int
        E_corr = 0.0
        for ind_nu, (i, j, a, b) in self.dict_ind_exops.items():
            term_ijab = 2 * two_int[i, j, a, b] - two_int[i, j, b, a]
            term_jiba = 2 * two_int[j, i, b, a] - two_int[j, i, a, b]
            E_corr += amplitudes[ind_nu] * (term_ijab + term_jiba)

        return E_corr

    def test_biorthogonality(self, atol=1e-8):
        r"""Diagnostic check: verifies <mu|nu> == delta_mu_nu for the bra
        actually implemented by _generate_sub_exops, i.e. checks whether the
        diagonal simplification used in calculate_a is actually justified
        for this system. Run this on a small test system (e.g. H4) before
        trusting production results.

        Returns
        -------
        max_off_diag : float
            Largest |<mu|nu>| for mu != nu (should be ~0).
        max_diag_error : float
            Largest |<mu|mu> - 1| (should be ~0).
        """
        overlap = np.zeros((self.nparams, self.nparams))
        sub_exops_cache = {
            ind: self._generate_sub_exops(exop) for ind, exop in self.dict_ind_exops.items()
        }
        for ind_mu, sub_mu_list in sub_exops_cache.items():
            mu_dets = {slater.excite(self.ref_sd, *s) for s in sub_mu_list}
            mu_dets.discard(None)
            for ind_nu, sub_nu_list in sub_exops_cache.items():
                nu_dets = {slater.excite(self.ref_sd, *s) for s in sub_nu_list}
                nu_dets.discard(None)
                overlap[ind_mu, ind_nu] = len(mu_dets & nu_dets)

        diag = np.diag(overlap)
        off_diag = overlap - np.diag(diag)
        return np.max(np.abs(off_diag)), np.max(np.abs(diag - 1))

