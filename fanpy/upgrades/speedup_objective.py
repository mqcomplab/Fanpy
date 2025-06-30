"""Must be imported after speedup_apg."""
from collections import Counter
import itertools as it
import numpy as np
from scipy.special import comb
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.geminal.base import BaseGeminal
from fanpy.wfn.geminal.apig import APIG
from fanpy.wfn.geminal.ap1rog import AP1roG
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.upgrades.cext_apg import _olp_deriv_internal_ap1rog
from fanpy.upgrades.cext_apg_parallel import _olp_deriv_internal_apig
from fanpy.upgrades.cext_apg_parallel2 import _olp_deriv_internal
import fanpy.tools.slater as slater
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.eqn.base import BaseSchrodinger
from fanpy.upgrades.cext_objective import get_energy_one_proj_deriv
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.eqn.constraints.norm import NormConstraint
from fanpy.eqn.constraints.energy import EnergyConstraint

from fanpy.tools import sd_list


def get_overlap(self, sd, deriv=None):
    # if no derivatization
    sd_alpha, sd_beta = slater.split_spin(sd, self.nspatial)
    if deriv is None:
        if self.seniority == 0 and sd_alpha != sd_beta:
            return 0
        return self._olp(sd)

    if self.seniority == 0 and sd_alpha != sd_beta:
        return np.zeros(self.nparams)[deriv]
    return self._olp_deriv(sd)[deriv]


def _olp_deriv(self, sd):
    # NOTE: This module requires the sped up objective functions
    occ_indices = list(slater.occ_indices(sd))

    if hasattr(self, "temp_generator"):
        orbpair_generator = self.temp_generator
    else:
        orbpair_generator = self.generate_possible_orbpairs(occ_indices)
    if not orbpair_generator:
        return np.zeros(self.nparams)

    return _olp_deriv_internal(orbpair_generator, self.params, self.nspin)


def _olp_deriv_apig(self, sd):
    # NOTE: This module requires the sped up objective functions
    occ_indices = list(slater.occ_indices(sd))

    if hasattr(self, "temp_generator"):
        orbpair_generator = self.temp_generator
    else:
        orbpair_generator = self.generate_possible_orbpairs(occ_indices)
    if not orbpair_generator:
        return np.zeros(self.nparams)

    return _olp_deriv_internal_apig(orbpair_generator, self.params)


def _olp_deriv_ap1rog(self, sd):
    # NOTE: This module requires the sped up objective functions
    spatial_sd, _ = slater.split_spin(sd, self.nspatial)
    spatial_ref_sd, _ = slater.split_spin(self.ref_sd, self.nspatial)
    orbs_annihilated, orbs_created = slater.diff_orbs(spatial_ref_sd, spatial_sd)
    inds_annihilated = np.array(
        [self.dict_reforbpair_ind[(i, i + self.nspatial)] for i in orbs_annihilated]
    )
    inds_created = np.array(
        [self.dict_orbpair_ind[(i, i + self.nspatial)] for i in orbs_created]
    )

    # FIXME: missing signature. see apig. Not a problem if alpha beta spin pairing
    return _olp_deriv_internal_ap1rog(self.params, inds_created, inds_annihilated)
    # return self.compute_permanent(row_inds=inds_annihilated, col_inds=inds_created, deriv=deriv)


def get_overlap_ap1rog(self, sd, deriv=None):
    # cut off beta part (for just the alpha/spatial part)
    spatial_ref_sd, _ = slater.split_spin(self.ref_sd, self.nspatial)
    spatial_sd, _ = slater.split_spin(sd, self.nspatial)
    # get indices of the occupied orbitals
    orbs_annihilated, orbs_created = slater.diff_orbs(spatial_ref_sd, spatial_sd)

    if deriv is None:
        zero = 0.0
    elif not isinstance(deriv, np.ndarray):
        raise TypeError
    else:
        zero = np.zeros(len(deriv))

    # if different number of electrons
    if len(orbs_annihilated) != len(orbs_created):
        return zero
    # if different seniority
    if slater.get_seniority(sd, self.nspatial) != 0:
        return zero

    # convert to spatial orbitals
    # NOTE: these variables are essentially the same as the output of
    #       generate_possible_orbpairs
    inds_annihilated = np.array(
        [self.dict_reforbpair_ind[(i, i + self.nspatial)] for i in orbs_annihilated]
    )
    inds_created = np.array(
        [self.dict_orbpair_ind[(i, i + self.nspatial)] for i in orbs_created]
    )

    # if no derivatization
    if deriv is None:
        if inds_annihilated.size == inds_created.size == 0:
            return 1.0

        return self._olp(sd)

    if inds_annihilated.size == inds_created.size == 0:
        return zero

    deriv = deriv[deriv < self.nparams]
    zero[deriv < self.nparams] = self._olp_deriv(sd)[deriv]
    return zero


def get_overlap_ci(self, sd, deriv=None):
    sd = slater.internal_sd(sd)
    if deriv is None:
        try:
            return self.params[self.dict_sd_index[sd]]
        except KeyError:
            return 0.0

    output = np.zeros(self.nparams)
    try:
        output[self.dict_sd_index[sd]] = 1.0
    except KeyError:
        pass
    return output[deriv]


old_integrate_sd_wfn = RestrictedMolecularHamiltonian.integrate_sd_wfn


def integrate_sd_wfn2(self, sd, wfn, wfn_deriv):
    nspatial = self.nspin // 2
    sd = slater.internal_sd(sd)
    occ_indices = np.array(slater.occ_indices(sd))
    vir_indices = np.array(slater.vir_indices(sd, self.nspin))
    # FIXME: hardcode slater determinant structure
    occ_alpha = occ_indices[occ_indices < nspatial]
    vir_alpha = vir_indices[vir_indices < nspatial]
    occ_beta = occ_indices[occ_indices >= nspatial]
    vir_beta = vir_indices[vir_indices >= nspatial]

    def fromiter(iterator, dtype, ydim, count):
        return np.fromiter(it.chain.from_iterable(iterator), dtype, count=int(count)).reshape(-1, ydim)

    output = np.zeros((3, wfn.nparams))

    overlaps_zero = np.array([wfn.get_overlap(sd, deriv=wfn_deriv)]).reshape(-1, wfn.nparams)
    output += np.sum(
        self._integrate_sd_sds_zero(occ_alpha, (occ_beta - nspatial))[:, :, None] * overlaps_zero,
        axis=1
    )

    occ_one_alpha = fromiter(it.combinations(occ_alpha.tolist(), 1), int, 1, count=len(occ_alpha))
    occ_one_alpha = np.left_shift(1, occ_one_alpha[:, 0])
    vir_one_alpha = fromiter(it.combinations(vir_alpha.tolist(), 1), int, 1, count=len(vir_alpha))
    vir_one_alpha = np.left_shift(1, vir_one_alpha[:, 0])
    overlaps_one_alpha = np.array(
        [
            wfn.get_overlap(sd_exc, deriv=wfn_deriv)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_one_alpha)[:, None],
                                                 vir_one_alpha[None, :]))
        ]
    ).reshape(-1, wfn.nparams)
    integrals_one_alpha = self._integrate_sd_sds_one_alpha(
        occ_alpha, (occ_beta - nspatial), vir_alpha
    )
    output += np.sum(integrals_one_alpha[:, :, None] * overlaps_one_alpha, axis=1)

    occ_one_beta = fromiter(it.combinations(occ_beta.tolist(), 1), int, 1, count=len(occ_beta))
    occ_one_beta = np.left_shift(1, occ_one_beta[:, 0])
    vir_one_beta = fromiter(it.combinations(vir_beta.tolist(), 1), int, 1, count=len(vir_beta))
    vir_one_beta = np.left_shift(1, vir_one_beta[:, 0])
    overlaps_one_beta = np.array(
        [
            wfn.get_overlap(sd_exc, deriv=wfn_deriv)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_one_beta)[:, None],
                                                 vir_one_beta[None, :]))
        ]
    ).reshape(-1, wfn.nparams)
    integrals_one_beta = self._integrate_sd_sds_one_beta(
        occ_alpha, (occ_beta - nspatial), (vir_beta - nspatial)
    )
    output += np.sum(integrals_one_beta[:, :, None] * overlaps_one_beta, axis=1)

    occ_two_aa = fromiter(it.combinations(occ_alpha.tolist(), 2),
                                int, ydim=2, count=2*comb(len(occ_alpha), 2))
    occ_two_aa = np.left_shift(1, occ_two_aa)
    occ_two_aa = np.bitwise_or(occ_two_aa[:, 0], occ_two_aa[:, 1])
    vir_two_aa = fromiter(it.combinations(vir_alpha.tolist(), 2),
                                int, 2, count=2*comb(len(vir_alpha), 2))
    vir_two_aa = np.left_shift(1, vir_two_aa)
    vir_two_aa = np.bitwise_or(vir_two_aa[:, 0], vir_two_aa[:, 1])
    overlaps_two_aa = np.array(
        [
            wfn.get_overlap(sd_exc, deriv=wfn_deriv)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_aa)[:, None],
                                                 vir_two_aa[None, :]))
        ]
    ).reshape(-1, wfn.nparams)
    if occ_alpha.size > 1 and vir_alpha.size > 1:
        integrals_two_aa = self._integrate_sd_sds_two_aa(
            occ_alpha, (occ_beta - nspatial), vir_alpha
        )
        output[1:] += np.sum(integrals_two_aa[:, :, None] * overlaps_two_aa, axis=1)

    occ_two_ab = fromiter(it.product(occ_alpha.tolist(), occ_beta.tolist()),
                          int, 2, count=2*len(occ_alpha) * len(occ_beta))
    occ_two_ab = np.left_shift(1, occ_two_ab)
    occ_two_ab = np.bitwise_or(occ_two_ab[:, 0], occ_two_ab[:, 1])
    vir_two_ab = fromiter(it.product(vir_alpha.tolist(), vir_beta.tolist()),
                          int, 2, count=2*len(vir_alpha) * len(vir_beta))
    vir_two_ab = np.left_shift(1, vir_two_ab)
    vir_two_ab = np.bitwise_or(vir_two_ab[:, 0], vir_two_ab[:, 1])
    overlaps_two_ab = np.array(
        [
            wfn.get_overlap(sd_exc, deriv=wfn_deriv)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_ab)[:, None],
                                                 vir_two_ab[None, :]))
        ]
    ).reshape(-1, wfn.nparams)
    if (
        occ_alpha.size > 0 and (occ_beta - nspatial).size > 0 and
        vir_alpha.size > 0 and (vir_beta - nspatial).size > 0
    ):
        integrals_two_ab = self._integrate_sd_sds_two_ab(
            occ_alpha, (occ_beta - nspatial), vir_alpha, (vir_beta - nspatial)
        )
        output[1] += np.sum(integrals_two_ab[:, None] * overlaps_two_ab, axis=0)

    occ_two_bb = fromiter(it.combinations(occ_beta.tolist(), 2),
                          int, 2, count=2*comb(len(occ_beta), 2))
    occ_two_bb = np.left_shift(1, occ_two_bb)
    occ_two_bb = np.bitwise_or(occ_two_bb[:, 0], occ_two_bb[:, 1])
    vir_two_bb = fromiter(it.combinations(vir_beta.tolist(), 2),
                          int, 2, count=2*comb(len(vir_beta), 2))
    vir_two_bb = np.left_shift(1, vir_two_bb)
    vir_two_bb = np.bitwise_or(vir_two_bb[:, 0], vir_two_bb[:, 1])
    overlaps_two_bb = np.array(
        [
            wfn.get_overlap(sd_exc, deriv=wfn_deriv)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_bb)[:, None],
                                                 vir_two_bb[None, :]))
        ]
    ).reshape(-1, wfn.nparams)
    if (occ_beta - nspatial).size > 1 and (vir_beta - nspatial).size > 1:
        integrals_two_bb = self._integrate_sd_sds_two_bb(
            occ_alpha, (occ_beta - nspatial), (vir_beta - nspatial)
        )
        output[1:] += np.sum(integrals_two_bb[:, :, None] * overlaps_two_bb, axis=1)

    return np.sum(output, axis=0)


def integrate_sd_wfn(self, sd, wfn, wfn_deriv=None):
    if wfn_deriv is not None:
        return integrate_sd_wfn2(self, sd, wfn, wfn_deriv)
    return old_integrate_sd_wfn(self, sd, wfn)


def wrapped_get_overlap(self, sd, deriv=None):
    if deriv is None:
        return self.wfn.get_overlap(sd)

    d_overlaps = self.wfn.get_overlap(sd, self.indices_component_params[self.wfn])
    output = np.zeros(self.active_nparams)
    # FIXME
    try:
        output[self.indices_objective_params[self.wfn]] = d_overlaps
    except KeyError:
        pass
    return output


def wrapped_integrate_wfn_sd(self, sd, deriv=None):
    # pylint: disable=C0103
    if deriv is None:
        # return np.sum(self.ham.integrate_sd_wfn(sd, self.wfn))
        return (self.ham.integrate_sd_wfn(sd, self.wfn))

    # NOTE: assume wavefunction parameters first, then hamiltonian parameters
    ham_nparams = self.ham.nparams

    output = np.zeros(self.active_nparams)
    try:
        # output[self.indices_objective_params[self.wfn]] = np.sum(
        #     self.ham.integrate_sd_wfn(sd, self.wfn, wfn_deriv=self.indices_component_params[self.wfn]), axis=0
        output[self.indices_objective_params[self.wfn]] = (
            self.ham.integrate_sd_wfn(sd, self.wfn, wfn_deriv=self.indices_component_params[self.wfn])
        )
    except KeyError:
        pass
    try:
        # output[self.indices_objective_params[self.ham]] = np.sum(
        #     self.ham.integrate_sd_wfn_deriv(sd, self.wfn, self.indices_component_params[self.ham]), axis=0
        output[self.indices_objective_params[self.ham]] = (
            self.ham.integrate_sd_wfn_deriv(sd, self.wfn, self.indices_component_params[self.ham])
        )
    except KeyError:
        pass
    return output


def parallel_energy_deriv(sds_objective):
    sds, objective = sds_objective
    import fanpy.upgrades.speedup_sd
    import fanpy.upgrades.speedup_sign
    if isinstance(objective.wfn, BaseGeminal):
        import fanpy.upgrades.speedup_apg

    return get_energy_one_proj_deriv(objective.wfn, objective.ham, sds)


def parallel_energy(sds_objective):
    sds, objective = sds_objective
    import fanpy.upgrades.speedup_sd
    import fanpy.upgrades.speedup_sign
    if isinstance(objective.wfn, BaseGeminal):
        import fanpy.upgrades.speedup_apg

    get_overlap = objective.wrapped_get_overlap
    integrate_wfn_sd = objective.wrapped_integrate_wfn_sd

    integral_update, norm_update = 0, 0

    overlaps = np.fromiter((get_overlap(sd) for sd in sds), float, count=len(sds))
    integrals = np.fromiter((integrate_wfn_sd(sd) for sd in sds), float, count=len(sds))
    norm_update = np.sum(overlaps ** 2)
    integral_update = np.sum(overlaps * integrals)

    return integral_update, norm_update


def get_energy_one_proj(self, pspace, deriv=None, parallel=None):
    if isinstance(pspace, int):
        pspace = [pspace]

    if deriv is not None:
        if parallel:
            ncores = parallel._processes
            parallel_pspace = [
                pspace[i * len(pspace)//ncores: (i+1) * len(pspace)//ncores] for i in range(ncores)
            ]
            if len(pspace) % ncores != 0:
                parallel_pspace[-1] += pspace[ncores * len(pspace)//ncores:]
            parallel_params = [(i, self) for i in parallel_pspace]

            energy_d_energy_norm_d_norm = np.array(
                parallel.map(parallel_energy_deriv, parallel_params)
            )
            energy_d_energy_norm_d_norm = np.sum(energy_d_energy_norm_d_norm, axis=0)
            energy, d_energy, norm, d_norm = energy_d_energy_norm_d_norm
            energy /= norm
            d_energy /= norm
            d_energy[:self.wfn.nparams] -= d_norm * energy / norm

            output = np.zeros(self.params.size)
            try:
                output[self.indices_objective_params[self.wfn]] = d_energy[:self.wfn.nparams]
            except KeyError:
                pass
            try:
                output[self.indices_objective_params[self.ham]] = d_energy[self.wfn.nparams:]
            except KeyError:
                pass
            return output

        energy, d_energy, norm, d_norm = get_energy_one_proj_deriv(self.wfn, self.ham, list(pspace))
        energy /= norm
        d_energy /= norm
        d_energy[:self.wfn.nparams] -= d_norm * energy / norm

        output = np.zeros(self.params.size)
        try:
            output[self.indices_objective_params[self.wfn]] = d_energy[:self.wfn.nparams][self.indices_component_params[self.wfn]]
        except KeyError:
            pass
        try:
            output[self.indices_objective_params[self.ham]] = d_energy[self.wfn.nparams:][self.indices_component_params[self.ham]]
        except KeyError:
            pass
        return output

    if parallel:
        ncores = parallel._processes
        parallel_pspace = [
            pspace[i * len(pspace)//ncores: (i+1) * len(pspace)//ncores] for i in range(ncores)
        ]
        if len(pspace) % ncores != 0:
            parallel_pspace[-1] += pspace[ncores * len(pspace)//ncores:]
        parallel_params = [(i, self) for i in parallel_pspace]

        energy_norm = np.array(parallel.map(parallel_energy, parallel_params))
        energy_norm = np.sum(energy_norm, axis=0)
        return energy_norm[0] / energy_norm[1]

    energy, norm = parallel_energy((pspace, self))
    return energy / norm


def objective_energy(self, params, parallel=None, assign=True, normalize=False, save=True):
    params = np.array(params)
    # Assign params
    if assign:
        self.assign_params(params)
    # Normalize
    if normalize:
        self.wfn.normalize(self.refwfn)
    # Save params
    if save:
        self.save_params()
    energy = self.get_energy_one_proj(self.refwfn, parallel=parallel)
    if hasattr(self, 'print_energy'):
        if self.print_energy:
            print("(Mid Optimization) Electronic Energy: {}".format(energy))
        else:
            try:
                self.print_queue['energy'] = energy
            except AttributeError:
                self.print_queue = {'energy': energy}
    return energy


def gradient_energy(self, params, parallel=None, assign=True, normalize=True, save=True):
    params = np.array(params)
    # Assign params
    if assign:
        self.assign_params(params)
    # Normalize
    if normalize:
        self.wfn.normalize(self.refwfn)
    # Save params
    if save:
        self.save_params()

    return self.get_energy_one_proj(self.refwfn, deriv=True, parallel=parallel)


def assign_pspace_system(self, pspace=None):
    if pspace is None:
        pspace = sd_list.sd_list(
            self.wfn.nelec, self.wfn.nspin, spin=self.wfn.spin, seniority=self.wfn.seniority
        )
    elif isinstance(pspace, (list, tuple)) and all(
        slater.is_sd_compatible(state) or isinstance(state, CIWavefunction) for state in pspace
    ):
        # Fixed a bug here (elif -> if)
        # Without this fix, pspace would be stored as a list by default, but tuple if provided
        # With this fix, pspace is stored as a list always
        pspace = list(pspace)
    else:
        raise TypeError(
            "Projected space must be given as a list/tuple of Slater determinants. "
            "See `tools.slater` for compatible Slater determinant formats."
        )
    self.pspace = pspace


def objective_system(self, params, return_energy=False, assign=True, normalize=True, save=False):
    params = np.array(params)
    # Assign params to wavefunction, hamiltonian, and other involved objects.
    if assign:
        self.assign_params(params)
    # Normalize
    if normalize:
        self.wfn.normalize(self.refwfn)
    # Save params
    # Commented out b/c trf_fanpy
    if save:
        self.save_params()

    get_overlap = self.wrapped_get_overlap
    integrate_wfn_sd = self.wrapped_integrate_wfn_sd

    overlaps = np.fromiter((get_overlap(sd) for sd in self.pspace), float, count=len(self.pspace))
    integrals = np.fromiter(
        (integrate_wfn_sd(sd) for sd in self.pspace), float, count=len(self.pspace)
    )
    # reference values
    if self.energy_type in ["variable", "fixed"]:
        energy = self.energy.params
    elif self.energy_type == "compute":
        if self.refwfn == self.pspace:
            overlaps_energy = overlaps
            integrals_energy = integrals
        else:
            overlaps_energy = np.fromiter(
                (get_overlap(sd) for sd in self.refwfn), float, count=len(self.refwfn)
            )
            integrals_energy = np.fromiter(
                (integrate_wfn_sd(sd) for sd in self.refwfn), float, count=len(self.refwfn)
            )
        norm = np.sum(overlaps_energy ** 2)
        energy = np.sum(overlaps_energy * integrals_energy) / norm
        self.energy.assign_params(energy)

    if hasattr(self, 'print_energy'):
        if self.energy_type == 'fixed':
            norm = np.sum(overlaps ** 2)
            energy_print = np.sum(overlaps * integrals) / norm
        else:
            energy_print = energy
        if self.print_energy:
            print("(Mid Optimization) Electronic Energy: {}".format(energy_print))
        else:
            try:
                self.print_queue['energy'] = energy_print
            except AttributeError:
                self.print_queue = {'energy': energy_print}

    # objective
    obj = np.empty(self.num_eqns)
    # <SD|H|Psi> - E<SD|Psi> == 0
    obj[: self.nproj] = integrals - energy * overlaps
    # Add constraints
    if self.nproj < self.num_eqns:
        obj[self.nproj :] = np.hstack([cons.objective(params) for cons in self.constraints])
    obj *= self.eqn_weights

    if hasattr(self, 'print_energy'):
        if self.print_energy:
            print("(Mid Optimization) Cost (without constraints): {}".format(0.5 * np.sum(obj[:self.nproj] ** 2)))
            if self.constraints:
                print("(Mid Optimization) Constraint Cost: {}".format(0.5 * obj[self.nproj:] ** 2))
            print("(Mid Optimization) Residual: {}".format(np.max(np.abs(obj[:self.nproj]))))
        else:
            self.print_queue['cost'] = 0.5 * np.sum(obj[:self.nproj] ** 2)
            if self.constraints:
                self.print_queue['constraint'] = 0.5 * obj[self.nproj:] ** 2
            self.print_queue['residual'] = np.max(np.abs(obj[:self.nproj]))

    if return_energy:
        return obj, energy
    else:
        return obj


def jacobian_system(self, params, return_d_energy=False, assign=False, normalize=False, save=True):
    params = np.array(params)
    # Assign params
    if assign:
        self.assign_params(params)
    # Normalize
    # commented out because trf would already normalized
    if normalize:
        self.wfn.normalize(self.refwfn)
    # Save params
    if save:
        self.save_params()

    get_overlap = self.wrapped_get_overlap
    integrate_wfn_sd = self.wrapped_integrate_wfn_sd

    overlaps = np.fromiter((get_overlap(sd) for sd in self.pspace), float, count=len(self.pspace))
    integrals = np.fromiter(
        (integrate_wfn_sd(sd) for sd in self.pspace), float, count=len(self.pspace)
    )
    d_overlaps = np.array([get_overlap(sd, True) for sd in self.pspace])
    d_integrals = np.array([integrate_wfn_sd(sd, True) for sd in self.pspace])

    # reference values
    if self.energy_type in ["variable", "fixed"]:
        energy = self.energy.params
        if self.energy_type == "variable":
            # FIXME: needs to be checked!
            # NOTE: ASSUME that energy is the last parameter
            d_energy = self.indices_objective_params[self.energy].astype(float)
        else:
            d_energy = np.zeros(params.size)
    elif self.energy_type == "compute":
        if self.refwfn == self.pspace:
            overlaps_energy = overlaps
            integrals_energy = integrals
            d_overlaps_energy = d_overlaps
            d_integrals_energy = d_integrals
        else:
            overlaps_energy = np.fromiter(
                (get_overlap(sd) for sd in self.refwfn), float, count=len(self.refwfn)
            )
            integrals_energy = np.fromiter(
                (integrate_wfn_sd(sd) for sd in self.refwfn), float, count=len(self.refwfn)
            )
            d_overlaps_energy = np.array([get_overlap(sd, True) for sd in self.refwfn])
            d_integrals_energy = np.array([integrate_wfn_sd(sd, True) for sd in self.refwfn])
        norm = np.sum(overlaps_energy ** 2)
        energy = np.sum(overlaps_energy * integrals_energy) / norm

        d_norm = 2 * np.sum(d_overlaps_energy * overlaps_energy[:, None], axis=0)
        d_energy = np.sum(d_overlaps_energy * integrals_energy[:, None], axis=0)
        d_energy += np.sum(overlaps_energy[:, None] * d_integrals_energy, axis=0)
        d_energy -= d_norm * energy
        d_energy /= norm
        self.energy.assign_params(energy)

    # jacobian
    jac = np.zeros((self.num_eqns, params.size))

    jac[: self.nproj, :] = d_integrals
    jac[: self.nproj, :] -= energy * d_overlaps
    jac[: self.nproj, :] -= d_energy[None, :] * overlaps[:, None]
    # Add constraints
    if self.nproj < self.num_eqns:
        jac[self.nproj :] = np.vstack([cons.gradient(params) for cons in self.constraints])
    # weigh equations
    jac *= self.eqn_weights[:, np.newaxis]

    if return_d_energy:
        obj = np.empty(self.num_eqns)
        obj[: self.nproj] = integrals - energy * overlaps
        # Add constraints
        if self.nproj < self.num_eqns:
            obj[self.nproj :] = np.hstack([cons.objective(params) for cons in self.constraints])
        # weigh equations
        obj *= self.eqn_weights
        return jac, obj, energy, d_energy
    else:
        return jac


def update_weights(self):
    # weighing equations by overlaps
    # self.eqn_weights[:self.nproj] *= overlaps
    # weighing constraints
    energy_constraint_index = [
        i for i, j in enumerate(self.constraints) if isinstance(j, EnergyConstraint)
    ]
    if energy_constraint_index:
        if not hasattr(self, 'counter'):
            self.counter = 0
        if not hasattr(self, 'num_count'):
            self.num_count = 10
        if not hasattr(self, 'decrease_factor'):
            self.decrease_factor = 10
        if self.counter % self.num_count == 0 and self.counter != 0:
            print("DECREASE ENERGY CONSTRAINT WEIGHT")
            self.eqn_weights[self.nproj + energy_constraint_index[0]] /= self.decrease_factor
            # self.counter = 0


def adapt_pspace(self):
    probable_sds, olps = zip(*self.wfn.probable_sds.items())
    prob = np.array(olps) ** 2
    prob /= np.sum(prob)

    # # adjust according to repetitions
    # pspace = np.random.choice(probable_sds, size=self.sample_size, p=prob, replace=True)
    # pspace_count = Counter(pspace)
    # weights = []
    # pspace = []
    # total_count = sum(pspace_count.values())
    # for sd, count in pspace_count.items():
    #     pspace.append(sd)
    #     weights.append(count / total_count)
    # weights = np.array(weights)

    pspace = np.random.choice(
        probable_sds, size=min(self.sample_size, len(probable_sds)), p=prob, replace=False
    )
    weights = np.array([self.wfn.probable_sds[sd]**2 for sd in pspace])
    weights /= np.sum(weights)

    if hasattr(self, 'weight_type'):
        if self.weight_type == 'ones':
            weights = np.ones(len(pspace))
        elif self.weight_type == 'decreasing':
            weights = weights ** ((self.counter // self.num_count + 1) ** -1)
            weights /= np.sum(weights)
    # weights *= len(pspace)

    print('Adapt pspace')
    print(len(pspace), len(probable_sds), max(olps), 'pspace')
    if hasattr(self, 'pspace_fill') and self.pspace_fill and len(pspace) < self.sample_size:
        n = self.sample_size // len(pspace)
        indices = np.zeros(len(pspace), dtype=bool)
        indices[:self.sample_size - n * len(pspace)] = True
        indices = np.tile(indices, n + 1)[:self.sample_size]

        pspace = np.tile(pspace, n + 1)[:self.sample_size]
        weights = np.tile(weights, n + 1)[:self.sample_size]
        weights[indices] /= (n + 1)
        weights[~indices] /= n

    self.pspace = list(pspace)
    if self.constraints:
        self.eqn_weights = np.hstack([weights, self.eqn_weights[-len(self.constraints):]])
    else:
        self.eqn_weights = np.array(weights)


def adapt_pspace_energy(self):
    probable_sds, olps = zip(*self.wfn.probable_sds.items())
    # prob = np.array(olps) ** 2
    # prob /= np.sum(prob)

    # # # adjust according to repetitions
    # # pspace = np.random.choice(probable_sds, size=self.sample_size, p=prob, replace=True)
    # # pspace_count = Counter(pspace)
    # # pspace = []
    # # for sd, count in pspace_count.items():
    # #     pspace.append(sd)

    # pspace = np.random.choice(
    #     probable_sds, size=min(self.sample_size, len(probable_sds)), p=prob, replace=False
    # )
    # print(len(pspace), len(probable_sds), max(olps))

    # self.refwfn = pspace
    if len(self.refwfn) <= self.sample_size:
        print('Adapt energy pspace')
        print(max(olps), len(self.refwfn), self.sample_size, 'refwfn')
        refwfn = set(self.refwfn)
        probable_sds = set(probable_sds)
        probable_sds = list(probable_sds - refwfn)
        probable_sds = sorted(probable_sds, key=lambda sd: abs(self.wfn.probable_sds[sd]), reverse=True)
        self.refwfn = list(refwfn) + probable_sds[:self.sample_size - len(refwfn)]


def objective_norm(self, params):
    params = np.array(params)
    # Assign params
    self.assign_params(params)
    # Establish shortcuts
    get_overlap = self.wrapped_get_overlap
    # Define reference
    overlaps = np.fromiter((get_overlap(sd) for sd in self.refwfn), float, count=len(self.refwfn))
    return np.sum(overlaps ** 2) - 1


def gradient_norm(self, params):
    params = np.array(params)
    # Assign params
    self.assign_params(params)
    # Establish shortcuts
    get_overlap = self.wrapped_get_overlap
    # Define reference
    overlaps = np.fromiter((get_overlap(sd) for sd in self.refwfn), float, count=len(self.refwfn))
    d_overlaps = np.array([get_overlap(sd, True) for sd in self.refwfn])
    return 2 * np.sum(d_overlaps * overlaps[:, None], axis=0)


def update_pspace_norm(self, refwfn=None):
    if refwfn:
        self.pspace_norm = refwfn
    else:
        self.pspace_norm.add(max(self.probable_sds.keys(), key=lambda x: self.probable_sds[x]))
    print('Adapt normalization pspace')
    print(
        sum(self.get_overlap(sd)**2 for sd in self.pspace_norm),
        len(self.probable_sds), len(self.pspace_norm), max(self.probable_sds.values()),
        'norm_pspace'
    )


BaseGeminal.get_overlap = get_overlap
BaseGeminal._olp_deriv = _olp_deriv
APIG._olp_deriv = _olp_deriv_apig
AP1roG.get_overlap = get_overlap_ap1rog
AP1roG._olp_deriv = _olp_deriv_ap1rog
CIWavefunction.get_overlap = get_overlap_ci
RestrictedMolecularHamiltonian.integrate_sd_wfn = integrate_sd_wfn
BaseSchrodinger.wrapped_get_overlap = wrapped_get_overlap
BaseSchrodinger.wrapped_integrate_wfn_sd = wrapped_integrate_wfn_sd
BaseSchrodinger.get_energy_one_proj = get_energy_one_proj
EnergyOneSideProjection.objective = objective_energy
EnergyOneSideProjection.gradient = gradient_energy
EnergyOneSideProjection.adapt_pspace = adapt_pspace_energy
ProjectedSchrodinger.assign_pspace = assign_pspace_system
ProjectedSchrodinger.objective = objective_system
ProjectedSchrodinger.jacobian = jacobian_system
ProjectedSchrodinger.update_weights = update_weights
ProjectedSchrodinger.adapt_pspace = adapt_pspace
NormConstraint.objective = objective_norm
NormConstraint.gradient = gradient_norm
NormConstraint.wrapped_get_overlap = wrapped_get_overlap
BaseWavefunction.update_pspace_norm = update_pspace_norm
