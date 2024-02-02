from fanpy.wfn.geminal.base import BaseGeminal
from fanpy.wfn.geminal.ap1rog import AP1roG
from fanpy.wfn.geminal.apg import APG
# from fanpy.wfn.geminal.apg2 import APG2
# from fanpy.wfn.geminal.apg3 import APG3
# from fanpy.wfn.geminal.apg4 import APG4
# from fanpy.wfn.geminal.apg5 import APG5
# from fanpy.wfn.geminal.apg6 import APG6
import fanpy.tools.slater as slater
# import fanpy.tools.graphs as graphs
import numpy as np
from fanpy.upgrades.cext_apg import generate_complete_pmatch, generate_general_pmatch, _olp_deriv_internal_ap1rog
from fanpy.upgrades.cext_apg_parallel import _olp_internal_apig, _olp_deriv_internal_apig
from fanpy.upgrades.cext_apg_parallel2 import _olp_internal, _olp_deriv_internal
from fanpy.wfn.geminal.apig import APIG


def _olp(self, sd):
    """Calculate the overlap with the Slater determinant.

    Parameters
    ----------
    sd : int
        Occupation vector of a Slater determinant given as a bitstring.

    Returns
    -------
    olp : {float, complex}
        Overlap of the current instance with the given Slater determinant.

    """
    # NOTE: Need to recreate occ_indices
    occ_indices = list(slater.occ_indices(sd))

    if hasattr(self, "temp_generator"):
        orbpair_generator = self.temp_generator
    else:
        orbpair_generator = self.generate_possible_orbpairs(occ_indices)

    if not orbpair_generator:
        return 0.0

    output = _olp_internal(orbpair_generator, self.params, self.nspin)

    if abs(output) > self.olp_threshold and len(self.probable_sds) < 1000000:
        self.probable_sds[sd] = output

    return output


def _olp_apig(self, sd):
    """Calculate the overlap with the Slater determinant.

    Parameters
    ----------
    sd : int
        Occupation vector of a Slater determinant given as a bitstring.

    Returns
    -------
    olp : {float, complex}
        Overlap of the current instance with the given Slater determinant.

    """
    # NOTE: Need to recreate occ_indices
    occ_indices = list(slater.occ_indices(sd))

    if hasattr(self, "temp_generator"):
        orbpair_generator = self.temp_generator
    else:
        orbpair_generator = self.generate_possible_orbpairs(occ_indices)

    if not orbpair_generator:
        return 0.0

    output = _olp_internal_apig(orbpair_generator, self.params)

    if abs(output) > self.olp_threshold and len(self.probable_sds) < 1000000:
        self.probable_sds[sd] = output

    return output


# def _olp_deriv(self, sd, deriv):
#     # NOTE: Need to recreate occ_indices, row_removed, col_removed
#     occ_indices = list(slater.occ_indices(sd))

#     val = 0.0
#     if hasattr(self, "temp_generator"):
#         orbpair_generator = self.temp_generator
#     else:
#         orbpair_generator = self.generate_possible_orbpairs(occ_indices)
#     if not orbpair_generator:
#         return val

#     for orbpairs_sign in orbpair_generator:
#         orbpairs = orbpairs_sign[:-1]
#         sign = orbpairs_sign[-1]
#         # ASSUMES: permanent evaluation is much more expensive than the lookup
#         if len(orbpairs) == 0:
#             continue
#         orbpairs = zip(orbpairs[::2], orbpairs[1::2])
#         col_inds = np.array([self.get_col_ind(orbp) for orbp in orbpairs], dtype=int)
#         # FIXME: converting all orbpairs is slow for some reason
#         # col_inds = self.get_col_inds(np.array(orbpairs))
#         val += sign * self.compute_permanent(col_inds, deriv=deriv)
#     return val


def apg_generate_possible_orbpairs(self, occ_indices):
    return generate_complete_pmatch(tuple(occ_indices))


# old_apg2_generate_possible_orbpairs = APG2.generate_possible_orbpairs
# old_apg3_generate_possible_orbpairs = APG3.generate_possible_orbpairs
# old_apg4_generate_possible_orbpairs = APG4.generate_possible_orbpairs
# old_apg5_generate_possible_orbpairs = APG5.generate_possible_orbpairs
# old_apg6_generate_possible_orbpairs = APG6.generate_possible_orbpairs


# @lru_cache(maxsize=1000)
# def generate_general_pmatch(indices, connectivity_flat):
#     """
#     Parameters
#     ----------
#     indices : tuple
#         Indices that are occupied.
#     connectivity_flat tuple
#         Components of the upper triangular matrix (offset 1) of the connectivity graph.
#     """
#     indices = tuple(indices)
#     #connectivity_matrix_full = np.ones((indices.size, indices.size), dtype=bool)
#     #connectivity_matrix_full[np.triu_indices(indices.size, k=1)] = connectivity_matrix
#     #connectivity_matrix_full[np.tril_indices(indices.size, k=-1)] = connectivity_matrix
#     #connectivity_matrix = connectivity_matrix_full
#     n = len(indices)
#     if n == 2:
#         return [(indices[0], indices[1], 1)]
#     elif n > 2:
#         output = []
#         ind_one = indices[0]
#         for j, is_connected in enumerate(connectivity_flat[:n-1]):
#             print(j, is_connected, indices, indices[0], indices[j+1], n)
#             print(connectivity_flat)
#             if not is_connected:
#                 continue
#             sign = (-1) ** j
#             j += 1
#             ind_two = indices[j]
#             # filter out indices that are not used
#             new_indices = indices[1:j] + indices[j+1:]
#             # mask_bool = ~np.isin(indices, [ind_one, ind_two])
#             # mask_ind = np.where(mask_bool)[0]
#             # mask_ind = mask_ind[mask_ind > 0]
#             # temp = (
#             #     connectivity_flat[n - 1 : (2 * n - j - 1) * j // 2] +
#             #     connectivity_flat[(2 * n - j - 2) * (j + 1) // 2:]
#             # )
#             # temp = []
#             j -= 1
#             temp = connectivity_flat[n - 1 : n - 1 + j - 1]
#             for k in range(2, j + 1):
#                 temp += connectivity_flat[
#                     (2 * n - 1 - (k - 1)) * (k - 1) // 2 + j - k + 2: (2 * n - 1 - k) * k // 2 + j - k
#                 ]
#             temp += connectivity_flat[(2 * n - 1 - j) * j // 2 + 1:]
#             # temp = (
#             #     connectivity_flat[(2 * n - 1 - 1) * 1 // 2 : (2 * n - 1 - 1) * 1 // 2 + j - 2] +
#             #     connectivity_flat[(2 * n - 1 - 1) * 1 // 2 + j - 3 : (2 * n - 1 - 2) * 2 // 2 + j - 3] +
#             #     connectivity_flat[(2 * n - 1 - 2) * 2 // 2 + j - 4 : (2 * n - 1 - 3) * 3 // 2 + j - 4] +
#             #     connectivity_flat[(2 * n - 1 - (j - 1)) * (j - 1) // 2 + 1:]
#             # )

#             # temp = connectivity_matrix[mask_ind[:, None], mask_ind[None, :]]
#             for scheme_inner_sign in generate_general_pmatch(
#                     new_indices, temp
#                 # tuple(indices[mask_ind]),
#                 # tuple(temp[np.triu_indices(temp.shape[0], k=1)])
#             ):
#                 n_inner = len(scheme_inner_sign)
#                 scheme = scheme_inner_sign[:n_inner-1]
#                 inner_sign = scheme_inner_sign[n_inner-1]
#                 output.append((ind_one, ind_two) + scheme + (sign * inner_sign,))
#         return output


# @lru_cache(maxsize=1000)
# def generate_general_pmatch(indices, connectivity_matrix):
#     if isinstance(indices, (list, tuple)):
#         indices = np.array(indices)
#     connectivity_matrix_full = np.ones((indices.size, indices.size), dtype=bool)
#     connectivity_matrix_full[np.triu_indices(indices.size, k=1)] = connectivity_matrix
#     connectivity_matrix_full[np.tril_indices(indices.size, k=-1)] = connectivity_matrix
#     connectivity_matrix = connectivity_matrix_full
#     if len(indices) == 2:
#         return [[indices[0], indices[1], 1]]
#     elif indices.size > 2:
#         output = []
#         ind_one = indices[0]
#         for j in np.where(connectivity_matrix[0, 1:])[0]:
#             sign = (-1) ** j
#             j += 1
#             ind_two = indices[j]
#             # filter out indices that are not used
#             mask_bool = ~np.isin(indices, [ind_one, ind_two])
#             mask_ind = np.where(mask_bool)[0]
#             mask_ind = mask_ind[mask_ind > 0]

#             temp = connectivity_matrix[mask_ind[:, None], mask_ind[None, :]]
#             for scheme_inner_sign in generate_general_pmatch(
#                 tuple(indices[mask_ind]),
#                 tuple(temp[np.triu_indices(temp.shape[0], k=1)])
#             ):
#                 n_inner = len(scheme_inner_sign)
#                 scheme = scheme_inner_sign[:n_inner-1]
#                 inner_sign = scheme_inner_sign[n_inner-1]
#                 output.append([ind_one, ind_two] + scheme + [sign * inner_sign])
#         return output


# FIXME: add APIG and AP1roG

def apg2_generate_possible_orbpairs(self, occ_indices):
    if self.connectivity is None:
        dead_indices = np.where(np.sum(np.abs(self.params), axis=0) < self.tol)[0]
        dead_orbpairs = np.array([self.get_orbpair(ind) for ind in dead_indices])
        connectivity = np.ones((self.nspin, self.nspin))
        if dead_orbpairs.size > 0:
            connectivity[dead_orbpairs[:, 0], dead_orbpairs[:, 1]] = 0
            connectivity[dead_orbpairs[:, 1], dead_orbpairs[:, 0]] = 0
        self.connectivity = connectivity.copy()
    else:
        connectivity = self.connectivity.copy()
    occ_indices = np.array(occ_indices)
    connectivity = connectivity[occ_indices[:, None], occ_indices[None, :]]
    connectivity = tuple(connectivity[np.triu_indices(connectivity.shape[0], k=1)])
    return generate_general_pmatch(tuple(occ_indices), connectivity)


def apg3_generate_possible_orbpairs(self, occ_indices):
    return [
        tuple(j for i in pmatch for j in i) + (sign,)
        for pmatch, sign in
        old_apg3_generate_possible_orbpairs(self, occ_indices)
    ]


def apg4_generate_possible_orbpairs(self, occ_indices):
    return [
        tuple(j for i in pmatch for j in i) + (sign,)
        for pmatch, sign in
        old_apg4_generate_possible_orbpairs(self, occ_indices)
    ]


def apg5_generate_possible_orbpairs(self, occ_indices):
    return [
        tuple(j for i in pmatch for j in i) + (sign,)
        for pmatch, sign in
        old_apg5_generate_possible_orbpairs(self, occ_indices)
    ]


def apg6_generate_possible_orbpairs(self, occ_indices):
    return [
        tuple(j for i in pmatch for j in i) + (sign,)
        for pmatch, sign in
        old_apg6_generate_possible_orbpairs(self, occ_indices)
    ]


def apig_generate_possible_orbpairs(self, occ_indices):
    npairs = len(occ_indices) // 2
    if (npairs * (npairs - 1) // 2) % 2 == 0:
        sign = -1
    else:
        sign = 1

    dict_orb_ind = {orbpair[0]: ind for orbpair, ind in self.dict_orbpair_ind.items()}
    output = tuple()
    for i in occ_indices:
        try:
            ind = dict_orb_ind[i]
        except KeyError:
            continue
        else:
            output += self.dict_ind_orbpair[ind]
    output += (sign,)

    # signature to turn orbpairs into strictly INCREASING order.
    if set(output[:-1]) != set(occ_indices):
        return []
    return [output]


BaseGeminal._olp = _olp
APG.generate_possible_orbpairs = apg_generate_possible_orbpairs
# graphs.generate_general_pmatch = generate_general_pmatch
# APG2.generate_possible_orbpairs = apg2_generate_possible_orbpairs
# APG3.generate_possible_orbpairs = apg3_generate_possible_orbpairs
# APG4.generate_possible_orbpairs = apg4_generate_possible_orbpairs
# APG5.generate_possible_orbpairs = apg5_generate_possible_orbpairs
# APG6.generate_possible_orbpairs = apg6_generate_possible_orbpairs
APIG.generate_possible_orbpairs = apig_generate_possible_orbpairs
APIG._olp = _olp_apig



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


BaseGeminal._olp_deriv = _olp_deriv
APIG._olp_deriv = _olp_deriv_apig
AP1roG.get_overlap = get_overlap_ap1rog
AP1roG._olp_deriv = _olp_deriv_ap1rog
