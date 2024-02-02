import numpy as np
import itertools as it
from scipy.special import comb
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
import fanpy.tools.slater as slater


def integrate_sd_wfn(self, sd, wfn, wfn_deriv=None):
    r"""Integrate the Hamiltonian with against a Slater determinant and a wavefunction.

    .. math::

        \left< \Phi \middle| \hat{H} \middle| \Psi \right>
        = \sum_{\mathbf{m} \in S_\Phi}
            f(\mathbf{m}) \left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>

    where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
    :math:`\Phi` is the Slater determinant. The :math:`S_{\Phi}` is the set of Slater
    determinants for which :math:`\left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>` is
    not zero, which are the :math:`\Phi` and its first and second order excitations for a
    chemical Hamiltonian.

    Parameters
    ----------
    sd : int
        Slater Determinant against which the Hamiltonian is integrated.
    wfn : Wavefunction
        Wavefunction against which the Hamiltonian is integrated.
        Needs to have the following in `__dict__`: `get_overlap`.
    wfn_deriv : {int, None}
        Index of the wavefunction parameter against which the integral is derivatized.
        Default is no derivatization.

    Returns
    -------
    integrals : np.ndarray(3,)
        Integrals of the given Slater determinant and the wavefunction.
        First element corresponds to the one-electron energy, second to the coulomb energy, and
        third to the exchange energy.

    """
    # pylint: disable=C0103
    nspatial = self.nspin // 2
    # sd = slater.internal_sd(sd)
    occ_indices = np.array(slater.occ_indices(sd))
    vir_indices = np.array(slater.vir_indices(sd, self.nspin))
    # FIXME: hardcode slater determinant structure
    occ_alpha = occ_indices[occ_indices < nspatial]
    vir_alpha = vir_indices[vir_indices < nspatial]
    occ_beta = occ_indices[occ_indices >= nspatial]
    vir_beta = vir_indices[vir_indices >= nspatial]

    overlaps_zero = np.array([[wfn.get_overlap(sd, deriv=wfn_deriv)]])

    def fromiter(iterator, dtype, ydim, count):
        return np.fromiter(it.chain.from_iterable(iterator), dtype, count=int(count)).reshape(-1, ydim)

    occ_one_alpha = fromiter(it.combinations(occ_alpha.tolist(), 1), int, 1, count=len(occ_alpha))
    occ_one_alpha = np.left_shift(1, occ_one_alpha[:, 0])

    occ_one_beta = fromiter(it.combinations(occ_beta.tolist(), 1), int, 1, count=len(occ_beta))
    occ_one_beta = np.left_shift(1, occ_one_beta[:, 0])

    vir_one_alpha = fromiter(it.combinations(vir_alpha.tolist(), 1), int, 1, count=len(vir_alpha))
    vir_one_alpha = np.left_shift(1, vir_one_alpha[:, 0])

    vir_one_beta = fromiter(it.combinations(vir_beta.tolist(), 1), int, 1, count=len(vir_beta))
    vir_one_beta = np.left_shift(1, vir_one_beta[:, 0])

    occ_two_aa = fromiter(it.combinations(occ_alpha.tolist(), 2),
                                int, ydim=2, count=2*comb(len(occ_alpha), 2))
    occ_two_aa = np.left_shift(1, occ_two_aa)
    occ_two_aa = np.bitwise_or(occ_two_aa[:, 0], occ_two_aa[:, 1])

    occ_two_ab = fromiter(it.product(occ_alpha.tolist(), occ_beta.tolist()),
                                int, 2, count=2*len(occ_alpha) * len(occ_beta))
    occ_two_ab = np.left_shift(1, occ_two_ab)
    occ_two_ab = np.bitwise_or(occ_two_ab[:, 0], occ_two_ab[:, 1])

    occ_two_bb = fromiter(it.combinations(occ_beta.tolist(), 2),
                                int, 2, count=2*comb(len(occ_beta), 2))
    occ_two_bb = np.left_shift(1, occ_two_bb)
    occ_two_bb = np.bitwise_or(occ_two_bb[:, 0], occ_two_bb[:, 1])

    vir_two_aa = fromiter(it.combinations(vir_alpha.tolist(), 2),
                                int, 2, count=2*comb(len(vir_alpha), 2))
    vir_two_aa = np.left_shift(1, vir_two_aa)
    vir_two_aa = np.bitwise_or(vir_two_aa[:, 0], vir_two_aa[:, 1])

    vir_two_ab = fromiter(it.product(vir_alpha.tolist(), vir_beta.tolist()),
                                int, 2, count=2*len(vir_alpha) * len(vir_beta))
    vir_two_ab = np.left_shift(1, vir_two_ab)
    vir_two_ab = np.bitwise_or(vir_two_ab[:, 0], vir_two_ab[:, 1])

    vir_two_bb = fromiter(it.combinations(vir_beta.tolist(), 2),
                                int, 2, count=2*comb(len(vir_beta), 2))
    vir_two_bb = np.left_shift(1, vir_two_bb)
    vir_two_bb = np.bitwise_or(vir_two_bb[:, 0], vir_two_bb[:, 1])

    overlaps_one_alpha = np.array(
        [
            wfn.get_overlap(sd_exc, deriv=wfn_deriv)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_one_alpha)[:, None],
                                                    vir_one_alpha[None, :]))
        ]
    )
    overlaps_one_beta = np.array(
        [
            wfn.get_overlap(sd_exc, deriv=wfn_deriv)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_one_beta)[:, None],
                                                    vir_one_beta[None, :]))
        ]
    )

    overlaps_two_aa = np.array(
        [
            wfn.get_overlap(sd_exc, deriv=wfn_deriv)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_aa)[:, None],
                                                    vir_two_aa[None, :]))
        ]
    )
    overlaps_two_ab = np.array(
        [
            wfn.get_overlap(sd_exc, deriv=wfn_deriv)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_ab)[:, None],
                                                    vir_two_ab[None, :]))
        ]
    )
    overlaps_two_bb = np.array(
        [
            wfn.get_overlap(sd_exc, deriv=wfn_deriv)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_bb)[:, None],
                                                    vir_two_bb[None, :]))
        ]
    )

    # FIXME: hardcode slater determinant structure
    occ_beta -= nspatial
    vir_beta -= nspatial

    output = np.zeros(3)

    output += np.sum(self._integrate_sd_sds_zero(occ_alpha, occ_beta) * overlaps_zero, axis=1)

    integrals_one_alpha = self._integrate_sd_sds_one_alpha(occ_alpha, occ_beta, vir_alpha)
    integrals_one_beta = self._integrate_sd_sds_one_beta(occ_alpha, occ_beta, vir_beta)
    output += np.sum(integrals_one_alpha * overlaps_one_alpha, axis=1) + np.sum(
        integrals_one_beta * overlaps_one_beta, axis=1
    )
    if occ_alpha.size > 1 and vir_alpha.size > 1:
        integrals_two_aa = self._integrate_sd_sds_two_aa(occ_alpha, occ_beta, vir_alpha)
        output[1:] += np.sum(integrals_two_aa * overlaps_two_aa, axis=1)
    if occ_alpha.size > 0 and occ_beta.size > 0 and vir_alpha.size > 0 and vir_beta.size > 0:
        integrals_two_ab = self._integrate_sd_sds_two_ab(
            occ_alpha, occ_beta, vir_alpha, vir_beta
        )
        output[1] += np.sum(integrals_two_ab * overlaps_two_ab)
    if occ_beta.size > 1 and vir_beta.size > 1:
        integrals_two_bb = self._integrate_sd_sds_two_bb(occ_alpha, occ_beta, vir_beta)
        output[1:] += np.sum(integrals_two_bb * overlaps_two_bb, axis=1)

    return np.sum(output)


def integrate_sd_wfn_deriv(self, sd, wfn, ham_derivs):
    r"""Integrate the Hamiltonian with against a Slater determinant and a wavefunction.

    .. math::

        \left< \Phi \middle| \hat{H} \middle| \Psi \right>
        = \sum_{\mathbf{m} \in S_\Phi}
            f(\mathbf{m}) \left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>

    where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
    :math:`\Phi` is the Slater determinant. The :math:`S_{\Phi}` is the set of Slater
    determinants for which :math:`\left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>` is
    chemical Hamiltonian.

    Parameters
    ----------
    sd : int
        Slater Determinant against which the Hamiltonian is integrated.
    wfn : Wavefunction
        Wavefunction against which the Hamiltonian is integrated.
    ham_derivs : np.ndarray(N_derivs)
        Indices of the Hamiltonian parameter against which the integrals are derivatized.

    Returns
    -------
    integrals : np.ndarray(3, N_params)
        Integrals of the given Slater determinant and the wavefunction.
        First element corresponds to the one-electron energy, second to the coulomb energy, and
        third to the exchange energy.

    Raises
    ------
    TypeError
        If ham_derivs is not a one-dimensional numpy array of integers.
    ValueError
        If ham_derivs has any indices than is less than 0 or greater than or equal to nparams.

    Notes
    -----
    Providing only some of the Hamiltonian parameter indices will not make the code any faster.
    The integrals are derivatized with respect to all of Hamiltonian parameters and the
    appropriate derivatives are selected afterwards.

    """
    nspatial = self.nspin // 2
    sd = slater.internal_sd(sd)
    occ_indices = np.array(slater.occ_indices(sd))
    vir_indices = np.array(slater.vir_indices(sd, self.nspin))

    # FIXME: hardcode slater determinant structure
    occ_alpha = occ_indices[occ_indices < nspatial]
    vir_alpha = vir_indices[vir_indices < nspatial]
    occ_beta = occ_indices[occ_indices >= nspatial]
    vir_beta = vir_indices[vir_indices >= nspatial]

    overlaps_zero = np.array([[[wfn.get_overlap(sd)]]])

    def fromiter(iterator, dtype, ydim, count):
        return np.fromiter(it.chain.from_iterable(iterator), dtype, count=int(count)).reshape(-1, ydim)

    occ_one_alpha = fromiter(it.combinations(occ_alpha.tolist(), 1), int, 1, count=len(occ_alpha))
    occ_one_alpha = np.left_shift(1, occ_one_alpha[:, 0])

    occ_one_beta = fromiter(it.combinations(occ_beta.tolist(), 1), int, 1, count=len(occ_beta))
    occ_one_beta = np.left_shift(1, occ_one_beta[:, 0])

    vir_one_alpha = fromiter(it.combinations(vir_alpha.tolist(), 1), int, 1, count=len(vir_alpha))
    vir_one_alpha = np.left_shift(1, vir_one_alpha[:, 0])

    vir_one_beta = fromiter(it.combinations(vir_beta.tolist(), 1), int, 1, count=len(vir_beta))
    vir_one_beta = np.left_shift(1, vir_one_beta[:, 0])

    occ_two_aa = fromiter(it.combinations(occ_alpha.tolist(), 2),
                                int, ydim=2, count=2*comb(len(occ_alpha), 2))
    occ_two_aa = np.left_shift(1, occ_two_aa)
    occ_two_aa = np.bitwise_or(occ_two_aa[:, 0], occ_two_aa[:, 1])

    occ_two_ab = fromiter(it.product(occ_alpha.tolist(), occ_beta.tolist()),
                                int, 2, count=2*len(occ_alpha) * len(occ_beta))
    occ_two_ab = np.left_shift(1, occ_two_ab)
    occ_two_ab = np.bitwise_or(occ_two_ab[:, 0], occ_two_ab[:, 1])

    occ_two_bb = fromiter(it.combinations(occ_beta.tolist(), 2),
                                int, 2, count=2*comb(len(occ_beta), 2))
    occ_two_bb = np.left_shift(1, occ_two_bb)
    occ_two_bb = np.bitwise_or(occ_two_bb[:, 0], occ_two_bb[:, 1])

    vir_two_aa = fromiter(it.combinations(vir_alpha.tolist(), 2),
                                int, 2, count=2*comb(len(vir_alpha), 2))
    vir_two_aa = np.left_shift(1, vir_two_aa)
    vir_two_aa = np.bitwise_or(vir_two_aa[:, 0], vir_two_aa[:, 1])

    vir_two_ab = fromiter(it.product(vir_alpha.tolist(), vir_beta.tolist()),
                                int, 2, count=2*len(vir_alpha) * len(vir_beta))
    vir_two_ab = np.left_shift(1, vir_two_ab)
    vir_two_ab = np.bitwise_or(vir_two_ab[:, 0], vir_two_ab[:, 1])

    vir_two_bb = fromiter(it.combinations(vir_beta.tolist(), 2),
                                int, 2, count=2*comb(len(vir_beta), 2))
    vir_two_bb = np.left_shift(1, vir_two_bb)
    vir_two_bb = np.bitwise_or(vir_two_bb[:, 0], vir_two_bb[:, 1])

    overlaps_one_alpha = np.array(
        [
            wfn.get_overlap(sd_exc)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_one_alpha)[:, None],
                                                 vir_one_alpha[None, :]))
        ]
    )
    overlaps_one_beta = np.array(
        [
            wfn.get_overlap(sd_exc)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_one_beta)[:, None],
                                                 vir_one_beta[None, :]))
        ]
    )

    overlaps_two_aa = np.array(
        [
            wfn.get_overlap(sd_exc)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_aa)[:, None],
                                                 vir_two_aa[None, :]))
        ]
    )
    overlaps_two_ab = np.array(
        [
            wfn.get_overlap(sd_exc)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_ab)[:, None],
                                                 vir_two_ab[None, :]))
        ]
    )
    overlaps_two_bb = np.array(
        [
            wfn.get_overlap(sd_exc)
            for sd_exc in np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_bb)[:, None],
                                                 vir_two_bb[None, :]))
        ]
    )

    # FIXME: hardcode slater determinant structure
    occ_beta -= nspatial
    vir_beta -= nspatial

    output = np.zeros((3, self.nparams))

    output += np.squeeze(
        self._integrate_sd_sds_deriv_zero_alpha(occ_alpha, occ_beta, vir_alpha) * overlaps_zero,
        axis=2,
    )
    output += np.squeeze(
        self._integrate_sd_sds_deriv_zero_beta(occ_alpha, occ_beta, vir_beta) * overlaps_zero,
        axis=2,
    )

    output += np.sum(
        self._integrate_sd_sds_deriv_one_aa(occ_alpha, occ_beta, vir_alpha)
        * overlaps_one_alpha,
        axis=2,
    )
    output[1, :] += np.sum(
        self._integrate_sd_sds_deriv_one_ab(occ_alpha, occ_beta, vir_beta)
        * overlaps_one_beta,
        axis=1,
    )
    output[1, :] += np.sum(
        self._integrate_sd_sds_deriv_one_ba(occ_alpha, occ_beta, vir_alpha)
        * overlaps_one_alpha,
        axis=1,
    )
    output += np.sum(
        self._integrate_sd_sds_deriv_one_bb(occ_alpha, occ_beta, vir_beta) * overlaps_one_beta,
        axis=2,
    )

    if occ_alpha.size > 1 and vir_alpha.size > 1:
        output[1:, :] += np.sum(
            self._integrate_sd_sds_deriv_two_aaa(occ_alpha, occ_beta, vir_alpha)
            * overlaps_two_aa,
            axis=2,
        )
    if occ_alpha.size > 0 and occ_beta.size > 0 and vir_alpha.size > 0 and vir_beta.size > 0:
        output[1, :] += np.sum(
            self._integrate_sd_sds_deriv_two_aab(occ_alpha, occ_beta, vir_alpha, vir_beta)
            * overlaps_two_ab,
            axis=1,
        )
        output[1, :] += np.sum(
            self._integrate_sd_sds_deriv_two_bab(occ_alpha, occ_beta, vir_alpha, vir_beta)
            * overlaps_two_ab,
            axis=1,
        )
    if occ_beta.size > 1 and vir_beta.size > 1:
        output[1:, :] += np.sum(
            self._integrate_sd_sds_deriv_two_bbb(occ_alpha, occ_beta, vir_beta)
            * overlaps_two_bb,
            axis=2,
        )

    return np.sum(output[:, ham_derivs], axis=0)


RestrictedMolecularHamiltonian.integrate_sd_wfn = integrate_sd_wfn
RestrictedMolecularHamiltonian.integrate_sd_wfn_deriv = integrate_sd_wfn_deriv


def internal_sd(identifier):
    return identifier


def is_internal_sd(sd):
    return True


def spatial_to_spin_indices(spatial_indices, nspatial, to_beta=True):
    if to_beta:
        return spatial_indices + nspatial
    else:
        return spatial_indices


slater.internal_sd = internal_sd
slater.is_internal_sd = is_internal_sd
slater.spatial_to_spin_indices = spatial_to_spin_indices
