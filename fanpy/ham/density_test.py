"""Functions for obtaining the density matrices."""

from fanpy.tools import sd_list, slater
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.ci.base import CIWavefunction


# FIXME: create base operator class, make hamiltonian child of this class, make this module child of
#        this class
def integrate_sd_wfn(sd, wfn, indices, deriv=None):
    r"""Integrate the density operator against the wavefunction and the Slater determinant.

    .. math::

        \Gamma_{i_1 \dots i_K j_1 \dots j_K}
        &= \left< \Phi \middle| \hat{\Gamma}_{i_1 \dots i_K}^{j_1 \dots j_K} \middle| \Psi \right>\\
        &= \left< \Phi \middle|
           a^\dagger_{i_1} \dots a^\dagger_{i_K} a_{j_K} \dots a_{j_1} \middle| \Psi \right>

    where :math:`\Psi` is the wavefunction, :math:`\hat{\Gamma}` is the density matrix operator,
    and :math:`\Phi` is the Slater determinant.

    Parameters
    ----------
    wfn : BaseWavefunction
        Wavefunction against which the operator is integrated.
    indices : tuple of int
        Spin orbital indices that describe the creators and annihilators used in the density
        matrix operators.
        List of indices will divided in two, where the first half corresponds to the creators
        and the second half corresponds to the annihilators.
    sd : int
        Slater Determinant against which the operator is integrated.
    deriv : {int, None}
        Index of the wavefunction parameter against which the integral is derivatized.
        Default is no derivatization.

    Returns
    -------
    density_element : float
        Density matrix element.

    Raises
    ------
    ValueError
        If number of indices is not even.
    TypeError
        If an index is not an integer.

    """
    # pylint: disable=C0103
    if len(indices) % 2 != 0:
        raise ValueError("There must be even number of indices.")
    if not all(isinstance(i, int) for i in indices):
        raise TypeError("All indices must be integers.")
    creators = indices[: len(indices)]
    annihilators = indices[len(indices) :]
    # reverse annhilators b/c indices are reversed for the annihilators
    annihilators = annihilators.reverse()

    sign = slater.sign_excite(sd, creators, annihilators)
    if sign is None:  # pylint: disable=R1705
        return 0.0
    else:
        return sign * wfn.get_overlap(slater.excite(sd, *creators, *annihilators), deriv=deriv)


# FIXME: store in a subclass of BaseObjective
# FIXME: much of the code was copied from EnergyOneSideProjection
# FIXME: not sure if this needs to be normalize
# FIXME: add threshold value below which the contribution from the Slater determinant is discarded
#        (useful for CIWavefunction and other wavefunctions where we can sort by the contribution of
#        each Slater determinant)
def get_density_matrix(wfn, refwfn=None, order=1, deriv=None, notation="physicist", orbtype="restricted"):
    r"""Return the density matrix of the given order.

    .. math::

        \Gamma_{i_1 \dots i_K j_1 \dots j_K}
        &= \left< \Phi \middle| \hat{\Gamma}_{i_1 \dots i_K}^{j_1 \dots j_K} \middle| \Psi \right>\\
        &= \left< \Phi \middle|
           a^\dagger_{i_1} \dots a^\dagger_{i_K} a_{j_K} \dots a_{j_1} \middle| \Psi \right>

    where :math:`\Psi` is the wavefunction, :math:`\hat{\Gamma}` is the density matrix operator,
    and :math:`\Phi` is a reference wavefunction. The reference wavefuntion can be a Slater
    determinant, CI wavefunction,

    .. math::

        \left| \Phi \right> = \sum_{\mathbf{m} \in S} c_{\mathbf{m}} \left| \mathbf{m} \right>

    or a projected form of wavefunction :math:`\Psi`

    .. math::

        \left| \Phi \right> = \sum_{\mathbf{m} \in S} \left< \Psi \middle| \mathbf{m} \right>
                              \left| \mathbf{m} \right>

    where :math:`S` is the projection space.

    Parameters
    ----------
    wfn : BaseWavefunction
        Wavefunction against which the operator is integrated.
    refwfn : {int, tuple of int, tuple of CIWavefunction, None}
        Wavefunction against which the density is obtained.
        If an int is given, then the reference wavefunction will be the corresponding Slater
        determinant.
        If a tuple/list of int is given, then the reference wavefunction will be the given
        wavefunction truncated to the given list of corresponding Slater determinants.
        By default, the given wavefunction is used.
    deriv : {int, None}
        Index of the wavefunction parameter against which the integral is derivatized.
        Default is no derivatization.
    notation : {'physicist', 'chemist'}
        Notation of the two electron density matrix.
        Default is Physicist's notation.
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital.

    Returns
    -------
    density_element : float
        Density matrix element.

    Raises
    ------
    TypeError
        If reference wavefunction is not a list or a tuple.
        If projection space (for the reference wavefunction) must be given as a list/tuple of
        Slater determinants.
    ValueError
        If given wavefunction is not a BaseWavefunction instance.
        If given Slater determinant in projection space (for the reference wavefunction) does
        not have the same number of electrons as the wavefunction.
        If given Slater determinant in projection space (for the reference wavefunction) does
        not have the same number of spin orbitals as the wavefunction.
        If given reference wavefunction does not have the same number of electrons as the
        wavefunction.
        If given reference wavefunction does not have the same number of spin orbitals as the
        wavefunction.

    """
    # pylint: disable=W0613,R0912
    if not isinstance(wfn, BaseWavefunction):
        raise TypeError("Given wavefunction is not an instance of BaseWavefunction (or its " "child).")

    # check refwfn (copied from EnergyOneSideProjection.assign_refwfn)
    if refwfn is None:
        refwfn = tuple(sd_list.sd_list(wfn.nelec, wfn.nspatial, spin=wfn.spin, seniority=wfn.seniority))
    if slater.is_sd_compatible(refwfn):
        pass
    elif isinstance(refwfn, (list, tuple)):
        for sd in refwfn:  # pylint: disable=C0103
            if slater.is_sd_compatible(sd):
                occs = slater.occ_indices(sd)
                if len(occs) != wfn.nelec:
                    raise ValueError(
                        "Given Slater determinant does not have the same number of"
                        " electrons as the given wavefunction."
                    )
                if any(i >= wfn.nspin for i in occs):
                    raise ValueError(
                        "Given Slater determinant does not have the same number of"
                        " spin orbitals as the given wavefunction."
                    )
            else:
                raise TypeError(
                    "Projection space (for the reference wavefunction) must only " "contain Slater determinants."
                )
        refwfn = tuple(refwfn)
    elif isinstance(refwfn, CIWavefunction):
        if refwfn.nelec != wfn.nelec:
            raise ValueError(
                "Given reference wavefunction does not have the same number of " "electrons as the given wavefunction."
            )
        if refwfn.nspin != wfn.nspin:
            raise ValueError(
                "Given reference wavefunction does not have the same number of "
                "spin orbitals as the given wavefunction."
            )
        refwfn = refwfn
    else:
        raise TypeError(
            "Reference wavefunction must be given as a Slater determinant, list/tuple "
            "of Slater determinants, or a CI wavefunction."
        )

    if notation not in ["chemist", "physicist"]:
        raise ValueError("The notation can only be one of 'chemist' and 'physicist'.")

    if orbtype not in ["restricted", "unrestricted", "generalized"]:
        raise ValueError("The orbital type must be one of 'restricted', 'unrestricted', and " "'generalized'.")
