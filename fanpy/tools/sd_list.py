"""Functions used to generate Slater determinants.

Functions
---------
satisfies_conditions(sd, nspatial, spin, seniority)
    Checks to see if Slater determinant has the desired spin and seniority
sd_list(nspatial, nelec, num_limit=None, exc_orders=None, spin=None, seniority=None)
    Generates a list of Slater determinants with the specified excitations from ground state, spin,
    and seniority

"""
from itertools import combinations, product

from fanpy.tools import slater


def satisfies_conditions(sd, nspatial, spin, seniority):
    r"""Check to see if Slater determinant has the desired spin and seniority.

    Paramaters
    ----------
    sd : int
        Slater determinant.
    nspatial : int
        Number of spatial orbitals.
    spin : int
        Spin of the desired Slater determinant.
        :math:`\frac{1}{2} (N_{\alpha} - N_{\beta})`.
        If `None` then all spin is allowed.
    seniority : int
        Seniority of the desired Slater determinant.
        Maximum number of unpaired electrons.
        If `None` then all seniority is allowed.

    Returns
    -------
    condition : bool
        True if Slater determinant has the desired spin and seniority.
        False if Slater determinant does not have the desired spin and seniority.

    """
    # pylint: disable=C0103
    return spin in [None, slater.get_spin(sd, nspatial)] and (
        seniority is None or seniority >= slater.get_seniority(sd, nspatial)
        #seniority == slater.get_seniority(sd, nspatial)
    )


def sd_list(nelec, nspin, num_limit=None, exc_orders=None, spin=None, seniority=None):
    r"""Return a list of Slater determinants.

    Parameters
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals.
    num_limit : {int, None}
        Maximum number of Slater determinants to be generated.
        Default is infinite.
    exc_orders : {list of int, None}
        Orders of excitations that is to be included (with respect to ground state Slater
        determinant) in the order that they appear.
        Default is all orders of excitations (in the order of lowest order to highest).
    spin : {int, float, None}
        Total spin of the generated Slater determinants.
        Default is no spin restrictions.
        If positive, then the number of alpha orbitals is greater than the number of beta orbitals.
        If negative, then the number of alpha orbitals is less than the number of beta orbitals.
        0.5 is singlet, 1 is doublet, 1.5 is triplet, etc.
    seniority : {int, None}
        Maximum number of unpaired electrons.
        Default is no seniority restritions.

    Returns
    -------
    sds : list of ints
        Integer that describes the occupation of a Slater determinant as a bitstring.

    Raises
    ------
    TypeError
        If nspin is not an integer.
        If nelec is not an integer.
        If num_limit is not an integer.
        If exc_orders is not a iterable of integers.
        If spin is not an integer or float.
        If seniority is not an integer.
    ValueError
        If num_limit is not greater than 0.
        If seniority is not compatible with the spin.

    """
    # pylint: disable=C0103,R0912
    if exc_orders is None:
        exc_orders = range(1, nelec + 1)

    if __debug__:
        if not isinstance(nspin, int):
            raise TypeError("Number of spin orbitals should be an integer")
        if not isinstance(nelec, int):  # pragma: no cover
            raise TypeError("Number of electrons should be an integer")
        if num_limit is not None:
            if not isinstance(num_limit, int):
                raise TypeError("Number of Slater determinants should be an integer")
            if num_limit is not None and num_limit <= 0:
                raise ValueError("Number of Slater determinants must be greater than 0.")
        if not (hasattr(exc_orders, "__iter__") and all(isinstance(i, int) for i in exc_orders)):
            raise TypeError("Orders of excitations should be given as a list or tuple of integers")
        if not isinstance(spin, (int, float, type(None))):
            raise TypeError("Spin should be given as an integer or a floating point")
        if not isinstance(seniority, (int, type(None))):
            raise TypeError("Seniority should be given as an integer")
        if None not in [spin, seniority] and seniority < abs(2 * spin):
            raise ValueError("Cannot have spin, {0}, with seniority, {1}.".format(spin, seniority))

    nspatial = nspin // 2

    sds = []
    # ASSUME: spin orbitals are ordered by increasing energy
    ground = slater.ground(nelec, nspin)
    if satisfies_conditions(ground, nspatial, spin, seniority):
        sds.append(ground)

    occ_indices = slater.occ_indices(ground).tolist()
    vir_indices = slater.vir_indices(ground, nspin).tolist()
    # order by energy
    occ_indices = sorted(
        occ_indices, key=lambda x: x - nspatial if x >= nspatial else x, reverse=True
    )
    vir_indices = sorted(vir_indices, key=lambda x: x - nspatial if x >= nspatial else x)

    if num_limit is not None and num_limit == 1:
        return sds

    count = 1
    for nexc in exc_orders:
        occ_combinations = combinations(occ_indices, nexc)
        vir_combinations = combinations(vir_indices, nexc)
        for occ, vir in product(occ_combinations, vir_combinations):
            sd = slater.excite(ground, *(occ + vir))
            if not satisfies_conditions(sd, nspatial, spin, seniority):
                continue
            sds.append(sd)
            count += 1
            if num_limit is not None and count == num_limit:
                return sds
    return sds
