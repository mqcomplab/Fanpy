"""Antisymmeterized Product of Interacting Geminals (APIG) wavefunction."""

from fanpy.tools.slater import sign_perm
from fanpy.wfn.geminal.base import BaseGeminal


class APIG(BaseGeminal):
    r"""Antisymmetrized Product of Interacting Geminals (APIG) Wavefunction.

    Each geminal is a linear combination of orbital pairs that do not share any orbitals with one
    another. Since there are :math:`2K` spin orbitals, there can only be :math:`K` orbital pairs.

    .. math::

        G^\dagger_p = \sum_{i=1}^{K} C_{pi} a^\dagger_i a^\dagger_{\bar{i}}

    where :math:`\bar{i}` corresponds to the spin orbital index that uniquely corresponds to the
    index :math:`i`.

    Then, there will be, at most, one way to represent a Slater determinant in terms of orbital
    pairs. The combinatorial sum present in more complex geminal flavors, such as APG and APsetG,
    reduces down to a single permanent.

    .. math::
        \left| \Psi_{\mathrm{APIG}} \right>
        &= \prod_{p=1}^P G^\dagger_p \left| \theta \right>\\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}}
        |C(\mathbf{m})|^+ \left| \mathbf{m} \right>

    By default, the spin orbitals that belong to the same spatial orbital, i.e. the alpha and beta
    spin orbitals, will be paired up. Then, the resulting wavefunction will only contain seniority
    zero (i.e. no unpaired electrons) Slater determinants.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.
    dict_orbpair_ind : dict of 2-tuple of int to int
        Dictionary of orbital pair (i, j) where i and j are spin orbital indices and i < j to the
        column index of the geminal coefficient matrix.
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix to the orbital pair (i, j)
        where i and j are spin orbital indices and i < j.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    spin : int
        Spin of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    dtype
        Data type of the wavefunction.
    npair : int
        Number of electron pairs.
    norbpair : int
        Number of orbital pairs used to construct the geminals.

    Methods
    -------
    __init__(self, nelec, nspin, memory=None)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_memory(self, memory=None)
        Assign memory available for the wavefunction.
    assign_ngem(self, ngem=None)
        Assign the number of geminals.
    assign_orbpairs(self, orbpairs=None)
        Assign the orbital pairs that will be used to construct the geminals.
    assign_params(self, params=None, add_noise=False)
        Assign the parameters of the geminal wavefunction.
    get_col_ind(self, orbpair)
        Get the column index that corresponds to the given orbital pair.
    get_orbpair(self, col_ind)
        Get the orbital pair that corresponds to the given column index.
    compute_permanent(self, col_inds, row_inds=None, deriv=None)
        Compute the permanent of the matrix that corresponds to the given orbital pairs.
    enable_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : {float, np.ndarray}
        Return the overlap (or derivative of the overlap) of the wavefunction with a Slater
        determinant.
    generate_possible_orbpairs(self, occ_indices)
        Yield the possible orbital pairs that can construct the given Slater determinant.

    """

    @property
    def spin(self):
        """Spin of geminal wavefunction.

        Returns
        -------
        spin : None
            Spin of the geminal geminal wavefunction.

        Notes
        -----
        Spin is not zero if you change the pairing scheme.

        """
        return 0.0

    @property
    def seniority(self):
        """Seniority of geminal wavefunction.

        Returns
        -------
        seniority : None
            Seniority of the geminal geminal wavefunction.

        Notes
        -----
        Seniority is not zero if you change the pairing scheme.

        """
        return 0

    def assign_orbpairs(self, orbpairs=None):
        """Assign the orbital pairs that will be used to construct the geminals.

        Parameters
        ----------
        orbpairs : iterable of 2-tuple/list of ints
            Indices of the orbital pairs that will be used to construct each geminal.
            Default is all possible orbital pairs.

        Raises
        ------
        TypeError
            If `orbpairs` is not an iterable.
            If an orbital pair is not given as a list or a tuple.
            If an orbital pair does not contain exactly two elements.
            If an orbital index is not an integer.
        ValueError
            If an orbital pair has the same integer.
            If an orbital pair occurs more than once.
            If any two orbital pair shares an orbital.
            If there is an orbital missing from the orbital pairs.

        Notes
        -----
        Must have `nspin` defined for the default option.

        """
        if orbpairs is None:
            orbpairs = ((i, i + self.nspatial) for i in range(self.nspatial))
        super().assign_orbpairs(orbpairs)

        if __debug__:
            all_orbs = [j for i in self.dict_orbpair_ind for j in i]
            if len(all_orbs) != len(set(all_orbs)):
                raise ValueError("At least two orbital pairs share an orbital")
            if len(all_orbs) != self.nspin:
                raise ValueError("Not all of the orbitals are included in orbital pairs")

    def generate_possible_orbpairs(self, occ_indices):
        """Yield the possible orbital pairs that can construct the given Slater determinant.

        The APIG wavefunction only contains one pairing scheme for each Slater determinant (because
        no two orbital pairs share the orbital). By default, the alpha and beta spin orbitals that
        correspond to the same spatial orbital will be paired up.

        Parameters
        ----------
        occ_indices : N-tuple of int
            Indices of the orbitals from which the Slater determinant is constructed.
            Must be strictly increasing.

        Yields
        ------
        orbpairs : P-tuple of 2-tuple of ints
            Indices of the creation operators (grouped by orbital pairs) that construct the Slater
            determinant.
        sign : int
            Signature of the transpositions required to shuffle the `orbpairs` back into the
            original order in `occ_indices`.

        Raises
        ------
        ValueError
            If the number of electrons in the Slater determinant does not match up with the number
            of electrons in the wavefunction.

        """
        if __debug__ and len(occ_indices) != self.nelec:
            raise ValueError(
                "The number of electrons in the Slater determinant does not match up "
                "with the number of electrons in the wavefunction."
            )
        # ASSUME each orbital is associated with exactly one orbital pair
        dict_orb_ind = {orbpair[0]: ind for orbpair, ind in self.dict_orbpair_ind.items()}
        orbpairs = []
        for i in occ_indices:
            try:
                ind = dict_orb_ind[i]
            except KeyError:
                continue
            else:
                orbpairs.append(self.dict_ind_orbpair[ind])

        # signature to turn orbpairs into strictly INCREASING order.
        constructed_occ_indices = [orb for orbpair in orbpairs for orb in orbpair]
        sign = sign_perm(constructed_occ_indices)

        # if Slater determinant cannot be constructed from the orbital pairing scheme
        if set(occ_indices) != set(constructed_occ_indices):
            yield [], 1
        else:
            yield tuple(orbpairs), sign
