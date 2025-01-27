"""Antisymmeterized Product of Set divided Geminals (APsetG) Wavefunction."""

from fanpy.tools import graphs
from fanpy.wfn.geminal.base import BaseGeminal


class BasicAPsetG(BaseGeminal):
    r"""Antisymmeterized Product of Set divided Geminals (APsetG) Wavefunction.

    .. math::

        G^\dagger_p = \sum_{i \in S_A} \sum_{j \in S_B} C_{pij} a^\dagger_i a^\dagger_j

    where :math:`S_A` and :math:`S_B` are two sets of orbitals that are mutually exclusive (no
    shared orbitals) and exhaustive (form the complete basis set as a whole).

    For simplicity, :math:`S_A` will be assumed to be a set of alpha orbitals, and :math:`S_B` will
    be assumed to be a set of beta orbitals.

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
    dtype
        Data type of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    npair : int
        Number of electron pairs.
    norbpair : int
        Number of orbital pairs used to construct the geminals.

    Methods
    -------
    __init__(self, nelec, nspin, memory=None, ngem=None, orbpairs=None, params=None)
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

    def assign_orbpairs(self, orbpairs=None):
        """Assign all possible orbital pairs for the basic APsetG wavefunction.

        It is not possible to configure the orbital pair ordering.

        Parameters
        ----------
        orbpairs : None
            Indices of the orbital pairs that will be used to construct each geminal.
            Default is all possible orbital pairs.

        Raises
        ------
        ValueError
            If orbpairs is not None.

        Notes
        -----
        Must have `nspin` defined for the default option.

        """
        if __debug__ and orbpairs is not None:
            raise ValueError(
                "Cannot specify the orbital pairs for the APsetG wavefunction. Only the default "
                "configuration will be used."
            )
        orbpairs = [(i, j) for i in range(self.nspatial) for j in range(self.nspatial, self.nspin)]
        # pylint: disable=R1721
        self.dict_orbpair_ind = {i: orbpair for i, orbpair in enumerate(orbpairs)}
        self.dict_ind_orbpair = {i: orbpair for orbpair, i in self.dict_orbpair_ind.items()}

    def get_col_ind(self, orbpair):
        """Get the column index that corresponds to the given orbital pair.

        Parameters
        ----------
        orbpair : 2-tuple of int
            Indices of the orbital pairs that will be used to construct each geminal.
            Default is all possible orbital pairs.

        Returns
        -------
        col_ind : int
            Column index that corresponds to the given orbital pair.

        Raises
        ------
        ValueError
            If given orbital pair is not valid.

        """
        i, j = orbpair
        # ASSUMES: specific structure for alpha and beta orbitals (first alpha then beta)
        if __debug__ and not 0 <= i < self.nspatial <= j < self.nspin:
            raise ValueError("Given orbital pair, {0}, is not included in the wavefunction.".format(orbpair))
        return int(self.nspatial * i + j - self.nspatial)

    def get_orbpair(self, col_ind):
        """Get the orbital pair that corresponds to the given column index.

        Parameters
        ----------
        col_ind : int
            Column index that corresponds to the given orbital pair.

        Returns
        -------
        orbpair : 2-tuple of int
            Indices of the orbital pairs that will be used to construct each geminal.
            Default is all possible orbital pairs.

        Raises
        ------
        ValueError
            If given orbital pair is not valid.

        """
        if __debug__ and not 0 <= col_ind < self.nspatial**2:
            raise ValueError("Given column index, {0}, is not used in the wavefunction".format(col_ind))
        i = col_ind // self.nspatial
        j = col_ind - i * self.nspatial + self.nspatial
        return (i, j)

    def generate_possible_orbpairs(self, occ_indices):
        """Yield the possible orbital pairs that can construct the given Slater determinant.

        Generates all possible orbital pairing schemes where one orbital is selected from one set
        and the other orbital is selected from the other. This is equivalent to finding all the
        perfect matchings (pairing schemes) within a bipartite graph with the given two sets
        (:math:`S_A` and :math:`S_B`) of vertices (occupied orbitals).

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

        """
        alpha_occ_indices = []
        beta_occ_indices = []
        for i in occ_indices:
            if i < self.nspatial:
                alpha_occ_indices.append(i)
            else:
                beta_occ_indices.append(i)
        yield from graphs.generate_biclique_pmatch(
            alpha_occ_indices, beta_occ_indices, occ_indices, is_decreasing=False
        )
