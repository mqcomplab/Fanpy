"""Antisymmeterized Product of Geminals (APG) Wavefunction."""

from fanpy.tools.graphs import generate_complete_pmatch
from fanpy.wfn.geminal.base import BaseGeminal

import numpy as np


class APG(BaseGeminal):
    r"""Antisymmeterized Product of Geminals (APG) Wavefunction.

    Each geminal is a linear combination of all possible (:math:`\binom{2K}{2}`) spin orbital pairs.

    .. math::

        G^\dagger_p = \sum_{i=1}^{2K} \sum_{j>i}^{2K} C_{pij} a^\dagger_i a^\dagger_j

    where all possible orbital pairs, :math:`a^\dagger_i a^\dagger_j`, are allowed to contribute to
    each geminal.

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
        Yield all possible orbital pairs that can construct the given Slater determinant.

    """

    def assign_orbpairs(self, orbpairs=None):
        """Assign all possible orbital pairs.

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
        if orbpairs is not None:
            raise ValueError(
                "Cannot specify the orbital pairs for the APG wavefunction. All possible orbital " "pairs will be used."
            )
        orbpairs = tuple((i, j) for i in range(self.nspin) for j in range(i + 1, self.nspin))
        self.dict_orbpair_ind = {orbpair: i for i, orbpair in enumerate(orbpairs)}
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
        TyperError
            If orbital pair is not a tuple of two integers.
        ValueError
            If any of the indices are less than 0 or greater than or equal to the number of spin
            orbitals.
            If the two indices are equal.

        """
        if __debug__ and not (
            isinstance(orbpair, tuple)
            and len(orbpair) == 2
            and all(isinstance(orb, int) or np.issubdtype(orb, np.integer) for orb in orbpair)
        ):
            raise TypeError("Orbital pair must be given as a tuple of two integers.")
        i, j = orbpair
        if i > j:
            i, j = j, i
        if __debug__ and not all(0 <= i < j < self.nspin for orb in orbpair):
            raise ValueError("Given orbital pair, {0}, is not included in the wavefunction.".format(orbpair))
        # col_ind = (iK - i(i+1)/2) + (j - i)
        return self.nspin * i - i * (i + 1) // 2 + (j - i - 1)

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
            If column index is less than zero or greater than the number of columns.

        """
        # pylint: disable=C0103
        if __debug__ and not 0 <= col_ind < self.nspin * (self.nspin - 1) / 2:
            raise ValueError(
                "Column index, {0}, is less than 0 or greater than or the number of columns." "".format(col_ind)
            )
        i = (2 * self.nspin - 1 - ((1 - 2 * self.nspin) ** 2 - 8 * col_ind) ** 0.5) // 2
        j = col_ind - (i * self.nspin - i * (i + 1) // 2 - i - 1)
        return (i, j)

    def generate_possible_orbpairs(self, occ_indices):
        """Yield all possible orbital pairs that can construct the given Slater determinant.

        Generates all possible orbital pairing schemes from a given set of occupied orbitals. This
        is equivalent to finding all the perfect matchings (pairing schemes) within a complete graph
        with the given set of vertices (occupied orbitals).

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
        yield from generate_complete_pmatch(occ_indices)
