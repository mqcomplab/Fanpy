"""Antisymmetric Product of Rank-2 (APr2G) Geminals wavefunction."""

from fanpy.wfn.geminal.apig import APIG
from fanpy.wfn.geminal.rank2_approx import RankTwoApprox


# FIXME: docstring copied from APIG
class APr2G(RankTwoApprox, APIG):
    """Antisymmetrized Product of Rank-2 Geminals (APr2G).

    APIG wavefunction with rank-2 geminal coefficient matrix.

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
