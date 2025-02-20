"""Antisymmeterized Product of Tetrets Wavefunctions."""

from fanpy.tools.graphs import generate_unordered_partition
from fanpy.wfn.quasiparticle.base import BaseQuasiparticle


class AntisymmeterizedProductTetrets(BaseQuasiparticle):
    """Antisymmeterized Product of Tetrets Wavefunction.

    Analogous to to the Antisymmeterized Product of Geminals (APG) wavefunction.

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
    nquasiparticle : int
        Number of quasisparticles used to make the wavefunction.
    dict_orbsubset_ind : dict
        Dictionary of the orbital subsets to the column indices of corresponding orbital subsets.
    dict_ind_orbsubset : dict
        Dictionary of the column indices to the orbital subsets of corresponding column indices.
    orbsubset_sizes : tuple of int
        Sizes of the orbital subsets used to construct the quasiparticle.

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
    norbsubsets : int
        Number of orbital subsets.

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
    assign_params(self, params=None, add_noise=False)
        Assign parameters of the wavefunction.
    enable_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    assign_nquasiparticle(self, nquasiparticle=None)
        Assign the number of creation operators that will be used to construct the wavefunction.
    assign_orbsubsets(self, orbsubsets=None)
        Assign the subsets of the orbitals that will be used to construct the quasiparticle.
    get_col_ind(self, orbsubset) : int
        Get the column index that corresponds to the given orbital subset.
    get_orbsubset(self, col_ind) : tuple of ind
        Get the orbital subset that corresponds to the given column index.
    compute_permsum(self, nboson, col_inds, row_inds=None, deriv=None) : float
        Compute the permutational sum that corespond to the given wavefunction.
        Default is th mix of permanents and determinants that are used to construct the generalized
        quasiparticle.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.
    _process_subsets(self, subsets) : np.ndarray of int, list of tuple of int, list of tuple of int
        Convert the given subsetss of orbitals to the column indices, subsets of even number of
        orbitals, and subsets of odd number of orbitals.
    _olp(self, sd) : float
        Calculate the overlap of the wavefunction with a Slater determinant.
    _olp_deriv(self, sd) : float
        Calculate the derivative of the overlap of the wavefunction with a Slater determinant.
    generate_possible_orbsubsets(self, occ_indices) : tuple
        Yield the possible 4-orbital subsets that can be used to construct the given Slater
        determinant.

    """

    def assign_nelec(self, nelec):
        """Assign the number of electrons.

        Parameters
        ----------
        nelec : int
            Number of electrons.

        Raises
        ------
        TypeError
            If number of electrons is not an integer.
        ValueError
            If number of electrons is not a positive number.
            If number of electrons is not a multiple of four.

        """
        super().assign_nelec(nelec)
        if __debug__ and self.nelec % 4 != 0:
            raise ValueError("Number of electrons must be a multiple of four.")

    def assign_nquasiparticle(self, nquasiparticle=None):
        """Assign the number of creation operators that will be used to construct the wavefunction.

        Parameters
        ----------
        nquasiparticle : int
            Number of quasiparticles used to create the wavefunction.

        Raises
        ------
        TypeError
            If the number of quasiparticles is not an integer.
        ValueError
            If the number of quasiparticle is not exactly N/4.

        Notes
        -----
        Must have attribute `nelec`.

        """
        if nquasiparticle is None:
            nquasiparticle = self.nelec // 4
        super().assign_nquasiparticle(nquasiparticle)
        if __debug__ and self.nquasiparticle * 4 != self.nelec:
            raise ValueError("Number of quasiparticles must be exactly N/4 where N is the number of electrons")

    # TODO: do we need to make sure that all orbitals indices are included by the subsets?
    # NOTE: what happens if the given subsets cannot be usd to build the given Slater determinant?
    def assign_orbsubsets(self, orbsubsets=None):
        """Assign the subsets of the orbitals that will be used to construct the quasiparticle.

        Parameters
        ----------
        orbsubsets : iterable of tuple/list of ints
            Subsets of orbitals that will be used to construct each quasiparticle.
            Default is all possible orbital groupings .

        Raises
        ------
        NotImplementedError
            If a non-default (`None`) orbsubsets is provided.

        Notes
        -----
        Must have `nspin` defined for the default option.

        """
        if orbsubsets is None:
            orbsubsets = tuple(
                (i, j, k, l)
                for i in range(self.nspin)
                for j in range(i + 1, self.nspin)
                for k in range(j + 1, self.nspin)
                for l in range(k + 1, self.nspin)  # noqa: E741
            )
        else:
            raise NotImplementedError("User provided orbital subsets is not yet supported.")

        super().assign_orbsubsets(orbsubsets)

    # TODO: this function can be generalized and moved to the parent class
    def generate_possible_orbsubsets(self, occ_indices):
        """Yield the possible orbital subsets that can construct the given Slater determinant.

        Parameters
        ----------
        occ_indices : N-tuple of int
            Indices of the orbitals from which the Slater determinant is constructed.
            Must be strictly increasing.

        Yields
        ------
        orbs : P-tuple of 2-tuple of ints
            Indices of the creation operators (grouped by orbital pairs) that construct the Slater
            determinant.

        Raises
        ------
        ValueError
            If the number of occupied orbital indices is not a multiple of four.

        """
        if __debug__ and len(occ_indices) % 4 != 0:
            raise ValueError("The number of orbital indices must be a multiple of four.")
        nsubsets = len(occ_indices) // 4
        for orbsubsets in generate_unordered_partition(occ_indices, [(4, nsubsets)]):
            # convert partition into tuples
            # NOTE: because the dictionary for orbital subsets (dict_orbsubset_ind and
            # dict_ind_orbsubset) was created assuming that each orbital subset is a tuple
            yield [tuple(orbsubset) for orbsubset in orbsubsets]
