"""Base class for quasiparticle wavefunctions."""

import abc
import itertools as it

from fanpy.tools import math_tools, slater
from fanpy.wfn.base import BaseWavefunction

import numpy as np


class BaseQuasiparticle(BaseWavefunction):
    r"""Base Quasiparticle Wavefunction.

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
    get_overlap(self, sd, deriv=None) : {float, np.ndarray}
        Return the overlap (or derivative of the overlap) of the wavefunction with a Slater
        determinant.
    _process_subsets(self, subsets) : np.ndarray of int, list of tuple of int, list of tuple of int
        Convert the given subsetss of orbitals to the column indices, subsets of even number of
        orbitals, and subsets of odd number of orbitals.
    _olp(self, sd) : float
        Calculate the overlap of the wavefunction with a Slater determinant.
    _olp_deriv(self, sd) : float
        Calculate the derivative of the overlap of the wavefunction with a Slater determinant.

    Abstract Methods
    ----------------
    generate_possible_orbsubsets(self, occ_indices) : tuple
        Yield the possible orbital subsets that can be used to construct the given Slater
        determinant.

    """

    def __init__(
        self,
        nelec,
        nspin,
        memory=None,
        nquasiparticle=None,
        orbsubsets=None,
        params=None,
        enable_cache=True,
    ):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            If number is provided, it is the number of bytes.
            If string is provided, it should end iwth either "mb" or "gb" to specify the units.
            Default does not limit memory usage (i.e. infinite).
        nquasiparticle : int
            Number of quasiparticles used to create the wavefunction.
        orbsubsets : iterable of tuple/list of ints
            Subsets of orbitals that will be used to construct each quasiparticle.
            Default is all possible orbital groupings .
        params : {np.ndarray, BaseQuasiparticle, None}
            Parameters of the quasiparticle wavefunction.
            If BaseQuasiparticle instance is given, then the parameters of this instance are used.
            Default corresponds to the ground state HF wavefunction.
        enable_cache : bool
            Option to cache the results of `_olp` and `_olp_deriv`.
            By default, `_olp` and `_olp_deriv` are cached.

        """
        super().__init__(nelec, nspin, memory=memory)
        self.assign_nquasiparticle(nquasiparticle=nquasiparticle)
        self.assign_orbsubsets(orbsubsets=orbsubsets)
        self.assign_params(params=params)
        if enable_cache:
            self.enable_cache()

    @property
    def norbsubsets(self):
        """Return the number of orbital subsets.

        Returns
        -------
        norbsubsets : int
            Number of orbital subsets.
            Number of columns in the coefficient matrix.

        """
        return len(self.dict_ind_orbsubset)

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
            If the number of quasiparticles is not greater than zero.
        NotImplementedError
            If the number of quasiparticles is not given.

        """
        if __debug__:
            if nquasiparticle is None:
                raise NotImplementedError("There is no default for the number quasiparticles.")
            if not isinstance(nquasiparticle, int):
                raise TypeError("Number of quasiparticles must be given as an integer.")
            if nquasiparticle <= 0:
                raise ValueError("Number of quasiparticles must be greater than zero.")
        self.nquasiparticle = nquasiparticle

    def assign_orbsubsets(self, orbsubsets=None):
        """Assign the subsets of the orbitals that will be used to construct the quasiparticle.

        Parameters
        ----------
        orbsubsets : iterable of tuple/list of ints
            Subsets of orbitals that will be used to construct each quasiparticle.
            Default is all possible orbital groupings .

        Raises
        ------
        TypeError
            If `orbsubsets` is not an iterable.
            If an orbital subset is not given as a list or a tuple.
            If an orbital index is not an integer.
        ValueError
            If an orbital subset has the same integer more than once.
            If an orbital subset occurs more than once.
        NotImplementedError
            If the orbital subsets is not given.

        """
        if __debug__:
            if orbsubsets is None:
                raise NotImplementedError("Default value for the orbital subsets is not set up yet.")
            if not hasattr(orbsubsets, "__iter__"):
                raise TypeError("`orbsubsets` must iterable.")

        dict_orbsubset_ind = {}
        orbsubset_sizes = []
        for i, orbsubset in enumerate(orbsubsets):
            if __debug__:
                if not isinstance(orbsubset, (list, tuple)):
                    raise TypeError("Each orbital subset must be a list or a tuple")
                if not all(isinstance(i, int) for i in orbsubset):
                    raise TypeError("Each orbital must be given as an integer")
                if len(orbsubset) != len(set(orbsubset)):
                    raise ValueError("Orbital subset cannot have the same orbital occuring more than once")

            orbsubset = tuple(sorted(orbsubset))
            if __debug__ and orbsubset in dict_orbsubset_ind:
                raise ValueError("The given orbital subset have multiple entries of {0}" "".format(orbsubset))
            dict_orbsubset_ind[orbsubset] = i

            if len(orbsubset) not in orbsubset_sizes:
                orbsubset_sizes.append(len(orbsubset))

        self.dict_orbsubset_ind = dict_orbsubset_ind
        self.dict_ind_orbsubset = {i: orbsubset for orbsubset, i in dict_orbsubset_ind.items()}
        self.orbsubset_sizes = tuple(orbsubset_sizes)

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the quasiparticle wavefunction.

        Parameters
        ----------
        params : {np.ndarray, BaseQuasiparticle, None}
            Parameters of the quasiparticle wavefunction.
            If BaseQuasiparticle instance is given, then the parameters of this instance are used.
            Default corresponds to the ground state HF wavefunction.

        Raises
        ------
        ValueError
            If given BaseQuasiparticle instance does not have the same number of electrons.
            If given BaseQuasiparticle instance does not have the same number of spin orbitals.
            If given BaseQuasiparticle instance does not have the same number of quasiparticles.

        """
        if params is None:
            # occupied orbitals of HF ground state
            occ_indices = slater.occ_indices(slater.ground(self.nelec, self.nspin))
            # take the first partition
            orbsubsets = next(self.generate_possible_orbsubsets(occ_indices))
            # ASSUME: these operators are the most important creators
            # FIXME: what happens if multiple null quasiparticle?
            # FIXME: what happens when the nquasiparticle != orbsubsets.size?
            # assign
            params = np.zeros((self.nquasiparticle, self.norbsubsets))
            for i, orbsubset in enumerate(orbsubsets):
                col_ind = self.get_col_ind(orbsubset)
                params[i, col_ind] += 1
        if isinstance(params, BaseQuasiparticle):
            other = params
            if __debug__:
                if self.nelec != other.nelec:
                    raise ValueError("The number of electrons in the two wavefunctions must be the same.")
                if self.nspin != other.nspin:
                    raise ValueError("The number of spin orbitals in the two wavefunctions must be the same.")
                if self.nquasiparticle < other.nquasiparticle:
                    raise ValueError(
                        "The number of quasiparticles must be greater than that of the given " "wavefunction."
                    )

            params = np.zeros((self.nquasiparticle, self.norbsubsets))
            for ind, orbsubset in other.dict_ind_orbsubset.items():
                try:
                    params[: other.nquasiparticle, self.get_col_ind(orbsubset)] = other.params[:, ind]
                except ValueError:
                    print(
                        "The orbital subset of the given wavefunction, {0}, is not possible in "
                        "the current wavefunction. Parameters corresponding to this orbital "
                        "subset will be ignored."
                    )

        params = params.reshape(self.nquasiparticle, self.norbsubsets)

        super().assign_params(params=params, add_noise=add_noise)
        self.clear_cache()

    # FIXME: necessary?
    # FIXME: convert orbsubset into a tuple?
    def get_col_ind(self, orbsubset):
        """Get the column index that corresponds to the given orbital subset.

        Parameters
        ----------
        orbsubset : tuple of int
            Indices of the orbitals that will be used to construct each creator.

        Returns
        -------
        col_ind : int
            Column index that corresponds to the given orbital subset.

        Raises
        ------
        ValueError
            If given orbital subset is not valid.

        """
        try:
            return self.dict_orbsubset_ind[orbsubset]
        except (KeyError, TypeError):
            raise ValueError("Given orbital subset, {0}, is not included in the wavefunction.".format(orbsubset))

    # FIXME: necessary?
    def get_orbsubset(self, col_ind):
        """Get the orbital subset that corresponds to the given column index.

        Parameters
        ----------
        col_ind : int
            Column index that corresponds to the given orbital subset.

        Returns
        -------
        orbsubset : tuple of int
            Indices of the orbitals in the subset.

        Raises
        ------
        ValueError
            If given orbital subset is not valid.

        """
        try:
            return self.dict_ind_orbsubset[col_ind]
        except (KeyError, TypeError):
            raise ValueError("Given column index, {0}, is not used in the wavefunction".format(col_ind))

    def compute_permsum(self, nboson, col_inds, row_inds=None, deriv=None):
        """Compute the permutational sum of the given matrix.

        Parameters
        ----------
        col_inds : np.ndarray
            Indices of the columns of coefficient matrices that will be used.
        row_inds : {np.ndarray, None}
            Indices of the rows of coefficient matrices that will be used.
            Default is all rows.
        deriv : {int, None}
            C-order index (in the flattened array) of the element with with respect to which the
            permanent is derivatized.
            Default is no derivatization.

        Returns
        -------
        permsum : float
            Permutational sum of the selected submatrix.

        Notes
        -----
        This permutational sum is almost always dependent on the ordering of the columns. Only in
        the case where all of the matrix belongs to even electron creators is the sum invariant.
        In the case where all of the matrix blongs to odd electron creators, the sum is a
        determinant, so the interchange results in a change in sign. In all other cases, the sum is
        nontrivialy different.

        """
        if row_inds is None:
            row_inds = np.arange(self.nquasiparticle)
        else:
            row_inds = np.array(row_inds)
        col_inds = np.array(col_inds)
        # FIXME: what if row_inds.size != col_inds.size?
        # FIXME: breaks down if empty column and nonempty row (occurs when number of rows is greater
        # than 1 and the number of columns is 1 and permanent is derivitized)
        # NOTE: what about zero electron operator?
        # FIXME: null particle not supported

        matrix = self.params[row_inds[:, None], col_inds[None, :]]
        nrow, ncol = matrix.shape
        rows_bos = np.arange(nboson)
        rows_ferm = np.arange(nboson, nrow)
        if deriv is None:
            output = 0.0
            for cols in it.combinations(range(ncol), nboson):
                cols_bos = np.array(cols)
                if nboson == 0:
                    matrix_bos = np.array([[1]])
                else:
                    matrix_bos = matrix[rows_bos[:, None], cols_bos[None, :]]
                permanent = math_tools.permanent_ryser(matrix_bos)

                cols_ferm = np.array([i for i in range(ncol) if i not in cols])
                if rows_ferm.size == 0 or cols_ferm.size == 0:
                    matrix_ferm = np.array([[1]])
                else:
                    matrix_ferm = matrix[rows_ferm[:, None], cols_ferm[None, :]]
                determinant = np.linalg.det(matrix_ferm)

                output += permanent * determinant
            return output

        # convert index to row and column indices
        row_removed = deriv // self.params.shape[1]
        col_removed = deriv % self.params.shape[1]
        # cut out rows and columns derivatized
        row_inds_trunc = np.array([i for i in row_inds if i != row_removed])
        col_inds_trunc = np.array([i for i in col_inds if i != col_removed])
        if row_removed < nboson:
            nboson -= 1
        # if nothing is cut
        if row_inds_trunc.size == row_inds.size or col_inds_trunc.size == col_inds.size:
            # FIXME: ideally, this outcome should be skipped (b/c it is trivial and can be computed
            #        on the fly instead of caching (this method is called by a cached method))
            return 0.0
        # if everything is cut
        if row_inds_trunc.size == col_inds_trunc.size == 0:
            return 1.0
        # if something remains
        return self.compute_permsum(nboson, col_inds_trunc, row_inds_trunc)

    def _process_subsets(self, subsets):
        """Extract the column indices, and list of bosons and fermions from the given subsets.

        Parameters
        ----------
        subsets : tuple of tuple of int
            Set of creation operators that are used to construct the Slater determinant.
            Each subset is a creation operator that creates the given orbitals.

        Returns
        -------
        col_inds : np.ndarray
            Column indices that correspond to the given subsets.
        bosons : list
            List of subsets that correspond to even-electron creators.
        fermions : list
            List of subsets that correspond to odd-electron creators.

        """
        # collect column indices
        col_inds_bosons = []
        col_inds_fermions = []
        # collect operators
        bosons = []
        fermions = []
        # separate bosons and fermions because bosons come first (left to right)
        for subset in subsets:
            if len(subset) % 2 == 0:
                bosons.append(subset)
                col_inds_bosons.append(self.get_col_ind(subset))
            else:
                fermions.append(subset)
                col_inds_fermions.append(self.get_col_ind(subset))
        # ensure that the first few columns are those of bosons
        col_inds = np.array(col_inds_bosons + col_inds_fermions)

        return col_inds, bosons, fermions

    def _olp(self, sd):  # pylint: disable=E0202
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
        occ_indices = slater.occ_indices(sd)

        val = 0.0
        for subsets in self.generate_possible_orbsubsets(occ_indices):
            # process subsets
            col_inds, bosons, fermions = self._process_subsets(subsets)

            # get sign
            # ASSUMES: bosons come first
            sign = slater.sign_perm([j for i in bosons + fermions for j in i], occ_indices)

            val += sign * self.compute_permsum(len(bosons), col_inds)
        return val

    def _olp_deriv(self, sd):  # pylint: disable=E0202
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.

        Returns
        -------
        olp_deriv : np.ndarray
            Derivatives of the overlap with respect to each parameter.

        """
        occ_indices = slater.occ_indices(sd)

        output = np.zeros(self.nparams)
        for subsets in self.generate_possible_orbsubsets(occ_indices):
            # process subsets
            col_inds, bosons, fermions = self._process_subsets(subsets)

            # get sign
            # ASSUMES: bosons come first
            sign = slater.sign_perm([j for i in bosons + fermions for j in i], occ_indices)

            for row_ind in range(self.params.shape[0]):
                for col_ind in col_inds:
                    i = row_ind * self.params.shape[1] + col_ind
                    output[i] += sign * self.compute_permsum(len(bosons), col_inds, deriv=i)
        return output

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::
            \left| \Psi \right>
            &= \prod_{p=1}^{N_{gem}} \sum_{ij}
               C_{pij} a^\dagger_i a^\dagger_j \left| \theta \right>\\
            &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} |C(\mathbf{m})|^+
            \left| \mathbf{m} \right>

        where :math:`N_{gem}` is the number of geminals, :math:`\mathbf{m}` is a Slater determinant.

        Parameters
        ----------
        sd : int
            Slater Determinant against which the overlap is taken.
        deriv : {np.ndarray, None}
            Indices of the parameters with respect to which the overlap is derivatized.
            Default returns the overlap without derivatization.

        Returns
        -------
        overlap : {float, np.ndarray}
            Overlap (or derivative of the overlap) of the wavefunction with the given Slater
            determinant.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.
            If deriv is not a one dimensional numpy array of integers.

        """
        if __debug__:
            if not slater.is_sd_compatible(sd):
                raise TypeError("Slater determinant must be given as an integer.")
            if deriv is not None and not (isinstance(deriv, np.ndarray) and deriv.ndim == 1 and deriv.dtype == int):
                raise TypeError("deriv must be given as a one dimensional numpy array of integers.")

        # if no derivatization
        if deriv is None:
            return self._olp(sd)
        return self._olp_deriv(sd)[deriv]

    @abc.abstractmethod
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

        """
