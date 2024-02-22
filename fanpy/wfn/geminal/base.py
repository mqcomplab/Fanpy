"""Base class for Geminal wavefunctions."""
import abc

from fanpy.tools import math_tools, slater
from fanpy.wfn.base import BaseWavefunction

import numpy as np


# FIXME: define some function to get the column indices of the parameters from the orbital pairs
#        so that it can be overwritten with something faster than dictionary lookup (the default can
#        be dictionary lookup)
class BaseGeminal(BaseWavefunction):
    r"""Base Geminal Wavefunction.

    A Geminal is a two-electron wavefunction.

    .. math::

        G^\dagger_p = \sum_{ij} C_{pij} a^\dagger_i a^\dagger_j

    All Geminal wavefunctions can be expresed as an antisymmeterized product of geminals:

    .. math::

        \left| \Psi \right> = \prod_{p=1}^P G^\dagger_p \left| \theta \right>

    These wavefunctions can be re-expressed in terms of orbital pairs (i.e. :math:`a^\dagger_p
    a^\dagger_q`), where the overlap of the wavefunction with a Slater determinant is the sum over
    all possible combinations of orbital pairs that construct the Slater determinant (each set of
    orbital pairs must be disjoint to follow Slater-Condon rule). These combinations are equivalent
    to the perfect matchings (disjoint and exhaustive pairing) of the orbitals of a Slater
    determinant. Different flavors of geminals can allow a different set of perfect matchings for a
    given Slater determinant. The method `generate_possible_orbpairs` yields a perfect matching for
    a given Slater determinant. The symmetery of electron pair interchange is captured through the
    evaluation of a permanent. Different approximations of the permanent can be implemented using
    the method `compute_permanent`.

    Alternatively, the sum over the different perfect matchings and the permanent evaluation can be
    merged to construct a different combinatorial sum (such as Pffafian). To implement these
    wavefunctions, the method `get_overlap` should be changed to use this sum and to ignore
    `generate_possible_orbpairs` and `compute_permanent`. These methods are only called in
    `get_overlap` so there should be no issue. If you'd like, you can always raise
    NotImplementedError.

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

    Abstract Methods
    ----------------
    generate_possible_orbpairs(self, occ_indices)
        Yield the possible orbital pairs that can construct the given Slater determinant.

    """

    def __init__(
        self, nelec, nspin, memory=None, ngem=None, orbpairs=None, params=None, enable_cache=True
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
        ngem : {int, None}
            Number of geminals.
        orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct each geminal.
        params : np.ndarray
            Geminal coefficient matrix.
        enable_cache : bool
            Option to cache the results of `_olp` and `_olp_deriv`.
            By default, `_olp` and `_olp_deriv` are cached.

        """
        super().__init__(nelec, nspin, memory=memory)
        self.assign_ngem(ngem=ngem)
        self.assign_orbpairs(orbpairs=orbpairs)
        self.assign_params(params=params)
        if enable_cache:
            self.enable_cache()
        # FIXME: product hack
        self.default_val = 0

    @property
    def npair(self):
        """Return number of electron pairs.

        Returns
        -------
        npair : int
            Number of electron pairs.

        """
        return self.nelec // 2

    @property
    def norbpair(self):
        """Return the number of orbital pairs used to construct the geminals.

        Returns
        -------
        norbpair : int
            Number of orbital pairs used to construct the geminals.

        """
        return len(self.dict_ind_orbpair)

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
        NotImplementedError
            If number of electrons is odd.

        """
        super().assign_nelec(nelec)
        if self.nelec % 2 != 0:
            raise NotImplementedError("Odd number of electrons is not supported.")

    def assign_ngem(self, ngem=None):
        """Assign the number of geminals.

        Parameters
        ----------
        ngem : {int, None}
            Number of geminals.
            Default is the number of electron pairs.

        Raises
        ------
        TypeError
            If number of geminals is not an integer.
        ValueError
            If number of geminals is less than the number of electron pairs.

        Notes
        -----
        Needs to have `nelec` defined.

        """
        if ngem is None:
            ngem = self.npair

        if __debug__:
            if not isinstance(ngem, int):
                raise TypeError("`ngem` must be an integer.")
            if ngem < self.npair:
                raise ValueError("`ngem` must be greater than the number of electron pairs.")

        self.ngem = ngem

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
            If an orbital pair is not given as a 2-tuple/list of integers.
        ValueError
            If an orbital pair has the same integer.
            If an orbital pair occurs more than once.

        Notes
        -----
        Must have `nspin` defined for the default option.

        """
        if orbpairs is None:
            orbpairs = tuple((i, j) for i in range(self.nspin) for j in range(i + 1, self.nspin))

        if __debug__ and not hasattr(orbpairs, "__iter__"):
            raise TypeError("`orbpairs` must iterable.")

        dict_orbpair_ind = {}
        for i, orbpair in enumerate(orbpairs):
            if __debug__:
                if not (
                    isinstance(orbpair, (list, tuple))
                    and len(orbpair) == 2
                    and all(isinstance(ind, int) for ind in orbpair)
                ):
                    raise TypeError("Each orbital pair must be a 2-tuple/list of integers.")
                if orbpair[0] == orbpair[1]:
                    raise ValueError("Orbital pair of the same orbital is invalid")

            orbpair = tuple(orbpair)
            # sort orbitals within the pair
            if orbpair[0] > orbpair[1]:
                orbpair = orbpair[::-1]
            if __debug__ and orbpair in dict_orbpair_ind:
                raise ValueError(
                    "The given orbital pairs have multiple entries of {0}.".format(orbpair)
                )
            dict_orbpair_ind[orbpair] = i

        self.dict_orbpair_ind = dict_orbpair_ind
        self.dict_ind_orbpair = {i: orbpair for orbpair, i in dict_orbpair_ind.items()}

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the geminal wavefunction.

        Parameters
        ----------
        params : {np.ndarray, BaseGeminal, None}
            Parameters of the geminal wavefunction.
            If BaseGeminal instance is given, then the parameters of this instance are used.
            Default corresponds to the ground state HF wavefunction.

        Raises
        ------
        ValueError
            If given BaseGeminal instance does not have the same number of electrons.
            If given BaseGeminal instance does not have the same number of spin orbitals.
            If given BaseGeminal instance does not have the same number of geminals.

        """
        if params is None:
            params = np.zeros((self.ngem, self.norbpair))
            for i in range(self.ngem):
                col_ind = int(self.get_col_ind((i, i + self.nspatial)))
                params[i, col_ind] += 1

        if isinstance(params, BaseGeminal):
            other = params
            if __debug__:
                if self.nelec != other.nelec:
                    raise ValueError(
                        "The number of electrons in the two wavefunctions must be the same."
                    )
                if self.nspin != other.nspin:
                    raise ValueError(
                        "The number of spin orbitals in the two wavefunctions must be the same."
                    )
                if self.ngem != other.ngem:
                    raise ValueError(
                        "The number of geminals in the two wavefunctions must be the same."
                    )
            params = np.zeros((self.ngem, self.norbpair))
            for ind, orbpair in other.dict_ind_orbpair.items():
                try:
                    params[:, self.get_col_ind(orbpair)] = other.params[:, ind]
                except ValueError:
                    print(
                        "The orbital pair of the given wavefunction, {0}, is not possible in the "
                        "current wavefunction. Parameters corresponding to this orbital pair will"
                        " be ignored.".format(orbpair)
                    )

        params = params.reshape(self.ngem, self.norbpair)

        super().assign_params(params=params, add_noise=add_noise)
        self.clear_cache()

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
        try:
            return self.dict_orbpair_ind[orbpair]
        except (KeyError, TypeError):
            raise ValueError(
                "Given orbital pair, {0}, is not included in the wavefunction.".format(orbpair)
            )

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
        try:
            return self.dict_ind_orbpair[col_ind]
        except (KeyError, TypeError):
            raise ValueError(
                "Given column index, {0}, is not used in the " "wavefunction".format(col_ind)
            )

    def compute_permanent(self, col_inds, row_inds=None, deriv=None):
        """Compute the permanent of the matrix that corresponds to the given orbital pairs.

        Parameters
        ----------
        col_inds : np.ndarray
            Indices of the columns of geminal coefficient matrices that will be used.
        row_inds : {np.ndarray, None}
            Indices of the rows of geminal coefficient matrices that will be used.
            Default is all rows.
        deriv : {int, None}
            C-order index (in the flattened array) of the element with with respect to which the
            permanent is derivatized.
            Default is no derivatization.

        Returns
        -------
        permanent : float
            Permanent of the selected submatrix.

        """
        if row_inds is None:
            row_inds = np.arange(self.ngem)

        if __debug__:
            row_inds = np.array(row_inds)
            col_inds = np.array(col_inds)

        # select function that evaluates the permanent
        # Ryser algorithm is faster if the number of rows and columns are greater than 3
        if len(col_inds) <= 3 >= len(row_inds):
            permanent = math_tools.permanent_ryser
        else:
            permanent = math_tools.permanent_combinatoric

        if deriv is None:
            return permanent(self.params[row_inds[:, None], col_inds[None, :]])

        row_removed = deriv // self.norbpair
        col_removed = deriv % self.norbpair
        # cut out rows and columns that corresponds to the element with which the permanent is
        # derivatized
        row_inds_trunc = row_inds[row_inds != row_removed]
        col_inds_trunc = col_inds[col_inds != col_removed]
        # pylint: disable=R1705
        if row_inds_trunc.size == row_inds.size or col_inds_trunc.size == col_inds.size:
            return 0.0
        elif row_inds_trunc.size == col_inds_trunc.size == 0:
            return 1.0
        else:
            return permanent(self.params[row_inds_trunc[:, None], col_inds_trunc[None, :]])

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
        # NOTE: Need to recreate occ_indices
        occ_indices = slater.occ_indices(sd)

        val = 0.0
        orbpair_generator = self.generate_possible_orbpairs(occ_indices)
        # FIXME: product hack
        counter = 0
        for orbpairs, sign in orbpair_generator:
            if len(orbpairs) == 0:
                continue

            col_inds = np.array([self.get_col_ind(orbp) for orbp in orbpairs], dtype=int)
            val += sign * self.compute_permanent(col_inds)
            counter += 1
        # FIXME: product hack
        if counter == 0:
            return self.default_val
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
        # NOTE: Need to recreate occ_indices, row_removed, col_removed
        occ_indices = slater.occ_indices(sd)

        orbpair_generator = self.generate_possible_orbpairs(occ_indices)

        output = np.zeros(self.nparams)
        for orbpairs, sign in orbpair_generator:
            # ASSUMES: permanent evaluation is much more expensive than the lookup
            if len(orbpairs) == 0:
                continue
            col_inds = np.array([self.get_col_ind(orbp) for orbp in orbpairs], dtype=int)
            for row_ind in range(self.params.shape[0]):
                for col_ind in col_inds:
                    i = row_ind * self.params.shape[1] + col_ind
                    output[i] += sign * self.compute_permanent(col_inds, deriv=i)
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
            if deriv is not None and not (
                isinstance(deriv, np.ndarray) and deriv.ndim == 1 and deriv.dtype == int
            ):
                raise TypeError("deriv must be given as a one dimensional numpy array of integers.")

        # if no derivatization
        if deriv is None:
            return self._olp(sd)
        # if derivatization
        return self._olp_deriv(sd)[deriv]

    @abc.abstractmethod
    def generate_possible_orbpairs(self, occ_indices):
        """Yield the possible orbital pairs that can construct the given Slater determinant.

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

    def normalize(self, pspace):
        """Normalize geminal wavefunction.

        Parameters
        ----------
        pspace : list of sd
            Slater determinant with respect to which the wavefunction is normalized.

        """
        norm = sum(self.get_overlap(i) ** 2 for i in pspace)
        self.assign_params(self.params * norm ** (-0.5 / self.ngem))
        self.clear_cache()
