"""Antisymmetric Product of One-Reference-Orbital (AP1roG) Geminals wavefunction."""

from fanpy.tools import slater
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.geminal.apig import APIG

import numpy as np


class AP1roG(APIG):
    r"""Antisymmetric Product of One-Reference-Orbital (AP1roG) Geminals Wavefunction.

    APIG wavefunction where a part of the geminal coefficient matrix, :math:`C`, is constrained to
    be an identity matrix. This part corresponds uniquely to a reference Slater determinant. Then,
    the wavefunction can be redescribed with respect to the reference Slater determinant, similar to
    Coupled Cluster wavefunctions.

    .. math::
        \left| \Psi_{\mathrm{AP1roG}} \right>
        &= \prod_{q=1}^P T_q^\dagger \left| \Phi_{\mathrm{ref}} \right>\\
        &= \prod_{q=1}^P \left( a_i^\dagger a_{\bar{i}}^\dagger +
           \sum_{i=P+1}^B c_{q;i}^{(\mathrm{AP1roG})} a_i^\dagger a_{\bar{i}}^\dagger \right)
           \left| \Phi_{\mathrm{ref}} \right>\\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}}
           | C(\mathbf{m})_{\mathrm{AP1roG}} |^+ \left| \mathbf{m} \right>

    where :math:`P` is the number of electron pairs and :math:`\mathbf{m}` is a seniority-zero
    Slater determinant.

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
        Dictionary of orbital pair that are not occupied in the reference Slater determinant to the
        column index of the geminal coefficient matrix (after removal of the identity submatrix).
    dict_reforbpair_ind : dict of int to 2-tuple of int
        Dictionary of orbital pair that correspond to the reference Slater determinant to the column
        index of the identity submatrix.
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix (after removal of the identity
        submatrix) to the orbital pair that are no occupied in the reference Slater determinant.

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
    __init__(self, nelec, nspin, memory=None, ngem=None, orbpairs=None, ref_sd=None,
             params=None)
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
    assign_ref_sd(self, sd=None)
        Assign the reference Slater determinant.

    """

    def __init__(
        self,
        nelec,
        nspin,
        memory=None,
        ngem=None,
        orbpairs=None,
        ref_sd=None,
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
        ngem : {int, None}
            Number of geminals.
        orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct each geminal.
        params : np.ndarray
            Geminal coefficient matrix.
        ref_sd : {int, None}
            Reference Slater determinant.
            Default is the HF ground state.
        enable_cache : bool
            Option to cache the results of `_olp` and `_olp_deriv`.
            By default, `_olp` and `_olp_deriv` are cached.

        Notes
        -----
        Need to skip over APIG.__init__ because `assign_ref_sd` must come before `assign_params`.

        """
        # pylint: disable=W0233,W0231
        BaseWavefunction.__init__(self, nelec, nspin, memory=memory)
        self.assign_ngem(ngem=ngem)
        self.assign_ref_sd(sd=ref_sd)
        self.assign_orbpairs(orbpairs=orbpairs)
        self.assign_params(params=params)
        if enable_cache:
            self.enable_cache()
        # FIXME: product hack
        self.default_val = 0

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the wavefunction.
            Default corresponds to the ground state HF wavefunction.
        add_noise : {bool, False}
            Option to add noise to the given parameters.
            Default is False.

        """
        if params is None:
            params = np.zeros((self.ngem, self.norbpair))

        super().assign_params(params=params, add_noise=add_noise)
        self.clear_cache()

    def assign_ref_sd(self, sd=None):
        """Assign the reference Slater determinant.

        Parameters
        ----------
        sd : {int, None}
            Slater determinant to use as a reference.
            Default is the HF ground state.

        Raises
        ------
        TypeError
            If given `sd` is not an integer.
        ValueError
            If given `sd` does not have the correct number of electrons.
            If given `sd` does not have the correct spin.
            If given `sd` does not have the correct seniority.

        Notes
        -----
        This method depends on `nelec`, `nspin`, `spin`, and `seniority`.

        """
        # pylint: disable=C0103
        if sd is None:
            sd = slater.ground(self.nelec, self.nspin)
        if __debug__:
            if not slater.is_sd_compatible(sd):
                raise TypeError("Slater determinant must be given as an integer.")
            if slater.total_occ(sd) != self.nelec:
                raise ValueError("Given Slater determinant does not have the correct number of " "electrons")
            if self.spin is not None and slater.get_spin(sd, self.nspatial):
                raise ValueError("Given Slater determinant does not have the correct spin.")
            if self.seniority is not None and slater.get_seniority(sd, self.nspatial):
                raise ValueError("Given Slater determinant does not have the correct seniority.")
        self.ref_sd = sd

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
        NotImplementedError
            If number of geminals is not equal to the number of electron pairs.

        """
        super().assign_ngem(ngem)
        if self.ngem != self.npair:
            raise NotImplementedError(
                "Currently, AP1roG does not support more than the exactly "
                " right number of geminals (i.e. number of electron pairs)."
            )

    def assign_orbpairs(self, orbpairs=None):
        """Assign the orbital pairs that will be used to construct the geminals.

        Since a part of the coefficient matrix is constrained, the column indices that correspond to
        these parts will be ignored. Instead, the column indices of the coefficient matrix after the
        removal of the constrained parts will be used.

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
            If orbital pair is not an alpha-beta spin orbital of a spatial orbital.

        Notes
        -----
        Must have `ref_sd` and `nspin` defined for the default option.

        """
        super().assign_orbpairs(orbpairs=orbpairs)
        if __debug__ and not (
            set(self.dict_orbpair_ind.keys()) <= set((i, i + self.nspatial) for i in range(self.nspatial))
        ):
            raise ValueError("Only the alpha-beta spin orbital pair of each spatial orbital is allowed.")
        # removing orbital indices that correspond to the reference Slater determinant
        dict_orbpair_ind = {}
        dict_reforbpair_ind = {}
        for orbpair in self.dict_ind_orbpair.values():
            # if orbital pair is occupied in the reference Slater determinant
            if slater.occ(self.ref_sd, orbpair[0]) and slater.occ(self.ref_sd, orbpair[1]):
                dict_reforbpair_ind[orbpair] = len(dict_reforbpair_ind)
            # otherwise
            else:
                dict_orbpair_ind[orbpair] = len(dict_orbpair_ind)

        self.dict_orbpair_ind = dict_orbpair_ind
        self.dict_reforbpair_ind = dict_reforbpair_ind
        self.dict_ind_orbpair = {i: orbpair for orbpair, i in self.dict_orbpair_ind.items()}

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
        # NOTE: Need to recreate spatial_ref_sd, inds_annihilated and inds_created
        spatial_sd, _ = slater.split_spin(sd, self.nspatial)
        spatial_ref_sd, _ = slater.split_spin(self.ref_sd, self.nspatial)
        orbs_annihilated, orbs_created = slater.diff_orbs(spatial_ref_sd, spatial_sd)
        inds_annihilated = np.array([self.dict_reforbpair_ind[(i, i + self.nspatial)] for i in orbs_annihilated])
        inds_created = np.array([self.dict_orbpair_ind[(i, i + self.nspatial)] for i in orbs_created])

        # FIXME: missing signature. see apig. Not a problem if alpha beta spin pairing
        return self.compute_permanent(row_inds=inds_annihilated, col_inds=inds_created)

    def _olp_deriv(self, sd, deriv):  # pylint: disable=E0202
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
        deriv : int
            Index of the parameter with respect to which the overlap is derivatized.

        Returns
        -------
        olp : {float, complex}
            Derivative of the overlap with respect to the given parameter.

        """
        spatial_sd, _ = slater.split_spin(sd, self.nspatial)
        spatial_ref_sd, _ = slater.split_spin(self.ref_sd, self.nspatial)
        orbs_annihilated, orbs_created = slater.diff_orbs(spatial_ref_sd, spatial_sd)
        inds_annihilated = np.array([self.dict_reforbpair_ind[(i, i + self.nspatial)] for i in orbs_annihilated])
        inds_created = np.array([self.dict_orbpair_ind[(i, i + self.nspatial)] for i in orbs_created])

        # FIXME: missing signature. see apig. Not a problem if alpha beta spin pairing
        return self.compute_permanent(row_inds=inds_annihilated, col_inds=inds_created, deriv=deriv)

    # FIXME: allow other pairing schemes.
    # FIXME: too many return statements
    def get_overlap(self, sd, deriv=None):
        """Return the overlap of the wavefunction with a Slater determinant.

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

        Notes
        -----
        Pairing scheme is assumed to be the alpha-beta spin orbital pair of each spatial orbital.
        This method will fail if another pairing scheme is used.

        """
        # pylint: disable=R0911
        if __debug__:
            if not slater.is_sd_compatible(sd):
                raise TypeError("Slater determinant must be given as an integer.")
            if deriv is not None and not (isinstance(deriv, np.ndarray) and deriv.ndim == 1 and deriv.dtype == int):
                raise TypeError("deriv must be given as a one dimensional numpy array of integers.")

        # cut off beta part (for just the alpha/spatial part)
        spatial_ref_sd, _ = slater.split_spin(self.ref_sd, self.nspatial)
        spatial_sd, _ = slater.split_spin(sd, self.nspatial)
        # get indices of the occupied orbitals
        orbs_annihilated, orbs_created = slater.diff_orbs(spatial_ref_sd, spatial_sd)

        # if different number of electrons
        if len(orbs_annihilated) != len(orbs_created):
            if deriv is not None:
                return np.zeros(deriv.size)
            return 0.0
        # if different seniority
        if slater.get_seniority(sd, self.nspatial) != 0:
            if deriv is not None:
                return np.zeros(deriv.size)
            # FIXME: product hack
            return self.default_val

        # convert to spatial orbitals
        # NOTE: these variables are essentially the same as the output of
        #       generate_possible_orbpairs
        inds_annihilated = np.array([self.dict_reforbpair_ind[(i, i + self.nspatial)] for i in orbs_annihilated])
        inds_created = np.array([self.dict_orbpair_ind[(i, i + self.nspatial)] for i in orbs_created])

        # if no derivatization
        if deriv is None:
            if inds_annihilated.size == inds_created.size == 0:
                return 1.0
            return self._olp(sd)
        # if derivatization
        if inds_annihilated.size == inds_created.size == 0:
            return np.zeros(len(deriv))

        return np.array([self._olp_deriv(sd, i) for i in deriv])
