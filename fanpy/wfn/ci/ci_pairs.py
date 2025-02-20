"""CI Pairs Wavefunction."""
from fanpy.tools import slater
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.ci.doci import DOCI
from fanpy.wfn.geminal.ap1rog import AP1roG

import numpy as np


class CIPairs(DOCI):
    r"""CI Pairs Wavefunction.

    DOCI wavefunction with only first order pairwise excitations.

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
    _spin : float
        Total spin of each Slater determinant.
        :math:`\frac{1}{2}(N_\alpha - N_\beta)`.
        Default is no spin (all spins possible).
    _seniority : int
        Number of unpaired electrons in each Slater determinant.
    sds : tuple of int
        List of Slater determinants used to construct the CI wavefunction.
    dict_sd_index : dictionary of int to int
        Dictionary from Slater determinant to its index in sds.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    spin : int
        Spin of the wavefunction
    seniority : int
        Seniority of the wavefunction
    dtype
        Data type of the wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin, memory=None, params=None, sds=None, spin=None, seniority=None):
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_memory(self, memory=None):
        Assign memory available for the wavefunction.
    assign_params(self, params=None, add_noise=False)
        Assign parameters of the wavefunction.
    import_params(self, guess)
        Transfers parameters from a guess wavefunction to the wavefunction.
    enable_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    assign_spin(self, spin=None)
        Assign the spin of the wavefunction.
    assign_seniority(self, seniority=None)
        Assign the seniority of the wavefunction.
    assign_sds(self, sds=None)
        Assign the list of Slater determinants from which the CI wavefunction is constructed.
    get_overlap(self, sd, deriv=None) : {float, np.ndarray}
        Return the overlap (or derivative of the overlap) of the wavefunction with a Slater
        determinant.

    """

    # pylint:disable=W0223
    def assign_sds(self, sds=None):
        """Assign the list of Slater determinants in the CI Pairs wavefunction.

        Ignores user input and uses the Slater determinants for the CI Pairs wavefunction (within
        the given spin).

        Parameters
        ----------
        sds : iterable of int
            List of Slater determinants.

        Raises
        ------
        ValueError
            If the sds is not `None` (default value).

        """
        if __debug__ and sds is not None:
            raise ValueError(
                "Only the default list of Slater determinants is allowed. i.e. sds "
                "is `None`. If you would like to customize your CI wavefunction, use "
                "CIWavefunction instead."
            )
        super().assign_sds(
            sd_list(
                self.nelec,
                self.nspin,
                num_limit=None,
                exc_orders=[2],
                spin=self.spin,
                seniority=self.seniority,
            )
        )

    def to_ap1rog(self):
        """Return the AP1roG wavefunction that corresponds to the CIPairs wavefunction.

        Returns
        -------
        ap1rog : fanpy.wavefunction.geminals.ap1rog.AP1roG
            AP1roG wavefunction.

        """
        # select Slater determinant with largest contributor as the reference Slater determinant
        ref_sd_ind = np.argsort(np.abs(self.params))[-1]
        ref_sd = self.sds[ref_sd_ind]
        spatial_ref_sd, _ = slater.split_spin(ref_sd, self.nspatial)
        # use ap1rog normalization scheme
        ci_params = self.params / self.params[ref_sd_ind]

        # create AP1roG
        ap1rog = AP1roG(
            self.nelec, self.nspin, ngem=None, orbpairs=None, ref_sd=ref_sd, params=None
        )
        # fill empty geminal coefficient
        gem_coeffs = np.zeros(ap1rog.params.shape)
        for occ_ind in slater.occ_indices(spatial_ref_sd):
            for vir_ind in slater.vir_indices(spatial_ref_sd, self.nspatial):
                # excite slater determinant
                sd_exc = slater.excite(ref_sd, occ_ind, vir_ind)
                sd_exc = slater.excite(sd_exc, occ_ind + self.nspatial, vir_ind + self.nspatial)
                # set geminal coefficient (`a` decremented b/c first npair columns are removed)
                row_ind = ap1rog.dict_reforbpair_ind[(occ_ind, occ_ind + self.nspatial)]
                col_ind = ap1rog.dict_orbpair_ind[(vir_ind, vir_ind + self.nspatial)]
                gem_coeffs[row_ind, col_ind] = ci_params[self.dict_sd_index[sd_exc]]
        ap1rog.assign_params(gem_coeffs)

        return ap1rog
