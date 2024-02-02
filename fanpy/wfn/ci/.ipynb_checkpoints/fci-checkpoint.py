"""Full Configuration Interaction wavefunction."""
from fanpy.wfn.ci.base import CIWavefunction


class FCI(CIWavefunction):
    r"""Full Configuration Interaction Wavefunction.

    Given :math:`2K` spin orbitals (or :math:`K` spatial orbitals) and :math:`N` electrons, there
    are a total of :math:`\binom{2K}{N}` different Slater determinants. While many of these are
    equivalent via certain symmetry operations (e.g. flipping the spin) under some conditions (e.g.
    alpha and beta orbitals are the same), no such considerations are taken in the class.

    .. math::

        \left| \Psi_{\mathrm{FCI}} \right> =
        \sum_{\mathbf{m} \in S_{\mathrm{FCI}}} c_{\mathbf{m}} \left| \mathbf{m} \right>

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
    def assign_seniority(self, seniority=None):
        r"""Assign the seniority of the wavefunction.

        All seniority must be allowed for a FCI wavefunction.

        Raises
        ------
        ValueError
            If the seniority is not `None` (default value)

        """
        if __debug__ and seniority is not None:
            raise ValueError(
                "Only seniority of `None` is supported with the FCI wavefunction. "
                "i.e. All seniorities must be enabled."
            )
        super().assign_seniority(seniority)

    def assign_sds(self, sds=None):
        """Assign the list of Slater determinants from which the CI wavefunction is constructed.

        Uses the Slater determinants for the FCI wavefunction (within the given spin).

        Raises
        ------
        ValueError
            If sds is not None.

        """
        if __debug__ and sds is not None:
            raise ValueError(
                "Only the default list of Slater determinants is allowed. i.e. sds "
                "is `None`. If you would like to customize your CI wavefunction, use "
                "CIWavefunction instead."
            )
        super().assign_sds(sds)
