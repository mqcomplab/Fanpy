"""DOCI wavefunction."""

from fanpy.wfn.ci.base import CIWavefunction


class DOCI(CIWavefunction):
    r"""Doubly Occupied Configuration Interaction (DOCI) Wavefunction.

    CI wavefunction constructed from all of the seniority zero Slater determinants within the given
    basis. Seniority zero means that only doubly occupied spatial orbitals are used in the
    construction of each Slater determinant.

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
    def assign_nelec(self, nelec):
        """Assign the number of electrons.

        Cannot create seniority zero Slater determinants with odd number of electrons.

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
            If number of electrons is not even.

        """
        super().assign_nelec(nelec)
        if __debug__:
            if self.nelec % 2 != 0:
                raise ValueError("`nelec` must be an even number")

    def assign_spin(self, spin=None):
        r"""Assign the spin of the wavefunction.

        :math:`\frac{1}{2}(N_\alpha - N_\beta)`

        Since the number of alpha and beta electrons are always equal (b/c seniority zero),
        wavefunction will always be a singlet.

        Parameters
        ----------
        spin : float
            Spin of each Slater determinant.
            Default is no spin (all spins possible).

        Raises
        ------
        TypeError
            If the spin is not an integer, float, or None.
        ValueError
            If the spin is not an integral multiple of `0.5` greater than zero.
            If the spin is not 0.

        """
        if spin is None:
            spin = 0
        super().assign_spin(spin)
        if __debug__ and self.spin != 0:
            raise ValueError("spin must be zero for DOCI wavefunction.")

    def assign_seniority(self, seniority=None):
        r"""Assign the seniority of the wavefunction.

        :math:`\frac{1}{2}(N_\alpha - N_\beta)`

        Parameters
        ----------
        seniority : float
            Seniority of each Slater determinant.
            Default is no seniority (all seniorities possible).

        Raises
        ------
        TypeError
            If the seniority is not an integer, float, or None.
        ValueError
            If the seniority is a negative integer.
            If the seniority is not zero.

        """
        if seniority is None:
            seniority = 0
        super().assign_seniority(seniority)
        if __debug__ and self.seniority != 0:
            raise ValueError("seniority must be zero for DOCI wavefunction.")
