"""Parent class of CI wavefunctions."""
import itertools

from fanpy.tools import slater
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.base import BaseWavefunction

import numpy as np


class CIWavefunction(BaseWavefunction):
    r"""Wavefunction that can be expressed as a linear combination of Slater determinants.

    .. math::

        \left| \Psi \right> &= \sum_i c_i \left| \Phi_i \right>\\
        &= \sum_{\mathbf{m} \in S} c_{\mathbf{m}} \left| \mathbf{m} \right>\\

    where :math:`\Phi_i` is Slater determinants. The :math:`\mathbf{m}` is the occupation vector of
    a Slater determinant (and therefore can be used interchangeably with the Slater determinant) and
    :math:`S` is the set of Slater determinants used to create the wavefunction.

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
    __init__(self, nelec, nspin, memory=None, params=None, sds=None, spin=None,
             seniority=None):
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
    def __init__(self, nelec, nspin, memory=None, params=None, sds=None, spin=None, seniority=None):
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
        params : np.ndarray
            Coefficients of the Slater determinants of a CI wavefunction.
        sds : iterable of int
            List of Slater determinants used to construct the CI wavefunction.
        spin : float
            Total spin of the wavefunction.
            Default is no spin (all spins possible).
            0 is singlet, 0.5 and -0.5 are doublets, 1 and -1 are triplets, etc.
            Positive spin means that there are more alpha orbitals than beta orbitals.
            Negative spin means that there are more beta orbitals than alpha orbitals.
        seniority : int
            Seniority of the wavefunction.
            Default is no seniority (all seniority possible).

        """
        super().__init__(nelec, nspin, memory=memory)
        self.assign_spin(spin=spin)
        self.assign_seniority(seniority=seniority)
        self.assign_sds(sds=sds)
        # FIXME: atleast doubling memory for faster lookup of sd coefficient
        self.dict_sd_index = {sd: i for i, sd in enumerate(self.sds)}
        self.assign_params(params=params)
        # FIXME: product hack
        self.default_val = 0

    @property
    def spin(self):
        r"""Return the spin of the wavefunction.

        .. math::

            \frac{1}{2}(N_\alpha - N_\beta)

        Returns
        -------
        spin : float
            Spin of the wavefunction.

        Notes
        -----
        `None` means that all possible spins are allowed.

        """
        return self._spin

    @property
    def seniority(self):
        """Return the seniority of the wavefunction.

        Seniority of a Slater determinant is its number of unpaired electrons. The seniority of the
        wavefunction is the expected number of unpaired electrons.

        Returns
        -------
        seniority : int
            Seniority of the wavefunction.

        Notes
        -----
        `None` means that all possible seniority are allowed.

        """
        return self._seniority

    @property
    def nsd(self):
        """Return number of Slater determinants.

        Returns
        -------
        nsd : int
            Number of Slater determinants.

        """
        return len(self.sds)

    def assign_spin(self, spin=None):
        r"""Assign the spin of the wavefunction.

        :math:`\frac{1}{2}(N_\alpha - N_\beta)`

        Parameters
        ----------
        spin : float
            Spin of each Slater determinant.
            `None` means spin is not specified.
            Default is no spin (all spins possible).

        Raises
        ------
        TypeError
            If the spin is not an integer, float, or None.
        ValueError
            If the spin is not an integral multiple of `0.5` greater than zero.

        """
        if __debug__:
            if not isinstance(spin, (int, float, type(None))):
                raise TypeError("Spin should be provided as an integer, float or `None`.")
            if isinstance(spin, (int, float)) and not ((2 * spin) % 1 == 0 and spin >= 0):
                raise ValueError("Spin should be an integral multiple of 0.5 greater than zero.")
        if isinstance(spin, (int, float)):
            spin = float(spin)
        self._spin = spin

    def assign_seniority(self, seniority=None):
        r"""Assign the seniority of the wavefunction.

        :math:`\frac{1}{2}(N_\alpha - N_\beta)`

        Parameters
        ----------
        seniority : float
            Seniority of each Slater determinant.
            `None` means seniority is not specified.
            Default is no seniority (all seniorities possible).

        Raises
        ------
        TypeError
            If the seniority is not an integer or None.
        ValueError
            If the seniority is a negative integer.

        """
        if __debug__:
            if not (seniority is None or isinstance(seniority, int)):
                raise TypeError("Seniority must be an integer or None.")
            if isinstance(seniority, int) and seniority < 0:
                raise ValueError("Seniority must be greater than or equal to zero.")
        self._seniority = seniority

    def assign_sds(self, sds=None):
        """Assign the list of Slater determinants from which the CI wavefunction is constructed.

        Parameters
        ----------
        sds : iterable of int
            List of Slater determinants.

        Raises
        ------
        TypeError
            If sds is not iterable.
            If a Slater determinant is not an integer.
        ValueError
            If an empty iterator was provided.
            If a Slater determinant does not have the correct number of electrons.
            If a Slater determinant does not have the correct spin.
            If a Slater determinant does not have the correct seniority.

        Notes
        -----
        Must have `nelec`, `nspin`, `spin` and `seniority` defined for default behaviour and type
        checking.

        """
        # pylint: disable=C0103
        # FIXME: terrible memory usage
        if sds is None:
            sds = sd_list(
                self.nelec,
                self.nspin,
                num_limit=None,
                exc_orders=None,
                spin=self.spin,
                seniority=self.seniority,
            )

        if __debug__:
            # FIXME: no check for repeated entries
            if not hasattr(sds, "__iter__"):
                raise TypeError("Slater determinants must be given as an iterable")
            sds, temp = itertools.tee(sds, 2)
            sds_is_empty = True
            for sd in temp:
                sds_is_empty = False
                if not slater.is_sd_compatible(sd):
                    raise TypeError("Slater determinant must be given as an integer.")
                if slater.total_occ(sd) != self.nelec:
                    raise ValueError(
                        "Slater determinant, {0}, does not have the correct number of "
                        "electrons, {1}".format(bin(sd), self.nelec)
                    )
                if isinstance(self.spin, float) and slater.get_spin(sd, self.nspatial) != self.spin:
                    raise ValueError(
                        "Slater determinant, {0}, does not have the correct spin, {1}".format(
                            bin(sd), self.spin
                        )
                    )
                if (
                    isinstance(self.seniority, int)
                    and slater.get_seniority(sd, self.nspatial) != self.seniority
                ):
                    raise ValueError(
                        "Slater determinant, {0}, does not have the correct seniority, {1}".format(
                            bin(sd), self.seniority
                        )
                    )
            if sds_is_empty:
                raise ValueError("No Slater determinants were provided.")

        self.sds = tuple(sds)

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
            params = np.zeros(len(self.sds))
            params[0] = 1

        super().assign_params(params=params, add_noise=add_noise)

    def import_params(self, guess):
        """Transfers parameters from a guess wavefunction to the wavefunction.

        Parameters
        ----------
        guess : BaseWavefunction
            The wavefunction object providing the guess parameters.

        """
        # Extract Slater determinants and parameters
        if isinstance(guess, CIWavefunction):
            for index, sd in enumerate(self.sds):
                if sd in guess.sds:
                    guess_index = guess.sds.index(sd)
                    self.params[index] = guess.params[guess_index]
        else:
            raise TypeError("Wavefunctions not supported. Both wavefunctions must be an instance of CIWavefunction.")

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the CI wavefunction with a Slater determinant.

        The overlap of the CI wavefunction with a Slater determinant is the coefficient of that
        Slater determinant in the wavefunction.

        .. math::

            \left< \Phi_i \middle| \Psi \right> = c_i

        where

        .. math::

            \left| \Psi \right> = \sum_i c_i \left| \Phi_i \right>

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

        """
        if __debug__:
            if not slater.is_sd_compatible(sd):
                raise TypeError("Slater determinant must be given as an integer.")
        # pylint:disable=R1705
        if deriv is None:
            try:
                return self.params[self.dict_sd_index[sd]]
            except KeyError:
                # FIXME: product hack
                return self.default_val

        output = np.zeros(self.nparams)
        try:
            output[self.dict_sd_index[sd]] = 1.0
        except KeyError:
            pass
        return output[deriv]

    def compute_1rdm(self):
        """Compute the one-particle reduced density matrix (1-RDM) for a CI wavefunction.

        Returns
        -------
        rdm1 : np.ndarray
            NumPy array representing the computed 1-RDM.

        """
        # Initialize the one-particle reduced density matrix (1-RDM)
        rdm1 = np.zeros((self.nspin, self.nspin))

        # Loop through Slater determinants and their corresponding parameters
        for sd, param in zip(self.sds, self.params):
            # Get occupied and virtual orbital indices
            occupied_indices = slater.occ_indices(sd)
            virtual_indices = slater.vir_indices(sd, self.nspin)

            # Diagonal contributions to 1-RDM
            rdm1[occupied_indices, occupied_indices] += param * param

            # Single excitation contributions to 1-RDM
            for occ_index in occupied_indices:
                for vir_index in virtual_indices:
                    # Generate the singly-excited determinant
                    excited_sd = slater.excite(sd, occ_index, vir_index)

                    # Check if the excited determinant exists in the wavefunction
                    if excited_sd in self.sds:
                        excited_param = self.params[self.dict_sd_index[excited_sd]]

                        # Compute the contribution with the excitation phase
                        phase = slater.sign_excite(sd, [occ_index], [vir_index])
                        contribution = param * excited_param * phase

                        # Update the 1-RDM with the off-diagonal term
                        rdm1[occ_index, vir_index] += contribution

        return rdm1
