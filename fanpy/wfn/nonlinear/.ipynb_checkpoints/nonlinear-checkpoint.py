"""Parent class of CI wavefunctions."""
import itertools

from fanpy.tools import slater
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.ci.fci import FCI
from fanpy.wfn.base import BaseWavefunction

import numpy as np
 


'''
BaseWavefuction 

params = np array

'''
class NonLinear(BaseWavefunction):
    def __init__(self, nelec, nspin, memory=None, params=None, sds=None, spin=None, seniority=None):
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
        if params is None:
            params = np.zeros(self.nspin)
            ground_state = slater.ground(self.nelec, self.nspin)
            occs = slater.occ_indices(ground_state)
            params[occs] = 4
        else:
            if params.shape != (self.nspin, ):
                raise ValueError('The number of the parameters must be equal to the number of spin orbitals.')
        super().assign_params(params=params, add_noise=add_noise)



    def get_overlap(self, sd, deriv=None):
        occ_indx = slater.occ_indices(sd)
        sd_params = np.array(self.params[occ_indx])
        if sd_params.size == len(occ_indx):
            tanh_sd_params = np.tanh(sd_params)
            output = np.prod(tanh_sd_params)
        else:
            print("err")

        return output
