"""Linear combination of different wavefunctions."""

import os

from fanpy.wfn.base import BaseWavefunction

import numpy as np


class LinearCombinationWavefunction(BaseWavefunction):
    """Linear combination of different wavefunctions.

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
    wfns : tuple of BaseWavefunction
        Wavefunctions that will be linearly combined.

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

    Methods
    -------
    __init__(self, nelec, nspin, wfns, memory=None, params=None)
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
    assign_wfns(self, wfns)
        Assign the wavefunctions that will be linearly combined.
    get_overlap(self, sd, deriv=None) : {float, np.ndarray}
        Return the overlap (or derivative of the overlap) of the wavefunction with a Slater
        determinant.

    """

    # pylint: disable=W0223
    def __init__(self, nelec, nspin, wfns, memory=None, params=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        wfns : tuple of BaseWavefunction
            Wavefunctions that will be linearly combined.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            If number is provided, it is the number of bytes.
            If string is provided, it should end iwth either "mb" or "gb" to specify the units.
            Default does not limit memory usage (i.e. infinite).
        params : np.ndarray
            Parameters of the wavefunction.

        """
        super().__init__(nelec, nspin, memory=memory)
        self.assign_wfns(wfns)
        self.assign_params(params=params)

    @property
    def spin(self):
        """Return the spin of the wavefunction.

        Spin is certain when all of the wavefunctions have the same spin.

        Returns
        -------
        spin : {float, None}
            Spin of the wavefunction.
            `None` means that all spins are allowed.

        """
        # pylint: disable=R1705
        if all(wfn.spin == self.wfns[0].spin for wfn in self.wfns):
            return self.wfns[0].spin
        else:
            return None

    @property
    def seniority(self):
        """Return the seniority of the wavefunction.

        Seniority is certain when all of the wavefunctions have the same seniority.

        Returns
        -------
        seniority : {int, None}
            Seniority of the wavefunction.
            `None` means that all senioritys are allowed.

        """
        # pylint: disable=R1705
        if all(wfn.seniority == self.wfns[0].seniority for wfn in self.wfns):
            return self.wfns[0].seniority
        else:
            return None

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the wavefunction.
            Default is first wavefunction.
        add_noise : {bool, False}
            Option to add noise to the given parameters.
            Default is False.

        """
        if params is None:
            params = np.zeros(len(self.wfns))
            params[0] = 1.0

        super().assign_params(params=params, add_noise=add_noise)

    # FIXME: long filenames due to long class names used to identify each component
    def save_params(self, filename):
        """Save parameters associated with the wavefunction.

        Since the parameters of the composite wavefunction and the parameters of the underlying
        wavefunctions are needed to replicated the overlaps, they are saved as separate files, using
        the given filename as the root (removing the extension). The class names of each underlying
        wavefunction and a counter are used to differentiate the files associated with each
        wavefunction.

        Parameters
        ----------
        filename : str

        """
        root, ext = os.path.splitext(filename)
        np.save(filename, self.params)
        names = [type(wfn).__name__ for wfn in self.wfns]
        names_totalcount = {name: names.count(name) for name in set(names)}
        names_count = {name: 0 for name in set(names)}

        for wfn in self.wfns:
            name = type(wfn).__name__
            if names_totalcount[name] > 1:
                names_count[name] += 1
                name = "{}{}".format(name, names_count[name])

            wfn.save_params("{}_{}{}".format(root, name, ext))

    def assign_wfns(self, wfns):
        """Assign the wavefunctions that will be linearly combined.

        Parameters
        ----------
        wfns : tuple of BaseWavefunction
             Wavefunctions that will be linearly combined.

        Raises
        ------
        TypeError
            If wavefunction is not an instance of `BaseWavefunction`
        ValueError
            If the given wavefunction does not have the same number of electrons as the initialized
            value.
            If the given wavefunction does not have the same data type as the initialized value.
            If the given wavefunction does not have the same memory as the initialized value.
            If only one wavefunction is given.
            If a wavefunction is given multiple times.

        """
        if __debug__:
            if any(not isinstance(wfn, BaseWavefunction) for wfn in wfns):
                raise TypeError("Each wavefunction must be a instance of `BaseWavefunction`.")
            if any(wfn.nelec != self.nelec for wfn in wfns):
                raise ValueError(
                    "Given wavefunction does not have the same number of electrons as the"
                    " the instantiated NonorthWavefunction."
                )
            if any(wfn.memory != self.memory for wfn in wfns):
                raise ValueError(
                    "Given wavefunction does not have the same memory as the " "instantiated NonorthWavefunction."
                )
            if len(wfns) == 1:
                raise ValueError("Only one wavefunction is given.")
            if len(set(wfns)) != len(wfns):
                raise ValueError("Same wavefunction was provided multiple times.")

        self.wfns = tuple(wfns)

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

        Since the overlap can be drivatized with respect to the parameters of the composite
        wavefunction and the underlying wavefunctions, the API of this method is different from the
        rest.

        Parameters
        ----------
        sd : int
            Slater Determinant against which the overlap is taken.
        deriv : {2-tuple, None}
            Wavefunction and the indices of the parameters with respect to which the overlap is
            derivatized.
            First element of the tuple is the wavefunction. Second element of the tuple is the
            indices of the parameters of the corresponding wavefunction. The overlap will be
            derivatized with respect to the selected parameters of this wavefunction.
            Default returns the overlap without derivatization.

        Returns
        -------
        overlap : {float, np.ndarray}
            Overlap (or derivative of the overlap) of the wavefunction with the given Slater
            determinant.

        Raises
        ------
        TypeError
            If given Slater determinant is not an integer.
            If `deriv` is not a 2-tuple where the first element is a BaseWavefunction instance and
            the second element is a one-dimensional numpy array of integers.
        ValueError
            If first element of `deriv` is not the composite wavefunction or the underlying
            waefunctions.
            If the provided indices is less than zero or greater than or equal to the number of the
            corresponding parameters.

        """
        wfn_contrib = np.array([wfn.get_overlap(sd, deriv=None) for wfn in self.wfns])
        if deriv is None:
            return np.sum(self.params * wfn_contrib)

        if __debug__:
            if not (
                isinstance(deriv, tuple)
                and len(deriv) == 2
                and isinstance(deriv[0], BaseWavefunction)
                and isinstance(deriv[1], np.ndarray)
                and deriv[1].ndim == 1
                and np.issubdtype(deriv[1].dtype, np.integer)
            ):
                raise TypeError(
                    "Derivative indices must be given as a 2-tuple whose first element is the "
                    "wavefunction and the second elment is the one-dimensional numpy array of "
                    "integer indices."
                )
            if deriv[0] not in (self,) + self.wfns:
                raise ValueError(
                    "Selected wavefunction (for derivatization) is not one of the composite "
                    "wavefunction or its underlying wavefunctions."
                )
            if deriv[0] == self and (np.any(deriv[1] < 0) or np.any(deriv[1] >= self.nparams)):
                raise ValueError(
                    "Provided indices must be greater than or equal to zero and less than the " "number of parameters."
                )

        wfn, indices = deriv
        if wfn == self:
            return wfn_contrib
        return self.params[self.wfns.index(wfn)] * wfn.get_overlap(sd, deriv=indices)
