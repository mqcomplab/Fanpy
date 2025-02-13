"""Product of different wavefunctions."""

from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.composite.lincomb import LinearCombinationWavefunction

import numpy as np


class ProductWavefunction(LinearCombinationWavefunction):
    """Product of different wavefunctions.

    Attributes
    ----------
    wfns : tuple of BaseWavefunction
        Wavefunctions whose overlaps will be multiplied together.

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
        Assign the wavefunctions whose overlaps will be multiplied together.
    get_overlap(self, sd, deriv=None) : {float, np.ndarray}
        Return the overlap (or derivative of the overlap) of the wavefunction with a Slater
        determinant.

    Notes
    -----
    In the case where one set of wavefunctions is already optimized and another set is added in
    hopes to improve the overall wavfunction, this second set must be initialized such that it
    produces an overlap of 1 for every Slater determinant. Otherwise, the work done optimizing the
    first set of wavefunction is lost.

    """

    # pylint: disable=W0223
    def __init__(self, wfns, memory=None):
        """Initialize the wavefunction.

        Parameters
        ----------
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
        self.assign_nelec(wfns[0].nelec)
        self.assign_nspin(wfns[0].nspin)

        # FIXME: uses memory in each wavefunction
        # self.assign_memory(memory)

        # assign wavefunctions
        if __debug__:
            if any(not isinstance(wfn, BaseWavefunction) for wfn in wfns):
                raise TypeError("Each wavefunction must be a instance of `BaseWavefunction`.")
            if any(wfn.nelec != wfns[0].nelec for wfn in wfns):
                raise ValueError("All of the wavefunctions must have the same number of electrons.")
            if len(set(wfns)) != len(wfns):
                raise ValueError("Same wavefunction was provided multiple times.")
        self.wfns = wfns

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

        Since the overlap can be drivatized with respect to the parameters of the underlying
        wavefunctions, the API of this method is different from the rest.

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
            return np.prod(wfn_contrib)

        if __debug__:
            if not (
                isinstance(deriv, tuple)
                and len(deriv) == 2
                and isinstance(deriv[0], BaseWavefunction)
                # and isinstance(deriv[1], np.ndarray)
                # and deriv[1].ndim == 1
                # and np.issubdtype(deriv[1].dtype, np.integer)
            ):
                raise TypeError(
                    "Derivative indices must be given as a 2-tuple whose first element is the "
                    "wavefunction and the second elment is the one-dimensional numpy array of "
                    "integer indices."
                )
            if deriv[0] not in self.wfns:
                raise ValueError(
                    "Selected wavefunction (for derivatization) is not one of the composite "
                    "wavefunction or its underlying wavefunctions."
                )
            # if deriv[0] == self and (np.any(deriv[1] < 0) or np.any(deriv[1] >= deriv[0].nparams)):
            #     raise ValueError(
            #         "Provided indices must be greater than or equal to zero and less than the "
            #         "number of parameters."
            #     )

        wfn, indices = deriv
        wfn_index = self.wfns.index(wfn)
        # NOTE: assume that parameters are not shared between wavefunctions
        output = np.prod(wfn_contrib[:wfn_index])
        output *= np.prod(wfn_contrib[wfn_index + 1 :])
        output *= wfn.get_overlap(sd, deriv=indices)
        return output

    def get_overlaps(self, sds, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

        Since the overlap can be drivatized with respect to the parameters of the underlying
        wavefunctions, the API of this method is different from the rest.

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
        # FIXME: MEMORY
        wfn_contrib = []
        for wfn in self.wfns:
            if hasattr(wfn, "get_overlaps"):
                wfn_contrib.append(wfn.get_overlaps(sds, deriv=None))
            else:
                wfn_contrib.append([wfn.get_overlap(sd, deriv=None) for sd in sds])
        wfn_contrib = np.array(wfn_contrib)
        if deriv is None:
            return np.prod(wfn_contrib, axis=0)

        if __debug__:
            if not (
                isinstance(deriv, tuple)
                and len(deriv) == 2
                and isinstance(deriv[0], BaseWavefunction)
                # and isinstance(deriv[1], np.ndarray)
                # and deriv[1].ndim == 1
                # and np.issubdtype(deriv[1].dtype, np.integer)
            ):
                raise TypeError(
                    "Derivative indices must be given as a 2-tuple whose first element is the "
                    "wavefunction and the second elment is the one-dimensional numpy array of "
                    "integer indices."
                )
            if deriv[0] not in self.wfns:
                raise ValueError(
                    "Selected wavefunction (for derivatization) is not one of the composite "
                    "wavefunction or its underlying wavefunctions."
                )
            # if deriv[0] == self and (np.any(deriv[1] < 0) or np.any(deriv[1] >= deriv[0].nparams)):
            #     raise ValueError(
            #         "Provided indices must be greater than or equal to zero and less than the "
            #         "number of parameters."
            #     )

        wfn, indices = deriv
        wfn_index = self.wfns.index(wfn)
        # NOTE: assume that parameters are not shared between wavefunctions
        output = np.prod(wfn_contrib[:wfn_index], axis=0)
        output *= np.prod(wfn_contrib[wfn_index + 1 :], axis=0)
        if hasattr(wfn, "get_overlaps"):
            output = output[:, None] * wfn.get_overlaps(sds, deriv=indices)
        else:
            output = output[:, None] * np.array([wfn.get_overlap(sd, deriv=indices) for sd in sds])
        return output
