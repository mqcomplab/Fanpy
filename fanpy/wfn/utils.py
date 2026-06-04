"""Utility function for constructing Wavefunction instances."""

import os
from typing import List

from fanpy.eqn.utils import ComponentParameterIndices
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.composite.product import ProductWavefunction
from fanpy.tools import slater

import numpy as np
import math

from scipy.optimize import OptimizeResult, least_squares, root, minimize
from fanpy.tools.performance import current_memory
import cma


def wfn_factory(olp, olp_deriv, nelec, nspin, params, memory=None, assign_params=None):
    """Return the instance of the Wavefunction class with the given overlaps.

    Parameters
    ----------
    olp(sd, params) : function
        Function that returns the overlap from the given Slater determinant and the wavefunction
        parameters.
        `sd` is an integer whose bitstring describes the occupation of the Slater determinant. See
        `fanpy.tools.slater` for more details.
        `params` is a numpy array that contains the parameters of the wavefunction. If other types
        of parameters is desired, `assign_params` should be provided.
    olp_deriv(sd, params) : function
        Function that returns the derivatives of the overlap with respect to the wavefunction
        parameters from the given Slater determinant and the wavefunction parameters.
        `sd` is an integer whose bitstring describes the occupation of the Slater determinant. See
        `fanpy.tools.slater` for more details.
        `params` is a numpy array that contains the parameters of the wavefunction. If other types
        of parameters is desired, `assign_params` should be provided.
    nelec : int
        Number of electrons in the wavefunction.
    nspin : int
        Number of spin orbitals in the wavefunction.
    params : np.ndarray
        Parameters of the wavefunction.
        If is not a numpy array, then `assign_params` should be provided.
    memory : {float, int, str, None}
        Memory available (in bytes) for the wavefunction.
        If number is provided, it is the number of bytes.
        If string is provided, it should end iwth either "mb" or "gb" to specify the units.
        Default does not limit memory usage (i.e. infinite).
    assign_params(self, params) : function
        Method for assigning the parameters in the wavefunction class. First argument is `self` and
        second argument is `params`.
        Default uses `BaseWavefunction.assign_params`.

    """

    class GeneratedWavefunction(BaseWavefunction):
        """Generated wavefunction class from the given olp and olp_deriv."""

        def __init__(self, nelec, nspin, memory=None, params=None):
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
                Parameters of the wavefunction.

            """
            super().__init__(nelec, nspin, memory=memory)
            self.assign_params(params)

        def assign_params(self, params=None):
            """Assign the parameters of the wavefunction.

            Parameters
            ----------
            params
                Parameters of the wavefunction.

            """
            if assign_params is None:
                super().assign_params(params)
            else:
                assign_params(self, params)

        def _olp(self, sd):  # pylint: disable=E0202
            """Return overlap of the wavefunction with the Slater determinant.

            Parameters
            ----------
            sd : int
                Occupation vector of a Slater determinant given as a bitstring.
                Assumed to have the same number of electrons as the wavefunction.

            Returns
            -------
            olp : {float, complex}
                Overlap of the current instance with the given Slater determinant.

            """
            return olp(sd, self.params)

        def _olp_deriv(self, sd):  # pylint: disable=E0202
            """Return the derivatives of the overlap with the Slater determinant.

            Parameters
            ----------
            sd : int
                Occupation vector of a Slater determinant given as a bitstring.
                Assumed to have the same number of electrons as the wavefunction.

            Returns
            -------
            olp_deriv : np.ndarray
                Derivatives of the overlap with respect to all of the wavefunction's parameters.

            """
            return olp_deriv(sd, self.params)

        def get_overlap(self, sd, deriv=None):
            r"""Return the overlap of the wavefunction with a Slater determinant.

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
            # if no derivatization
            if deriv is None:
                return self._olp(sd)
            # if derivatization
            return self._olp_deriv(sd)[deriv]

    return GeneratedWavefunction(nelec, nspin, params=params, memory=memory)