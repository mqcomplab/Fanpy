"""Tools for keeping track of active parameters in the optimization process."""

from collections import OrderedDict

from fanpy.ham.base import BaseHamiltonian
from fanpy.wfn.base import BaseWavefunction

import numpy as np


class ParamContainer:
    """Container for parameters.

    This class is used to provide the basic functionality that would be available when updating the
    parameters in a given object. This class provides the bare minimum API to be compatible with the
    BaseSchrodinger class.

    Attributes
    ----------
    params : np.ndarray
        Parameters.

    Properties
    ----------
    nparams : int
        Number of parameters.

    Methods
    -------
    __init__(self, params)
        Initialize.
    assign_params(self, params)
        Assign parameters.

    """

    def __init__(self, params):
        """Initialize.

        Parameters
        ----------
        params : {np.ndarray, number}
            Parameters.

        """
        self.assign_params(params)

    @property
    def nparams(self):
        """Return the number of parameters.

        Returns
        -------
        nparams : int
            Number of parameters.

        """
        return self.params.size

    def assign_params(self, params):
        r"""Assign parameters.

        Parameters
        ----------
        params : {np.ndarray, number}
            Parameters.

        Raises
        ------
        TypeError
            If given parameters are not given as a numpy array (or list or tuple) of numbers.
        ValueError
            If number of parameters is zero.

        """
        if isinstance(params, (list, tuple)):
            params = np.array(params)
        elif np.array(params).ndim == 0 and np.issubdtype(np.array(params).dtype, np.number):
            params = np.array([params])

        if __debug__:
            if not (isinstance(params, np.ndarray) and params.ndim == 1 and np.issubdtype(params.dtype, np.number)):
                raise TypeError("Parameters must be given as a numpy array (or list or tuple) of numbers.")
            if params.size == 0:
                raise ValueError("Number of parameters cannot be zero.")

        self.params = params

    def save_params(self, filename):
        """Save parameters.

        Parameters
        ----------
        filename : str

        """
        np.save(filename, self.params)


class ComponentParameterIndices(OrderedDict):
    """Ordered dictionary modified for `indices_component_params` in BaseSchrodinger.

    This dictionary is used to keep track of the parameters that are active (contributing to the
    optimization) in the objective in BaseSchrodinger. Objects that contribute to the building of
    equations will be referred to as a `component` of the objective. The parameters that are
    selected for optimization will be referred to as `active`.

    """

    def __getitem__(self, component):
        """Return the indices of the component parameters that are active in the optimization."""
        try:
            return super().__getitem__(component)
        except KeyError:
            return np.array([], dtype=int)

    def __setitem__(self, component, indices):
        """Set the indices of the active parameters for the given component.

        If boolean indices are provided, then they are converted into integer indices.
        If indices are not ordered, they are ordered before storing.

        Parameters
        ----------
        component : {BaseWavefunction, BaseHamiltonian, ParamContainer}
            Component of the BaseSchrodinger.
        indices : np.ndarray
            Indices that select the active parameters in the given component.
            Must be a one-dimensional numpy array of integral or boolean indices.

        Raises
        ------
        TypeError
            If `component` is not a `BaseWavefunction`, `BaseHamiltonian`, or `ParamContainer`.
            If `indices` is not a one-dimensional numpy array of integers or booleans.
        ValueError
            If `indices` is boolean indices and the number of indices is not equal to the number of
            parameters in the component.
            If `indices` is integer indices and there are repeating indices.
            If `indices` is integer indices and there are integers that are less than zero or are
            greater than or equal to the number of parameters.

        """
        if isinstance(indices, (list, tuple)):
            indices = np.array(indices)

        if __debug__:
            if not isinstance(component, (BaseWavefunction, BaseHamiltonian, ParamContainer)):
                raise TypeError("Provided key must be a `BaseWavefunction`, `BaseHamiltonian`, or " "`ParamContainer`.")
            if not (
                isinstance(indices, np.ndarray)
                and indices.ndim == 1
                and (indices.dtype == np.bool_ or np.issubdtype(indices.dtype, np.integer) or indices.size == 0)
            ):
                raise TypeError(
                    "Indices for the selected (active) parameters must be given as a "
                    "one-dimensional numpy array (or list or tuple) of booleans or integers."
                )
            if indices.dtype == np.bool_ and indices.size != component.nparams:
                raise ValueError(
                    "If boolean indices are used to select parameters for optimization, the number "
                    "of indices must be equal to the number of parameters must have the same size."
                )
            if np.issubdtype(indices.dtype, np.integer):
                if np.unique(indices).size != indices.size:
                    raise ValueError(
                        "If integer indices are used to select parameters for optimization, the "
                        "given indices cannot have repetitions."
                    )
                if np.any(indices < 0) or np.any(indices >= component.nparams):
                    raise ValueError(
                        "If integer indices are used to select parameters for optimization, the "
                        "given indices must be greater than or equal to zero and less than the "
                        "number of parameters."
                    )

        if indices.size == 0:
            indices = indices.astype(int)

        if indices.dtype == bool:
            indices = np.where(indices)[0]

        super().__setitem__(component, np.sort(indices))

    def __eq__(self, other):
        """Check if equal to the given ComponentParameterIndices instance."""
        if not isinstance(other, ComponentParameterIndices):
            return False

        if self.keys() == other.keys():
            return all(np.array_equal(self[key], other[key]) for key in self.keys())
        return False

    def __ne__(self, other):
        """Check if not equal to the given ComponentParameterIndices instance."""
        return not self.__eq__(other)
