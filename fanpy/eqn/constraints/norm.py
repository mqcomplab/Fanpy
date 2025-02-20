"""Normalization constraint on the wavefunction."""

from fanpy.eqn.base import BaseSchrodinger
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.utils import ComponentParameterIndices
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np


class NormConstraint(BaseSchrodinger):
    r"""Normalization constraint on the wavefunction.

    .. math::

        \left<\Phi \middle| \Psi\right> - 1 = 0

    where :math:`\Phi` is some reference wavefunction.

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system.
    indices_component_params : ComponentParameterIndices
        Indices of the component parameters that are active in the objective.
    refwfn : {tuple of int, CIWavefunction}
        Wavefunction against which the Schrodinger equation is integrated.
        Tuple of Slater determinants will be interpreted as a projection space, and the reference
        wavefunction will be the given wavefunction truncated to the given projection space.

    Properties
    ----------
    indices_objective_params : dict
        Indices of the (active) objective parameters that corresponds to each component.
    all_params : np.ndarray
        All of the parameters associated with the objective.
    active_params : np.ndarray
        Parameters that are selected for optimization.
    active_nparams : int
        Number of active parameters in the objective.

    Methods
    -------
    __init__(self, wfn, ham, param_selection=None, tmpfile="")
        Initialize the objective.
    assign_params(self, params)
        Assign the parameters to the wavefunction and/or Hamiltonian.
    wrapped_get_overlap(self, sd, deriv=False)
        Wrap `get_overlap` to be derivatized with respect to the (active) parameters of the
        objective.

    """

    # pylint: disable=W0223
    def __init__(self, wfn, refwfn=None, param_selection=None):
        r"""Initialize the norm constraint.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction that defines the state of the system (number of electrons and excited
            state).
        refwfn : {tuple/list of int, CIWavefunction, None}
            Wavefunction against which the norm is integrated.
            Tuple of Slater determinants will be interpreted as a projection space, and the
            reference wavefunction will be the given wavefunction truncated to the given projection
            space.
            By default, the given wavefunction is used as the reference by using a complete
            projection space.
        param_selection : tuple/list of 2-tuple/list
            Selection of the parameters that will be used in the objective.
            First element of each entry is a component of the objective: a wavefunction,
            Hamiltonian, or `ParamContainer` instance.
            Second element of each entry is a numpy index array (boolean or indices) that will
            select the parameters from each component that will be used in the objective.
            Default selects the wavefunction parameters.

        """
        # pylint: disable=W0231
        if not isinstance(wfn, BaseWavefunction):
            raise TypeError("Given wavefunction is not an instance of BaseWavefunction (or its " "child).")
        self.wfn = wfn
        self.assign_refwfn(refwfn)

        if param_selection is None:
            param_selection = [(self.wfn, np.arange(self.wfn.nparams))]

        if isinstance(param_selection, dict):
            self.indices_component_params = param_selection
        else:
            self.indices_component_params = ComponentParameterIndices()
            for component, indices in param_selection:
                self.indices_component_params[component] = indices

    # NOTE: the reference wavefunction assignment is already defined in EnergyOneSideProjection
    assign_refwfn = EnergyOneSideProjection.assign_refwfn

    @property
    def num_eqns(self):
        """Return the number of equations in the objective.

        Returns
        -------
        num_eqns : int
            Number of equations in the objective.

        """
        return 1

    # NOTE: much of this code is copied from BaseSchrodinger.get_energy_one_proj
    def objective(self, params):
        r"""Return the norm of the wavefunction.

        .. math::

            \left< \Phi \middle| \Psi \right> - 1
            = \sum_{\mathbf{m} \in S} f^*(\mathbf{m}) \left< \mathbf{m} \middle| \Psi \right> - 1

        where :math:`S` is a set of Slater determinants that are used to build :math:`\Phi`. Then,
        the vector of Slate determinants :math:`[\mathbf{m}]`, is denoted with `ref_sds` and the
        vector :math:`[f^*(\mathbf{m})]` is denoted by `ref_coeffs`.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        objective_value : float
            Value of the objective for the given parameters.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Establish shortcuts
        get_overlap = self.wrapped_get_overlap
        ref = self.refwfn  # pylint: disable=E1101
        # Define reference
        if isinstance(ref, CIWavefunction):
            ref_sds = ref.sds
            ref_coeffs = ref.params
        else:
            ref_sds = ref
            ref_coeffs = np.array([get_overlap(i) for i in ref])
        # Compute
        overlaps = np.array([get_overlap(i) for i in ref_sds])
        return np.sum(ref_coeffs * overlaps) - 1

    # NOTE: much of this code is copied from BaseSchrodinger.get_energy_one_proj
    def gradient(self, params):
        r"""Gradient of the normalization constraint of the wavefunction.

        .. math::

            \frac{d}{dx} (\left< \Phi \middle| \Psi \right> - 1)
            &= \frac{d}{dx} \left< \Phi \middle| \Psi \right>\\
            &= \sum_{\mathbf{m} \in S}
               \frac{d}{dx} f^*(\mathbf{m}) \left< \mathbf{m} \middle| \Psi \right> +
               f^*(\mathbf{m}) \frac{d}{dx} \left< \mathbf{m} \middle| \Psi \right>\\

        where :math:`S` is a set of Slater determinants that are used to build :math:`\Phi`. Then,
        the vector of Slate determinants :math:`[\mathbf{m}]`, is denoted with `ref_sds` and the
        vector :math:`[f^*(\mathbf{m})]` is denoted by `ref_coeffs`.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        gradient : np.array(N,)
            Derivative of the objective with respect to each of the parameters.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Establish shortcuts
        get_overlap = self.wrapped_get_overlap
        ref = self.refwfn  # pylint: disable=E1101
        # Define reference
        if isinstance(ref, CIWavefunction):
            ref_sds = ref.sds
            ref_coeffs = ref.params
        else:
            ref_sds = ref
            ref_coeffs = np.array([get_overlap(i) for i in ref])
        overlaps = np.array([get_overlap(i) for i in ref_sds])

        d_norm = np.zeros(params.size)

        # get derivatives of reference wavefunction
        if isinstance(ref, CIWavefunction):
            d_ref_coeffs = np.zeros((ref.nparams, self.active_nparams), dtype=float)
            inds_component = self.indices_component_params[ref]
            if inds_component.size > 0:
                inds_objective = self.indices_objective_params[ref]
                d_ref_coeffs[inds_component, inds_objective] = 1.0
        else:
            d_ref_coeffs = np.array([get_overlap(k, True) for k in ref])
        # Compute
        d_norm = np.sum(d_ref_coeffs * overlaps[:, None], axis=0)
        d_norm += np.sum(ref_coeffs[:, None] * np.array([get_overlap(i, True) for i in ref_sds]), axis=0)
        return d_norm
