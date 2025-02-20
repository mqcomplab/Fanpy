"""Base class for objectives related to solving the Schrodinger equation."""

import abc
import os

import fanpy.tools.slater as slater
from fanpy.eqn.utils import ComponentParameterIndices
from fanpy.ham.base import BaseHamiltonian
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.composite.base_one import BaseCompositeOneWavefunction
from fanpy.wfn.composite.lincomb import LinearCombinationWavefunction
from fanpy.wfn.composite.product import ProductWavefunction

import numpy as np


class BaseSchrodinger:
    """Base class for objectives related to solving the Schrodinger equation.

    Combines the wavefunction and Hamiltonian (and possibly other components) of the Schrodinger
    equation to create an equation or a set of equations whose solution corresponds to the solution
    of the Schrodinger equation.

    Objects that contribute to the building of equations will be referred to as a `component` of the
    objective. The parameters that are selected for optimization will be referred to as `active`.

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system.
    ham : BaseHamiltonian
        Hamiltonian that defines the system under study.
    indices_component_params : ComponentParameterIndices
        Indices of the component parameters that are active in the objective.
    step_print : bool
        Option to print relevant information when the objective is evaluated.
    step_save : bool
        Option to save parameters when the objective is evaluated.
    tmpfile : str
        Name of the file that will store the parameters used by the objective method.
        If a file name is provided, then parameters are stored upon execution of the objective
        method.

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
    __init__(self, wfn, ham, param_selection=None, optimize_orbitals=False, tmpfile="")
        Initialize the objective.
    assign_params(self, params)
        Assign the parameters to the wavefunction and/or Hamiltonian.
    save_params(self)
        Save all of the parameters to the temporary file.
    wrapped_get_overlap(self, sd, deriv=False)
        Wrap `get_overlap` to be derivatized with respect to the (active) parameters of the
        objective.
    wrapped_integrate_sd_wfn(self, sd, deriv=False)
        Wrap `integrate_sd_wfn` to be derivatized wrt the (active) parameters of the objective.
    wrapped_integrate_sd_sd(self, sd1, sd2, deriv=False)
        Wrap `integrate_sd_sd` to be derivatized wrt the (active) parameters of the objective.
    get_energy_one_proj(self, refwfn, deriv=False)
        Return the energy with respect to a reference wavefunction.
    get_energy_two_proj(self, pspace_l, pspace_r=None, pspace_norm=None, deriv=False)
        Return the energy after projecting out both sides.

    Abstract Properties
    -------------------
    num_eqns : int
        Number of equations in the objective.

    Abstract Methods
    ----------------
    objective(self, params) : float
        Return the value of the objective for the given parameters.
    gradient(self, params) : np.ndarray(K)
        Return the values of the gradient of the objective with respect to the (active) parameters,
        evaluated at the given parameters.
    jacobian(self, params) : np.ndarray(M, K)
        Return the values of the Jacobian of the objective equations with respect to the (active)
        parameters, evaluated at the given parameters.

    """

    # pylint: disable=W0223
    def __init__(
        self,
        wfn,
        ham,
        param_selection=None,
        optimize_orbitals=False,
        step_print=True,
        step_save=True,
        tmpfile="",
    ):
        """Initialize the objective instance.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction.
        ham : BaseHamiltonian
            Hamiltonian that defines the system under study.
        param_selection : tuple/list of 2-tuple/list
            Selection of the parameters that will be used in the objective.
            First element of each entry is a component of the objective: a wavefunction,
            Hamiltonian, or `ParamContainer` instance.
            Second element of each entry is a numpy index array (boolean or indices) that will
            select the parameters from each component that will be used in the objective.
            Default selects the wavefunction parameters.
        optimize_orbitals : bool
            Option to optimize orbitals.
            If Hamiltonian parameters are not selected, all of the orbital optimization parameters
            are optimized.
            If Hamiltonian parameters are selected, then only optimize the selected parameters.
            Default is no orbital optimization.
        step_print : bool
            Option to print relevant information when the objective is evaluated.
            Default is True.
        step_save : bool
            Option to save parameters with every evaluation of the objective.
            Default is True
        tmpfile : str
            Name of the file that will store the parameters used by the objective method.
            By default, the parameter values are not stored.
            If a file name is provided, then parameters are stored upon execution of the objective
            method.

        Raises
        ------
        TypeError
            If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
            If Hamiltonian is not an instance (or instance of a child) of BaseHamiltonian.
            If tmpfile is not a string.
        ValueError
            If wavefunction and Hamiltonian do not have the same number of spin orbitals.

        """
        if __debug__:
            if not isinstance(wfn, BaseWavefunction):
                raise TypeError("Given wavefunction is not an instance of BaseWavefunction (or its child).")
            if not isinstance(ham, BaseHamiltonian):
                raise TypeError("Given Hamiltonian is not an instance of BaseWavefunction (or its child).")
            if wfn.nspin != ham.nspin:
                raise ValueError("Wavefunction and Hamiltonian do not have the same number of spin orbitals.")
            if not isinstance(tmpfile, str):
                raise TypeError("`tmpfile` must be a string.")

        self.wfn = wfn
        self.ham = ham

        if param_selection is None:
            param_selection = [(self.wfn, np.arange(self.wfn.nparams))]
        self._param_selection = param_selection

        if isinstance(param_selection, ComponentParameterIndices):
            self.indices_component_params = param_selection
        else:
            self.indices_component_params = ComponentParameterIndices()
            for component, indices in param_selection:
                self.indices_component_params[component] = indices

        self._optimize_orbitals = optimize_orbitals
        if optimize_orbitals and (
            self.ham not in self.indices_component_params or self.indices_component_params[self.ham].size == 0
        ):
            self.indices_component_params[self.ham] = np.arange(self.ham.nparams)

        self.tmpfile = tmpfile

        self.step_print = step_print
        self.step_save = step_save
        self.print_queue = {}

    @property
    def optimize_orbitals(self):
        """
        Flag to select if orbitals are optimized in the objective.

        """
        return self._optimize_orbitals

    @property
    def param_selection(self):
        """
        Parameter selection of the objective.

        """
        return self._param_selection

    @property
    def indices_objective_params(self):
        """Return the indices of the active objective parameters for each component.

        Returns
        -------
        indices_objctive_params : dict
            Indices of the (active) objective parameters associated with each component.

        """
        output = {}
        count = 0
        for component, indices in self.indices_component_params.items():
            output[component] = np.arange(count, count + indices.size)
            count += indices.size
        return output

    @property
    def all_params(self):
        """Return all of the associated parameters, even if they were not selected.

        Returns
        -------
        params : np.ndarray
            All of the associated parameters.
            Parameters are first ordered by the ordering of each container, then they are ordered by
            the order in which they appear in the container.

        Examples
        --------
        Suppose you have `wfn` and `ham` with parameters `[1, 2, 3]` and `[4, 5, 6, 7]`,
        respectively.

        >>> eqn = BaseSchrodinger((wfn, [True, False, True]), (ham, [3, 1]))
        >>> eqn.all_params
        np.ndarray([1, 2, 3, 4, 5, 6, 7])

        """
        # pylint: disable=E1101
        return np.hstack([component.params.ravel() for component in self.indices_component_params])

    @property
    def active_params(self):
        """Return the parameters selected for optimization.

        Returns
        -------
        params : np.ndarray
            Parameters that are selected for optimization.
            Parameters are first ordered by the ordering of each component, then they are ordered by
            the order in which they appear in the component.

        Examples
        --------
        Suppose you have `wfn` and `ham` with parameters `[1, 2, 3]` and `[4, 5, 6, 7]`,
        respectively.

        >>> eqn = BaseSchrodinger((wfn, [True, False, True]), (ham, [3, 1]))
        >>> eqn.active_params
        np.ndarray([1, 3, 5, 7])

        """
        return np.hstack([comp.params.ravel()[inds] for comp, inds in self.indices_component_params.items()])

    @property
    def active_nparams(self):
        """Return the number of active parameters in the objective.

        Returns
        -------
        active_nparams : int
            Number of active parameters in the objective.

        """
        return sum(indices.size for indices in self.indices_component_params.values())

    # FIXME: long filenames due to long class names used to identify each component
    def save_params(self):
        """Save the parameters associated with the Schrodinger equation.

        All of the parameters are saved, even if it was frozen in the objective.

        The parameters of each component of the Schrodinger equation is saved separately using the
        name in the `tmpfile` as the root (removing the extension). The class name of each component
        and a counter are used to differentiate the files associated with each component.

        """
        if self.tmpfile != "":
            root, ext = os.path.splitext(self.tmpfile)
            names = [type(component).__name__ for component in self.indices_component_params]
            names_totalcount = {name: names.count(name) for name in set(names)}
            names_count = {name: 0 for name in set(names)}

            for component in self.indices_component_params:
                name = type(component).__name__
                if names_totalcount[name] > 1:
                    names_count[name] += 1
                    name = "{}{}".format(name, names_count[name])

                # pylint: disable=E1101
                component.save_params("{}_{}{}".format(root, name, ext))

    def assign_params(self, params):
        """Assign the parameters to the wavefunction and/or Hamiltonian.

        Parameters
        ----------
        params : {np.ndarray(K, )}
            Parameters that are active in the objective function.

        Raises
        ------
        TypeError
            If `params` is not a one-dimensional numpy array.
        ValueError
            If number of parameters must be equal to the number of active parameters.

        """
        indices_objective_params = self.indices_objective_params

        if __debug__:
            if not (isinstance(params, np.ndarray) and params.ndim == 1):
                raise TypeError("Given parameter must be a one-dimensional numpy array.")
            if sum(indices.size for indices in indices_objective_params.values()) != params.size:
                raise ValueError(
                    "Number of given parameters must be equal to the number of active/selected " "parameters."
                )

        for component, indices in self.indices_component_params.items():
            new_params = component.params.ravel()
            new_params[indices] = params[indices_objective_params[component]]
            component.assign_params(new_params)

    def wrapped_get_overlap(self, sd, deriv=False):  # pylint: disable=C0103
        """Wrap `get_overlap` to be derivatized with respect to the parameters of the objective.

        Parameters
        ----------
        sd : {int, np.int64, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : bool
            Option for derivatizing the overlap with respect to the active objective parameters.
            Default is no derivatization.

        Returns
        -------
        overlap : {float, np.ndarray(active_nparams, )}
            Overlap of the wavefunction with the given Slater determinant if `deriv` is False.
            Derivative of the overlap of the wavefunction with the given Slater determinant if
            `deriv` is True.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.

        """
        if __debug__ and not isinstance(deriv, bool):
            raise TypeError("`deriv` must be given as a boolean.")
        # pylint: disable=C0103
        if not deriv:
            return self.wfn.get_overlap(sd)

        output = np.zeros(self.active_nparams)
        if not isinstance(
            self.wfn,
            (BaseCompositeOneWavefunction, LinearCombinationWavefunction, ProductWavefunction),
        ):
            inds_component = self.indices_component_params[self.wfn]
            if inds_component.size > 0:
                inds_objective = self.indices_objective_params[self.wfn]
                output[inds_objective] = self.wfn.get_overlap(sd, inds_component)
        else:
            if isinstance(self.wfn, BaseCompositeOneWavefunction):  # pragma: no cover
                wfns = [self.wfn, self.wfn.wfn]
            elif isinstance(self.wfn, ProductWavefunction):
                wfns = self.wfn.wfns
            elif isinstance(self.wfn, LinearCombinationWavefunction):
                wfns = (self.wfn,) + self.wfn.wfns
            for wfn in wfns:
                if wfn not in self.indices_component_params:
                    continue
                inds_component = self.indices_component_params[wfn]
                if inds_component.size > 0:
                    inds_objective = self.indices_objective_params[wfn]
                    output[inds_objective] = self.wfn.get_overlap(sd, (wfn, inds_component))

        return output

    # FIXME: there are problems when ham is a composite hamiltonian (ham must distinguish between
    # different derivs)
    def wrapped_integrate_sd_wfn(self, sd, deriv=False):  # pylint: disable=C0103
        r"""Wrap `integrate_sd_wfn` to be derivatized wrt the parameters of the objective.

        Parameters
        ----------
        sd : {int, np.int64, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : bool
            Option for derivatizing the integral with respect to the active objective parameters.
            Default is no derivatization.

        Returns
        -------
        integral : {float, np.ndarray(active_nparams, )}
            Integral :math:`\left< \Phi \middle| \hat{H} \middle| \Psi \right>` if `deriv` is False.
            Derivative of the integral :math:`\left< \Phi \middle| \hat{H} \middle| \Psi \right>` if
            `deriv` is True.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.

        """
        if __debug__ and not isinstance(deriv, bool):
            raise TypeError("`deriv` must be given as a boolean.")
        # pylint: disable=C0103
        if not deriv:
            return self.ham.integrate_sd_wfn(sd, self.wfn)

        output = np.zeros(self.active_nparams)

        if not isinstance(
            self.wfn,
            (BaseCompositeOneWavefunction, LinearCombinationWavefunction, ProductWavefunction),
        ):
            wfn_inds_component = self.indices_component_params[self.wfn]
            if wfn_inds_component.size > 0:
                wfn_inds_objective = self.indices_objective_params[self.wfn]
                output[wfn_inds_objective] = self.ham.integrate_sd_wfn(sd, self.wfn, wfn_deriv=wfn_inds_component)
        else:
            if isinstance(self.wfn, BaseCompositeOneWavefunction):  # pragma: no cover
                wfns = [self.wfn, self.wfn.wfn]
            elif isinstance(self.wfn, ProductWavefunction):
                wfns = self.wfn.wfns
            elif isinstance(self.wfn, LinearCombinationWavefunction):
                wfns = (self.wfn,) + self.wfn.wfns
            for wfn in wfns:
                if wfn not in self.indices_component_params:
                    continue
                wfn_inds_component = self.indices_component_params[wfn]
                if wfn_inds_component.size > 0:
                    wfn_inds_objective = self.indices_objective_params[wfn]
                    output[wfn_inds_objective] = self.ham.integrate_sd_wfn(
                        sd, self.wfn, wfn_deriv=(wfn, wfn_inds_component)
                    )

        ham_inds_component = self.indices_component_params[self.ham]
        if ham_inds_component.size > 0:
            ham_inds_objective = self.indices_objective_params[self.ham]
            output[ham_inds_objective] = self.ham.integrate_sd_wfn(sd, self.wfn, ham_deriv=ham_inds_component)

        return output

    # FIXME: there are problems when ham is a composite hamiltonian (ham must distinguish between
    #        different derivs)
    def wrapped_integrate_sd_sd(self, sd1, sd2, deriv=False):
        r"""Wrap `integrate_sd_sd` to be derivatized wrt the parameters of the objective.

        Parameters
        ----------
        sd1 : int
            Slater determinant against which the Hamiltonian is integrated.
        sd2 : int
            Slater determinant against which the Hamiltonian is integrated.
        deriv : bool
            Option for derivatizing the integral with respect to the active objective parameters.
            Default is no derivatization.

        Returns
        -------
        integral : {float, np.ndarray(active_nparams, )}
            Integral :math:`\left< \mathbf{m}_i \middle| \hat{H} \middle| \mathbf{m}_j \right>` if
            `deriv` is False.
            Derivative of the integral :math:`\left< \mathbf{m}_i \middle| \hat{H} \middle|
            \mathbf{m}_j \right>` if `deriv` is True.

        Raises
        ------
        TypeError
            If Slater determinant is not an integer.

        """
        if __debug__ and not isinstance(deriv, bool):
            raise TypeError("`deriv` must be given as a boolean.")
        if not deriv:
            return self.ham.integrate_sd_sd(sd1, sd2)

        output = np.zeros(self.active_nparams)

        inds_component = self.indices_component_params[self.ham]
        if inds_component.size > 0:
            inds_objective = self.indices_objective_params[self.ham]
            output[inds_objective] = self.ham.integrate_sd_sd(sd1, sd2, deriv=inds_component)

        return output

    def get_energy_one_proj(self, refwfn, deriv=False):
        r"""Return the energy of the Schrodinger equation with respect to a reference wavefunction.

        .. math::

            E \approx \frac{\left< \Phi_{ref} \middle| \hat{H} \middle| \Psi \right>}
                           {\left< \Phi_{ref} \middle| \Psi \right>}

        where :math:`\Phi_{ref}` is some reference wavefunction. Let

        .. math::

            \left| \Phi_{ref} \right> = \sum_{\mathbf{m} \in S}
                                        g(\mathbf{m}) \left| \mathbf{m} \right>

        Then,

        .. math::

            \left< \Phi_{ref} \middle| \hat{H} \middle| \Psi \right>
            = \sum_{\mathbf{m} \in S}
              g^*(\mathbf{m}) \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>

        and

        .. math::

            \left< \Phi_{ref} \middle| \Psi \right> =
            \sum_{\mathbf{m} \in S} g^*(\mathbf{m}) \left< \mathbf{m} \middle| \Psi \right>

        Ideally, we want to use the actual wavefunction as the reference, but :math:`\Psi` often
        uses too many Slater determinants to be computationally tractible. Instead, a CI
        wavefunction

        .. math::

            \left| \Phi \right> = \sum_{\mathbf{m} \in S} c_{\mathbf{m}} \left| \mathbf{m} \right>

        or a projected form of the wavefunction :math:`\Psi`

        .. math::

            \left| \Phi \right> = \sum_{\mathbf{m} \in S}
                                \left< \Psi \middle| \mathbf{m} \middle> \left| \mathbf{m} \right>

        can be used as a reference wavefunction.

        Parameters
        ----------
        refwfn : {CIWavefunction, list/tuple of int, int}
            Reference wavefunction used to calculate the energy.
            If list/tuple of Slater determinants are given, then the reference wavefunction will be
            the truncated form (according to the given Slater determinants) of the provided
            wavefunction.
        deriv : bool
            Option for derivatizing the energy with respect to the active objective parameters.
            Default is no derivatization.

        Returns
        -------
        energy : {float, np.ndarray(active_nparams, )}
            Energy with respect to the reference if `deriv` is False.
            Derivatives of the energy with respect to the reference if `deriv` is True.

        Raises
        ------
        TypeError
            If `deriv` is not a boolean.
            If `refwfn` is not a CIWavefunction, int, or list/tuple of int.

        """
        if __debug__ and not isinstance(deriv, bool):
            raise TypeError("`deriv` must be given as a boolean.")
        get_overlap = self.wrapped_get_overlap
        integrate_sd_wfn = self.wrapped_integrate_sd_wfn

        # define reference
        if isinstance(refwfn, CIWavefunction):
            ref_sds = refwfn.sds
            ref_coeffs = refwfn.params
            if deriv:
                d_ref_coeffs = np.zeros((refwfn.nparams, self.active_nparams), dtype=float)
                inds_component = self.indices_component_params[refwfn]
                if inds_component.size > 0:
                    inds_objective = self.indices_objective_params[refwfn]
                    d_ref_coeffs[inds_component, inds_objective] = 1.0
        elif slater.is_sd_compatible(refwfn) or (
            isinstance(refwfn, (list, tuple, np.ndarray)) and all(slater.is_sd_compatible(sd) for sd in refwfn)
        ):
            if slater.is_sd_compatible(refwfn):
                refwfn = [refwfn]
            ref_sds = refwfn
            ref_coeffs = np.array([get_overlap(i) for i in refwfn])
            if deriv:
                d_ref_coeffs = np.array([get_overlap(i, deriv) for i in refwfn])
        else:
            raise TypeError(
                "Reference state must be given as a Slater determinant, a CI "
                "Wavefunction, or a list/tuple of Slater determinants. See "
                "`tools.slater` for compatible representations of the Slater "
                "determinants."
            )

        # overlaps and integrals
        overlaps = np.array([get_overlap(i) for i in ref_sds])
        integrals = np.array([integrate_sd_wfn(i) for i in ref_sds])

        # norm
        norm = np.sum(ref_coeffs * overlaps)

        # energy
        energy = np.sum(ref_coeffs * integrals) / norm

        if not deriv:
            return energy

        d_norm = np.sum(d_ref_coeffs * overlaps[:, None], axis=0)
        d_norm += np.sum(ref_coeffs[:, None] * np.array([get_overlap(i, deriv) for i in ref_sds]), axis=0)
        d_energy = np.sum(d_ref_coeffs * integrals[:, None], axis=0) / norm
        d_energy += (
            np.sum(
                ref_coeffs[:, None] * np.array([integrate_sd_wfn(i, deriv) for i in ref_sds]),
                axis=0,
            )
            / norm
        )
        d_energy -= d_norm * energy / norm
        return d_energy

    def get_energy_two_proj(self, pspace_l, pspace_r=None, pspace_norm=None, deriv=False):
        r"""Return the energy after projecting out both sides.

        The variational form of the energy,

        .. math::

            E = \frac{\left< \Phi \middle| \hat{H} \middle| \Psi \right>}
                     {\left< \Psi \middle| \Psi \right>}

        can be approximated by inserting projection operators to the numerator:

        .. math::

            \left< \Psi \middle| \hat{H} \middle| \Psi \right>
            &\approx \left< \Psi \right|
            \sum_{\mathbf{m} \in S_\mathrm{left}}
            \left| \mathbf{m} \middle> \middle< \mathbf{m} \right|
            \hat{H}
            \sum_{\mathbf{n} \in S_\mathrm{right}}
            \left| \mathbf{n} \middle> \middle< \mathbf{n} \middle| \Psi_\mathbf{n} \right>\\
            &\approx \sum_{\mathbf{m} \in S_\mathrm{left}} \sum_{\mathbf{n} \in S_\mathrm{right}}
            \left< \Psi \middle| \mathbf{m} \right>
            \left< \mathbf{m} \middle| \hat{H} \middle| \mathbf{n} \right>
            \left< \mathbf{n} \middle| \Psi \right>\\

        Likewise, the denominator can be approximated by inserting a projection operator:

        .. math::

            \left< \Psi \middle| \Psi \right>
            &\approx \left< \Psi \right|
            \sum_{\mathbf{m} \in S_{norm}} \left| \mathbf{m} \middle> \middle< \mathbf{m} \middle|
            \middle| \Psi \right>\\
            &\approx \sum_{\mathbf{m} \in S_{\mathrm{norm}}}
            \left< \Psi \middle| \mathbf{m} \right>^2

        Parameters
        ----------
        pspace_l : {list/tuple of int, int}
            Projection space used to truncate the numerator of the energy on the left.
        pspace_r : {list/tuple of int, int, None}
            Projection space used to truncate the numerator of the energy on the right.
            By default, the same space as `l_pspace` is used.
        pspace_norm : {list/tuple of int, int, None}
            Projection space used to truncate the denominator of the energy.
            By default, the same space as `l_pspace` is used.
        deriv : bool
            Option for derivatizing the energy with respect to the active objective parameters.
            Default is no derivatization.

        Returns
        -------
        energy : {float, np.ndarray(active_nparams, )}
            Energy by inserting projection operators if `deriv` is False.
            Derivative of the energy by inserting projection operators if `deriv` is True.

        Raises
        ------
        TypeError
            If deriv is not a boolean.
            If projection space is not a list/tuple of int.

        """
        if pspace_r is None:
            pspace_r = pspace_l
        if pspace_norm is None:
            pspace_norm = pspace_l

        if __debug__:
            if not isinstance(deriv, bool):
                raise TypeError("`deriv` must be given as a boolean.")
            for pspace in [pspace_l, pspace_r, pspace_norm]:
                if not (
                    slater.is_sd_compatible(pspace)
                    or (isinstance(pspace, (list, tuple)) and all(slater.is_sd_compatible(sd) for sd in pspace))
                ):
                    raise TypeError(
                        "Projection space must be given as a Slater determinant or a "
                        "list/tuple of Slater determinants. See `tools.slater` for "
                        "compatible representations of the Slater determinants."
                    )

        if slater.is_sd_compatible(pspace_l):
            pspace_l = [pspace_l]
        if slater.is_sd_compatible(pspace_r):
            pspace_r = [pspace_r]
        if slater.is_sd_compatible(pspace_norm):
            pspace_norm = [pspace_norm]
        pspace_l = np.array(pspace_l)
        pspace_r = np.array(pspace_r)
        pspace_norm = np.array(pspace_norm)

        get_overlap = self.wrapped_get_overlap
        integrate_sd_sd = self.wrapped_integrate_sd_sd

        # overlaps and integrals
        overlaps_l = np.array([[get_overlap(i)] for i in pspace_l])
        overlaps_r = np.array([[get_overlap(i) for i in pspace_r]])
        ci_matrix = np.array([[integrate_sd_sd(i, j) for j in pspace_r] for i in pspace_l])
        overlaps_norm = np.array([get_overlap(i) for i in pspace_norm])

        # norm
        norm = np.sum(overlaps_norm**2)

        # energy
        if not deriv:
            return np.sum(overlaps_l * ci_matrix * overlaps_r) / norm

        d_norm = 2 * np.sum(overlaps_norm[:, None] * np.array([get_overlap(i, deriv) for i in pspace_norm]), axis=0)
        d_energy = (
            np.sum(
                np.array([[get_overlap(i, deriv)] for i in pspace_l]) * ci_matrix[:, :, None] * overlaps_r[:, :, None],
                axis=(0, 1),
            )
            / norm
        )
        d_energy += (
            np.sum(
                overlaps_l[:, :, None]
                * np.array([[integrate_sd_sd(i, j, deriv) for j in pspace_r] for i in pspace_l])
                * overlaps_r[:, :, None],
                axis=(0, 1),
            )
            / norm
        )
        d_energy += (
            np.sum(
                overlaps_l[:, :, None] * ci_matrix[:, :, None] * np.array([[get_overlap(i, deriv) for i in pspace_r]]),
                axis=(0, 1),
            )
            / norm
        )
        d_energy -= d_norm * np.sum(overlaps_l * ci_matrix * overlaps_r) / norm**2
        return d_energy

    @abc.abstractproperty
    def num_eqns(self):
        """Return the number of equations in the objective.

        Returns
        -------
        num_eqns : int
            Number of equations in the objective.

        """

    @abc.abstractmethod
    def objective(self, params):
        """Return the value(s) of the equation(s) that represent the Schrodinger equation.

        Parameters
        ----------
        params : np.ndarray
            Parameters of the objective.

        Returns
        -------
        objective_value : float
            Value of the objective for the given parameters.

        """

    def gradient(self, params):
        """Return the gradient of the equation that represent the Schrodinger equation.

        Parameters
        ----------
        params : np.ndarray(K)
            Parameters of the objective.

        Returns
        -------
        gradient_value : np.ndarray(K)
            Values of the gradient of the objective with respect to the (active) parameters.
            Evaluated at the given parameters.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError(
            "Gradient is not implemented. May need to use a derivative-free optimization algorithm" " (e.g. cma)."
        )

    def jacobian(self, params):
        """Return the Jacobian of the equations that represent the Schrodinger equation.

        Parameters
        ----------
        params : np.ndarray(K)
            Parameters of the objective.

        Returns
        -------
        jacobian_value : np.ndarray(M, K)
            Values of the Jacobian of the objective equations with respect to the (active)
            parameters.
            Evaluated at the given parameters.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError(
            "Jacobian is not implemented. At the moment, derivative-free optimization algorithm is "
            "not supported for vector-valued function. May need to condense the equations down to a"
            " single equation and use cma."
        )
