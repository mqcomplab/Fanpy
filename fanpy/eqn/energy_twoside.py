"""Energy of the Schrodinger equation integrated against projected forms of the wavefunction."""

from fanpy.eqn.base import BaseSchrodinger
from fanpy.tools import sd_list, slater

import numpy as np


class EnergyTwoSideProjection(BaseSchrodinger):
    r"""Energy of the Schrodinger equations integrated against projected forms of the wavefunction.

    .. math::

        E &= \frac{
          \left< \Psi \right|
          \left(
            \sum_{\mathbf{m}_{\mathrm{l}} \in S_\mathrm{left}}
            \left| \mathbf{m}_{\mathrm{l} \middle> \middle< \mathbf{m}_{\mathrm{l} \right|
          \right)
          \hat{H}
          \left(
            \sum_{\mathbf{m}_{\mathrm{r}} \in S_\mathrm{right}}
            \left| \mathbf{m}_{\mathrm{r}} \middle> \middle< \mathbf{m}_{\mathrm{r}} \right|
          \right)
          \left| \Psi \right>
        }{
          \left< \Psi \right|
          \left(
            \sum_{\mathbf{m}_\mathrm{n} \in S_\mathrm{norm}}
            \left| \mathbf{m}_\mathrm{n} \middle> \middle< \mathbf{m}_\mathrm{n} \right|
          \right)
          \left| \Psi \right>
        }\\
        &= \frac{
          \sum_{\mathbf{m}_\mathrm{l} \in S_\mathrm{left}}
          \sum_{\mathbf{m}_\mathrm{r} \in S_\mathrm{right}}
          \left< \Psi \middle| \mathbf{m}_\mathrm{l} \right>
          \left< \mathbf{m}_\mathrm{l} \middle| \hat{H} \middle| \mathbf{m}_\mathrm{r} \right>
          \left< \mathbf{m}_\mathrm{r} \middle| \Psi \right>
        }{
          \sum_{\mathbf{m}_{\mathrm{n}} \in S_{\mathrm{norm}}}
          \left< \Psi \middle| \mathbf{m}_{\mathrm{n}} \right>^2
        }

    where :math:`S_{left}` and  :math:`S_{right}` are the projection spaces for the left and right
    side of the integral :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>`, respectively,
    and :math:`S_{norm}` is the projection space for the norm,
    :math:`\left< \Psi \middle| \Psi \right>`.

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited state).
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
    pspace_l : {list/tuple of int, int}
        Projection space used to truncate the numerator of the energy on the left.
    pspace_r : {list/tuple of int, int, None}
        Projection space used to truncate the numerator of the energy on the right.
    pspace_norm : {list/tuple of int, int, None}
        Projection space used to truncate the denominator of the energy.

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
    num_eqns : int
        Number of equations in the objective.

    Methods
    -------
    __init__(self, wfn, ham, param_selection=None, optimize_orbitals=False, tmpfile="")
        Initialize the objective.
    assign_params(self, params)
        Assign the parameters to the wavefunction and/or hamiltonian.
    save_params(self)
        Save all of the parameters in the `param_selection` to the temporary file.
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
    objective(self, params)
        Return the energy after inserting projection operators.
    gradient(self, params)
        Return the gradient of the energy after inserting projection operators.

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
        pspace_l=None,
        pspace_r=None,
        pspace_n=None,
    ):
        r"""Initialize the objective instance.

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
        pspace_l : {tuple/list of int, None}
            States in the projection space of the left side of the integral
            :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>`.
            By default, the largest space is used.
        pspace_r : {tuple/list of int, None}
            States in the projection space of the right side of the integral
            :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>`.
            By default, the same space as `pspace_l` is used.
        pspace_n : {tuple/list of int, None}
            States in the projection space of the norm :math:`\left< \Psi \middle| \Psi \right>`.
            By default, the same space as `pspace_l` is used.

        Raises
        ------
        TypeError
            If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
            If Hamiltonian is not an instance (or instance of a child) of BaseHamiltonian.
            If tmpfile is not a string.
        ValueError
            If wavefunction and Hamiltonian do not have the same number of spin orbitals.

        """
        super().__init__(
            wfn,
            ham,
            param_selection=param_selection,
            optimize_orbitals=optimize_orbitals,
            step_print=step_print,
            step_save=step_save,
            tmpfile=tmpfile,
        )
        self.assign_pspaces(pspace_l, pspace_r, pspace_n)

    def assign_pspaces(self, pspace_l=None, pspace_r=None, pspace_n=None):
        r"""Assign the projection space.

        Parameters
        ----------
        pspace_l : {tuple/list of int, None}
            States in the projection space of the left side of the integral
            :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>`.
            By default, the largest space is used.
        pspace_r : {tuple/list of int, None}
            States in the projection space of the right side of the integral
            :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>`.
            By default, the same space as `pspace_l` is used.
        pspace_n : {tuple/list of int, None}
            States in the projection space of the norm :math:`\left< \Psi \middle| \Psi \right>`.
            By default, the same space as `pspace_l` is used.

        Raises
        ------
        TypeError
            If projection space is not a list or a tuple.
            If state in projection space is not given as a Slater determinant or a CI wavefunction.
        ValueError
            If given state in projection space does not have the same number of electrons as the
            wavefunction.
            If given state in projection space does not have the same number of spin orbitals as the
            wavefunction.

        """
        if pspace_l is None:
            pspace_l = sd_list.sd_list(self.wfn.nelec, self.wfn.nspin, spin=self.wfn.spin, seniority=self.wfn.seniority)

        for pspace in [pspace_l, pspace_r, pspace_n]:
            if pspace is None:
                continue
            if not isinstance(pspace, (list, tuple)):
                raise TypeError("Projection space must be given as a list or a tuple.")
            for state in pspace:
                if slater.is_sd_compatible(state):
                    occs = slater.occ_indices(state)
                    if len(occs) != self.wfn.nelec:
                        raise ValueError(
                            "Given state does not have the same number of electrons as" " the given wavefunction."
                        )
                    if any(i >= self.wfn.nspin for i in occs):
                        raise ValueError(
                            "Given state does not have the same number of spin " "orbitals as the given wavefunction."
                        )
                else:
                    raise TypeError("Projection space must only contain Slater determinants.")

        self.pspace_l = tuple(pspace_l)
        self.pspace_r = tuple(pspace_r) if pspace_r is not None else pspace_r
        self.pspace_n = tuple(pspace_n) if pspace_n is not None else pspace_n

    @property
    def num_eqns(self):
        """Return the number of equations in the objective.

        Returns
        -------
        num_eqns : int
            Number of equations in the objective.

        """
        return 1

    def objective(self, params):
        """Return the energy after inserting projection operators.

        See `BaseSchrodinger.get_energy_two_proj` for details.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        objective : float
            Energy after inserting projection operators.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Save params
        if self.step_save:
            self.save_params()

        energy = self.get_energy_two_proj(self.pspace_l, self.pspace_r, self.pspace_n)

        if self.step_print:
            print("(Mid Optimization) Electronic energy: {}".format(energy))
        else:
            self.print_queue["Electronic energy"] = energy

        return energy

    def gradient(self, params):
        """Return the gradient of the energy after inserting projection operators.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        gradient : np.array(active_nparams, )
            Derivative of the energy after inserting projection operators.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)

        grad = self.get_energy_two_proj(self.pspace_l, self.pspace_r, self.pspace_n, True)

        grad_norm = np.linalg.norm(grad)
        if self.step_print:
            print("(Mid Optimization) Norm of the gradient of the energy: {}".format(grad_norm))
        else:
            self.print_queue["Norm of the gradient of the energy"] = grad_norm

        return grad
