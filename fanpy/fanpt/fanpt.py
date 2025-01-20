""" FANPT wrapper"""

import numpy as np
import pyci
import pytest

from fanpy.interface.fanci import ProjectedSchrodingerFanCI, ProjectedSchrodingerPyCI
from .fanpt import FANPTUpdater, FANPTContainerEParam, FANPTContainerEFree


class FANPT:

    @property
    def fanpt_container_class(self):
        """
        FANPT container object.

        """
        return self._fanpt_container_class

    @property
    def fanci_objective(self):
        """
        Projected Schrodinger Objective object from FancI code.

        """
        return self._fanci_objective

    @property
    def ham0(self) -> pyci.hamiltonian:
        """
        PyCI Hamiltonian of the ideal system.

        """
        return self._ham0

    @property
    def ham1(self) -> pyci.hamiltonian:
        """
        PyCI Hamiltonian of the real system.

        """
        return self._ham1

    @property
    def nequation(self):
        """
        Number of equations in the FANPT problem.

        """
        return self._nequation

    @property
    def nactive(self):
        """
        Number of active parameters in the FANPT problem.

        """
        return self._nactive

    @property
    def inorm(self):
        """
        Intermediate normalization flag.

        """
        return self._inorm

    @property
    def norm_det(self):
        """
        Normalization determinant.

        """
        return self._norm_det

    def __init__(
        self,
        fanci_objective,
        guess_params=None,
        energy_active=True,
        resum=False,
        ref_sd=0,
        final_order=1,
        lambda_i=0.0,
        lambda_f=1.0,
        steps=1,
        **kwargs,
    ):
        """
        Initialize the FANPT problem class.

        Parameters
        ----------
            fanci_objective : FanCI class
                FanCI objective.
            guess_params : np.ndarray, optional
                Initial guess for wave function parameters.
            energy_active : bool, optional
                Whether the energy is an active parameter. It determines which FANPT
                method is used. If set to true, FANPTContainerEParam is used.
                Defaults to True.
            resum : bool, optional
                Indicates if we will solve the FANPT equations by re-summing the series.
                Defaults to False.
            ref_sd : int, optional
                Index of the Slater determinant used to impose intermediate normalization.
                <n[ref_sd]|Psi(l)> = 1. Defaults to 0.
            final_order : int, optional
                Final order of the FANPT calculation. Defaults to 1.
            lambda_i : float, optional
                Initial lambda value for the solution of the FANPT equations. Defaults to 0.0.
            lambda_f : float, optional
                Lambda value up to which the FANPT calculation will be performed. Defaults to 1.0.
            steps (int, optional): int, optional
                Solve FANPT in n stepts between lambda_i and lambda_f. Defaults to 1.
            kwargs (dict, optional):
                Additional keyword arguments for self.fanpt_container_class class. Defaults to {}.

        """
        if not isinstance(fanci_objective, (ProjectedSchrodingerFanCI, ProjectedSchrodingerPyCI)):
            raise TypeError("fanci_objective must be a FanCI wavefunction")
        if kwargs is None:
            kwargs = {}

        # Check for normalization constraint in FANCI wfn
        # Assumes intermediate normalization relative to ref_sd only
        if ref_sd is None:
            ref_sd = fanci_objective.fanpy_objective.refwfn

        if f"<\\psi_{{{ref_sd}}}|\\Psi> - v_{{{ref_sd}}}" in fanci_objective.constraints:
            inorm = True
            norm_det = [(ref_sd, 1.0)]
        else:
            inorm = False
            norm_det = None

        # Select FANPT method
        if energy_active:
            self._fanpt_container_class = FANPTContainerEParam
            if not fanci_objective.mask[-1]:
                fanci_objective.unfreeze_parameter(-1)
        else:
            self._fanpt_container_class = FANPTContainerEFree
            if fanci_objective.mask[-1]:
                fanci_objective.freeze_parameter(-1)

        if resum:
            if energy_active:
                raise ValueError("The energy parameter must be inactive with the resumation option.")
            nequation = fanci_objective.nequation
            nactive = fanci_objective.nactive
            steps = 1
            if not inorm and (nequation == nactive):
                norm_det = [(ref_sd, 1.0)]
            elif inorm and (nequation - 1) == nactive:
                fanci_objective.remove_constraint(f"<\\psi_{{{ref_sd}}}|\\Psi> - v_{{{ref_sd}}}")
                inorm = False
            else:
                raise ValueError("The necesary condition of a determined system of equations is not met.")

        # Obtain Hamiltonian objects from FanCI objective
        self._ham1 = fanci_objective.ham
        self._ham0 = pyci.hamiltonian(
            fanci_objective.ham.ecore, fanci_objective.ham.one_mo, reduce_to_fock(fanci_objective.ham.two_mo)
        )

        # Assign parameters to instance
        self._nequation = nequation
        self._nactive = nactive
        self._inorm = inorm
        self._norm_det = norm_det

        # Assign attributes to instance
        self._fanci_objective = fanci_objective

        if not guess_params:
            self.guess_params = fanci_objective.active_params

        self.fill = fanci_objective.fill
        self.energy_active = energy_active
        self.resum = resum

        self.ref_sd = ref_sd
        self.final_order = final_order
        self.lambda_i = lambda_i
        self.lambda_f = lambda_f
        self.steps = steps
        self.kwargs = kwargs

    def optimize(
        self,
        guess_params=None,
        energy_active=None,
        resum=None,
        ref_sd=None,
        final_order=None,
        lambda_i=None,
        lambda_f=None,
        steps=None,
        **solver_kwargs,
    ):
        """
        Solve the FANPT equations.

        Returns
        -------
            params: np.ndarray
            Solution of the FANPT calculation.
        """

        # Assign attributes
        guess_params = guess_params or self.guess_params
        energy_active = energy_active or self.energy_active
        resum = resum or self.resum
        ref_sd = ref_sd or self.ref_sd
        final_order = final_order or self.final_order
        lambda_i = lambda_i or self.lambda_i
        lambda_f = lambda_f or self.lambda_f
        steps = steps or self.steps

        # Get initial guess for parameters at initial lambda value.
        results = self.fanci_objective.optimize(guess_params, **solver_kwargs)

        guess_params[self.fanci_objective.mask] = results.x
        params = self.guess_params

        # Solve FANPT equations
        for l in np.linspace(lambda_i, lambda_f, steps, endpoint=False):
            fanpt_container = self.self.fanpt_container_class(
                fanci_objective=self.fanci_objective,  # TODO: Check this
                params=params,
                ham0=self.ham0,
                ham1=self.ham1,
                l=l,
                inorm=self.inorm,
                ref_sd=self.ref_sd,
                **self.kwargs,
            )

            final_l = l + (lambda_f - lambda_i) / steps
            print(f"Solving FanPT problem at lambda={final_l}")

            fanpt_updater = FANPTUpdater(
                fanpt_container=fanpt_container,
                final_order=final_order,
                final_l=final_l,
                solver=None,
                resum=resum,
            )
            new_wfn_params = fanpt_updater.new_wfn_params
            new_energy = fanpt_updater.new_energy

            # These params serve as initial guess to solve the fanci equations for the given lambda.
            fanpt_params = np.append(new_wfn_params, new_energy)
            print("Frobenius Norm of parameters: {}".format(np.linalg.norm(fanpt_params - params)))
            print("Energy change: {}".format(np.linalg.norm(fanpt_params[-1] - params[-1])))

            # Initialize perturbed Hamiltonian with the current value of lambda using the static method of fanpt_container.
            ham = fanpt_container.linear_comb_ham(self.ham1, self.ham0, final_l, 1 - final_l)

            # Initialize fanci wfn with the perturbed Hamiltonian.
            self._fanci_objective = self.update_fanci_objective(ham, self.fanci_objective, self.norm_det, self.fill)

            # Solve the fanci problem with fanpt_params as initial guess.
            # Take the params given by fanci and use them as initial params in the
            # fanpt calculation for the next lambda.
            results = self.fanci_objective.optimize(fanpt_params, **solver_kwargs)

            fanpt_params[self.fanci_objective.mask] = results.x
            params = fanpt_params

            if not energy_active:
                self.fanci_objective.freeze_parameter([-1])

        return results

    def update_fanci_objective(self, new_ham, fanci_objective):

        # Get the class of the fanci_objective
        fanpy_objective_class = fanci_objective.fanpy_objective.__class__

        # Create new Fanpy objective
        new_fanpy_objective = fanpy_objective_class(
            new_ham,
            fanci_objective._fanpy_wfn,
            param_selection=fanci_objective.param_selection,
            optimize_orbitals=fanci_objective.optimize_orbitals,
            step_print=fanci_objective.step_print,
            step_save=fanci_objective.step_save,
            tmpfile=fanci_objective.tmpfile,
            pspace=fanci_objective.pspace,
            refwfn=fanci_objective.refwfn,
            eqn_weights=fanci_objective.eqn_weights,
            energy_type=fanci_objective.energy_type,
            energy=fanci_objective.energy,
            constraints=fanci_objective.constraints,
        )

        # Build FanCI objective as PyCI interface
        from fanpy.interface.pyci import PYCI

        fanci_interface = PYCI(
            new_fanpy_objective,
            fanci_objective.energy_nuc,
            norm_det=self.norm_det,
            max_memory=fanci_objective.max_memory,
            legacy=fanci_objective.legacy_fanci,
        )

        return fanci_interface.objective


def reduce_to_fock(two_int, lambda_val=0):
    """Reduce given two electron integrals to that of the correspoding Fock operator.

    Parameters
    ----------
    two_int : np.ndarray(K, K, K, K)
        Two electron integrals of restricted orbitals.

    """
    fock_two_int = two_int * lambda_val
    nspatial = two_int.shape[0]
    indices = np.arange(nspatial)

    fock_two_int[indices[:, None, None], indices[None, :, None], indices[None, None, :], indices[None, :, None]] = (
        two_int[indices[:, None, None], indices[None, :, None], indices[None, None, :], indices[None, :, None]]
    )

    fock_two_int[indices[:, None, None], indices[None, :, None], indices[None, :, None], indices[None, None, :]] = (
        two_int[indices[:, None, None], indices[None, :, None], indices[None, :, None], indices[None, None, :]]
    )

    return fock_two_int
