"""FANPT wrapper"""

import numpy as np
import pyci

import fanpy.interface.pyci
from fanpy.fanpt.utils import reduce_to_fock
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.fanpt.containers import FANPTUpdater, FANPTContainerEParam, FANPTContainerEFree


class FANPT:

    @property
    def fanpt_container_class(self):
        """
        FANPT container object.

        """
        return self._fanpt_container_class

    @property
    def fanci_interface(self):
        """
        Projected Schrodinger Objective as interface to PyCI code.

        """
        return self._fanci_interface

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

    @property
    def energy_nuc(self):
        """
        Nuclear repulsion energy in Hartree.

        """
        return self._energy_nuc

    def __init__(
        self,
        fanpy_objective,
        energy_nuc,
        legacy_fanci=True,
        energy_active=True,
        resum=False,
        ref_sd=0,
        final_order=1,
        lambda_i=0.0,
        lambda_f=1.0,
        steps=1,
        step_print=False,
        **kwargs,
    ):
        """
        Initialize the FANPT problem class.

        Arguments
        ---------
            fanpy_objective : ProjectedSchrodinger
                Projected Schrodinger Equation Fanpy objective.
            energy_nuc : float
                Nuclear repulsion energy in Hartree.
            legacy_fanci : bool, optional
                Select Legacy FanCI code as interface to PyCI. It will be removed in future. Defaults to True.
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
            step_print : bool
                Option to print relevant information when the objective is evaluated.
            kwargs (dict, optional):
                Additional keyword arguments for self.fanpt_container_class class. Defaults to {}.

        """
        if not isinstance(fanpy_objective, ProjectedSchrodinger):
            raise TypeError("fanpy_objective must be a Fanpy objective.")
        if kwargs is None:
            kwargs = {}

        # Convert ProjectedSchrodinger to PyCI objective interface
        fanci_interface = fanpy.interface.pyci.PYCI(fanpy_objective, energy_nuc, legacy_fanci=legacy_fanci)
        fanci_objective = fanci_interface.objective

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
        self._nequation = fanci_objective.nequation
        self._nactive = fanci_objective.nactive
        self._inorm = inorm
        self._norm_det = norm_det
        self._energy_nuc = energy_nuc

        # Assign attributes to instance
        self._fanci_interface = fanci_interface

        self.fill = fanci_objective.fill
        self.energy_active = energy_active
        self.resum = resum

        self.ref_sd = ref_sd
        self.final_order = final_order
        self.lambda_i = lambda_i
        self.lambda_f = lambda_f
        self.steps = steps
        self.step_print = step_print
        self.kwargs = kwargs

    def optimize(
        self,
        guess_params,
        final_order=None,
        lambda_i=None,
        lambda_f=None,
        steps=None,
        **solver_kwargs,
    ):
        """
        Solve the FANPT equations.

        Arguments
        ---------
            guess_params : np.ndarray
                Initial guess for wave function parameters.
            final_order : int, optional
                Final order of the FANPT calculation. Defaults to 1.
            lambda_i : float, optional
                Initial lambda value for the solution of the FANPT equations. Defaults to 0.0.
            lambda_f : float, optional
                Lambda value up to which the FANPT calculation will be performed. Defaults to 1.0.
            steps (int, optional): int, optional
                Solve FANPT in n stepts between lambda_i and lambda_f. Defaults to 1.
            solver_kwargs (dict, optional)
                Additional keyword arguments for the solver.

        Returns
        -------
            params: np.ndarray
            Solution of the FANPT calculation.
        """

        # Assign attributes
        final_order = final_order or self.final_order
        lambda_i = lambda_i or self.lambda_i
        lambda_f = lambda_f or self.lambda_f
        steps = steps or self.steps

        # Initialize FanCI objective with Hamiltonian of ideal system
        print(f"Solving FanPT problem using the ideal Hamiltonian")
        self.fanci_interface.update_objective(self.ham0)
        fanci_objective = self.fanci_interface.objective

        # Get initial guess for parameters at initial lambda value.
        results = fanci_objective.optimize(guess_params, **solver_kwargs)
        guess_params[fanci_objective.mask] = results.x

        # Solve FANPT equations
        for l in np.linspace(lambda_i, lambda_f, steps, endpoint=False):
            fanpt_container = self.fanpt_container_class(
                fanci_interface=self.fanci_interface,
                params=guess_params,
                ham0=self.ham0,
                ham1=self.ham1,
                l=l,
                inorm=self.inorm,
                norm_det=self.norm_det,
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
                resum=self.resum,
            )
            new_wfn_params = fanpt_updater.new_wfn_params
            new_energy = fanpt_updater.new_energy

            # These params serve as initial guess to solve the fanci equations for the given lambda.
            fanpt_params = np.append(new_wfn_params, new_energy)
            print("Frobenius Norm of parameters: {}".format(np.linalg.norm(fanpt_params - guess_params)))
            print("Energy change: {}".format(np.linalg.norm(fanpt_params[-1] - guess_params[-1])))

            # Initialize perturbed Hamiltonian with the current value of lambda using the static method of fanpt_container.
            self.fanci_interface.update_objective(fanpt_updater.new_ham)
            fanci_objective = self.fanci_interface.objective

            # Solve the fanci problem with fanpt_params as initial guess.
            # Take the params given by fanci and use them as initial params in the FANPT calculation for the next lambda.
            results = fanci_objective.optimize(fanpt_params, **solver_kwargs)

            fanpt_params[fanci_objective.mask] = results.x
            guess_params = fanpt_params

            # Rebuild active parameters mask according to energy_active
            if not self.energy_active:
                fanci_objective.freeze_parameter(-1)

        return results
