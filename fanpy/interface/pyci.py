from fanpy.eqn.projected import ProjectedSchrodinger

import numpy as np
from scipy.special import comb
from typing import Any

# Log Levels
INFO = 4
NOTE = 3

__all__ = [
    "PYCI",
]


class PYCI:

    def __init__(
        self,
        fanpy_objective,
        energy_nuc,
        norm_det=None,
        norm_param=None,
        constraints=None,
        max_memory=8192,
        legacy_fanci=False,
        verbose=4,
        **kwargs: Any,
    ):
        """
        Initiate an interface object to run FanCI calculations under PyCI framework.

        Arguments
        ---------
        fanpy_objective : ProjectedSchrodinger)
            Fanpy problem built as an objective.
        energy_nuc : float
            Nuclear repulsion energy.
        norm_det : (optional)
            Determinant normalization from FanCI PyCI objective.
        norm_param : (optional)
            Parameters normalization from FanCI PyCI objective.
        constraints : (optional)
            Constraints conditions from FanCI PyCI objective.
        max_memory : (float (in Mb), optional)
            Set maximum memory available to be used in specific memory demanding tasks.
        legacy_fanci : (bool, optional)
            Flag to run legacy FanCI version instead of PyCI actual code. Defaults to True.
        verbose : (int, optional)
            Selects print level. Defaults to 4.

        """

        try:
            import pyci
        except ImportError:
            print("# ERROR: PyCI package not found.")

        # Output settings
        def info(str, *args):
            if self.verbose >= INFO:
                print(str % args)

        def note(str, *args):
            if self.verbose >= NOTE:
                print(str % args)

        self.verbose = verbose

        # Obtain required data from Fanpy objective object
        note("Importing Fanpy ProjectedSchrodinger object...")

        if not isinstance(fanpy_objective, ProjectedSchrodinger):
            raise TypeError("Invalid object: Objective object from Fanpy not found.")

        self.fanpy_objective = fanpy_objective
        self.fanpy_wfn = fanpy_objective.wfn
        self.fanpy_ham = fanpy_objective.ham

        self.legacy_fanci = legacy_fanci
        self.step_print = fanpy_objective.step_print
        self.step_save = fanpy_objective.step_save
        self.tmpfile = fanpy_objective.tmpfile
        self.param_selection = fanpy_objective.indices_component_params

        self.objective_type = "projected"  # TODO: Should it follow fanpy_objective type?
        self.kwargs = kwargs

        # Build PyCI Hamiltonian Object
        self.pyci_ham = pyci.hamiltonian(energy_nuc, self.fanpy_ham.one_int, self.fanpy_ham.two_int)

        # Obtain required data from Fanpy Wavefunction object
        self.seniority = self.fanpy_wfn.seniority
        self.nocc = self.fanpy_wfn.nelec // 2

        # Define default parameters to buld FanCI object
        self.norm_det = norm_det
        self.norm_param = norm_param
        self.constraints = constraints
        self.max_memory = max_memory

        # Build list of indices for objective parameters
        # TODO: Check if it can be built by Fanpy objective settings
        # TODO: Check if mask can (or even must) be related to norm_param and norm_det
        self.mask = []
        for component, indices in self.fanpy_objective.indices_component_params.items():
            bool_indices = np.zeros(component.nparams, dtype=bool)
            bool_indices[indices] = True
            self.mask.append(bool_indices)

        # Optimize energy
        self.mask.append(True)
        self.mask = np.hstack(self.mask)

        # Compute number of parameters including energy as a parameter
        self.nparam = np.sum(self.mask)

        # Handle default wfn (P space == single pair excitations)
        if self.seniority == 0:
            self.fill = "seniority"
            self.pspace_wfn = pyci.doci_wfn(self.pyci_ham.nbasis, self.nocc, self.nocc)
        else:
            self.fill = "excitation"
            self.pspace_wfn = pyci.fullci_wfn(self.pyci_ham.nbasis, self.fanpy_wfn.nelec - self.nocc, self.nocc)

        # Check if the number of projections is valid for PyCI
        self.nproj = fanpy_objective.nproj

        max_pyci_nproj = int(
            comb(self.fanpy_wfn.nspin // 2, self.fanpy_wfn.nelec - self.fanpy_wfn.nelec // 2)
            * comb(self.fanpy_wfn.nspin // 2, self.fanpy_wfn.nelec // 2)
        )

        if self.nproj > max_pyci_nproj:
            print(
                f"WARNING: Invalid number of projections ({self.nproj} > {max_pyci_nproj}). "
                f"PyCI only supports projections with Sz = 0.\n"
                f"Reassigning 'nproj' to {max_pyci_nproj}..."
            )
            self.nproj = max_pyci_nproj

        self.build_pyci_objective(legacy=legacy_fanci)

    # Define ProjectedSchrodingerPyCI objective interface class
    def build_pyci_objective(self, legacy=True):

        # Select PyCI objective class based on Fanpy or PyCI
        if legacy:
            from fanpy.interface.fanci import ProjectedSchrodingerFanCI

            self.objective = ProjectedSchrodingerFanCI(
                fanpy_objective=self.fanpy_objective,
                ham=self.pyci_ham,
                wfn=self.pspace_wfn,
                nocc=self.nocc,
                seniority=self.seniority,
                nproj=self.nproj,
                fill=self.fill,
                mask=self.mask,
                constraints=self.constraints,
                param_selection=self.param_selection,
                norm_det=self.norm_det,
                objective_type=self.objective_type,
                max_memory=self.max_memory,
                step_print=self.step_print,
                step_save=self.step_save,
                tmpfile=self.tmpfile,
                **self.kwargs,
            )

        else:
            from fanpy.interface.fanci import ProjectedSchrodingerPyCI

            self.objective = ProjectedSchrodingerPyCI(
                fanpy_objective=self.fanpy_objective,
                ham=self.pyci_ham,
                wfn=self.pspace_wfn,
                nocc=self.nocc,
                seniority=self.seniority,
                nproj=self.nproj,
                fill=self.fill,
                mask=self.mask,
                constraints=self.constraints,
                param_selection=self.param_selection,
                norm_param=self.norm_param,
                norm_det=self.norm_det,
                objective_type=self.objective_type,
                max_memory=self.max_memory,
                step_print=self.step_print,
                step_save=self.step_save,
                tmpfile=self.tmpfile,
                **self.kwargs,
            )
