from fanpy.eqn.projected import ProjectedSchrodinger

import numpy as np
from typing import Any

# Log Levels
INFO = 4
NOTE = 3


class PYCI:

    def __init__(
        self,
        fanpy_objective,
        energy_nuc,
        legacy=True,
        verbose=4,
        **kwargs: Any,
    ):

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
        info("Importing Fanpy ProjectedSchrodinger object...")

        if not isinstance(fanpy_objective, ProjectedSchrodinger):
            raise TypeError("Invalid object: Objective object from Fanpy not found.")

        self.fanpy_objective = fanpy_objective
        self.fanpy_wfn = fanpy_objective.wfn
        self.fanpy_ham = fanpy_objective.ham

        self.nproj = fanpy_objective.nproj
        self.step_print = fanpy_objective.step_print
        self.step_save = fanpy_objective.step_save
        self.tmpfile = fanpy_objective.tmpfile
        self.param_selection = fanpy_objective.indices_component_params

        self.objective_type = "projected"  # TODO: Should it follow fanpy_objective type?
        self.kwargs = kwargs  # TODO: Is it needed?

        # Build PyCI Hamiltonian Object
        self.pyci_ham = pyci.hamiltonian(energy_nuc, self.fanpy_ham.one_int, self.fanpy_ham.two_int)

        # Obtain required data from Fanpy Wavefunction object
        self.seniority = self.fanpy_wfn.seniority
        self.nocc = self.fanpy_wfn.nelec // 2

        # Define default parameters to buld FanCI object
        self.mask = None
        self.norm_det = None
        self.norm_param = None
        self.constraints = None
        self.max_memory = 8192

        # Build list of indices for objective parameters
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

        self.build_pyci_objective(legacy=legacy)

    # Define ProjectedSchrodingerPyCI objective interface class
    def build_pyci_objective(self, legacy=True):

        # Select PyCI objective class based on Fanpy or PyCI
        if legacy:
            from fanpy.interface.fanci.legacy import ProjectedSchrodingerFanCI

            self.objective = ProjectedSchrodingerFanCI(
                fanpy_wfn=self.fanpy_wfn,
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
            from pyci.fanci.fanci import FanCI as ProjectedSchrodingerFanCI
