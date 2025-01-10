from fanpy.tools import slater
from fanpy.ham.base import BaseHamiltonian
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.composite.product import ProductWavefunction
from fanpy.eqn.utils import ComponentParameterIndices

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import numpy as np
import pyci

# Log Levels
INFO = 4
NOTE = 3


class PYCI:

    def __init__(self,
                fanpy_wfn,
                fanpy_ham,
                energy_nuc,
                #  keep_fanpy_objects=False, # Idea to be developed
                verbose=4):

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

        if isinstance(fanpy_wfn, BaseWavefunction):
            self.type = "pyci"
        else:
            raise TypeError("Invalid object: Wavefunction object from Fanpy not found.")

        if isinstance(fanpy_ham, BaseHamiltonian):
            self.type = "pyci"
        else:
            raise TypeError("Invalid object: Hamiltonian object from Fanpy not found.")

        self.verbose = verbose
        # self.keep_fanpy_objects = keep_fanpy_objects

        info("Importing Fanpy objects...")

        # Build PyCI Hamiltonian Object
        self.pyci_ham = pyci.hamiltonian(energy_nuc, fanpy_ham.one_int, fanpy_ham.two_int)
        # if not self.keep_fanpy_objects:
        #     del(fanpy_ham)

        # Obtain required data from Fanpy Wavefunction object
        self.fanpy_wfn = fanpy_wfn
        self.seniority = fanpy_wfn.seniority

        # Define default parameters to buld FanCI object
        self.mask = None
        self.norm_det = None
        self.norm_param = None
        self.constraints = None

        # Build list of indices for objective parameters
        self.param_selection = [(fanpy_wfn, np.arange(fanpy_wfn.nparams))]
        if isinstance(self.param_selection, ComponentParameterIndices):
            self.indices_component_params = self.param_selection
        else:
            self.indices_component_params = ComponentParameterIndices()
            for component, indices in self.param_selection:
                self.indices_component_params[component] = indices

        self.mask = []
        for component, indices in self.indices_component_params.items():
            bool_indices = np.zeros(component.nparams, dtype=bool)
            bool_indices[indices] = True
            self.mask.append(bool_indices)

        # Optimize energy
        self.mask.append(True)
        self.mask = np.hstack(self.mask)

        # Compute number of parameters including energy as a parameter
        self.nparam = np.sum(self.mask)

        # Handle default nproj
        self.nproj = self.nparam

        # Handle default wfn (P space == single pair excitations)
        if self.seniority == 0:
            self.fill = "seniority"
            self.pspace_wfn = pyci.doci_wfn(
                self.pyci_ham.nbasis,
                self.fanpy_wfn.nelec // 2,
                self.fanpy_wfn.nelec // 2
                )
        else:
            self.fill = "excitation"
            self.pspace_wfn = pyci.fullci_wfn(
                self.pyci_ham.nbasis,
                self.fanpy_wfn.nelec - self.fanpy_wfn.nelec // 2,
                self.fanpy_wfn.nelec // 2
                )

    def build_pyci_wavefunction(self, legacy=False, **kwargs):
        """
        Build FanCI Wavefunction object based on PyCI.
        """

        if legacy:
            from fanpy.interface.fanci.legacy import LegacyFanCI
        else:
            from pyci.fanci.fanci import FanCI as LegacyFanCI

        class FanCI(LegacyFanCI):

            def __init__(
                self,
                pyci_ham: pyci.hamiltonian,
                fanpy_wfn: BaseWavefunction,
                pspace_wfn,
                nproj: int = None,
                nparam: int = None,
                fill: str = "excitation",
                mask = None,
                constraints = None,
                norm_det = None,
                norm_param = None
                ):

                self._fanpy_wfn = fanpy_wfn

                # Build list of indices for objective parameters
                self.param_selection = [(fanpy_wfn, np.arange(fanpy_wfn.nparams))]
                if isinstance(self.param_selection, ComponentParameterIndices):
                    self.indices_component_params = self.param_selection
                else:
                    self.indices_component_params = ComponentParameterIndices()
                    for component, indices in self.param_selection:
                        self.indices_component_params[component] = indices

                # Initialize constraints
                if constraints is None and norm_det is None:
                    constraints = {"<\\Phi|\\Psi> - 1>": self.make_norm_constraint()}

                LegacyFanCI.__init__(
                    self,
                    pyci_ham,
                    pspace_wfn,
                    nproj,
                    nparam,
                    norm_param,
                    norm_det,
                    constraints,
                    mask,
                    fill
                )

            def make_norm_constraint(self):
                def f(x: np.ndarray) -> float:
                    r""" "
                    Constraint function <\psi_{i}|\Psi> - v_{i}.

                    """
                    norm = np.sum(self.compute_overlap(x[:-1], "S") ** 2)
                    if self.step_print:
                        print(f"(Mid Optimization) Norm of wavefunction: {norm}")
                    return norm - 1

                def dfdx(x: np.ndarray) -> np.ndarray:
                    r""" "
                    Constraint gradient d(<\psi_{i}|\Psi>)/d(p_{k}).

                    """
                    y = np.zeros(self._nactive, dtype=pyci.c_double)
                    ovlp = self.compute_overlap(x[:-1], "S")

                    chunks = self.calculate_overlap_deriv_chunks()
                    for s_chunk, f_chunk in chunks:

                        # Compute overlap derivative for the current chunk
                        d_ovlp_chunk = self.compute_overlap_deriv(x[:-1], "S", [s_chunk, f_chunk])

                        # Compute the partial contribution to y
                        y[: self._nactive - self._mask[-1]] += np.einsum(
                            'i,ij->j', 2 * ovlp[s_chunk:f_chunk], d_ovlp_chunk, optimize='greedy'
                        )

                    return y

                return f, dfdx

            def compute_overlap(
                self, x: np.ndarray, occs_array: Union[np.ndarray, str]
            ) -> np.ndarray:
                """
                Compute the FanCI overlap vector.

                Parameters
                ----------
                x : np.ndarray
                    Parameter array, [p_0, p_1, ..., p_n].
                occs_array : (np.ndarray | 'P' | 'S')
                    Array of determinant occupations for which to compute overlap. A string "P" or "S" can
                    be passed instead that indicates whether ``occs_array`` corresponds to the "P" space
                    or "S" space, so that a more efficient, specialized computation can be done for these.

                Returns
                -------
                ovlp : np.ndarray
                    Overlap array.

                """
                if isinstance(occs_array, np.ndarray):
                    pass
                elif occs_array == "P":
                    occs_array = self._pspace
                elif occs_array == "S":
                    occs_array = self._sspace
                else:
                    raise ValueError("invalid `occs_array` argument")

                # FIXME: converting occs_array to slater determinants to be converted back to indices is
                # a waste
                # convert slater determinants
                sds = []
                if isinstance(occs_array[0, 0], np.ndarray):
                    for i, occs in enumerate(occs_array):
                        # FIXME: CHECK IF occs IS BOOLEAN OR INTEGERS
                        # convert occupation vector to sd
                        if occs.dtype == bool:
                            occs = np.where(occs)[0]
                        sd = slater.create(0, *occs[0])
                        sd = slater.create(sd, *(occs[1] + self.ham.nbasis))
                        sds.append(sd)
                else:
                    for i, occs in enumerate(occs_array):
                        if occs.dtype == bool:
                            occs = np.where(occs)
                        sd = slater.create(0, *occs)
                        sds.append(sd)

                # Feed in parameters into fanpy wavefunction
                for component, indices in self.indices_component_params.items():
                    new_params = component.params.ravel()
                    new_params[indices] = x[self.indices_objective_params[component]]
                    component.assign_params(new_params)

                # initialize
                y = np.zeros(occs_array.shape[0], dtype=pyci.c_double)

                # Compute overlaps of occupation vectors
                if hasattr(self._fanpy_wfn, "get_overlaps"):
                    y += self._fanpy_wfn.get_overlaps(sds)
                else:
                    for i, sd in enumerate(sds):
                        y[i] = self._fanpy_wfn.get_overlap(sd)
                return y

            def compute_overlap_deriv(
                self, x: np.ndarray, occs_array: Union[np.ndarray, str], chunk_idx = [0, -1]
            ) -> np.ndarray:
                r"""
                Compute the FanCI overlap derivative matrix.

                Parameters
                ----------
                x : np.ndarray
                    Parameter array, [p_0, p_1, ..., p_n].
                occs_array : (np.ndarray | 'P' | 'S')
                    Array of determinant occupations for which to compute overlap. A string "P" or "S" can
                    be passed instead that indicates whether ``occs_array`` corresponds to the "P" space
                    or "S" space, so that a more efficient, specialized computation can be done for these.
                chunk_idx : np.array
                    List of start and end positions of the chunks to be computed.

                Returns
                -------
                ovlp : np.ndarray
                    Overlap derivative array.

                """
                if isinstance(occs_array, np.ndarray):
                    pass
                elif occs_array == "P":
                    occs_array = self._pspace
                elif occs_array == "S":
                    occs_array = self._sspace
                else:
                    raise ValueError("invalid `occs_array` argument")

                # FIXME: converting occs_array to slater determinants to be converted back to indices is
                # a waste
                # convert slater determinants
                sds = []
                if isinstance(occs_array[0, 0], np.ndarray):
                    for i, occs in enumerate(occs_array):
                        # FIXME: CHECK IF occs IS BOOLEAN OR INTEGERS
                        # convert occupation vector to sd
                        if occs.dtype == bool:
                            occs = np.where(occs)[0]
                        sd = slater.create(0, *occs[0])
                        sd = slater.create(sd, *(occs[1] + self.ham.nbasis))
                        sds.append(sd)
                else:
                    for i, occs in enumerate(occs_array):
                        if occs.dtype == bool:
                            occs = np.where(occs)
                        sd = slater.create(0, *occs)
                        sds.append(sd)

                # Select sds according to selected chunks
                s_chunk, f_chunk = chunk_idx
                sds = sds[s_chunk:f_chunk]

                # Feed in parameters into fanpy wavefunction
                for component, indices in self.indices_component_params.items():
                    new_params = component.params.ravel()
                    new_params[indices] = x[self.indices_objective_params[component]]
                    component.assign_params(new_params)

                # Shape of y is (no. determinants, no. active parameters excluding energy)
                y = np.zeros(
                    (occs_array.shape[0], self._nactive - self._mask[-1]),
                    dtype=pyci.c_double,
                )

                # Select parameters according to selected chunks
                y = y[s_chunk:f_chunk]

                # Compute derivatives of overlaps
                deriv_indices = self.indices_component_params[self._fanpy_wfn]
                deriv_indices = np.arange(self.nparam - 1)[self._mask[:-1]]

                if isinstance(self._fanpy_wfn, ProductWavefunction):
                    wfns = self._fanpy_wfn.wfns
                    for wfn in wfns:
                        if wfn not in self.indices_component_params:
                            continue
                        inds_component = self.indices_component_params[wfn]
                        if inds_component.size > 0:
                            inds_objective = self.indices_objective_params[wfn]
                            y[:, inds_objective] = self._fanpy_wfn.get_overlaps(
                                sds, (wfn, inds_component)
                            )
                elif hasattr(self._fanpy_wfn, "get_overlaps"):
                    y += self._fanpy_wfn.get_overlaps(sds, deriv=deriv_indices)
                else:
                    for i, sd in enumerate(sds):
                        y[i] = self._fanpy_wfn.get_overlap(sd, deriv=deriv_indices)

                return y

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

        return FanCI(
            self.pyci_ham,
            self.fanpy_wfn,
            self.pspace_wfn,
            nproj = self.nproj,
            nparam = self.nparam,
            fill = self.fill,
            mask = self.mask,
            constraints = self.constraints,
            norm_det = self.norm_det,
            norm_param = self.norm_param
        )
