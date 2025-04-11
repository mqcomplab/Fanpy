"""
Legacy version of FanCI objective class.
Based of the original class from FanCI code.
"""

from fanpy.tools import slater
from fanpy.solver.least_squares_fanci import least_squares
from fanpy.wfn.base import BaseWavefunction
from fanpy.eqn.base import BaseSchrodinger
from fanpy.wfn.composite.product import ProductWavefunction
from fanpy.tools.performance import current_memory

from typing import Any, List, Tuple, Union, Sequence

import os
import math

import pyci
from pyci.fanci import FanCI

import cma
import numpy as np
from scipy.optimize import OptimizeResult, least_squares, root, minimize

__all__ = [
    "ProjectedSchrodingerFanCI",
]


# Parent Interface Class to add PyCI features to Fanpy objective class
class ProjectedSchrodingerPyCI(FanCI):
    """
    Generated PyCI objective class from the Fanpy objective.
    """

    @property
    def fanpy_objective(self) -> BaseSchrodinger:
        """
        BaseSchrodinger object from Fanpy code.

        """
        return self._fanpy_objective

    @property
    def fanpy_wfn(self) -> BaseWavefunction:
        """
        BaseWavefunction object from Fanpy code.

        """
        return self._fanpy_wfn

    @property
    def fill(self) -> str:
        """
        Fill type for p-space in PyCI code.

        """
        return self._fill

    @property
    def seniority(self) -> int:
        """
        Seniority number from Fanpy BaseWavefunction.

        """
        return self._seniority

    @property
    def step_print(self) -> bool:
        """
        Option to print relevant information when the objective is evaluated.

        """
        return self._step_print

    @property
    def step_save(self) -> bool:
        """
        Option to save parameters with every evaluation of the objective.

        """
        return self._step_save

    @property
    def tmpfile(self) -> str:
        """
        Name of the file that will store the parameters used by the objective method.

        """
        return self._tmpfile

    @property
    def param_selection(self) -> str:
        """
        Parameters selection from Fanpy objective.

        """
        return self._param_selection

    @property
    def objective_type(self) -> str:
        """
        Set objective type for the FanCI problem as "projected" or "energy".

        """
        return self._objective_type

    @property
    def max_memory(self) -> int:
        """
        Maximum memory available for this calculations in Megabytes.

        """
        return self._max_memory

    @property
    def nactive(self) -> int:
        """
        Number of active parameters for this FanCI problem.

        """
        return self._nactive

    @property
    def mask(self) -> np.ndarray:
        """
        Frozen parameter mask.

        """
        return self._mask_view

    def freeze_parameter(self, *params: Sequence[int]) -> None:
        """
        Set a FanCI parameter to be frozen during optimization.

        Parameters
        ----------
        params : Sequence[int]
            Indices of parameters to freeze.

        """
        for param in params:
            self._mask[param] = False

        # Update nactive
        self._nactive = self._mask.sum()

    def unfreeze_parameter(self, *params: Sequence[int]) -> None:
        """
        Set a FanCI parameter to be active during optimization.

        Parameters
        ----------
        params : Sequence[int]
            Indices of parameters to unfreeze.

        """
        for param in params:
            self._mask[param] = True

        # Update nactive
        self._nactive = self._mask.sum()

    def __init__(
        self,
        fanpy_objective: BaseSchrodinger,
        ham: pyci.hamiltonian,
        wfn,
        nocc: int,
        seniority: int,
        nproj: int,
        fill: str,
        mask: list,
        constraints: dict,
        param_selection: dict,
        norm_param,
        norm_det,
        objective_type: str,
        max_memory: int,
        step_print: bool,
        step_save: bool,
        tmpfile: str,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the FanCI problem.

        Arguments
        ---------
        fanpy_objective : BaseSchrodinger
            FanCI problem defined by Fanpy.
        ham : pyci.hamiltonian
            PyCI Hamiltonian.
        wfn : pyci.doci_wfn or pyci.fullci_wfn
            If specified, this PyCI wave function defines the projection ("P") space.
        nocc : int
            Number of occupied orbitals.
        seniority: int
            Seniority number from Fanpy BaseWavefunction.
        nproj : int
            Number of determinants in projection ("P") space.
        fill : ('excitation' | 'seniority' | None)
            Whether to fill the projection ("P") space by excitation level, by seniority, or not at all (in which case ``wfn`` must already be filled).
        mask : list
            List of parameters to freeze.
        constraints : dict
            Pairs of functions (f, dfdx) corresponding to additional constraints.
        param_selection : dict
            Parameters selection from Fanpy objective.
        norm_param : Sequence[Tuple[int, float]], optional
            Indices of parameters whose values to constrain, and the value to which to constrain them.
        norm_det : Sequence[Tuple[int, float]]
            Indices of determinant whose overlaps to constrain, and the value to which to constrain them.
        objective_type: str
            Set objective type for the FanCI problem as "projected" or "energy".
        max_memory = int
            Maximum memory available for this calculations in Megabytes.
            It is utilized in specific loops to avoid potential memory leaks.
        step_print : bool
            Option to print relevant information when the objective is evaluated.
        step_save : bool
            Option to save parameters with every evaluation of the objective.
        tmpfile : str
            Name of the file that will store the parameters used by the objective method.
            By default, the parameter values are not stored.
            If a file name is provided, then parameters are stored upon execution of the objective method.
        kwargs : Any
            Additional keyword arguments for base FanCI class.

        """
        if not isinstance(ham, pyci.hamiltonian):
            raise TypeError(f"Invalid `ham` type `{type(ham)}`; must be `pyci.hamiltonian`")

        # Save sub-class -specific attributes
        self._fanpy_objective = fanpy_objective
        self._fanpy_wfn = fanpy_objective.wfn
        self._nocc = nocc
        self._fill = fill
        self._mask = mask
        self._mask_view = mask[...]
        self._nactive = mask.sum()
        self._seniority = seniority
        self._step_print = step_print
        self._step_save = step_save
        self._tmpfile = tmpfile
        self._param_selection = param_selection
        self._objective_type = objective_type
        self._max_memory = max_memory

        self.print_queue = {}

        # Set read-only flag on public array attributes
        self._mask_view.setflags(write=False)

        # Define constraints
        if constraints is None and norm_det is None:
            constraints = {"<\\Phi|\\Psi> - 1>": self.make_norm_constraint()}

        # Initialize base class
        FanCI.__init__(
            self,
            ham,
            wfn,
            nproj,
            nparam=self.nactive,
            norm_param=norm_param,
            norm_det=norm_det,
            constraints=constraints,
            fill=fill,
            **kwargs,
        )

    def compute_overlap(self, x: np.ndarray, occs_array: Union[np.ndarray, str]) -> np.ndarray:
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
            occs_array = self.pspace
        elif occs_array == "S":
            occs_array = self.sspace
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
                sd = slater.create(sd, *(occs[1] + self.fanpy_wfn.nspatial))
                sds.append(sd)
        else:
            for i, occs in enumerate(occs_array):
                if occs.dtype == bool:
                    occs = np.where(occs)
                sd = slater.create(0, *occs)
                sds.append(sd)

        # Feed in parameters into fanpy wavefunction
        for component, indices in self.param_selection.items():
            new_params = component.params.ravel()
            new_params[indices] = x[self.param_selection[component]]
            component.assign_params(new_params)

        # initialize
        y = np.zeros(occs_array.shape[0], dtype=pyci.c_double)

        # Compute overlaps of occupation vectors
        if hasattr(self.fanpy_wfn, "get_overlaps"):
            y += self.fanpy_wfn.get_overlaps(sds)
        else:
            for i, sd in enumerate(sds):
                y[i] = self.fanpy_wfn.get_overlap(sd)
        return y

    def compute_overlap_deriv(
        self, x: np.ndarray, occs_array: Union[np.ndarray, str], chunk_idx=[None, None]
    ) -> np.ndarray:
        """
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
            occs_array = self.pspace
        elif occs_array == "S":
            occs_array = self.sspace
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
                sd = slater.create(sd, *(occs[1] + self.fanpy_wfn.nspatial))
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
        for component, indices in self.param_selection.items():
            new_params = component.params.ravel()
            new_params[indices] = x[self.indices_objective_params[component]]
            component.assign_params(new_params)

        # Shape of y is (no. determinants, no. active parameters excluding energy)
        y = np.zeros(
            (occs_array.shape[0], self.nactive - self.mask[-1]),
            dtype=pyci.c_double,
        )

        # Select parameters according to selected chunks
        y = y[s_chunk:f_chunk]

        # Compute derivatives of overlaps
        deriv_indices = self.param_selection[self.fanpy_wfn]
        deriv_indices = np.arange(self.nparam - 1)[self.mask[:-1]]

        if isinstance(self.fanpy_wfn, ProductWavefunction):
            wfns = self.fanpy_wfn.wfns
            for wfn in wfns:
                if wfn not in self.param_selection:
                    continue
                inds_component = self.param_selection[wfn]
                if inds_component.size > 0:
                    inds_objective = self.indices_objective_params[wfn]
                    y[:, inds_objective] = self.fanpy_wfn.get_overlaps(sds, (wfn, inds_component))
        elif hasattr(self.fanpy_wfn, "get_overlaps"):
            y += self.fanpy_wfn.get_overlaps(sds, deriv=deriv_indices)
        else:
            for i, sd in enumerate(sds):
                y[i] = self.fanpy_wfn.get_overlap(sd, deriv=deriv_indices)

        return y

    def compute_objective(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the FanCI objective function.

            f : x[k] -> y[n]

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n, E].

        Returns
        -------
        obj : np.ndarray
            Objective vector.

        """
        if self.objective_type == "projected":
            output = super().compute_objective(x)
            self.print_queue["Electronic Energy"] = x[-1]
            self.print_queue["Cost"] = np.sum(output[: self.nproj] ** 2)
            self.print_queue["Cost from constraints"] = np.sum(output[self.nproj :] ** 2)
            if self.step_print:
                print("(Mid Optimization) Electronic Energy: {}".format(self.print_queue["Electronic Energy"]))
                print("(Mid Optimization) Cost: {}".format(self.print_queue["Cost"]))
                if self.constraints:
                    print(
                        "(Mid Optimization) Cost from constraints: {}".format(self.print_queue["Cost from constraints"])
                    )
        else:
            # NOTE: ignores energy and constraints
            # Allocate objective vector
            output = np.zeros(self.nproj, dtype=pyci.c_double)

            # Compute overlaps of determinants in sspace:
            #
            #   c_m
            #
            ovlp = self.compute_overlap(x[:-1], "S")

            # Compute objective function:
            #
            #   f_n = (\sum_n <\Psi|n> <n|H|\Psi>) / \sum_n <\Psi|n> <n|\Psi>
            #
            # Note: we update ovlp in-place here
            self.ci_op(ovlp, out=output)
            output = np.sum(output * ovlp[: self.nproj])
            output /= np.sum(ovlp[: self.nproj] ** 2)
            self.print_queue["Electronic Energy"] = output
            if self.step_print:
                print("(Mid Optimization) Electronic Energy: {}".format(self.print_queue["Electronic Energy"]))

        if self.step_save:
            self.save_params()

        return output

    def compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian of the FanCI objective function.

            j : x[k] -> y[n, k]

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n, E].

        Returns
        -------
        jac : np.ndarray
            Jacobian matrix.

        """
        if self.objective_type == "projected":
            output = super().compute_jacobian(x)
            self.print_queue["Norm of the Jacobian"] = np.linalg.norm(output)
            if self.step_print:
                print("(Mid Optimization) Norm of the Jacobian: {}".format(self.print_queue["Norm of the Jacobian"]))
        else:
            # NOTE: ignores energy and constraints
            # Allocate Jacobian matrix (in transpose memory order)
            output = np.zeros((self.nproj, self.nactive), order="F", dtype=pyci.c_double)
            integrals = np.zeros(self.nproj, dtype=pyci.c_double)

            # Compute Jacobian:
            #
            #   J_{nk} = d(<n|H|\Psi>)/d(p_k) - E d(<n|\Psi>)/d(p_k) - dE/d(p_k) <n|\Psi>
            #   J_{nk} = (\sum_n d<\Psi|n> <n|H|\Psi> + <\Psi|n> d<n|H|\Psi>) / \sum_n <\Psi|n>^2 -
            #            (\sum_n <\Psi|n> <n|H|\Psi>) / (\sum_n <\Psi|n> <n|\Psi>)^2 * (2 \sum_n <\Psi|n>)
            #   J_{nk} = ((\sum_n d<\Psi|n> <n|H|\Psi> + <\Psi|n> d<n|H|\Psi>) (\sum_n <\Psi|n>^2)
            #             - (\sum_n <\Psi|n> <n|H|\Psi>) * (2 \sum_n <\Psi|n> d<\Psi|n>))
            #            / (\sum_n <\Psi|n>^2)^2
            #   J_{nk} = ((\sum_n d<\Psi|n> <n|H|\Psi> + <\Psi|n> d<n|H|\Psi>) N
            #             - H * (2 \sum_n <\Psi|n> d<\Psi|n>))
            #            / N^2
            #   J_{nk} = (\sum_n N (d<\Psi|n> <n|H|\Psi> + <\Psi|n> d<n|H|\Psi>) - 2 H <\Psi|n> d<\Psi|n>)
            #            / N^2
            #
            # Compute overlap derivatives in sspace:
            #
            #   d(c_m)/d(p_k)
            #
            overlaps = self.compute_overlap(x[:-1], "S")
            norm = np.sum(overlaps[: self.nproj] ** 2)
            self.ci_op(overlaps, out=integrals)
            energy_integral = np.sum(overlaps[: self.nproj] * integrals)

            d_ovlp = self.compute_overlap_deriv(x[:-1], "S")

            # Iterate over remaining columns of Jacobian and d_ovlp
            for output_col, d_ovlp_col in zip(output.transpose(), d_ovlp.transpose()):
                #
                # Compute each column of the Jacobian:
                #
                #   d(<n|H|\Psi>)/d(p_k) = <m|H|n> d(c_m)/d(p_k)
                #
                #   E d(<n|\Psi>)/d(p_k) = E \delta_{nk} d(c_n)/d(p_k)
                #
                # Note: we update d_ovlp in-place here
                self.ci_op(d_ovlp_col, out=output_col)
                output_col *= overlaps[: self.nproj]
                output_col += d_ovlp_col[: self.nproj] * integrals
                output_col *= norm
                output_col -= 2 * energy_integral * overlaps[: self.nproj] * d_ovlp_col[: self.nproj]
                output_col /= norm**2
            output = np.sum(output, axis=0)
            self.print_queue["Norm of the gradient of the energy"] = np.linalg.norm(output)
            if self.step_print:
                print(
                    "(Mid Optimization) Norm of the gradient of the energy: {}".format(
                        self.print_queue["Norm of the gradient of the energy"]
                    )
                )

        if self.step_save:
            self.save_params()
        return output

    def save_params(self):
        """Save the parameters associated with the Schrodinger equation.

        All of the parameters are saved, even if it was frozen in the objective.

        The parameters of each component of the Schrodinger equation is saved separately using the
        name in the `tmpfile` as the root (removing the extension). The class name of each component
        and a counter are used to differentiate the files associated with each component.

        """
        if self.tmpfile != "":
            root, ext = os.path.splitext(self.tmpfile)
            names = [type(component).__name__ for component in self.param_selection]
            names_totalcount = {name: names.count(name) for name in set(names)}
            names_count = {name: 0 for name in set(names)}

            for component in self.param_selection:
                name = type(component).__name__
                if names_totalcount[name] > 1:
                    names_count[name] += 1
                    name = "{}{}".format(name, names_count[name])

                # pylint: disable=E1101
                component.save_params("{}_{}{}".format(root, name, ext))

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
        for component, indices in self.param_selection.items():
            output[component] = np.arange(count, count + indices.size)
            count += indices.size
        return output

    @property
    def active_params(self):
        """Return the parameters selected for optimization EXCLUDING ENERGY.

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
        return np.hstack([comp.params.ravel()[inds] for comp, inds in self.param_selection.items()])

    def make_norm_constraint(self):
        def f(x: np.ndarray) -> float:
            """ "
            Constraint function <\psi_{i}|\Psi> - v_{i}.

            """
            norm = np.sum(self.compute_overlap(x[:-1], "S") ** 2)
            if self.step_print:
                print(f"(Mid Optimization) Norm of wavefunction: {norm}")
            return norm - 1

        def dfdx(x: np.ndarray) -> np.ndarray:
            """ "
            Constraint gradient d(<\psi_{i}|\Psi>)/d(p_{k}).

            """
            y = np.zeros(self.nactive, dtype=pyci.c_double)
            ovlp = self.compute_overlap(x[:-1], "S")

            chunks = self.calculate_overlap_deriv_chunks()
            for s_chunk, f_chunk in chunks:

                # Compute overlap derivative for the current chunk
                d_ovlp_chunk = self.compute_overlap_deriv(x[:-1], "S", [s_chunk, f_chunk])

                # Compute the partial contribution to y
                y[: self.nactive - self.mask[-1]] += np.einsum(
                    "i,ij->j", 2 * ovlp[s_chunk:f_chunk], d_ovlp_chunk, optimize="greedy"
                )

            return y

        return f, dfdx

    def optimize(
        self,
        x0: np.ndarray,
        mode: str = "lstsq",
        use_jac: bool = False,
        **kwargs: Any,
    ) -> OptimizeResult:
        """
        Optimize the wave function parameters.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for wave function parameters.
        mode : ('lstsq' | 'root' | 'cma'), default='lstsq'
            Solver mode.
        use_jac : bool, default=False
            Whether to use the Jacobian function or a finite-difference approximation.
        kwargs : Any, optional
            Additional keyword arguments to pass to optimizer.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Result of optimization.

        """
        # Check if system is underdetermined
        if self.nequation < self.nactive:
            print(
                "WARNING: System is underdetermined with dimensions {:}, {:}. Continuing anyways".format(
                    self.nequation, self.nactive
                )
            )

        # Convert x0 to proper dtype array
        x0 = np.asarray(x0, dtype=pyci.c_double)
        # Check input x0 length
        if x0.size != self.nparam:
            raise ValueError("length of `x0` does not match `param`")

        # Prepare objective, Jacobian, x0
        if self.nactive < self.nparam:
            # Generate objective, Jacobian, x0 with frozen parameters
            x_ref = np.copy(x0)
            f = self.mask_function(self.compute_objective, x_ref)
            j = self.mask_function(self.compute_jacobian, x_ref)
            x0 = np.copy(x0[self.mask])
        else:
            # Use bare functions
            f = self.compute_objective
            j = self.compute_jacobian

        # Set up initial arguments to optimizer
        opt_args = f, x0
        opt_kwargs = kwargs.copy()
        if use_jac:
            opt_kwargs["jac"] = j

        # Parse mode parameter; choose optimizer and fix arguments
        if mode == "lstsq":
            optimizer = least_squares
            opt_kwargs.setdefault("xtol", 1.0e-8)
            opt_kwargs.setdefault("ftol", 1.0e-8)
            opt_kwargs.setdefault("gtol", 1.0e-8)
            opt_kwargs.setdefault("max_nfev", 1000 * self.nactive)
            opt_kwargs.setdefault("verbose", 2)
            # opt_kwargs.setdefault("callback", self.print)
            if self.objective_type != "projected":
                raise ValueError("objective_type must be projected")
        elif mode == "root":
            if self.nequation != self.nactive:
                raise ValueError("'root' does not work with over-determined system")
            optimizer = root
            opt_kwargs.setdefault("method", "hybr")
            opt_kwargs.setdefault("options", {})
            opt_kwargs["options"].setdefault("xtol", 1.0e-9)
            opt_kwargs.setdefault("callback", self.print)
        elif mode == "cma":
            optimizer = cma.fmin
            opt_kwargs.setdefault("sigma0", 0.01)
            opt_kwargs.setdefault("options", {})
            opt_kwargs["options"].setdefault("ftarget", None)
            opt_kwargs["options"].setdefault("timeout", np.inf)
            opt_kwargs["options"].setdefault("tolfun", 1e-11)
            opt_kwargs["options"].setdefault("verb_log", 0)
            if self.objective_type != "energy":
                raise ValueError("objective_type must be energy")
        elif mode == "bfgs":
            if self.objective_type != "energy":
                raise ValueError("objective_type must be energy")
            optimizer = minimize
            opt_kwargs["method"] = "bfgs"
            opt_kwargs.setdefault("options", {"gtol": 1e-8})
            # opt_kwargs["options"]['schrodinger'] = objective
            opt_kwargs.setdefault("callback", self.print)
        elif mode == "trustregion":
            raise NotImplementedError
        elif mode == "trf":
            if self.objective_type != "projected":
                raise ValueError("objective_type must be energy")
            raise NotImplementedError
        else:
            raise ValueError("invalid mode parameter")

        # Run optimizer
        results = optimizer(*opt_args, **opt_kwargs)
        return results

    def print(self, *args, **kwargs):
        for data_type, data in self.print_queue.items():
            print(f"(Mid Optimization) {data_type}: {data}")

    def optimize_stochastic(
        self,
        nsamp: int,
        x0: np.ndarray,
        mode: str = "lstsq",
        use_jac: bool = False,
        fill: str = "excitation",
        **kwargs: Any,
    ) -> List[Tuple[np.ndarray]]:
        """
        Run a stochastic optimization of a FanCI wave function.

        Parameters
        ----------
        nsamp: int
            Number of samples to compute.
        x0 : np.ndarray
            Initial guess for wave function parameters.
        mode : ('lstsq' | 'root' | 'cma'), default='lstsq'
            Solver mode.
        use_jac : bool, default=False
            Whether to use the Jacobian function or a finite-difference approximation.
        fill : ('excitation' | 'seniority' | None)
            Whether to fill the projection ("P") space by excitation level, by seniority, or not
            at all (in which case ``wfn`` must already be filled).
        kwargs : Any, optional
            Additional keyword arguments to pass to optimizer.

        Returns
        -------
        result : List[Tuple[np.ndarray]]
            List of (occs, coeffs, params) vectors for each solution.

        """
        # Get wave function information
        ham = self.ham
        nproj = self.nproj
        nparam = self.nparam
        nbasis = self.wfn.nbasis
        nocc_up = self.wfn.nocc_up
        nocc_dn = self.wfn.nocc_dn
        constraints = self.constraints
        mask = self.mask
        ci_cls = self.wfn.__class__
        # Start at sample 1
        isamp = 1
        result = []
        # Iterate until nsamp samples are reached
        # **kwargs: Any,
        while True:
            # Optimize this FanCI wave function and get the result
            opt = self.optimize(x0, mode=mode, use_jac=use_jac, **kwargs)
            energy = opt.x[-1]
            if opt.success:
                print("Optimization was successful")
            else:
                print("Optimization was not successful: {}".format(opt.message))
            print("Final Electronic Energy for sample {isamp}: {}".format(energy))
            x0 = opt.x
            coeffs = self.compute_overlap(x0[:-1], "S")
            prob = coeffs**2
            prob /= np.sum(prob)
            nonzero_prob = prob[prob > 0]
            if nproj > nonzero_prob.size:
                print(
                    f"Number of nonzero coefficients, {nonzero_prob.size}, is less than the projection space, {nproj}. Truncating projectionspace"
                )
                nproj = nonzero_prob.size
            # Add the result to our list
            result.append((np.copy(self.sspace), coeffs, x0))
            # Check if we're done manually each time; this avoids an extra
            # CI matrix preparation with an equivalent "for" loop
            if isamp >= nsamp:
                return result
            # Try to get the garbage collector to remove the old CI matrix
            del self.ci_op
            self.ci_op = None
            # Make new FanCI wave function in-place
            self.__init__(
                self.fanpy_wfn,
                ham,
                # Generate new determinants from "S" space via alias method
                wfn=ci_cls(
                    nbasis,
                    nocc_up,
                    nocc_dn,
                    self.sspace[
                        sorted(
                            np.random.choice(
                                np.arange(prob.size),
                                size=nproj,
                                p=prob,
                                replace=False,
                            )
                        )
                    ],
                ),
                # wfn=ci_cls(nbasis, nocc_up, nocc_dn, self.sspace[Alias(coeffs ** 2)(nproj)]),
                nocc=nocc_up + nocc_dn,
                seniority=self.seniority,
                nproj=nproj,
                fill=fill,
                mask=mask,
                constraints=constraints,
                param_selection=self.param_selection,
                objective_type=self.objective_type,
                max_memory=self.max_memory,
                step_print=self.step_print,
                step_save=self.step_save,
                tmpfile=self.tmpfile,
            )
            # Go to next iteration
            isamp += 1

    def calculate_overlap_deriv_chunks(self):

        tensor_mem = self.nactive * 8 / 1e6
        avail_mem = (self.max_memory - current_memory()) * 0.9

        chunk_size = max(1, math.floor(avail_mem / tensor_mem))
        chunk_size = min(chunk_size, self.nactive)

        chunks_list = []
        for s_chunk in range(0, self.nactive, chunk_size):
            f_chunk = min(self.nactive, s_chunk + chunk_size)
            chunks_list.append([s_chunk, f_chunk])

        return chunks_list
