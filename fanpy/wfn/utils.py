"""Utility function for constructing Wavefunction instances."""

import os
from typing import List

from fanpy.eqn.utils import ComponentParameterIndices
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.composite.product import ProductWavefunction
from fanpy.tools import slater

import numpy as np

# from scipy.optimize import OptimizeResult, least_squares, root, minimi
from scipy.optimize import OptimizeResult, root, minimize
from fanpy.solver.least_squares_fanci import least_squares
from fanpy.tools.performance import current_memory
import cma


def wfn_factory(olp, olp_deriv, nelec, nspin, params, memory=None, assign_params=None):
    """Return the instance of the Wavefunction class with the given overlaps.

    Parameters
    ----------
    olp(sd, params) : function
        Function that returns the overlap from the given Slater determinant and the wavefunction
        parameters.
        `sd` is an integer whose bitstring describes the occupation of the Slater determinant. See
        `fanpy.tools.slater` for more details.
        `params` is a numpy array that contains the parameters of the wavefunction. If other types
        of parameters is desired, `assign_params` should be provided.
    olp_deriv(sd, params) : function
        Function that returns the derivatives of the overlap with respect to the wavefunction
        parameters from the given Slater determinant and the wavefunction parameters.
        `sd` is an integer whose bitstring describes the occupation of the Slater determinant. See
        `fanpy.tools.slater` for more details.
        `params` is a numpy array that contains the parameters of the wavefunction. If other types
        of parameters is desired, `assign_params` should be provided.
    nelec : int
        Number of electrons in the wavefunction.
    nspin : int
        Number of spin orbitals in the wavefunction.
    params : np.ndarray
        Parameters of the wavefunction.
        If is not a numpy array, then `assign_params` should be provided.
    memory : {float, int, str, None}
        Memory available (in bytes) for the wavefunction.
        If number is provided, it is the number of bytes.
        If string is provided, it should end iwth either "mb" or "gb" to specify the units.
        Default does not limit memory usage (i.e. infinite).
    assign_params(self, params) : function
        Method for assigning the parameters in the wavefunction class. First argument is `self` and
        second argument is `params`.
        Default uses `BaseWavefunction.assign_params`.

    """

    class GeneratedWavefunction(BaseWavefunction):
        """Generated wavefunction class from the given olp and olp_deriv."""

        def __init__(self, nelec, nspin, memory=None, params=None):
            """Initialize the wavefunction.

            Parameters
            ----------
            nelec : int
                Number of electrons.
            nspin : int
                Number of spin orbitals.
            memory : {float, int, str, None}
                Memory available for the wavefunction.
                If number is provided, it is the number of bytes.
                If string is provided, it should end iwth either "mb" or "gb" to specify the units.
                Default does not limit memory usage (i.e. infinite).
            params : np.ndarray
                Parameters of the wavefunction.

            """
            super().__init__(nelec, nspin, memory=memory)
            self.assign_params(params)

        def assign_params(self, params=None):
            """Assign the parameters of the wavefunction.

            Parameters
            ----------
            params
                Parameters of the wavefunction.

            """
            if assign_params is None:
                super().assign_params(params)
            else:
                assign_params(self, params)

        def _olp(self, sd):  # pylint: disable=E0202
            """Return overlap of the wavefunction with the Slater determinant.

            Parameters
            ----------
            sd : int
                Occupation vector of a Slater determinant given as a bitstring.
                Assumed to have the same number of electrons as the wavefunction.

            Returns
            -------
            olp : {float, complex}
                Overlap of the current instance with the given Slater determinant.

            """
            return olp(sd, self.params)

        def _olp_deriv(self, sd):  # pylint: disable=E0202
            """Return the derivatives of the overlap with the Slater determinant.

            Parameters
            ----------
            sd : int
                Occupation vector of a Slater determinant given as a bitstring.
                Assumed to have the same number of electrons as the wavefunction.

            Returns
            -------
            olp_deriv : np.ndarray
                Derivatives of the overlap with respect to all of the wavefunction's parameters.

            """
            return olp_deriv(sd, self.params)

        def get_overlap(self, sd, deriv=None):
            r"""Return the overlap of the wavefunction with a Slater determinant.

            Parameters
            ----------
            sd : int
                Slater Determinant against which the overlap is taken.
            deriv : {np.ndarray, None}
                Indices of the parameters with respect to which the overlap is derivatized.
                Default returns the overlap without derivatization.

            Returns
            -------
            overlap : {float, np.ndarray}
                Overlap (or derivative of the overlap) of the wavefunction with the given Slater
                determinant.

            Raises
            ------
            TypeError
                If Slater determinant is not an integer.
                If deriv is not a one dimensional numpy array of integers.

            """
            # if no derivatization
            if deriv is None:
                return self._olp(sd)
            # if derivatization
            return self._olp_deriv(sd)[deriv]

    return GeneratedWavefunction(nelec, nspin, params=params, memory=memory)


def convert_to_fanci(
    wfn,
    ham,
    nproj=None,
    proj_wfn=None,
    fill=None,
    seniority=None,
    step_print=None,
    step_save=None,
    tmpfile=None,
    param_selection=None,
    mask=None,
    objective_type=None,
    constraints=None,
    norm_det=None,
    max_memory=None,
    **kwargs,
):
    """Covert the given wavefunction instance to that of FanCI class.

    https://github.com/QuantumElephant/FanCI

    Parameters
    ----------
    wfn : BaseWavefunction
    ham : pyci.hamiltonian
        PyCI Hamiltonian.
    nproj : int, optional
        Number of determinants in projection ("P") space.
    proj_wfn : pyci.doci_wfn, optional
        If specified, this PyCI wave function defines the projection ("P") space.
    kwargs : Any, optional
        Additional keyword arguments for base FanCI class.

    Returns
    -------
    new_wfn : FanCI

    """
    from typing import Any, Tuple, Union
    import pyci
    from fanci.fanci import FanCI
    from fanci.alias import Alias

    class GeneratedFanCI(FanCI):
        """Generated FanCI wavefunction class from the fanpy wavefunction.

        Does not work for composite wavefunctions.

        """

        def __init__(
            self,
            ham: pyci.hamiltonian,
            fanpy_wfn: BaseWavefunction,
            nocc: int,
            nproj: int = None,
            wfn: pyci.doci_wfn = None,
            fill: str = "excitation",
            seniority: int = None,
            step_print: bool = True,
            step_save: bool = True,
            tmpfile: str = "",
            param_selection=None,
            mask=None,
            objective_type: str = "projected",
            constraints=None,
            norm_det=None,
            max_memory=1024,
            **kwargs: Any,
        ) -> None:
            r"""
            Initialize the FanCI problem.

            Parameters
            ----------
            fanpy_wfn : BaseWavefunction
                Wavefunction from fanpy.
            ham : pyci.hamiltonian
                PyCI Hamiltonian.
            nocc : int
                Number of occupied orbitals.
            nproj : int, optional
                Number of determinants in projection ("P") space.
            wfn : pyci.doci_wfn, optional
                If specified, this PyCI wave function defines the projection ("P") space.
            fill : ('excitation' | 'seniority' | None)
                Whether to fill the projection ("P") space by excitation level, by seniority, or not
            at all (in which case ``wfn`` must already be filled).
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
            max_memory = int
                Maximum memory available for this calculations in Megabytes.
                It is utilized in specific loops to avoid potential memory leaks.
                Default is 1024MB (1GB).
            kwargs : Any, optional
                Additional keyword arguments for base FanCI class.

            """
            if not isinstance(ham, pyci.hamiltonian):
                raise TypeError(
                    f"Invalid `ham` type `{type(ham)}`; must be `pyci.hamiltonian`"
                )

            # Save sub-class -specific attributes
            self._fanpy_wfn = fanpy_wfn

            self.step_print = step_print
            self.step_save = step_save
            self.max_memory = max_memory

            if param_selection is None:
                param_selection = [(fanpy_wfn, np.arange(fanpy_wfn.nparams))]
            if isinstance(param_selection, ComponentParameterIndices):
                self.indices_component_params = param_selection
            else:
                self.indices_component_params = ComponentParameterIndices()
                for component, indices in param_selection:
                    self.indices_component_params[component] = indices
            self.tmpfile = tmpfile
            self.objective_type = objective_type
            self.seniority = seniority
            self.print_queue = {}

            mask = []
            for component, indices in self.indices_component_params.items():
                bool_indices = np.zeros(component.nparams, dtype=bool)
                bool_indices[indices] = True
                mask.append(bool_indices)
            # optimize energy
            mask.append(True)
            mask = np.hstack(mask)

            # NOTE: energy is always a parameter
            # Compute number of parameters
            nparam = np.sum(mask)

            # Handle default nproj
            nproj = nparam if nproj is None else nproj

            # Handle default wfn (P space == single pair excitations)
            # FIXME: unable to generate enough Slater determinants due to spin constraint
            if wfn is None:
                if seniority == 0:
                    wfn = pyci.doci_wfn(ham.nbasis, nocc // 2, nocc // 2)
                else:
                    wfn = pyci.fullci_wfn(ham.nbasis, nocc - nocc // 2, nocc // 2)

            # constraints
            if constraints is None and norm_det is None:
                constraints = {"<\\Phi|\\Psi> - 1>": self.make_norm_constraint()}

            # Initialize base class
            FanCI.__init__(
                self,
                ham,
                wfn,
                nproj,
                nparam,
                fill=fill,
                mask=mask,
                constraints=constraints,
                norm_det=norm_det,
                **kwargs,
            )

        def compute_overlap(
            self, x: np.ndarray, occs_array: Union[np.ndarray, str]
        ) -> np.ndarray:
            r"""
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
                    sd = slater.create(sd, *(occs[1] + self._fanpy_wfn.nspatial))
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
            self, x: np.ndarray, occs_array: Union[np.ndarray, str], chunk_idx = None
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

            if chunk_idx is not None:
                s_chunk, f_chunk = chunk_idx
            else:
                s_chunk = 0
                f_chunk = -1

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
                    sd = slater.create(sd, *(occs[1] + self._fanpy_wfn.nspatial))
                    sds.append(sd)
            else:
                for i, occs in enumerate(occs_array):
                    if occs.dtype == bool:
                        occs = np.where(occs)
                    sd = slater.create(0, *occs)
                    sds.append(sd)

            # Select sds according to selected chunks
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

        def compute_objective(self, x: np.ndarray) -> np.ndarray:
            r"""
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
                self.print_queue["Cost"] = np.sum(output[: self._nproj] ** 2)
                self.print_queue["Cost from constraints"] = np.sum(
                    output[self._nproj :] ** 2
                )
                if self.step_print:
                    print(
                        "(Mid Optimization) Electronic Energy: {}".format(
                            self.print_queue["Electronic Energy"]
                        )
                    )
                    print(
                        "(Mid Optimization) Cost: {}".format(self.print_queue["Cost"])
                    )
                    if self.constraints:
                        print(
                            "(Mid Optimization) Cost from constraints: {}".format(
                                self.print_queue["Cost from constraints"]
                            )
                        )
            else:
                # NOTE: ignores energy and constraints
                # Allocate objective vector
                output = np.zeros(self._nproj, dtype=pyci.c_double)

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
                self._ci_op(ovlp, out=output)
                output = np.sum(output * ovlp[: self._nproj])
                output /= np.sum(ovlp[: self._nproj] ** 2)
                self.print_queue["Electronic Energy"] = output
                if self.step_print:
                    print(
                        "(Mid Optimization) Electronic Energy: {}".format(
                            self.print_queue["Electronic Energy"]
                        )
                    )

            if self.step_save:
                self.save_params()

            return output

        def compute_jacobian(self, x: np.ndarray) -> np.ndarray:
            r"""
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
                    print(
                        "(Mid Optimization) Norm of the Jacobian: {}".format(
                            self.print_queue["Norm of the Jacobian"]
                        )
                    )
            else:
                # NOTE: ignores energy and constraints
                # Allocate Jacobian matrix (in transpose memory order)
                output = np.zeros(
                    (self._nproj, self._nactive), order="F", dtype=pyci.c_double
                )
                integrals = np.zeros(self._nproj, dtype=pyci.c_double)

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
                norm = np.sum(overlaps[: self._nproj] ** 2)
                self._ci_op(overlaps, out=integrals)
                energy_integral = np.sum(overlaps[: self._nproj] * integrals)

                d_ovlp = self.compute_overlap_deriv(x[:-1], "S")

                # Iterate over remaining columns of Jacobian and d_ovlp
                for output_col, d_ovlp_col in zip(
                    output.transpose(), d_ovlp.transpose()
                ):
                    #
                    # Compute each column of the Jacobian:
                    #
                    #   d(<n|H|\Psi>)/d(p_k) = <m|H|n> d(c_m)/d(p_k)
                    #
                    #   E d(<n|\Psi>)/d(p_k) = E \delta_{nk} d(c_n)/d(p_k)
                    #
                    # Note: we update d_ovlp in-place here
                    self._ci_op(d_ovlp_col, out=output_col)
                    output_col *= overlaps[: self._nproj]
                    output_col += d_ovlp_col[: self._nproj] * integrals
                    output_col *= norm
                    output_col -= (
                        2
                        * energy_integral
                        * overlaps[: self._nproj]
                        * d_ovlp_col[: self._nproj]
                    )
                    output_col /= norm**2
                output = np.sum(output, axis=0)
                self.print_queue["Norm of the gradient of the energy"] = np.linalg.norm(
                    output
                )
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
                names = [
                    type(component).__name__
                    for component in self.indices_component_params
                ]
                names_totalcount = {name: names.count(name) for name in set(names)}
                names_count = {name: 0 for name in set(names)}

                for component in self.indices_component_params:
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
            for component, indices in self.indices_component_params.items():
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
            return np.hstack(
                [
                    comp.params.ravel()[inds]
                    for comp, inds in self.indices_component_params.items()
                ]
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

        def optimize(
            self,
            x0: np.ndarray,
            mode: str = "lstsq",
            use_jac: bool = False,
            **kwargs: Any,
        ) -> OptimizeResult:
            r"""
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
            # if self.nequation < self.nactive:
            #     raise ValueError("system is underdetermined")

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
                # self.step_print = False
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
                self.step_print = False
                opt_kwargs.setdefault("callback", self.print)
            elif mode == "cma":
                optimizer = cma.fmin
                opt_kwargs.setdefault("sigma0", 0.01)
                opt_kwargs.setdefault("options", {})
                opt_kwargs["options"].setdefault("ftarget", None)
                opt_kwargs["options"].setdefault("timeout", np.inf)
                opt_kwargs["options"].setdefault("tolfun", 1e-11)
                opt_kwargs["options"].setdefault("verb_log", 0)
                self.step_print = False
                if self.objective_type != "energy":
                    raise ValueError("objective_type must be energy")
            elif mode == "bfgs":
                if self.objective_type != "energy":
                    raise ValueError("objective_type must be energy")
                optimizer = minimize
                opt_kwargs["method"] = "bfgs"
                opt_kwargs.setdefault("options", {"gtol": 1e-8})
                # opt_kwargs["options"]['schrodinger'] = objective
                self.step_print = False
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
            r"""
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
            ham = self._ham
            nproj = self._nproj
            nparam = self._nparam
            nbasis = self._wfn.nbasis
            nocc_up = self._wfn.nocc_up
            nocc_dn = self._wfn.nocc_dn
            constraints = self._constraints
            mask = self._mask
            ci_cls = self._wfn.__class__
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
                del self._ci_op
                self._ci_op = None
                # Make new FanCI wave function in-place
                self.__init__(
                    ham,
                    self._fanpy_wfn,
                    nocc_up + nocc_dn,
                    nproj=nproj,
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
                    constraints=constraints,
                    mask=mask,
                    fill=fill,
                    seniority=self.seniority,
                    step_print=self.step_print,
                    step_save=self.step_save,
                    tmpfile=self.tmpfile,
                    param_selection=self.indices_component_params,
                    objective_type=self.objective_type,
                )
                # Go to next iteration
                isamp += 1

        def calculate_overlap_deriv_chunks(self):

            tensor_mem = self._nactive * 8/1e6
            avail_mem = (self.max_memory - current_memory()) * 0.9

            chunk_size = int(avail_mem / tensor_mem)

            if chunk_size <= 0:
                chunk_size = 1

            chunks_list = []
            for s_chunk in range(0, self._nactive, chunk_size):
                f_chunk = min(self._nactive, s_chunk + chunk_size)
                chunks_list.append([s_chunk, f_chunk])

            return chunks_list

    return GeneratedFanCI(
        ham,
        wfn,
        wfn.nelec,
        nproj=nproj,
        wfn=proj_wfn,
        fill=fill,
        seniority=seniority,
        step_print=step_print,
        step_save=step_save,
        tmpfile=tmpfile,
        param_selection=param_selection,
        mask=mask,
        objective_type=objective_type,
        constraints=constraints,
        norm_det=norm_det,
        max_memory=max_memory,
        **kwargs,
    )
