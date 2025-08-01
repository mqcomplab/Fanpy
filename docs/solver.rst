.. _solver:

Solver
======
Once the objective has been selected, its parameters, e.g. wavefunction parameters or the orbital
optimization coefficients, must be optimized such that the corresponding Schrödinger equation is
satisfied. Different solvers can be implemented for an objective and one solver can be used to solve
multiple different objectives. Since the techniques involved in a solver is not necessarily
exclusive to an objective, the solver and objective are separated by design. Though sophisticated
algorithms can be implemented in the FANCI module, only the solvers for :mod:`minimizing an
equation <fanpy.solver.equation>` and for :mod:`solving a system of equations <fanpy.solver.system>`
are available.

The FANCI module is not limited to its solvers, because most solvers have similar API. They usually
take in an input of the objective function and the initial guess with more method specific keyword
arguments. The output typically includes the state of the optimization, optimized parameters, and
the value of the objective at the optimized parameters. Since there are variations of the input and
output between different modules, a wrapper can be used to standardize the API of the different
solvers. In the FANCI module, the standard input for a solver includes an argument of
:class:`BaseObjective <fanpy.eqn.base.BaseObjective>` instance and a keyword argument of
`save_file` for storing the final results. Additional keyword arguments are passed directly to the
solver. The (minimal) standardized output is a dictionary with keys `success` (for success of the
optimization), `params` (for parameters at the end of the optimization), `message` (for message
returned by the solver) and the `internal` (for the returned value of the solver). The
:mod:`wrappers <fanpy.solver.wrappers>` module contains the wrappers used to standardize the solvers
from `scipy` and `skopt`.
