.. _script_make_script:

Making Calculation Scripts
==========================
The script :py:mod:`fanpy_make_script` runs a calculation using a set of arguments and optional
arguments:

.. code:: bash

   fanpy_make_script [-h] --nelec NELEC --nspin NSPIN
                     --one_int_file ONE_INT_FILE --two_int_file TWO_INT_FILE
                     --wfn_type WFN_TYPE
                     [--nuc_repulsion NUC_NUC] [--optimize_orbs]
                     [--pspace PSPACE_EXC [PSPACE_EXC ...]]
                     [--objective OBJECTIVE] [--solver SOLVER]
                     [--solver_kwargs SOLVER_KWARGS]
                     [--wfn_kwargs WFN_KWARGS] [--load_orbs LOAD_ORBS]
                     [--load_ham LOAD_HAM] [--load_wfn LOAD_WFN]
                     [--load_chk LOAD_CHK] [--save_orbs SAVE_ORBS]
                     [--save_ham SAVE_HAM] [--save_wfn SAVE_WFN]
                     [--save_chk SAVE_CHK] [--memory MEMORY]
                     [--filename FILENAME]


where the arguments inside square brackets are optional.

Mandatory Keywords
------------------

:code:`--nelec NELEC`
   Number of electrons in the wavefunction.
   :code:`NELEC` is an integer.

:code:`--one_int_file ONE_INT_FILE`
   One-electron integrals used to construct the chemical Hamiltonian.
   The integrals are expressed with respect to spatial orbitals.
   :code:`ONE_INT_FILE` is a numpy file that contains a two-dimensional square array of floats.
   The dimension of each axis is the number of spatial orbitals.

:code:`--two_int_file TWO_INT_FILE`
   Two-electron integrals used to construct the chemical Hamiltonian.
   The integrals are expressed with respect to spatial orbitals.
   :code:`TWO_INT_FILE` is a numpy file that contains a four-dimensional array of floats with equal
   dimension in all four axis.
   The dimension of each axis is the number of spatial orbitals.

:code:`--wfn_type WFN_TYPE`
   Type of the wavefunction.
   :code:`WFNTYPE` must be one of :code:`fci`, :code:`doci`, :code:`mps`, `determinant-ratio`,
   :code:`ap1rog`, :code:`apr2g`, :code:`apig`, :code:`apsetg`, or :code:`apg`.

Optional Keywords
-----------------
:code:`--nuc_repulsion NUC_NUC`
   Nuclear-nuclear repulsion energy (in Hartree).
   If not provided, then it is set to :code:`0.0`.

:code:`--optimize_orbs`
   Flag for optimizing the orbitals.
   Currently, orbitals can be optimized only when the solver does not use the gradient (i.e.
   :code:`cma`).
   If not provided, then orbitals are not optimized.

:code:`--pspace PSPACE_EXC [PSPACE_EXC ...]`
   Orders of excitations that wil be used to construct the projection space. For the objectives
   :code:`system` and :code:`least_squares`, the projection space corresponds to the equations that
   will be created. For the objective :code:`variational`, the projection space correspond to the
   projection operators that are inserted to the left and right of the Hamiltonian. See
   :ref:`Objective <objective>` for more details.

   :code:`PSPACE_EXC` is a sequence of integers (orders of excitations) separated by space.
   Zeroth order excitation (i.e. HF ground state) is always included.
   If not provided, then first and second order excitations (and HF ground state) are used as the
   projection space (equivalent to :code:`--pspace 1 2`).

:code:`--objective OBJECTIVE`
   Objective function that will be used to optimize the parameters.
   :code:`OBJECTIVE` must be one of :code:`system`, :code:`least_squares`, or :code:`variational`.
   If not provided, then :code:`least_squares` is selected.

   See :ref:`Objective <objective>` for more details.

:code:`--solver SOLVER`
   Solver used to optimize the objective.
   :code:`SOLVER` must be one of :code:`cma`, :code:`diag`, :code:`minimize`,
   :code:`least_squares`, or :code:`root`.
   If not provided, then :code:`cma` is selected.

   Solvers :code:`cma` and :code:`minimize` can only be used with the objectives
   :code:`least_squares` and :code:`variational`.
   Solvers :code:`least_squares` and :code:`root` can only be used with objective :code:`system`.
   Solver :code:`diag` can only be used with CI wavefunctions (wavefunction types of :code:`fci` and
   :code:`doci`).

   See XXX for more details.

:code:`--solver_kwargs SOLVER_KWARGS`
   Keyword arguments that will be passed to the solver method.
   :code:`SOLVER_KWARGS` must be provided as a string that will be passed into the solver method,
   overwriting *all* keyword arguments.
   For example, :code:`--solver_kwargs 'sigma0=0.001'` will change the sigma value in the
   :code:`cma` solver.

   If not provided, then no keyword argument will be provided to the solver and its default setting
   will be used.
   See XXX for more details.

:code:`--wfn_kwargs WFN_KWARGS`
   Keyword arguments that will be passed to the wavefunction instance.
   :code:`WFN_KWARGS` must be provided as a string that will be passed into the wavefunction
   instance, overwriting *all* keyword arguments.
   For example, :code:`--wfn_kwargs 'dimension=10'` will change the dimension of the matrices in the
   MPS wavefunction.

   If not provided, then no keyword argument will be provided to the wavefunction and its default
   setting will be used.
   See XXX for more details.

:code:`--load_orbs LOAD_ORBS`
   Transformation matrix that will be used to rotate the integrals in the Hamiltonian. This keyword
   can be used to port over the orbitals from another calculation (for example, from a different
   wavefunction calculation).

   :code:`LOAD_ORBS` must be provided as a numpy file of a two-dimension array with correct
   dimensions. The number of rows must correspond with the dimension of an axis in the integrals.

   If the keyword :code:`--load_ham` is also provided, then the integrals (orbitals) are rotated
   after instantiating the Hamiltonian. However, it is not recommended to use both keywords
   :code:`--load_orbs` and :code:`--load_ham`.

   If not provided, then integrals will not be rotated.

:code:`--load_ham LOAD_HAM`
   Parameters of the Hamiltonian that will be used to instantiate the Hamiltonian. This keyword can
   be used to port over the Hamiltonian parameters from another calculation (for example, from a
   different wavefunction calculation).

   :code:`LOAD_HAM` must be provided as a numpy file of a one-dimension array with the correct
   dimension. The number of parameters must correspond with the number of elements in the upper
   triangular matrix of the anti-Hermitian matrix in the transformation operator.

   If the keyword :code:`--load_orbs` is also provided, then the integrals (orbitals) are rotated
   after instantiating the Hamiltonian. However, it is not recommended to use both keywords
   :code:`--load_orbs` and :code:`--load_ham`.

   If not provided, then default Hamiltonian parameters (zeros) will be used.

:code:`--load_wfn LOAD_WFN`
   Parameters of the wavefunction that will be used to instantiate the wavefunction. This keyword
   can be used to port over the wavefunction parameters from another calculation (for example, from
   a different Hamiltonian/system).

   :code:`LOAD_WFN` must be provided as a numpy file of a one-dimension array with the correct
   dimension. The number of parameters varies depending on the wavfunction type.
   If not provided, then the default parameters of the wavefunction will be used (almost always HF
   ground state).

   See XXX for more details.

:code:`--load_chk LOAD_CHK`
   Checkpoint in the optimization process. This keyword can be used to restart a calculation.
   :code:`LOAD_CHK` must be provided as a numpy file of one-dimension array with the correct
   dimension. The number of parameters can vary depending on the number of active (not frozen)
   parameters in the optimization.

   See XXX for more details.

:code:`--save_orbs SAVE_ORBS`
   Transformation matrix that was used to rotate the integrals in the Hamiltonian. This keyword can
   be used to save the orbitals for use in another calculation. (for example, in a different
   wavefunction calculation).

   :code:`SAVE_ORBS` is the name of the numpy file used to save the transformation matrix.
   Since transformation matrix is produced from the Hamiltonian parameters, it is not recommended to
   use both keywords :code:`--save_orbs` and :code:`--save_ham`.

   If not provided, then the transformation matrix will not be stored.

:code:`--save_ham SAVE_HAM`
   Parameters of the Hamiltonian that was used in Hamiltonian instance. This keyword can be used to
   save the Hamiltonian parameters for use in another calculation (for example, in a different
   wavefunction calculation).

   :code:`SAVE_HAM` is the name of the numpy file used to save the Hamiltonian parameters.
   Since transformation matrix is produced from the Hamiltonian parameters, it is not recommended to
   use both keywords :code:`--save_orbs` and :code:`--save_ham`.

   If not provided, then Hamiltonian parameters are not saved.

:code:`--save_wfn SAVE_WFN`
   Parameters of the wavefunction that was used in wavefunction instance. This keyword can be used to
   save the wavefunction parameters for use in another calculation (for example, in a different
   Hamiltonian/system calculation).

   :code:`SAVE_WFN` is the name of the numpy file used to save the wavefunction parameters.

   If not provided, then wavefunction parameters are not saved.

:code:`--save_chk SAVE_CHK`
   Checkpoint file that saves the values of all active (not frozen) parameters in the optimization
   process. This keyword can be used to save the progress of the optimization so that it can be
   restarted should the optimization fails prematurely.

   :code:`SAVE_CHK` is the name of the numpy file used to save the checkpoint.

   If not provided, then wavefunction parameters are not saved.

:code:`--memory MEMORY`
   Memory available for the wavefunction.
   :code:`MEMORY` must be a string that ends with :code:`mb` (for MB) or :code:`gb` (for GB).
   If not provided, then no restrictions will be put on cache for the overlaps of wavefunction,
   which may result in memory overflow.

:code:`--filename FILENAME`
   Name of the script that will be produced.
   :code:`FILENAME` must be a string.
   If not provided, then the script is printed out in the :code:`STDOUT`
