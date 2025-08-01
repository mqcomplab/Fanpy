.. _tutorial_calc_scripts:

How to run a calculation using scripts
======================================
There are currently two scripts that help users run calculations. In this context, a script is
python code that configures (and executes) a calculation using preset options. The first script
configures and runs the calculation. The second script configures and produces a script that
corresponds to the configured calculation. It is recommended to use the second script because it
provides a written record of the configuration and it can be configured for functionality beyond
those provided by the two scripts.

Using a script that runs a calculation
--------------------------------------
The script :ref:`fanpy_run_calc <script_run_calc>` runs a calculation using a set of
arguments and optional arguments:

.. code:: bash

   fanpy_run_calc [-h] --nelec NELEC --one_int_file ONE_INT_FILE
                      --two_int_file TWO_INT_FILE --wfn_type WFN_TYPE
                      [--nuc_repulsion NUC_NUC] [--optimize_orbs]
                      [--pspace PSPACE_EXC [PSPACE_EXC ...]]
                      [--objective OBJECTIVE] [--solver SOLVER]
                      [--ham_noise HAM_NOISE] [--wfn_noise WFN_NOISE]
                      [--solver_kwargs SOLVER_KWARGS]
                      [--wfn_kwargs WFN_KWARGS] [--load_orbs LOAD_ORBS]
                      [--load_ham LOAD_HAM] [--load_ham_um LOAD_HAM_UM]
                      [--load_wfn LOAD_WFN] [--save_chk SAVE_CHK]
                      [--memory MEMORY]


where the arguments inside square brackets are optional.

If the optional keywords are not provided, a sufficient default will be used. However, the keywords
:code:`--optimize_orbs`, :code:`--objective OBJECTIVE`, and :code:`--solver SOLVER` may require some
special attention.

:code:`--optimize-orbs`
   When this keyword is provided, the orbital optimization is enabled. However, gradients of the
   objective with respect to the orbital optimization parameters have not yet been implemented so
   the solver cannot use the gradient. In other words, only the solver :code:`cma` is compatible
   with this keyword.

:code:`--objective OBJECTIVE`
   There are three different objectives available in this script. The keyword :code:`system` will
   build a system of equations with the given projection space (can be modified by keyword
   :code:`--pspace`), :code:`least_squares` will flatten down this system of equations as a squared
   sum of each equation, and :code:`variational` will make an equation for the energy using the
   projection operator that correspond to the projection space. For more information on the
   different objectives, go to :ref:`Objective <objective>`

:code:`--solver SOLVER`
   There are five different solvers available in this script. However, some of the solvers are
   incompatible with other keywords. Only the solver :code:`cma` supports orbital optimization. The
   solvers :code:`cma` and :code:`minimize` can only be used with the objectives
   :code:`least_squares` and :code:`variational`.  The solvers :code:`least_squares` and
   :code:`root` can only be used with objective :code:`system`. The solver :code:`diag` can
   only be used with CI wavefunctions (wavefunction types of :code:`fci` and :code:`doci`).


Following are some examples:

- FCI calculation on a 4 electron system defined by the integrals :code:`oneint.npy` and
  :code:`twoint.npy`

.. code:: bash

   fanpy_run_calc --nelec 4 --one_int_file oneint.npy \
                  --two_int_file twoint.npy --wfn_type fci

- Same FCI calculation except the CI matrix is diagonalized

.. code:: bash

   fanpy_run_calc --nelec 4 --one_int_file oneint.npy \
                  --two_int_file twoint.npy --wfn_type fci \
                  --solver diag

- AP1roG calculation without orbital optimization solved as a system of equations using a least
  squares solver

.. code:: bash

   fanpy_run_calc --nelec 4 --one_int_file oneint.npy \
                  --two_int_file twoint.npy --wfn_type ap1rog \
                  --objective system --solver least_squares

- AP1roG calculation with orbital optimization solved as a squared sum of equations using a CMA
  solver

.. code:: bash

   fanpy_run_calc --nelec 4 --one_int_file oneint.npy \
                  --two_int_file twoint.npy --wfn_type ap1rog \
                  --optimize_orbs --objective system --solver cma


For a detailed explanation of each keyword, go to :ref:`fanpy_run_calc
<script_run_calc>`.

.. _tutorial_calc_make_script:

Using a script that makes a script
----------------------------------
The script :ref:`fanpy_make_script <script_make_script>` creates a script that can
be executed to run the calculation:

.. code:: bash

   fanpy_make_script.py [-h] --nelec NELEC --nspin NSPIN
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


where the arguments inside square brackets are optional. This script shares all of the keywords with
:ref:`fanpy_run_calc <script_run_calc>` with addition of two more keywords:
:code:`--nspin NSPIN` and :code:`--filename FILENAME`. The :code:`--nspin NSPIN` specifies the
number of spin orbitals used in the system. This number must match the number of orbitals associated
with the integrals (number of spatial orbitals times two). The :code:`--filename FILENAME` is the
name of the script that will be produced. If it not provided then the script is printed onto the
screen (:code:`STDOUT`).

Though the functionality of this script is very close to the :ref:`fanpy_run_calc
<script_run_calc>`, this script can be used to produce a template. The template can
be modified for functionality that is not included in the scripts :ref:`fanpy_run_calc
<script_run_calc>` and :ref:`fanpy_make_script
<script_make_script>`. For details on modifying the template script, see
:ref:`How to make a script <tutorial_calc_code>`.

Following are some examples:

- FCI calculation on a 4 electron system defined by the integrals :code:`oneint.npy` and
  :code:`twoint.npy`

.. code:: bash

   fanpy_make_script.py --nelec 4 --nspin 8 --one_int_file oneint.npy \
                       --two_int_file twoint.npy --wfn_type fci \
                       --filename example.py
   python3 example.py

- Same FCI calculation except the CI matrix is diagonalized

.. code:: bash

   fanpy_make_script.py --nelec 4 --nspin 8 --one_int_file oneint.npy \
                       --two_int_file twoint.npy --wfn_type fci \
                       --solver diag --filename example.py
   python3 example.py

- AP1roG calculation without orbital optimization solved as a system of equations using a least
  squares solver

.. code:: bash

   fanpy_make_script.py --nelec 4 --nspin 8 --one_int_file oneint.npy \
                       --two_int_file twoint.npy --wfn_type ap1rog \
                       --objective system --solver least_squares \
                       --filename example.py
   python3 example.py


- AP1roG calculation with orbital optimization solved as a squared sum of equations using a CMA
  solver

.. code:: bash

   fanpy_make_script.py --nelec 4 --nspin 8 --one_int_file oneint.npy \
                       --two_int_file twoint.npy --wfn_type ap1rog \
                       --optimize_orbs --objective least_squares \
                       --solver cma --filename example.py
   python3 example.py


For a detailed explanation of each keyword, go to :ref:`fanpy_make_script
<script_make_script>`.
