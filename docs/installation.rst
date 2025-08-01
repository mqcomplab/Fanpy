.. installation:

============
Installation
============

Dependencies
============

The FANCI module has been developed using the following dependencies:

* Python >= 3.5: http://www.python.org/
* SciPy >= 0.18.0: http://www.scipy.org/
* NumPy >= 1.11.0: http://www.numpy.org/
* cma >= 2.7.0: https://github.com/CMA-ES/pycma
* pytest >= 4.0.0: https://docs.pytest.org/en/latest/

The latest versions of these modules can be accessed with conda or pip.

Conda
-----

  .. code-block:: bash

     conda install numpy scipy cma pytest

Pip
---

  .. code-block:: bash

     pip install numpy scipy cma pytest


Optional Dependencies
=====================

Some modules may require additional dependencies, such as `scikit-optimize` (for decision tree and
Bayesian optimization algorithms) or `tensorflow` (for neural network wavefunction):

* skopt >= 0.5.2: https://scikit-optimize.github.io/stable/
* tensorflow == 2.2.0: https://www.tensorflow.org/

In addition, the one and two-electron integrals are not generated within FanCI and must be obtained
from some external sources. The `horton` module can be used to generate integrals from Gaussian fchk
file and from its HF code. The `pyscf` and `psi4` modules can be used to generate integrals from
their HF routines. Scripts for generating the integrals are provided in the `scripts` module.

* HORTON == 2.1.0: http://theochem.github.io/horton/2.0.1/index.html
* PySCF >= 1.6.5: https://github.com/sunqm/pyscf
* Psi4 >= 1.1: https://github.com/psi4/psi4

Please note that `horton` is only compatible with Python 2.7, so it must be installed on a different
version of Python. It is recommended that you use some form of virtual environment to isolate the
two versions of Python.

  .. code-block:: bash

     conda create --name integrals python=2.7
     conda install -n integrals -c theochem horton
     source activate integrals

The Python interpreters for HORTON must be specified in the environment variables `HORTONPYTHON`. To
find the location of the Python interpreter, either use `which python` in the terminal or `import
sys; print(sys.executable)`

  .. code-block:: bash

     export HORTONPYTHON=/abspath/to/hortonpython


For Developers
==============

Though `Fanpy` in continuously integrated and tested when contributed through GitHub, developers may be
interested in applying the same set of tests locally. To do so, only the module `tox` is needed.

* tox >= 3.12.1: https://tox.readthedocs.io/en/latest/

Then, from the project home directory, the following command executes the unit tests:

  .. code-block:: bash

     tox -e py37

To check the code quality, linters, such as `flake8` and `pylint`, can be used via the following command:

  .. code-block:: bash

     tox -e linters


Documentation Building
======================
* sphinx
