.. installation:

============
Installation
============

Quick Steps
===========

Step 1: Create conda environment 
--------------------------------

Fanpy requires **Python 3.9 or newer.** It is recommended to install fanpy in a conda environment:

.. code-block:: bash
    
    conda create -n fanpy python 
    conda activate fanpy

Step 2: Clone the Fanpy code base
---------------------------------

The code for Fanpy is available in the GitHub repository of `mqcomplab` and can be cloned by running the following commands:

.. code-block:: bash

  git clone https://github.com/mqcomplab/Fanpy.git
  cd Fanpy
  
Step 3: Installing Fanpy 
------------------------
Once you have cloned the repository, you can `pip` install the module.  

.. code-block:: bash

  pip install -e .

This will install the required dependencies listed below:

  * SciPy >= 1.9.0: http://www.scipy.org/
  * NumPy >= 1.25.0: http://www.numpy.org/
  * Cython: https://cython.org/
  * Pandas: https://pandas.pydata.org/
  * cma: https://github.com/CMA-ES/pycma
  * psutil: https://psutil.readthedocs.io/stable/

Fanpy has multiple optional dependencies, some install a single package while others install multiple packages:

* `pyscf`: Fanpy needs one and two-electron integrals from Hartree Fock (HF) calculations. 
* `tensorflow`: there are some wavefunctions that require tensorflow
* `horton`: This has been used to convert HF results from Gaussian to numpy arrays. Note: running HF calculations with Gaussian is depricated. 
* `test`: these dependencies are necessary to run the unit tests for Fanpy. List of packages installed: `pytest`, `pytest-cov`, `numdifftools`, `scikit-learn`, `scikit-optimize`
* `dev`: installs linters and other development tools. List of packages: `tox`, `pytest`, `pytest-cov`, `flake8`, `flake8-pydocstyle`, `flake8-import-order`, `pep8-naming`, `pylint`, `bandit`, `black`

Run the following command to install any of the optional dependencies:

.. code-block:: bash

  pip install -e ".[pyscf, tensorflow]"

The one and two-electron integrals are not generated within Fanpy and must be obtainedfrom some 
external sources. The recommended approach is with PySCF, though there is also the option to provide these
integrals in the form of `numpy` files. Please refer to the tutorials for more information.  

For Developers
==============

`Fanpy` in continuously integrated and tested when contributed through GitHub, developers may be
interested in applying the same set of tests locally. For this, the optional `test` dependencies 
in addition to PySCF are required. There are currently no linter tests required when contributing to 
`Fanpy`.


Documentation Building
======================

`Fanpy` has a documentation which can be built locally with `sphinx`. The following packages are required to build the documentation:

* `sphinx`
* `sphinx-rtd-theme`
