## Fanpy: A Python library for prototyping multideterminant methods in ab initio Electronic Structure Calculations


**Web: [Miranda Quintana Group](https://quintana.chem.ufl.edu/)**   


Fanpy is a free, open-source, and cross-platform Python 3 library designed for ab initio electronic structure calculations. The Fanpy implementation is based on the mathematical framework called [Flexible Ansatz for N-electron Configuration Interaction (FANCI)](https://doi.org/10.1016/j.comptc.2021.113187). The adoption of the FANCI framework gives a highly modular structure to Fanpy resulting in 5 modules - Hamiltonian, Wavefunction, Objective, Solver, and Tools. The modular structure offers two greatest virtues, The first is its 'sandbox-like' ability to handle any combinations of wavefunction ansatz and different methods, and the second is the ease of transition from the formal conception of a method to its working implementation.

## Installing Fanpy
We recommend setting up a virtual environment for Fanpy. A virtual environment is essentially a blank slate, where we can install all the required codes for a given project. We can create different environments for different projects. This keeps the coding cleaner, since the dependencies are always assigned to one single main package. There are multiple virtual environment managers. Here we are going to use Conda (or Miniconda), but feel free to use a different virtual environment manager if you want to. 

### Step 1: set up conda virtual environment
First, create a new environment for Fanpy:

```
$ conda create -n fanpy python
```

This will create a new environment `fanpy` and install Python in that environment.

Next, activate the conda environment:
```
$ conda activate fanpy
```
### Step 2: Clone repository
Then, clone the repository: 
```
$ git clone https://github.com/mqcomplab/Fanpy.git
```
and go to the Fanpy folder:
```
$ cd Fanpy/
```
### Step 3: Install & Optional dependencies
Fanpy has optional dependencies, which can be chosen during installation:
* dev: installs some code development tools, such as linters and pytest
* test: concise dependencies that we need to run the unit tests in Fanpy
* horton: an outdated dependency suite that helped us convert data from Gaussian to Fanpy
* tensorflow: this is required for a small number of wavefunctions
* pyscf: installs the PySCF package to run HF calculations before the Fanpy calculation.

Note: running the tests requires the test and pyscf dependencies. Additionally, PyCI needs to be installed. See the documentation of PyCI for installation instructions: https://pyci.qcdevs.org/install.html

In the activated fanpy conda environment run:

```
$ pip install ".[optional_dependency1, optional_dependency2]"
```
to install Fanpy.

### (Optional) Step 4: Test install
Make sure to install the dependencies for the tests from the previous step. Then go to the tests folder and run the pytest command:

```
$ pytest test_*
```
This will run all the `test_` files in the folder. 

---
## Modules in Fanpy


### 1. Wavefunctions
The following wavefunctions are already implemented in Fanpy.     

**Configuration Interaction**  
 - Configuration Interaction with singles and doubles (CISD)   
 - Doubly-occupied Configuration Interaction (DOCI)   
 - Full CI   
 - Selected CI wavefunctions with a user-specified set of Slater determinants   

**Coupled-Cluster**   
 - Standard Coupled Cluster (CCSD, CCSDT, ...)   
 - CC with seniority-specific excitations
 - Seniority-restricted CC    

**Geminal wavefunctions**   
 - Antisymmetrized Product of Geminals (APG)     
 - Antisymmetrized Product of Interacting Geminals (APIG)   
 - Antisymmetrized Product of rank-two Interacting Geminals (APr2G)
 - Matrix Product States (MPS)    

**Coupled Cluster-Inspired Geminal Wavefunctions**   
 - The following 1-reference orbital geminal wavefunctions are implemented incorporating single-like excitations. 
 - Antisymmetrized Product of 1-reference Orbital Interacting Geminals (AP1roG)    
 - Antisymmetrized Product of Set-separated 1-reference Orbital Geminals (APset1roG)  
 - Antisymmetrized Product of Geminals with 1-reference Orbital (APG1ro)    


### 2. Hamiltonians
The following Hamiltonians are implemented:    
Electronic Hamiltonian - restricted, unrestricted, and generalized basis     


### 3. Objective
The Objective module combines the wavefunction and Hamiltonian to represent the following forms of Schrodinger Equations.
- Variational (solving for the expectation value of the energy)    
- Projected (solving for the system of equations generated)    

### 4. Solver
The Solver module supports the following optimizers to optimize/solve the equations from the Objective module.   
- For CI, it supports brute-force eigenvalue decomposition.  
- Optimizers from [SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.html#) 
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm from [pycma](https://pypi.org/project/cma)     
- Decision trees and Bayesian optimization algorithms from [scikit-optimize](https://scikit-optimize.github.io/stable/)   


### 5. Tool
The Tool module provides different utility functions used throughout the Fanpy package—for example, tools for generating and manipulating Slater determinants. 


### Publications
For detailed information about the mathematical formulation, please take a look at the [FANCI publication](https://doi.org/10.1016/j.comptc.2021.113187). The official notes of the Fanpy library can be found in [this article](https://doi.org/10.1002/jcc.27034).    


