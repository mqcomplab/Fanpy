## Fanpy: A Python library for prototyping multideterminant methods in ab initio Electronic Structure Calculations


**Web: [Miranda Quintana Group](https://quintana.chem.ufl.edu/)**   


Fanpy is a free, open-source, and cross-platform Python 3 library designed for ab initio electronic structure calculations. The Fanpy implementation is based on the mathematical framework called [Flexible Ansatz for N-electron Configuration Interaction (FANCI)](https://doi.org/10.1016/j.comptc.2021.113187). The adoption of the FANCI framework gives a highly modular structure to Fanpy resulting in 5 modules - Hamiltonian, Wavefunction, Objective, Solver, and Tools. The modular structure offers two greatest virtues, The first is its 'sandbox-like' ability to handle any combinations of wavefunction ansatz and different methods, and the second is the ease of transition from the formal conception of a method to its working implementation.


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


