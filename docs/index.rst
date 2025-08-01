=================
 Fanpy |version|
=================

:code:`Fanpy` is a free and open-source Python 3 library for developing and testing multideterminant
wavefunctions and related *ab initio* methods in electronic structure theory. The main use of
:code:`Fanpy` is to quickly prototype new methods by decreasing the barrier required to translate the
mathematical conception of a new wavefunction ans\"{a}tze to a working implementation. :code:`Fanpy` uses
the framework of our recently introduced Flexible Ansatz for N-electron Configuration Interaction
(FANCI), where multideterminant wavefunctions are represented by their overlaps with Slater
determinants of orthonormal spin-orbitals. In the simplest case, then, a new wavefunction ansatz can
be implemented by simply writing a function for evaluating its overlap with an arbitrary Slater
determinant. :code:`Fanpy` is modular both in implementation and in theory. Each electronic structure
method is represented as a collection of four components, each of which is represented by an
independent module:

(a) the (multideterminant) wavefunction model
(b) the system Hamiltonian, as represented by its one- and two-electron integrals
(c) an equation (or system of equations) to solve that is equivalent to the Schrödinger equation
(d) an algorithm for optimizing the objective function(s).

This modular structure makes it easy for users to mix and match different methods and for developers
to quickly try new ideas. :code:`Fanpy` is written purely in Python with standard dependencies, making it
accessible for most operating systems; it adheres to principles of modern software development,
including comprehensive documentation, extensive testing, and continuous integration and delivery
protocols.

.. toctree::
  :maxdepth: 2

  installation
  overview
  slater
  hamiltonian
  wavefunction
  objective
  solver
  tutorial_index
  example_index
  tech_api
