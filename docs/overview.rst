Overview
========

We seek to find the wavefunction, :math:`\Psi`, and the energy, :math:`E`, that satisfies the
Schrödinger equation,

.. math::

    \hat{H} \left| \Psi \right> = E \left| \Psi \right>

for the given electronic Hamiltonian, :math:`\hat{H}`. Since finding the exact solution in the full
Hilbert space (function space) is not practical, we use a finite set of one-electron basis functions
(orbitals) from which we build the N-electron basis functions (Slater determinants). Since the wavefunction is
antisymmeteric with respect to the interchange of the electrons, we use the antisymmeterized
product of orbitals (Slater determinant) as the foundation from which the wavefunction is built.
Then, the Hamiltonian, which operates on the wavefunction, can also be expressed with respect to
Slater determinants. We represent Slater determinants and transformations between them using Second
Quantization. All operations involving Slater determinants are handled by the module,
:mod:`slater <fanpy.tools.slater>`.

Within a finite one-electron basis set and the Second Quantization framework, the Schrödinger
equation can be decomposed into four components:

  * :ref:`Hamiltonian <hamiltonian>`

    * The Hamiltonian describes the system and the interactions within the system.

  * :ref:`Wavefunction <wavefunction>`

    * The wavefunction is the function that approximates the eigenfunction of the Hamiltonian.

  * :ref:`Objective <objective>`

    * The objective is a function whose optimization corresponds to (approximately) solving the
      Schrödinger equation.

  * :ref:`Solver <solver>`

    * The solver is an algorithm that will optimize the objective.

Each component is independent of the others, except through the modules in
:mod:`tools <fanpy.tools>`. Almost all combinations of the components are possible, though
special care needs to be taken to ensure that the the given combination is meaningful. Please see
their respective sections for more details.
