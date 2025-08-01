.. _wavefunction:

Multideterminant Wavefunction
=============================

Because the set of all :math:`N`-electron Slater determinants is a complete basis for the
:math:`N`-electron wavefunction, all :math:`N`-electron wavefunctions can be represented exactly
(within the strictures of the presupposed one-electron basis set) as a linear combination of Slater
determinants. Often the exact wavefunction for a chemical system has a single dominant determinant;
such systems are called single-reference. Closed-shell organic molecules near their equilibrium
geometry are often single reference. However, for chemical systems that have many valence orbitals
that are nearly degenerate in energy, like transition-metal complexes, the ground-state electron
configuration is ambiguous and a single electron configuration cannot sufficiently describe the
system, often leading to catastrophically wrong results. Building a :math:`N`-electron wavefunction
for this type of multiconfigurational molecule requires multiple Slater determinants.

The wavefunctions in :code:`Fanpy` are represented as a linear combination of Slater determinants
whose coefficients are parameterized with a function.

.. math::

  \ket{\Psi} = \sum_{\mathbf{m} \in S} f(\mathbf{m}) \ket{\mathbf{m}}

Since the Slater determinants are orthogonal to one another, these functions represent the overlap
of the wavefunction with the given Slater determinant. These functions are provided by the
:code:`get_overlap` method of the wavefunction class. Since evaluating these methods is often the
time-limiting step within a electronic structure calculation, the overlaps can be memoized, i.e.
results of these functions for a Slater determinant can be cached to avoid recomputing the overlaps.
To do so, defining the time-critical steps in computing overlap and its derivative can be defined in
:code:`_olp` and :code:`_olp_deriv`, respectively. Then, the cache can be enabled by calling the
:code:`enable_cache` method. The framework for wavefunctions is established in the abstract base
class, :class:`BaseWavefunction <fanpy.wfn.base.BaseWavefunction>`.
