.. _objective:

Objective
=========
Given the Hamiltonian, :math:`\hat{H}`, and the wavefunction, :math:`\Psi`, we seek to optimize the
parameters such that the Schrödinger equation is satisfied:

.. math::

    \hat{H} \left| \Psi \right> = E \left| \Psi \right>

We can convert this equation into other objectives more amenable for computation.

.. _energy:

Energy
------
Multiplying the Schrödinger equation on the left by the complex conjugate of the wavefunction
and integrating gives an explicit expression for the expectation value of the energy for the
wavefunction in question:

.. math::

    E &= \frac{\left< \Psi \middle| \hat{H} \middle| \Psi \right>}{\left< \Psi \middle| \Psi \right>}

This formulation of energy satisfies the variational principle: it is always greater than or equal
to the true energy and is equal to the true energy only when the wavefunction represents the exact
ground state. This means that the parameters in the wavefunction model can be determined by
minimizing the energy expression. However, except for some special cases, the integrals,
:math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>` and :math:`\left< \Psi \middle| \Psi \right>`,
are often too expensive to compute for larger systems. In addition, the energy is typically a
non-convex function of the wavefunction parameters, and many local minima exist. Global optimization
algorithms are usually intractable except for systems so small that FCI is tractable.
Nonetheless, local minima of the energy often provide reasonable approximate energies, especially
when good initial guesses are available.

At the expense of accuracy, the cost for evaluating (and optimizing) the wavefunction can be reduced
by replacing the wavefunctions with more computationally tractable functions. Replacing the
wavefunction on the left side of the integrals (i.e. :math:`\left< \Psi \right|`) with a reference
wavefunction,

.. math::

    E = \frac{\left< \Phi \middle| \hat{H} \middle| \Psi \right>}{\left< \Phi \middle| \Psi \right>}

where :math:`\Phi` is the reference wavefunction. In :code:`Fanpy`, the reference wavefunction can
be a Slater determinant, :math:`\left| \mathbf{m} \right>`, a linear combination of Slater
determinants (CI wavefunction), :math:`\sum_{\mathbf{m} \in S_{ci}} c_{\mathbf{m}} \left| \mathbf{m} \right>`
or the wavefunction projected onto a given set of Slater determinants,
:math:`\sum_{\mathbf{m} \in S_{trunc}} f(\mathbf{m}) \left| \mathbf{m} \right>`.
This set is oftened denoted as "projection space" throughout :code:`Fanpy`.
These objectives are implemented in class
:class:`EnergyOneSideProjection <fanpy.eqn.energy_oneside.OneSidedEnergy>`.

Similarly, the wavefunctions on both sides can be replaced with similar wavefunctions by expressing
both functions with respect to sets of Slater determinants:

.. math::

    E &= \frac{\left< \Psi \middle| \hat{H} \middle| \Psi \right>}{\left< \Psi \middle| \Psi \right>}\\
    &= \frac{
        \left< \Psi \right|
        \bigg( \sum_{\mathbf{m}_l \in S_l}  \left| \mathbf{m}_l \middle> \middle< \mathbf{m}_l \right| \bigg)
        \hat{H}
        \bigg( \sum_{\mathbf{m}_r \in S_r} \left| \mathbf{m}_r \middle> \middle< \mathbf{m}_r \right|  \bigg)
        \left| \Psi \right>
    }{
        \left< \Psi \right|
        \bigg( \sum_{\mathbf{m}_n \in S_n}  \left| \mathbf{m}_n \middle> \middle< \mathbf{m}_n \right| \bigg)
        \left| \Psi \right>
    }\\
    &= \frac{
        \sum_{\mathbf{m}_l  \in S_l} \sum_{\mathbf{m}_r \in S_r}
        \left< \Psi \middle| \mathbf{m}_l \middle>
        \middle< \mathbf{m}_l \middle| \hat{H} \middle| \mathbf{m}_r \middle>
        \middle< \mathbf{m}_r \middle| \Psi \right>
    }{
        \sum_{\mathbf{m}_n \in S_n}
        \left< \Psi \middle| \mathbf{m}_n \middle> \middle< \mathbf{m}_n \middle| \Psi \right>
    }

This objective is implemented in class :class:`EnergyTwoSideProjection <fanpy.eqn.energy_twoside.TwoSidedEnergy>`.

When the projection space for the left side, :math:`S_l`, right side, :math:`S_r`, and the
denominator, :math:`S_n`, are the same, then the energy is variational. This is equivalent to
reducing the wavefunction down to the CI wavefunction that corresponds to the given projection
space. This objective is implemented in class
:class:`EnergyVariational <fanpy.eqn.energy_variational.EnergyVariational>`.


Projected Schrödinger Equation
------------------------------
Notice that the solutions to the Schrödinger equation will also satisfy the equation after
integrating on the left with an arbitrary function. In other words, if

.. math::

  \hat{H} \ket{\Psi} = E \ket{\Psi}

then

.. math::

  \braket{\Phi | \hat{H} | \Psi} = E \braket{\Psi | \Phi} \mbox{  $\forall$ $\Phi$}

This equation is called the weak form of the eigenproblem, or the projected Schrödinger
equation. In the projected Schrödinger equation, the Schrödinger equation is integrated with a set
of functions, termed the projection space, to form a system of nonlinear equations:

.. math::

    \braket{\mathbf{m}_1 | \hat{H} | \Psi} - E \braket{\mathbf{m}_1 | \Psi} &= 0\\
    &\hspace{0.5em} \vdots\\
    \braket{\mathbf{m}_M | \hat{H} | \Psi} - E \braket{\mathbf{m}_M | \Psi} &= 0\\

To avoid the trivial solution, :math:`\Psi = 0`, a normalization constraint is often added as an
extra equation. If all of the equations are satisfied, the wavefunction and energy are the solutions
to the Schr\"{o}dinger equation within the selected projection space. When the projection space is
complete, i.e. it encompasses all possible Slater determinants, the solutions are exact within the
basis-set approximation. If the formulation of the wavefunction cannot produce a solution for the
given Hamiltonian, the cost function for the system of nonlinear equations solver (typically a
residual sum of squares) provides a measure of the error in the optimized wavefunction and energy.
The least-squares solution for a complete projection space is an upper bound to the energy, and this
energetic upper bound, minus the residual sum of squares from the equations, is a lower bound to the
energy\cite{completeprojection}.

The objective for the projected Schrödinger equation is implemented in class
:class:`ProjectedSchrodinger <fanpy.eqn.projected.ProjectedSchrodinger>`.


Projection Space
~~~~~~~~~~~~~~~~
In essence, the insignificant ("trivially" satisfied) parts of the Schrödinger equation can be
removed with a projection operator. In the FANCI module, the projection space can include

* Slater determinants
* CI wavefunctions, which is equivalent to linearly combining the equations that correspond to the
  Slater determinants in the CI wavefunction:

.. math::

    \left< \Phi \middle| \hat{H} \middle| \Psi \right> - E \left< \Phi \middle| \Psi \right> &= 0\\
    \sum_{\mathbf{m} \in S_{ci}} c^*_{\mathbf{m}} \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>
    - E \sum_{\mathbf{m} \in S_{ci}} c^*_{\mathbf{m}} \left< \mathbf{m} \middle| \Psi \right> &= 0\\
    \sum_{\mathbf{m} \in S_{ci}} c^*_{\mathbf{m}}
    \left(
        \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right> - E \left< \mathbf{m} \middle| \Psi \right>
    \right)
    &= 0

Energy
~~~~~~
The energy in the projected Schrödinger equation can be treated in different ways. The energy can
be

* a fixed number - it would not change in the course of the optimization.

* a variable - it will be optimizied like all the other parameters in the Schrödinger equation.

* computed by integrating the Schrödinger equation with respect to a reference wavefunction - see
  :ref:`Energy <energy>` for details.

Constraints
~~~~~~~~~~~
Since the Schrödinger equation is treated as a system of equations, it is quite easy to put
constraints into the objective - simply add more equations to the system. So far, only the
normalization constraint is implemented:

.. math::

    \left< \Phi \middle| \Psi \right> - 1 = 0

where :math:`\Phi` can be

* a Slater determinant

.. math::

    \left< \mathbf{m} \middle| \Psi \right> - 1 = 0

* a CI wavefunction

.. math::

    \sum_{\mathbf{m} \in S_{ci}} c^*_{\mathbf{m}} \left< \mathbf{m} \middle| \Psi \right> - 1 = 0

* a truncated form of the wavefunction

.. math::

    \sum_{\mathbf{m} \in S_{trunc}} f^*(\mathbf{m}) \left< \mathbf{m} \middle| \Psi \right> - 1 = 0

Though there is no abstract base class for the constraints specifically, they should follow the same
structure as the abstract base class, :class:`BaseSchrodinger <fanpy.eqn.base.BaseSchrodinger>`.
