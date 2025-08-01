.. _hamiltonian:

Hamiltonian
===========

The non-relativistic electronic Hamiltonian of molecular systems, in atomic units, is given by

.. math::

    \hat{H} &=
    -\frac{1}{2} \sum_i \nabla^2_i
    - \sum_i \sum_A \frac{Z_A}{r_{iA}}
    + \sum_{i<j} \frac{1}{r_{ij}}\\
    &= \sum_i \hat{h}_i + \sum_{i<j} \hat{g}_{ij}

where :math:`\nabla_i` is the gradient with respect to the position of electron :math:`i`,
:math:`Z_A` is the charge of the atomic nucleus :math:`A`, :math:`r_{iA}` is the distance between
electron :math:`i` and nucleus :math:`A`, and :math:`r_{ij}` is the distance between electrons
:math:`i` and :math:`j`. For each electron :math:`i`, the operators for the kinetic energy,
:math:`-\frac{1}{2} \nabla^2_i`, and the nuclear electron attraction, :math:`- \sum_A
\frac{Z_A}{r_{iA}}`, are grouped together to form the one-electron operator, :math:`\hat{h}_i`.


Since the wavefunction is a function of the positions of the electrons, the electron indices,
:math:`i` and :math:`j`, are needed to keep track of the distinct electronic positions even though
all electrons are equivalent particles. It is helpful to project the Hamiltonian onto the orbital
basis set and reexpress it with second quantization, so that the electronic positions need not be
tracked explicitly. The matrix representations of the one- and two-electron operators in the orbital
basis set are denoted as one-electron integrals, :math:`h_{ij}`, and two-electron integrals,
:math:`g_{ijkl}`, respectively. Explicitly,

.. math::

    h_{ij}
    &= \int \phi^*_i(\mathbf{r}_1) \hat{h} \phi_j(\mathbf{r}_1) d\mathbf{r}_1\\
    &= \int \phi^*_i(\mathbf{r}_1)
    \left( -\frac{1}{2} \nabla^2  - \sum_A \frac{Z_A}{|\mathbf{r}_1 - \mathbf{r}_{A}|} \right)
    \phi_j(\mathbf{r}_1) d\mathbf{r}_1

.. math::

    g_{ijkl}
    &= \int \phi^*_i(\mathbf{r}_1) \phi^*_j(\mathbf{r}_2)
    \hat{g}
    \phi_k(\mathbf{r}_1) \phi_l(\mathbf{r}_2)
    d\mathbf{r}_1 d\mathbf{r}_2\\
    &= \int \phi^*_i(\mathbf{r}_1) \phi^*_j(\mathbf{r}_2)
    \frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|}
    \phi_k(\mathbf{r}_1) \phi_l(\mathbf{r}_2)
    d\mathbf{r}_1 d\mathbf{r}_2\\

Note that in the literature two-electron integrals are sometimes denoted using physicists' notation
and sometimes using chemists' notation. The equation above is in the physicists' notation. The
following equation is in the chemists' notation:

.. math::

    g_{ijkl}
    &= \int \phi^*_i(\mathbf{r}_1) \phi^*_k(\mathbf{r}_2)
    \hat{g}
    \phi_j(\mathbf{r}_1) \phi_l(\mathbf{r}_2)
    d\mathbf{r}_1 d\mathbf{r}_2\\

:code:`Fanpy` defaults to the physicists' notation, unless an option is provided otherwise.

Using the one- and two-electron integrals, the second-quantized Hamiltonian is:

.. math::

    \hat{H}
    &= \sum_{ij} h_{ij} a^\dagger_i a_j
    + \frac{1}{2} \sum_{ijkl} g_{ijkl} a^\dagger_i a^\dagger_j a_l a_k\\


Note that this second-quantized Hamiltonian is valid for any number of fermions, interacting by any
type of 1-body and 2-body forces. Thus this equation, and the numerical methods used to solve it,
are applicable not only to molecules but to any other system of interacting elementary fermionic
particles. At this moment, only the electronic molecular Hamiltonians are implemented in
:code:`Fanpy`, though its structure supports Hamiltonians of other fermionic particles.

In :code:`Fanpy`, the Hamiltonians can be expressed with respect to restricted, unrestricted, and
generalized orbitals which differ by the ways in which they treat spin orbitals of opposing spins.
In generalized orbitals, no assumptions are made in regads to the spin of the orbitals. Thus, the
following properties hold for the one- and two-electron integrals:

.. math::

    h^\chi_{ij} &= (h^\chi_{ji})^*\\
    g^{\chi}_{ijkl} &= g^\chi_{jilk}\\
    g^{\chi}_{ijkl} &= (g^\chi_{klij})^*

If the orbitals are unrestricted, only the spin orbitals of the same spin can be integrated with one
another:

.. math::

    h_{ij}^\chi &=
    \begin{cases}
      h^\alpha_{ij} & \mbox{if $\sigma_i = \sigma_j = \alpha$}\\
      h^\beta_{ij} & \mbox{if $\sigma_i = \sigma_j = \beta$}\\
      0 & \mbox{else}\\
    \end{cases}\\
    g_{ijkl}^{\chi} &=
    \begin{cases}
      g^{\alpha \alpha}_{ijkl} & \mbox{if $\sigma_i = \sigma_j = \sigma_k = \sigma_l$}\\
      g^{\alpha \beta}_{ijkl} & \mbox{if $\sigma_i = \sigma_k = \alpha$ and $\sigma_j = \sigma_l = \beta$}\\
      g^{\beta \alpha}_{ijkl} & \mbox{if $\sigma_i = \sigma_k = \beta$ and $\sigma_j = \sigma_l = \alpha$}\\
      g^{\beta \beta}_{ijkl} & \mbox{if $\sigma_i = \sigma_j = \sigma_k = \sigma_l$}\\
      0 & \mbox{else}\\
    \end{cases}

and the following properties hold:

.. math::

    h^\alpha_{ij} &= (h^\alpha_{ji})^*\\
    h^\beta_{ij} &= (h^\beta_{ji})^*\\
    g^{\alpha \alpha}_{ijkl} &= g^{\alpha \alpha}_{jilk}\\
    g^{\alpha \beta}_{ijkl} &= g^{\beta \alpha}_{jilk}\\
    g^{\beta \beta}_{ijkl} &= g^{\beta \beta}_{jilk}\\
    g^{\alpha \alpha}_{ijkl} &= (g^{\alpha \alpha}_{klij})^*\\
    g^{\alpha \beta}_{ijkl} &= (g^{\alpha \beta}_{klij})^*\\
    g^{\beta \beta}_{ijkl} &= (g^{\beta \beta}_{klij})^*\\

If the orbitals are restricted, the spin orbitals of opposing spins are constrained to be equal to
one another (except for their spins) and only the spin orbitals of the same spin can be integrated
with one another:

.. math::

    h_{ij}^\phi &=
    \begin{cases}
      h^\alpha_{ij}  =  h^\beta_{ij} = h^\phi_{ij} & \mbox{if $\sigma_i = \sigma_j}\\
      0 & \mbox{else}\\
    \end{cases}\\
    g_{ijkl}^{\chi} &=
    \begin{cases}
      g^{\alpha \alpha}_{ijkl} = g^{\alpha \beta}_{ijkl} = g^{\beta \alpha}_{ijkl} =
      g^{\beta \beta}_{ijkl} =  = g^\phi_{ijkl}
      & \mbox{if $\sigma_i = \sigma_k and $\sigma_j = \sigma_l}\\
      0 & \mbox{else}\\
    \end{cases}

The following properties hold for the restricted orbitals:

.. math::

    h^\phi_{ij} &= (h^\phi_{ji})^*\\
    g^{\phi}_{ijkl} &= g^\phi_{jilk}\\
    g^{\phi}_{ijkl} &= (g^\phi_{klij})^*

Orbital rotations of the electronic molecular Hamiltonian is supported by explicitly transforming
the integrals with a Jacobi matrix, a general transformation matrix, and with a unitary matrix (that
corresponds to an anti-Hermitian matrix). The integrals of the Hamiltonian with Slater determinants
(:code:`integrate_sd_sd` and :code:`integrate_sd_wfn`) can be derivatized with respect to the
anti-Hermitian matrix elements, :math:`\kappa_{ij}`, in the unitary transformation operator,
:math:`\exp^{\hat{\mathbf{\kappa}}} = \sum_{i>j} \kappa_{ij} (a^\dagger_i a_j - a_j^\dagger a_i)`:

.. math::

  \left.
    \frac{
      \partial \braket{\mathbf{m} | e^{-\hat{\mathbf{\kappa}}} \hat{H} e^{\hat{\mathbf{\kappa}}} | \mathbf{n}}
    }{\partial \kappa_{ij}}
  \right|_{\kappa = 0}
  =
  \left.
    \frac{
      \partial \braket{\mathbf{m} | [\hat{H}, \hat{\mathbf{\kappa}}]  | \mathbf{n}}
    }{\partial \kappa_{ij}}
  \right|_{\kappa = 0}

In the :code:`Fanpy`, the objectives represent the Schrödinger equation with the integrals
:math:`\left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>` and
:math:`\left< \mathbf{m} \middle| \hat{H} \middle| \mathbf{n} \right>`, where
:math:`\mathbf{m}` and :math:`\mathbf{n}` are Slater determinants and :math:`\Psi` is the
wavefunction. The electronic molecular Hamiltonians of the restricted, unrestricted, and generalized
orbitals are given in
:class:`RestrictedMolecularHamiltonian <fanpy.ham.restricted_chemical.RestrictedMolecularHamiltonian>`,
:class:`UnrestrictedMolecularHamiltonian <fanpy.ham.unrestricted_chemical.UnrestrictedMolecularHamiltonian>`,
and :class:`GeneralizedMolecularHamiltonian <fanpy.ham.generalized_chemical.GeneralizedMolecularHamiltonian>`,
respectively.
