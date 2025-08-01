.. _api:

*****************
API Documentation
*****************

Hamiltonians
============

* :class:`Base Hamiltonian <fanpy.ham.base.BaseHamiltonian>`
* :class:`Generalized Base Hamiltonian <fanpy.ham.generalized_base.BaseGeneralizedHamiltonian>`
* :class:`Unrestricted Base Hamiltonian <fanpy.ham.unrestricted_base.BaseUnrestrictedHamiltonian>`
* :class:`Generalized Molecular Hamiltonian <fanpy.ham.generalized_chemical.GeneralizedChemicalHamiltonian>`
* :class:`Unrestricted Molecular Hamiltonian <fanpy.ham.unrestricted_chemical.UnrestrictedChemicalHamiltonian>`
* :class:`Restricted Molecular Hamiltonian <fanpy.ham.restricted_chemical.RestrictedChemicalHamiltonian>`
* :class:`Seniority Zero Hamiltonian <fanpy.ham.senzero.SeniorityZeroHamiltonian>`

Wavefunctions
=============

* :class:`Base Wavefunction <fanpy.wfn.base.BaseWavefunction>`
* CI Wavefunction

  * :class:`Base CI Wavefunction <fanpy.wfn.ci.base.CIWavefunction>`
  * :class:`FCI Wavefunction <fanpy.wfn.ci.fci.FCI>`
  * :class:`DOCI Wavefunction <fanpy.wfn.ci.doci.DOCI>`
  * :class:`CISD Wavefunction <fanpy.wfn.ci.cisd.CISD>`
  * :class:`CI-Pair Wavefunction <fanpy.wfn.ci.ci_pairs.CIPairs>`

* Geminal Wavefunction

  * :class:`Base Geminal Wavefunction <fanpy.wfn.geminal.base.BaseGeminal>`
  * :class:`APG Wavefunction <fanpy.wfn.geminal.apg.APG>`
  * :class:`APsetG Wavefunction <fanpy.wfn.geminal.apsetg.BasicAPsetG>`
  * :class:`APIG Wavefunction <fanpy.wfn.geminal.apig.APIG>`
  * :class:`AP1roG Wavefunction <fanpy.wfn.geminal.ap1rog.AP1roG>`
  * :class:`APr2g Wavefunction <fanpy.wfn.geminal.apr2g.APr2G>`

* Network Wavefunction

  * :class:`MatrixProductState <fanpy.wfn.network.mps.MatrixProductState>`

* Quasiparticle Wavefunction

  * :class:`BaseQuasiparticle <fanpy.wfn.quasiparticle.base.BaseQuasiparticle>`
  * :class:`DeterminantRatio <fanpy.wfn.quasiparticle.det_ratio.DeterminantRatio>`
  * :class:`AntisymmeterizedProductTetrets <fanpy.wfn.quasiparticle.tetret.AntisymmeterizedProductTetrets>`

* Composite Wavefunction

  * :class:`Base Composite of One Wavefunction <fanpy.wfn.composite.base_one.BaseCompositeOneWavefunction>`
  * :class:`Wavefunction with Nonorthogonal Orbitals <fanpy.wfn.composite.nonorth.NonorthWavefunction>`
  * :class:`Wavefunction with Jacobi Rotated Orbitals <fanpy.wfn.composite.jacobi.JacobiWavefunction>`
  * :class:`Linear Combination of Wavefunctions <fanpy.wfn.composite.lincomb.LinearCombinationWavefunction>`


Objectives
==========

* :class:`Base Schrodinger Equation <fanpy.eqn.base.BaseSchrodinger>`
* :class:`System of Equations <fanpy.eqn.projected.ProjectedSchrodinger>`
* :class:`Least Squared Sum of Equations <fanpy.eqn.least_squares.LeastSquaresEquations>`
* :class:`One Sided Energy <fanpy.eqn.energy_oneside.EnergyOneSideProjection>`
* :class:`Two Sided Energy <fanpy.eqn.energy_twoside.EnergyTwoSideProjection>`
* :class:`Variational Energy <fanpy.eqn.energy_variational.EnergyVariational>`
* :class:`Local Energy <fanpy.eqn.local_energy.LocalEnergy>`
* :class:`Normalization Constraint <fanpy.eqn.constraints.norm.NormConstraint>`

Solvers
=======

* :func:`Brute CI Solver <fanpy.solver.ci.brute>`
* Single Equation Solver

  * :func:`CMA-ES Solver <fanpy.solver.equation.cma>`
  * :func:`scipy.optimize.minimize Solver <fanpy.solver.equation.minimize>`

* System of Equations Solver

  * :func:`Least Squares Solver <fanpy.solver.system.least_squares>`
  * :func:`Root Solver <fanpy.solver.system.root>`

* Wrapper for External Solver

  * :func:`SciPy Solver Wrapper <fanpy.solver.wrappers.wrap_scipy>`
  * :func:`scikit-optimize Solver Wrapper <fanpy.solver.wrappers.wrap_skopt>`

Backend
=======
* General Math Tools

  * :func:`Binomial Coefficient <fanpy.tools.math_tools.binomial>`
  * :func:`Adjugate <fanpy.tools.math_tools.adjugate>`
  * :func:`Permanent Using Combinatorics <fanpy.tools.math_tools.permanent_combinatoric>`
  * :func:`Permanent Using Ryser Algorithm <fanpy.tools.math_tools.permanent_ryser>`
  * :func:`Permanent Using Borchardt Theorem <fanpy.tools.math_tools.permanent_borchardt>`

* :mod:`Slater Determinant <fanpy.tools.slater>`

  * :func:`Check if occupied <fanpy.tools.slater.occ>`
  * :func:`Check if alpha <fanpy.tools.slater.is_alpha>`
  * :func:`Convert spin to spatial <fanpy.tools.slater.spatial_index>`
  * :func:`Get occupation number <fanpy.tools.slater.total_occ>`
  * :func:`Annhilation Operator <fanpy.tools.slater.annihilate>`
  * :func:`Creation Operator <fanpy.tools.slater.create>`
  * :func:`Excitation Operator<fanpy.tools.slater.excite>`
  * :func:`Ground state Slater determinant <fanpy.tools.slater.ground>`
  * :func:`Check if internal Slater determinant <fanpy.tools.slater.is_internal_sd>`
  * :func:`Convert to internal Slater determinant <fanpy.tools.slater.internal_sd>`
  * :func:`Get occupied orbital indices <fanpy.tools.slater.occ_indices>`
  * :func:`Get virtual orbital indices <fanpy.tools.slater.vir_indices>`
  * :func:`Get orbitals shared between Slater determinants <fanpy.tools.slater.shared_orbs>`
  * :func:`Get orbitals different between Slater determinants <fanpy.tools.slater.diff_orbs>`
  * :func:`Combine alpha and beta parts <fanpy.tools.slater.combine_spin>`
  * :func:`Split a Slater determinant into alpha and beta parts <fanpy.tools.slater.split_spin>`
  * :func:`Get index after interleaving <fanpy.tools.slater.interleave_index>`
  * :func:`Get index after deinterleaving <fanpy.tools.slater.deinterleave_index>`
  * :func:`Interleave Slater determinant <fanpy.tools.slater.interleave>`
  * :func:`Deinterleave Slater determinant <fanpy.tools.slater.deinterleave>`
  * :func:`Get spin of Slater determinant <fanpy.tools.slater.get_spin>`
  * :func:`Get seniority of Slater determinant <fanpy.tools.slater.get_seniority>`
  * :func:`Get signature of the permutation that sorts a set of annihilators. <fanpy.tools.slater.sign_perm>`
  * :func:`Get signature of moving a creation operator to a specific position. <fanpy.tools.slater.sign_swap>`
  * :func:`Generate Slater determinants <fanpy.tools.sd_list.sd_list>`

* Perfect Matching Generator

  * :func:`Complete Graph Perfect Matching Generator <fanpy.tools.graphs.generate_complete_pmatch>`
  * :func:`Bipartite Graph Perfect Matching Generator <fanpy.tools.graphs.generate_biclique_pmatch>`

Scripts
=======
* :func:`Run calculation <fanpy.scripts.run_calc.run_calc>`
* :func:`Make calculation script <fanpy.scripts.make_script.make_script>`

.. Silent api generation
    .. autosummary::
      :toctree: modules/generated

      fanpy.ham.base.BaseHamiltonian
      fanpy.ham.generalized_base.BaseGeneralizedHamiltonian
      fanpy.ham.unrestricted_base.BaseUnrestrictedHamiltonian
      fanpy.ham.generalized_chemical.GeneralizedMolecularHamiltonian
      fanpy.ham.unrestricted_chemical.UnrestrictedMolecularHamiltonian
      fanpy.ham.restricted_chemical.RestrictedMolecularHamiltonian
      fanpy.ham.senzero.SeniorityZeroHamiltonian

      fanpy.solver.ci.brute
      fanpy.solver.equation.cma
      fanpy.solver.equation.minimize
      fanpy.solver.system.least_squares
      fanpy.solver.system.root
      fanpy.solver.wrappers
      fanpy.solver.wrappers.wrap_scipy
      fanpy.solver.wrappers.wrap_skopt

      fanpy.eqn.constraints.norm.NormConstraint
      fanpy.eqn.base.BaseSchrodinger
      fanpy.eqn.projected.ProjectedSchrodinger
      fanpy.eqn.least_squares.LeastSquaresEquations
      fanpy.eqn.energy_oneside.EnergyOneSideProjection
      fanpy.eqn.energy_twoside.EnergyTwoSideProjection

      fanpy.wfn.base.BaseWavefunction
      fanpy.wfn.ci.base.CIWavefunction
      fanpy.wfn.ci.fci.FCI
      fanpy.wfn.ci.doci.DOCI
      fanpy.wfn.ci.cisd.CISD
      fanpy.wfn.ci.ci_pairs.CIPairs
      fanpy.wfn.geminal.base.BaseGeminal
      fanpy.wfn.geminal.apg.APG
      fanpy.wfn.geminal.apsetg.BasicAPsetG
      fanpy.wfn.geminal.apig.APIG
      fanpy.wfn.geminal.ap1rog.AP1roG
      fanpy.wfn.geminal.apr2g.APr2G
      fanpy.wfn.composite.base_one.BaseCompositeOneWavefunction
      fanpy.wfn.composite.nonorth.NonorthWavefunction
      fanpy.wfn.composite.jacobi.JacobiWavefunction
      fanpy.wfn.composite.lincomb.LinearCombinationWavefunction
      fanpy.wfn.network.mps.MatrixProductState
      fanpy.wfn.quasiparticle.base.BaseQuasiparticle
      fanpy.wfn.quasiparticle.det_ratio.DeterminantRatio
      fanpy.wfn.quasiparticle.tetret.AntisymmeterizedProductTetrets

      fanpy.tools.math_tools.binomial
      fanpy.tools.math_tools.adjugate
      fanpy.tools.math_tools.permanent_combinatoric
      fanpy.tools.math_tools.permanent_ryser
      fanpy.tools.math_tools.permanent_borchardt
      fanpy.tools.math_tools.unitary_matrix

      fanpy.tools.slater
      fanpy.tools.slater.is_sd_compatible
      fanpy.tools.slater.occ
      fanpy.tools.slater.occ_indices
      fanpy.tools.slater.vir_indices
      fanpy.tools.slater.total_occ
      fanpy.tools.slater.is_alpha
      fanpy.tools.slater.spatial_index
      fanpy.tools.slater.annihilate
      fanpy.tools.slater.create
      fanpy.tools.slater.excite
      fanpy.tools.slater.ground
      fanpy.tools.slater.shared_orbs
      fanpy.tools.slater.diff_orbs
      fanpy.tools.slater.combine_spin
      fanpy.tools.slater.split_spin
      fanpy.tools.slater.interleave_index
      fanpy.tools.slater.deinterleave_index
      fanpy.tools.slater.interleave
      fanpy.tools.slater.deinterleave
      fanpy.tools.slater.get_spin
      fanpy.tools.slater.get_seniority
      fanpy.tools.slater.sign_perm
      fanpy.tools.slater.sign_swap

      fanpy.tools.sd_list.sd_list

      fanpy.tools.graphs.generate_complete_pmatch
      fanpy.tools.graphs.generate_biclique_pmatch

      fanpy.scripts.make_script.make_script
      fanpy.scripts.run_calc.run_calc
