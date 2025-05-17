"""Test fanpy.eqn.projected."""
from fanpy.eqn.constraints.norm import NormConstraint
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.eqn.utils import ComponentParameterIndices, ParamContainer
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np

import pytest

from utils import skip_init


def test_system_init_energy():
    """Test energy initialization in ProjectedSchrodinger.__init__."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedMolecularHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )

    test = ProjectedSchrodinger(wfn, ham, energy=None, energy_type="compute")
    assert isinstance(test.energy, ParamContainer)
    assert test.energy.params == test.get_energy_one_proj(0b0101)

    test = ProjectedSchrodinger(wfn, ham, energy=2.0, energy_type="compute")
    assert test.energy.params == 2.0

    test = ProjectedSchrodinger(wfn, ham, energy=np.complex128(2.0), energy_type="compute")
    assert test.energy.params == 2.0

    with pytest.raises(TypeError):
        ProjectedSchrodinger(wfn, ham, energy="1", energy_type="compute")

    with pytest.raises(ValueError):
        ProjectedSchrodinger(wfn, ham, energy=None, energy_type="something else")
    with pytest.raises(ValueError):
        ProjectedSchrodinger(wfn, ham, energy=None, energy_type=0)

    test = ProjectedSchrodinger(wfn, ham, energy=0.0, energy_type="variable")
    assert np.allclose(test.indices_component_params[test.energy], np.array([0]))
    assert np.allclose(test.indices_objective_params[test.energy], np.array([6]))

    test = ProjectedSchrodinger(wfn, ham, energy=0.0, energy_type="fixed")
    assert np.allclose(test.indices_component_params[test.energy], np.array([]))
    assert np.allclose(test.indices_objective_params[test.energy], np.array([]))


def test_system_nproj():
    """Test SystemEquation.nproj."""
    test = skip_init(ProjectedSchrodinger)
    test.assign_pspace([0b0101, 0b1010])
    assert test.nproj == 2
    test.assign_pspace([0b0101, 0b1010, 0b0110])
    assert test.nproj == 3


def test_system_assign_pspace():
    """Test ProjectedSchrodinger.assign_pspace."""
    test = skip_init(ProjectedSchrodinger)
    test.wfn = CIWavefunction(2, 4)
    test.pspace_exc_orders=None
    test.assign_pspace()
    for sd, sol_sd in zip(test.pspace, [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]):
        assert sd == sol_sd

    test.assign_pspace([0b0101, 0b1010])
    for sd, sol_sd in zip(test.pspace, [0b0101, 0b1010]):
        assert sd == sol_sd

    with pytest.raises(TypeError):
        test.assign_pspace(0b0101)
    with pytest.raises(TypeError):
        test.assign_pspace("0101")


def test_system_assign_refwfn():
    """Test ProjectedSchrodinger.assign_refwfn."""
    test = skip_init(ProjectedSchrodinger)
    test.wfn = CIWavefunction(2, 4)

    test.assign_refwfn()
    assert test.refwfn == (0b0101,)

    test.assign_refwfn(0b0110)
    assert test.refwfn == (0b0110,)

    test.assign_refwfn([0b0101, 0b0110])
    assert test.refwfn == (0b0101, 0b0110)

    ciwfn = CIWavefunction(2, 4)
    test.assign_refwfn(ciwfn)
    assert test.refwfn == ciwfn

    with pytest.raises(TypeError):
        test.assign_refwfn([ciwfn, ciwfn])
    with pytest.raises(TypeError):
        test.assign_refwfn("0101")
    with pytest.raises(TypeError):
        test.assign_refwfn(np.array([0b0101, 0b0110]))


def test_system_assign_eqn_weights():
    """Test ProjectedSchrodinger.assign_eqn_weights."""
    test = skip_init(ProjectedSchrodinger)
    test.wfn = CIWavefunction(2, 4)
    test.pspace_exc_orders=None
    test.assign_pspace()
    test.assign_refwfn()
    test.indices_component_params = ComponentParameterIndices()
    test.indices_component_params[test.wfn] = np.arange(test.wfn.nparams)
    test.energy_type = "compute"
    test.assign_constraints()

    test.assign_eqn_weights()
    assert np.allclose(test.eqn_weights, np.array([1, 1, 1, 1, 1, 1, 6]))

    test.assign_eqn_weights(np.array([0, 0, 0, 0, 0, 0, 0], dtype=float))
    assert np.allclose(test.eqn_weights, np.array([0, 0, 0, 0, 0, 0, 0]))

    with pytest.raises(TypeError):
        test.assign_eqn_weights([1, 1, 1, 1, 1, 1, 1])

    with pytest.raises(ValueError):
        test.assign_eqn_weights(np.array([0, 0, 0, 0, 0, 0, 0, 0]))


def test_system_assign_constraints():
    """Test ProjectedSchrodinger.assign_constraints."""
    test = skip_init(ProjectedSchrodinger)
    test.wfn = CIWavefunction(2, 4)
    test.assign_refwfn(0b0101)
    test.indices_component_params = ComponentParameterIndices()
    test.indices_component_params[test.wfn] = np.arange(test.wfn.nparams)
    test.energy_type = "compute"

    test.assign_constraints()
    assert isinstance(test.constraints, list)
    assert len(test.constraints) == 1
    assert isinstance(test.constraints[0], NormConstraint)
    assert test.constraints[0].wfn == test.wfn
    assert test.constraints[0].refwfn == (0b0101,)

    norm_constraint = NormConstraint(test.wfn, param_selection=test.indices_component_params)
    test.assign_constraints(norm_constraint)
    assert isinstance(test.constraints, list)
    assert len(test.constraints) == 1
    assert isinstance(test.constraints[0], NormConstraint)
    assert test.constraints[0].wfn == test.wfn
    assert test.constraints[0].refwfn == (0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010)

    with pytest.raises(TypeError):
        test.assign_constraints(lambda x: None)
    with pytest.raises(TypeError):
        test.assign_constraints(np.array(norm_constraint))
    with pytest.raises(TypeError):
        test.assign_constraints([norm_constraint, lambda x: None])
    norm_constraint.indices_component_params = ComponentParameterIndices()
    norm_constraint.indices_component_params[norm_constraint.wfn] = np.arange(
        norm_constraint.wfn.nparams - 1
    )
    with pytest.raises(ValueError):
        test.assign_constraints(norm_constraint)
    with pytest.raises(ValueError):
        test.assign_constraints([norm_constraint])


def test_num_eqns():
    """Test SystemEquation.num_eqns."""
    test = skip_init(ProjectedSchrodinger)
    test.wfn = CIWavefunction(2, 4)
    test.assign_refwfn()
    test.indices_component_params = ComponentParameterIndices()
    test.indices_component_params[test.wfn] = np.arange(test.wfn.nparams)
    test.energy_type = "compute"
    test.assign_constraints()
    test.assign_pspace((0b0101, 0b1010))
    assert test.num_eqns == 3


def test_system_objective():
    """Test SystemEquation.objective."""
    wfn = CIWavefunction(2, 4)

    one_int = np.random.rand(2, 2)
    one_int = one_int + one_int.T
    two_int = np.random.rand(2, 2, 2, 2)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    ham = RestrictedMolecularHamiltonian(one_int, two_int)

    weights = np.random.rand(7)
    # check assignment
    test = ProjectedSchrodinger(wfn, ham, eqn_weights=weights)
    test.objective(np.arange(1, 7, dtype=float))
    np.allclose(wfn.params, np.arange(1, 7))
    # check save
    test.step_save = False
    test.objective(np.arange(1, 7, dtype=float))
    np.allclose(wfn.params, np.arange(1, 7))
    # check print
    test = ProjectedSchrodinger(wfn, ham, step_print=True, constraints=[])
    test.objective(np.arange(1, 7, dtype=float))

    # <SD1 | H | Psi> - E <SD | Psi>
    ciref = CIWavefunction(2, 4)
    ciref.assign_params(np.random.rand(6))
    for refwfn in [0b0101, [0b0101, 0b1010], ciref]:
        guess = np.random.rand(7)
        # computed energy
        test = ProjectedSchrodinger(wfn, ham, eqn_weights=weights, refwfn=refwfn)
        wfn.assign_params(guess[:6])
        if refwfn == 0b0101:
            norm_answer = weights[-1] * (wfn.get_overlap(0b0101) ** 2 - 1)
        elif refwfn == [0b0101, 0b1010]:
            norm_answer = weights[-1] * (
                wfn.get_overlap(0b0101) ** 2 + wfn.get_overlap(0b1010) ** 2 - 1
            )
        elif refwfn == ciref:
            norm_answer = weights[-1] * (
                sum(ciref.get_overlap(sd) * wfn.get_overlap(sd) for sd in ciref.sds) - 1
            )

        objective = test.objective(guess[:6])
        for eqn, sd, weight in zip(
            objective[:-1], [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]
        ):
            assert np.allclose(
                eqn,
                weight
                * (
                    (ham.integrate_sd_wfn(sd, wfn))
                    - test.get_energy_one_proj(refwfn) * wfn.get_overlap(sd)
                ),
            )
        assert np.allclose(objective[-1], norm_answer)

        # variable energy
        test = ProjectedSchrodinger(
            wfn, ham, energy=1.0, energy_type="variable", eqn_weights=weights, refwfn=refwfn
        )
        objective = test.objective(guess)
        for eqn, sd, weight in zip(
            objective[:-1], [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]
        ):
            assert np.allclose(
                eqn, weight * ((ham.integrate_sd_wfn(sd, wfn)) - guess[-1] * wfn.get_overlap(sd))
            )
        assert np.allclose(objective[-1], norm_answer)

        # fixed energy
        test = ProjectedSchrodinger(
            wfn, ham, energy=1.0, energy_type="fixed", eqn_weights=weights, refwfn=refwfn
        )
        objective = test.objective(guess[:6])
        for eqn, sd, weight in zip(
            objective[:-1], [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]
        ):
            assert np.allclose(
                eqn, weight * ((ham.integrate_sd_wfn(sd, wfn)) - 1.0 * wfn.get_overlap(sd))
            )
        assert np.allclose(objective[-1], norm_answer)


def test_system_jacobian():
    """Test SystemEquation.jacobian with only wavefunction parameters active."""
    wfn = CIWavefunction(2, 4)

    one_int = np.random.rand(2, 2)
    one_int = one_int + one_int.T
    two_int = np.random.rand(2, 2, 2, 2)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    ham = RestrictedMolecularHamiltonian(one_int, two_int)

    weights = np.random.rand(7)

    # check assignment
    test = ProjectedSchrodinger(wfn, ham, eqn_weights=weights)
    test.jacobian(np.arange(1, 7, dtype=float))
    np.allclose(wfn.params, np.arange(1, 7))

    # df_1/dx_1 = d/dx_1 <SD_1 | H | Psi> - dE/dx_1 <SD_1 | Psi> - E d/dx_1 <SD_1 | Psi>
    ciref = CIWavefunction(2, 4)
    ciref.assign_params(np.random.rand(6))
    for refwfn in [0b0101, [0b0101, 0b1010], ciref]:
        guess = np.random.rand(7)
        # computed energy
        test = ProjectedSchrodinger(wfn, ham, eqn_weights=weights, refwfn=refwfn)
        wfn.assign_params(guess[:6])
        if refwfn == 0b0101:
            norm_answer = [
                weights[-1] * (2 * wfn.get_overlap(0b0101) * wfn.get_overlap(0b0101, deriv=i))
                for i in range(6)
            ]
        elif refwfn == [0b0101, 0b1010]:
            norm_answer = [
                weights[-1]
                * (
                    2 * wfn.get_overlap(0b0101) * wfn.get_overlap(0b0101, deriv=i)
                    + 2 * wfn.get_overlap(0b1010) * wfn.get_overlap(0b1010, deriv=i)
                )
                for i in range(6)
            ]
        elif refwfn == ciref:
            norm_answer = [
                weights[-1]
                * (sum(ciref.get_overlap(sd) * wfn.get_overlap(sd, deriv=i) for sd in ciref.sds))
                for i in range(6)
            ]

        jacobian = test.jacobian(guess[:6])
        for eqn, sd, weight in zip(
            jacobian[:-1], [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]
        ):
            assert np.allclose(
                eqn,
                weight
                * (
                    (ham.integrate_sd_wfn(sd, wfn, wfn_deriv=np.arange(6)))
                    - test.get_energy_one_proj(refwfn, deriv=True) * wfn.get_overlap(sd)
                    - test.get_energy_one_proj(refwfn) * wfn.get_overlap(sd, deriv=np.arange(6))
                ),
            )
        assert np.allclose(jacobian[-1], norm_answer)

        # variable energy
        test = ProjectedSchrodinger(
            wfn, ham, energy=3.0, energy_type="variable", eqn_weights=weights, refwfn=refwfn
        )
        jacobian = test.jacobian(guess, normalize=False, assign=True)
        for eqn, sd, weight in zip(
            jacobian[:-1], [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]
        ):
            assert np.allclose(
                eqn[:6],
                weight
                * (
                    (ham.integrate_sd_wfn(sd, wfn, wfn_deriv=np.arange(6)))
                    - guess[-1] * wfn.get_overlap(sd, deriv=np.arange(6))
                ),
            )
            assert np.allclose(eqn[6], -weight * wfn.get_overlap(sd))
        assert np.allclose(jacobian[-1], norm_answer + [0.0])

        # fixed energy
        test = ProjectedSchrodinger(
            wfn, ham, energy=1.0, energy_type="fixed", eqn_weights=weights, refwfn=refwfn
        )
        jacobian = test.jacobian(guess[:6])
        for eqn, sd, weight in zip(
            jacobian[:-1], [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]
        ):
            assert np.allclose(
                eqn,
                weight
                * (
                    (ham.integrate_sd_wfn(sd, wfn, wfn_deriv=np.arange(6)))
                    - 0.0 * wfn.get_overlap(sd)
                    - 1 * wfn.get_overlap(sd, deriv=np.arange(6))
                ),
            )
        assert np.allclose(jacobian[-1], norm_answer)


def test_system_jacobian_active_ciref():
    """Test SystemEquation.jacobian with CIWavefunction reference with active parameters."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))

    one_int = np.random.rand(2, 2)
    one_int = one_int + one_int.T
    two_int = np.random.rand(2, 2, 2, 2)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int
    ham = RestrictedMolecularHamiltonian(one_int, two_int)

    weights = np.random.rand(7)

    ciref = CIWavefunction(2, 4)
    ciref.assign_params(np.random.rand(6))

    # computed energy
    test = ProjectedSchrodinger(
        wfn,
        ham,
        eqn_weights=weights,
        refwfn=ciref,
        param_selection=((wfn, np.ones(6, dtype=bool)), (ciref, np.ones(6, dtype=bool))),
    )

    jacobian = test.jacobian(np.random.rand(12), assign=True)
    for eqn, sd, weight in zip(
        jacobian[:-1], [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010], weights[:-1]
    ):
        results = np.zeros(12)
        results[:6] += ham.integrate_sd_wfn(sd, wfn, wfn_deriv=np.arange(6))
        results -= test.get_energy_one_proj(ciref, deriv=True) * wfn.get_overlap(sd)
        results[:6] -= test.get_energy_one_proj(ciref) * wfn.get_overlap(sd, deriv=np.arange(6))

        assert np.allclose(eqn, weight * results)
    assert np.allclose(
        jacobian[-1],
        [
            weights[-1]
            * (sum(ciref.get_overlap(sd) * wfn.get_overlap(sd, deriv=i) for sd in ciref.sds))
            for i in range(6)
        ]
        + [
            weights[-1]
            * (sum(ciref.get_overlap(sd, deriv=i) * wfn.get_overlap(sd) for sd in ciref.sds))
            for i in range(6)
        ],
    )
