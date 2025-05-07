"""Test fanpy.ham.senzero."""

from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.ham.senzero import SeniorityZeroHamiltonian
from fanpy.tools.math_tools import unitary_matrix
from fanpy.tools.slater import get_seniority
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np

import pytest

from utils import find_datafile


def test_integrate_sd_sd_decomposed_trivial():
    """Test SeniorityZeroHamiltonian.integrate_sd_sd for trivial cases."""
    one_int = np.random.rand(4, 4)
    two_int = np.random.rand(4, 4, 4, 4)
    test = SeniorityZeroHamiltonian(one_int, two_int)

    with pytest.raises(NotImplementedError):
        test.integrate_sd_sd(0b00010001, 0b01000100, deriv=0)

    assert np.allclose((0, 0, 0), test.integrate_sd_sd_decomposed(0b00010001, 0b00100001))
    assert np.allclose(0, test.integrate_sd_sd(0b00010001, 0b00100001))
    assert np.allclose((0, 0, 0), test.integrate_sd_sd_decomposed(0b00100001, 0b00010001))
    assert np.allclose((0, 0, 0), test.integrate_sd_sd_decomposed(0b01010101, 0b00010001))
    assert np.allclose(0, test.integrate_sd_sd(0b01010101, 0b00010001))
    assert np.allclose((0, 0, 0), test.integrate_sd_sd_decomposed(0b11001100, 0b00110011))
    assert np.allclose((0, two_int[0, 0, 1, 1], 0), test.integrate_sd_sd_decomposed(0b00010001, 0b00100010))

    with pytest.raises(TypeError):
        test.integrate_sd_sd(0b110001, "1")
    with pytest.raises(TypeError):
        test.integrate_sd_sd("1", 0b101010)


def test_integrate_sd_sd_h2_631gdp():
    """Test SeniorityZeroHamiltonian.integrate_sd_sd using H2 HF/6-31G** orbitals.

    Compare CI matrix with the PySCF result.

    """
    one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    full_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    test_ham = SeniorityZeroHamiltonian(one_int, two_int)

    ref_pspace = np.load(find_datafile("data_h2_hf_631gdp_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        sd1 = int(sd1)
        if get_seniority(sd1, one_int.shape[0]) != 0:
            continue
        for j, sd2 in enumerate(ref_pspace):
            sd2 = int(sd2)
            if get_seniority(sd2, one_int.shape[0]) != 0:
                continue
            assert np.allclose(full_ham.integrate_sd_sd(sd1, sd2), test_ham.integrate_sd_sd(sd1, sd2))


def test_integrate_sd_sd_lih_631g_full():
    """Test SeniorityZeroHamiltonian.integrate_sd_sd using LiH HF/6-31G orbitals.

    Compared to all of the CI matrix.

    """
    one_int = np.load(find_datafile("data_lih_hf_631g_oneint.npy"))
    two_int = np.load(find_datafile("data_lih_hf_631g_twoint.npy"))
    full_ham = RestrictedMolecularHamiltonian(one_int, two_int)
    test_ham = SeniorityZeroHamiltonian(one_int, two_int)

    ref_pspace = np.load(find_datafile("data_lih_hf_631g_civec.npy"))

    for i, sd1 in enumerate(ref_pspace):
        sd1 = int(sd1)
        if get_seniority(sd1, one_int.shape[0]) != 0:
            continue
        for j, sd2 in enumerate(ref_pspace):
            sd2 = int(sd2)
            if get_seniority(sd2, one_int.shape[0]) != 0:
                continue
            assert np.allclose(full_ham.integrate_sd_sd(sd1, sd2), test_ham.integrate_sd_sd(sd1, sd2))


def test_integrate_sd_wfn_2e():
    """Test SeniorityZeroHamiltonian.integrate_sd_wfn with 2 electron wavefunction."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    test_wfn = type(
        "Temporary class with desired overlap",
        (object,),
        {"get_overlap": lambda sd, deriv=None: 1 if sd == 0b0101 else 2 if sd == 0b1010 else 0},
    )

    assert np.allclose((0, 0, 0), ham.integrate_sd_wfn(0b1001, test_wfn, components=True))
    assert np.allclose(0, ham.integrate_sd_wfn(0b1001, test_wfn, components=False))

    one_energy, coulomb, exchange = ham.integrate_sd_wfn(0b0101, test_wfn, components=True)
    assert one_energy == 1 * 1 + 1 * 1
    assert coulomb == 1 * 5 + 2 * 8
    assert exchange == 0

    one_energy, coulomb, exchange = ham.integrate_sd_wfn(0b1010, test_wfn, components=True)
    assert one_energy == 2 * 4 + 2 * 4
    assert coulomb == 1 * 17 + 2 * 20
    assert exchange == 0

    with pytest.raises(ValueError):
        ham.integrate_sd_wfn(0b0101, test_wfn, wfn_deriv=0, ham_deriv=0)


def test_integrate_sd_wfn_4e():
    """Test SeniorityZeroHamiltonian.integrate_sd_wfn with 4 electron wavefunction."""
    one_int = np.arange(1, 10, dtype=float).reshape(3, 3)
    two_int = np.arange(1, 82, dtype=float).reshape(3, 3, 3, 3)
    ham = SeniorityZeroHamiltonian(one_int, two_int)
    ham_full = RestrictedMolecularHamiltonian(one_int, two_int)
    test_wfn = type(
        "Temporary class with desired overlap",
        (object,),
        {
            "get_overlap": lambda sd, deriv=None: (
                1 if sd == 0b011011 else 2 if sd == 0b101101 else 3 if sd == 0b110110 else 0
            )
        },
    )

    assert np.allclose(ham.integrate_sd_wfn(0b011011, test_wfn), ham_full.integrate_sd_wfn(0b011011, test_wfn))

    assert np.allclose(ham.integrate_sd_wfn(0b101101, test_wfn), ham_full.integrate_sd_wfn(0b101101, test_wfn))

    assert np.allclose(ham.integrate_sd_wfn(0b110110, test_wfn), ham_full.integrate_sd_wfn(0b110110, test_wfn))

    with pytest.raises(ValueError):
        ham.integrate_sd_wfn(0b0101, test_wfn, wfn_deriv=0, ham_deriv=0)
    with pytest.raises(TypeError):
        ham.integrate_sd_wfn("1", test_wfn)

    with pytest.raises(TypeError):
        ham.integrate_sd_wfn(0b01101010, test_wfn, ham_deriv=np.arange(10).tolist())
    with pytest.raises(TypeError):
        ham.integrate_sd_wfn(0b01101010, test_wfn, ham_deriv=np.arange(10).astype(float))
    with pytest.raises(TypeError):
        ham.integrate_sd_wfn(0b01101010, test_wfn, ham_deriv=np.arange(10).reshape(2, 5))
    with pytest.raises(ValueError):
        bad_indices = np.arange(10)
        bad_indices[0] = -1
        ham.integrate_sd_wfn(0b01101010, test_wfn, ham_deriv=bad_indices)
    with pytest.raises(ValueError):
        bad_indices = np.arange(10)
        bad_indices[0] = 10
        ham.integrate_sd_wfn(0b01101010, test_wfn, ham_deriv=bad_indices)


def test_integrate_sd_wfn_deriv_fdiff():
    """Test SeniorityZeroHamiltonian.integrate_sd_wfn_deriv with finite difference."""
    nd = pytest.importorskip("numdifftools")

    wfn = CIWavefunction(3, 6)
    wfn.assign_params(np.random.rand(*wfn.params.shape))

    one_int = np.random.rand(3, 3)
    one_int = one_int + one_int.T

    two_int = np.random.rand(3, 3, 3, 3)
    two_int = np.einsum("ijkl->jilk", two_int) + two_int
    two_int = np.einsum("ijkl->klij", two_int) + two_int

    ham = SeniorityZeroHamiltonian(one_int, two_int, update_prev_params=True)
    original = np.random.rand(ham.params.size)
    step1 = np.random.rand(ham.params.size)
    step2 = np.random.rand(ham.params.size)
    ham.assign_params(original.copy())
    ham.assign_params(original + step1)
    ham.assign_params(original + step1 + step2)

    temp_ham = SeniorityZeroHamiltonian(one_int, two_int)
    temp_ham.orb_rotate_matrix(unitary_matrix(original).dot(unitary_matrix(step1)).dot(unitary_matrix(step2)))
    assert np.allclose(ham.one_int, temp_ham.one_int)
    assert np.allclose(ham.two_int, temp_ham.two_int)

    def objective(params):
        temp_ham = SeniorityZeroHamiltonian(one_int, two_int)
        temp_ham.orb_rotate_matrix(unitary_matrix(original).dot(unitary_matrix(step1)).dot(unitary_matrix(step2)))
        temp_ham.set_ref_ints()
        temp_ham._prev_params = ham.params.copy()
        temp_ham.assign_params(params.copy())
        return temp_ham.integrate_sd_wfn(wfn.sds[0], wfn)

    assert np.allclose(
        nd.Gradient(objective)(ham.params),
        ham.integrate_sd_wfn(wfn.sds[0], wfn, ham_deriv=np.arange(ham.nparams)),
    )
