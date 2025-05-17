"""Test fanpy.wavefunction.composite.nonorth."""
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.composite.nonorth import NonorthWavefunction

import numpy as np

import pytest

from utils import find_datafile, skip_init


class TempWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abstract class structure."""

    _spin = None
    _seniority = None

    def __init__(self):
        """Do nothing."""
        pass

    def get_overlap(self):
        """Do nothing."""
        pass

    @property
    def spin(self):
        """Return spin of wavefunction."""
        return self._spin

    @property
    def seniority(self):
        """Return seniority of wavefunction."""
        return self._seniority

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction."""
        if params is None:
            params = np.identity(10)
        super().assign_params(params=params, add_noise=add_noise)


def test_nonorth_assign_params():
    """Tests NonorthWavefunction.assign_params."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 4
    test.nspin = 10
    test.memory = 10
    test._cache_fns = {}

    test_wfn = TempWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.memory = 10
    test_wfn._spin = 2
    test.assign_wfn(test_wfn)

    test.assign_params(None)
    assert isinstance(test.params, tuple)
    assert len(test.params) == 1
    assert np.allclose(test.params[0], np.eye(5, 6))

    test_params = np.random.rand(5, 6)
    with pytest.raises(TypeError):
        test.assign_params((test_params,) * 3)
    with pytest.raises(TypeError):
        test.assign_params([])
    with pytest.raises(TypeError):
        test.assign_params({0: test_params})
    with pytest.raises(TypeError):
        test.assign_params([test_params.tolist()])
    with pytest.raises(TypeError):
        test.assign_params([np.random.rand(5, 6, 1)])
    with pytest.raises(ValueError):
        test.assign_params([np.random.rand(6, 5)])
    with pytest.raises(ValueError):
        test.assign_params([np.random.rand(9, 9)])
    with pytest.raises(ValueError):
        test.assign_params([np.random.rand(5, 6), np.random.rand(5, 5)])
    with pytest.raises(ValueError):
        test.assign_params([np.random.rand(5, 5), np.random.rand(5, 6)])

    test.assign_params(test_params)
    assert isinstance(test.params, tuple)
    assert len(test.params) == 1
    assert np.allclose(test.params[0], test_params)
    test_params_alpha = np.random.rand(5, 6)
    test_params_beta = np.random.rand(5, 6)
    test.assign_params([test_params_alpha, test_params_beta])
    assert isinstance(test.params, tuple)
    assert len(test.params) == 2
    assert np.allclose(test.params[0], test_params_alpha)
    assert np.allclose(test.params[1], test_params_beta)

    test_params = np.random.rand(10, 12)
    test.assign_params(test_params)
    assert isinstance(test.params, tuple)
    assert len(test.params) == 1
    assert np.allclose(test.params[0], test_params)

    test_params = np.random.rand(10, 12)
    test.assign_params(test_params.copy(), add_noise=True)
    assert np.all(np.abs(test.params - test_params) <= 0.2 / 120)
    assert not np.allclose(test.params[0], test_params)

    test_params = np.random.rand(10, 12).astype(complex)
    test.assign_params(test_params.copy(), add_noise=True)
    assert np.all(np.abs(np.real(test.params - test_params)) <= 0.2 / 120)
    assert np.all(np.abs(np.imag(test.params - test_params)) <= 0.01 * 0.1 / 120)
    assert not np.allclose(test.params[0], test_params)


def test_nonorth_spin():
    """Test NonorthWavefunction.spin."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 4
    test.nspin = 10
    test.memory = 10
    test._cache_fns = {}

    test_wfn = TempWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.memory = 10
    test_wfn._spin = 2
    test.assign_wfn(test_wfn)

    # restricted
    test.assign_params(np.random.rand(5, 6))
    assert test.spin == 2
    # restricted
    test.assign_params([np.random.rand(5, 6), np.random.rand(5, 6)])
    assert test.spin == 2
    # generalized
    test.assign_params(np.random.rand(10, 12))
    assert test.spin is None


def test_nonorth_seniority():
    """Test NonorthWavefunction.seniority."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 4
    test.nspin = 10
    test.memory = 10
    test._cache_fns = {}

    test_wfn = TempWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.memory = 10
    test_wfn._seniority = 2
    test.assign_wfn(test_wfn)

    # restricted
    test.assign_params(np.random.rand(5, 6))
    assert test.seniority == 2
    # restricted
    test.assign_params([np.random.rand(5, 6), np.random.rand(5, 6)])
    assert test.seniority == 2
    # generalized
    test.assign_params(np.random.rand(10, 12))
    assert test.seniority is None


def test_nonorth_default_params():
    """Test NonorthWavefunction.default_params."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 4
    test.nspin = 10
    test.memory = 10
    test._cache_fns = {}

    test_wfn = TempWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.memory = 10
    test.assign_wfn(test_wfn)
    test.assign_params()

    assert isinstance(test.params, tuple)
    assert len(test.params) == 1
    assert np.allclose(test.params[0], np.eye(5, 6))


def test_nonorth_nparams():
    """Test NonorthWavefunction.nparams."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 4
    test.nspin = 10
    test.memory = 10
    test._cache_fns = {}

    test_wfn = TempWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.memory = 10
    test.assign_wfn(test_wfn)

    # restricted
    test.assign_params(np.random.rand(5, 6))
    assert test.nparams == 30
    # unrestricted
    test.assign_params([np.random.rand(5, 6), np.random.rand(5, 6)])
    assert test.nparams == 60
    # generalized
    test.assign_params(np.random.rand(10, 12))
    assert test.nparams == 120


def test_nonorth_params_shape():
    """Test NonorthWavefunction.params_shape."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 4
    test.nspin = 10
    test.memory = 10
    test._cache_fns = {}

    test_wfn = TempWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.memory = 10
    test.assign_wfn(test_wfn)

    # restricted
    test.assign_params(np.random.rand(5, 6))
    assert test.params_shape == ((5, 6),)
    # unrestricted
    test.assign_params([np.random.rand(5, 6), np.random.rand(5, 6)])
    assert test.params_shape == ((5, 6), (5, 6))
    # generalized
    test.assign_params(np.random.rand(10, 12))
    assert test.params_shape == ((10, 12),)


def test_nonorth_orbtype():
    """Test NonorthWavefunction.orbtype."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 4
    test.nspin = 10
    test.memory = 10
    test._cache_fns = {}

    test_wfn = TempWavefunction()
    test_wfn.nspin = 12
    test_wfn.nelec = 4
    test_wfn.memory = 10
    test.assign_wfn(test_wfn)

    # restricted
    test.assign_params(np.random.rand(5, 6))
    assert test.orbtype == "restricted"
    # unrestricted
    test.assign_params([np.random.rand(5, 6), np.random.rand(5, 6)])
    assert test.orbtype == "unrestricted"
    # generalized
    test.assign_params(np.random.rand(10, 12))
    assert test.orbtype == "generalized"


def test_nonorth_olp_generalized():
    """Test NonorthWavefunction._olp for generalized orbitals."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 2
    test.nspin = 4
    test.memory = 10
    test.assign_wfn(CIWavefunction(2, 4, memory=10))
    test._cache_fns = {}
    test.wfn.params = np.arange(1, 7)
    wfn_sd_coeff = {0b0101: 1, 0b0110: 2, 0b1100: 3, 0b0011: 4, 0b1001: 5, 0b1010: 6}

    test.params = [np.arange(1, 17).reshape(4, 4)]
    test.enable_cache()
    # 0b0101 uses [[1, 2, 3, 4],
    #              [9, 10, 11, 12]]
    assert np.isclose(
        test._olp(0b0101),
        (
            (1 * 10 - 2 * 9) * wfn_sd_coeff[0b0011]
            + (1 * 11 - 3 * 9) * wfn_sd_coeff[0b0101]
            + (1 * 12 - 4 * 9) * wfn_sd_coeff[0b1001]
            + (2 * 11 - 3 * 10) * wfn_sd_coeff[0b0110]
            + (2 * 12 - 4 * 10) * wfn_sd_coeff[0b1010]
            + (3 * 12 - 4 * 11) * wfn_sd_coeff[0b1100]
        ),
        rtol=0,
        atol=1e-12,
    )
    # 0b0110 uses [[5, 6, 7, 8],
    #              [9, 10, 11, 12]]
    assert np.isclose(
        test._olp(0b0110),
        (
            (5 * 10 - 6 * 9) * wfn_sd_coeff[0b0011]
            + (5 * 11 - 7 * 9) * wfn_sd_coeff[0b0101]
            + (5 * 12 - 8 * 9) * wfn_sd_coeff[0b1001]
            + (6 * 11 - 7 * 10) * wfn_sd_coeff[0b0110]
            + (6 * 12 - 8 * 10) * wfn_sd_coeff[0b1010]
            + (7 * 12 - 8 * 11) * wfn_sd_coeff[0b1100]
        ),
        rtol=0,
        atol=1e-12,
    )
    # 0b1100 uses [[9, 10, 11, 12],
    #              [13, 14, 15, 16]]
    assert np.isclose(
        test._olp(0b1100),
        (
            (9 * 14 - 10 * 13) * wfn_sd_coeff[0b0011]
            + (9 * 15 - 11 * 13) * wfn_sd_coeff[0b0101]
            + (9 * 16 - 12 * 13) * wfn_sd_coeff[0b1001]
            + (10 * 15 - 11 * 14) * wfn_sd_coeff[0b0110]
            + (10 * 16 - 12 * 14) * wfn_sd_coeff[0b1010]
            + (11 * 16 - 12 * 15) * wfn_sd_coeff[0b1100]
        ),
        rtol=0,
        atol=1e-12,
    )


def test_nonorth_olp_unrestricted():
    """Test NonorthWavefunction._olp for unrestricted orbitals."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 2
    test.nspin = 4
    test.memory = 10
    test.assign_wfn(CIWavefunction(2, 4, memory=10))
    test._cache_fns = {}
    test.wfn.params = np.arange(1, 7)
    wfn_sd_coeff = {0b0101: 1, 0b0110: 2, 0b1100: 3, 0b0011: 4, 0b1001: 5, 0b1010: 6}

    test.params = [np.array([[1, 2], [5, 6]]), np.array([[11, 12], [15, 16]])]
    test.enable_cache()
    # 0b0101 uses [[1, 2, 0, 0],
    #              [0, 0, 11, 12]]
    assert np.isclose(
        test._olp(0b0101),
        (
            (1 * 0 - 2 * 0) * wfn_sd_coeff[0b0011]
            + (1 * 11 - 0 * 0) * wfn_sd_coeff[0b0101]
            + (1 * 12 - 0 * 0) * wfn_sd_coeff[0b1001]
            + (2 * 11 - 0 * 0) * wfn_sd_coeff[0b0110]
            + (2 * 12 - 0 * 0) * wfn_sd_coeff[0b1010]
            + (0 * 12 - 0 * 11) * wfn_sd_coeff[0b1100]
        ),
        rtol=0,
        atol=1e-12,
    )
    # 0b0110 uses [[5, 6, 0, 0],
    #              [0, 0, 11, 12]]
    assert np.isclose(
        test._olp(0b0110),
        (
            (5 * 0 - 6 * 0) * wfn_sd_coeff[0b0011]
            + (5 * 11 - 0 * 0) * wfn_sd_coeff[0b0101]
            + (5 * 12 - 0 * 0) * wfn_sd_coeff[0b1001]
            + (6 * 11 - 0 * 0) * wfn_sd_coeff[0b0110]
            + (6 * 12 - 0 * 0) * wfn_sd_coeff[0b1010]
            + (0 * 12 - 0 * 11) * wfn_sd_coeff[0b1100]
        ),
        rtol=0,
        atol=1e-12,
    )
    # 0b1100 uses [[0, 0, 11, 12],
    #              [0, 0, 15, 16]]
    assert np.isclose(
        test._olp(0b1100),
        (
            (0 * 0 - 0 * 0) * wfn_sd_coeff[0b0011]
            + (0 * 15 - 11 * 0) * wfn_sd_coeff[0b0101]
            + (0 * 16 - 12 * 0) * wfn_sd_coeff[0b1001]
            + (0 * 15 - 11 * 0) * wfn_sd_coeff[0b0110]
            + (0 * 16 - 12 * 0) * wfn_sd_coeff[0b1010]
            + (11 * 16 - 12 * 15) * wfn_sd_coeff[0b1100]
        ),
        rtol=0,
        atol=1e-12,
    )


def test_nonorth_olp_restricted():
    """Test NonorthWavefunction._olp for restricted orbitals."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 2
    test.nspin = 4
    test.memory = 10
    test.assign_wfn(CIWavefunction(2, 4, memory=10))
    test._cache_fns = {}
    test.wfn.params = np.arange(1, 7)
    wfn_sd_coeff = {0b0101: 1, 0b0110: 2, 0b1100: 3, 0b0011: 4, 0b1001: 5, 0b1010: 6}

    test.params = [np.array([[1, 2], [5, 6]])]
    test.enable_cache()
    # 0b0101 uses [[1, 2, 0, 0],
    #              [0, 0, 1, 2]]
    assert np.isclose(
        test._olp(0b0101),
        (
            (1 * 0 - 2 * 0) * wfn_sd_coeff[0b0011]
            + (1 * 1 - 0 * 0) * wfn_sd_coeff[0b0101]
            + (1 * 2 - 0 * 0) * wfn_sd_coeff[0b1001]
            + (2 * 1 - 0 * 0) * wfn_sd_coeff[0b0110]
            + (2 * 2 - 0 * 0) * wfn_sd_coeff[0b1010]
            + (0 * 2 - 0 * 1) * wfn_sd_coeff[0b1100]
        ),
        rtol=0,
        atol=1e-12,
    )
    # 0b0110 uses [[5, 6, 0, 0],
    #              [0, 0, 1, 2]]
    assert np.isclose(
        test._olp(0b0110),
        (
            (5 * 0 - 6 * 0) * wfn_sd_coeff[0b0011]
            + (5 * 1 - 0 * 0) * wfn_sd_coeff[0b0101]
            + (5 * 2 - 0 * 0) * wfn_sd_coeff[0b1001]
            + (6 * 1 - 0 * 0) * wfn_sd_coeff[0b0110]
            + (6 * 2 - 0 * 0) * wfn_sd_coeff[0b1010]
            + (0 * 1 - 0 * 2) * wfn_sd_coeff[0b1100]
        ),
        rtol=0,
        atol=1e-12,
    )
    # 0b1100 uses [[0, 0, 1, 2],
    #              [0, 0, 5, 6]]
    assert np.isclose(
        test._olp(0b1100),
        (
            (0 * 0 - 0 * 0) * wfn_sd_coeff[0b0011]
            + (0 * 1 - 0 * 5) * wfn_sd_coeff[0b0101]
            + (0 * 2 - 0 * 6) * wfn_sd_coeff[0b1001]
            + (0 * 1 - 0 * 5) * wfn_sd_coeff[0b0110]
            + (0 * 2 - 0 * 6) * wfn_sd_coeff[0b1010]
            + (1 * 6 - 5 * 2) * wfn_sd_coeff[0b1100]
        ),
        rtol=0,
        atol=1e-12,
    )


def test_nonorth_olp_deriv_generalized():
    """Test NonorthWavefunction._olp_deriv for generalized orbitals."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 2
    test.nspin = 4
    test.memory = 10
    test.assign_wfn(CIWavefunction(2, 4, memory=10))
    test._cache_fns = {}
    test.wfn.params = np.arange(1, 7)
    wfn_sd_coeff = {0b0101: 1, 0b0110: 2, 0b1100: 3, 0b0011: 4, 0b1001: 5, 0b1010: 6}

    # generalized
    test.params = [np.arange(1, 17).reshape(4, 4)]
    test.enable_cache()
    # 0b0101 uses [[1, 2, 3, 4],
    #              [9, 10, 11, 12]]
    assert np.isclose(
        test._olp_deriv(0b0101, 0),
        (10 * wfn_sd_coeff[0b0011] + 11 * wfn_sd_coeff[0b0101] + 12 * wfn_sd_coeff[0b1001]),
        rtol=0,
        atol=1e-12,
    )
    assert np.isclose(test._olp_deriv(0b0101, 6), 0, rtol=0, atol=1e-12)
    assert np.isclose(
        test._olp_deriv(0b0101, 8),
        (-2 * wfn_sd_coeff[0b0011] + -3 * wfn_sd_coeff[0b0101] + -4 * wfn_sd_coeff[0b1001]),
        rtol=0,
        atol=1e-12,
    )


def test_nonorth_olp_deriv_unrestricted():
    """Test NonorthWavefunction._olp_deriv for unrestricted orbitals."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 2
    test.nspin = 4
    test.memory = 10
    test.assign_wfn(CIWavefunction(2, 4, memory=10))
    test._cache_fns = {}
    test.wfn.params = np.arange(1, 7)
    wfn_sd_coeff = {0b0101: 1, 0b0110: 2, 0b1100: 3, 0b0011: 4, 0b1001: 5, 0b1010: 6}

    test.params = [np.array([[1, 2], [5, 6]]), np.array([[11, 12], [15, 16]])]
    test.enable_cache()
    # 0b0101 uses [[1, 2, 0, 0],
    #              [0, 0, 11, 12]]
    assert np.isclose(
        test._olp_deriv(0b0101, 0),
        (0 * wfn_sd_coeff[0b0011] + 11 * wfn_sd_coeff[0b0101] + 12 * wfn_sd_coeff[0b1001]),
        rtol=0,
        atol=1e-12,
    )
    assert np.isclose(
        test._olp_deriv(0b0101, 5),
        (1 * wfn_sd_coeff[0b1001] + 2 * wfn_sd_coeff[0b1010] + 0 * wfn_sd_coeff[0b1100]),
        rtol=0,
        atol=1e-12,
    )
    assert np.isclose(test._olp_deriv(0b0101, 6), 0, rtol=0, atol=1e-12)
    assert np.isclose(
        test._olp_deriv(0b0101, 5),
        (1 * wfn_sd_coeff[0b1001] + 2 * wfn_sd_coeff[0b1010] + 0 * wfn_sd_coeff[0b1100]),
        rtol=0,
        atol=1e-12,
    )
    # 0b1100 uses [[0, 0, 11, 12],
    #              [0, 0, 15, 16]]
    assert np.isclose(test._olp_deriv(0b1100, 6), -12 * wfn_sd_coeff[0b1100], rtol=0, atol=1e-12)

    # check trivial
    assert test._olp_deriv(0b1100, 0) == 0


def test_nonorth_olp_deriv_restricted():
    """Test NonorthWavefunction._olp_deriv for restricted orbitals."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 2
    test.nspin = 4
    test.memory = 10
    test.assign_wfn(CIWavefunction(2, 4, memory=10))
    test._cache_fns = {}
    test.wfn.params = np.arange(1, 7)
    wfn_sd_coeff = {0b0101: 1, 0b0110: 2, 0b1100: 3, 0b0011: 4, 0b1001: 5, 0b1010: 6}

    # restricted (one block)
    test.clear_cache()
    test.params = [np.array([[1, 2], [5, 6]])]
    test.enable_cache()
    # 0b0110 uses [[5, 6, 0, 0],
    #              [0, 0, 1, 2]]
    assert np.isclose(
        test._olp_deriv(0b0110, 0),
        (5 * wfn_sd_coeff[0b0101] + 6 * wfn_sd_coeff[0b0110] + 0 * wfn_sd_coeff[0b1100]),
        rtol=0,
        atol=1e-12,
    )

    # restricted (two block)
    # 0b0101 uses [[1, 2, 0, 0],
    #              [0, 0, 1, 2]]
    assert np.isclose(
        test._olp_deriv(0b0101, 0),
        (
            +2 * 1 * wfn_sd_coeff[0b0101]
            + 2 * wfn_sd_coeff[0b1001]
            + 2 * wfn_sd_coeff[0b0110]
            + 0 * wfn_sd_coeff[0b1100]
        ),
        rtol=0,
        atol=1e-12,
    )

    # check trivial
    assert test._olp_deriv(0b1010, 0) == 0
    assert test._olp_deriv(0b1111, 0) == 0
    assert test._olp_deriv(0b1111, 2) == 0

    test = skip_init(NonorthWavefunction)
    test.nelec = 3
    test.nspin = 4
    test.assign_wfn(CIWavefunction(3, 4, memory=10))
    test.wfn.params = np.arange(1, 4)
    wfn_sd_coeff = {0b0111: 1, 0b1011: 2, 0b1101: 3, 0b1110: 4}
    test.params = [np.array([[1, 2], [5, 6]])]

    # restricted (two block)
    # 0b1101 uses [[1, 2, 0, 0],
    #              [0, 0, 1, 2]]
    assert np.isclose(
        test._olp_deriv(0b1101, 0), 3 * 6 * test.wfn.get_overlap(0b1101) + 2 * (1 * -4 + 1 * 6)
    )


def test_nonorth_get_overlap():
    """Test NonorthWavefunction.get_overap."""
    test = skip_init(NonorthWavefunction)
    test.nelec = 2
    test.nspin = 4
    test.memory = 10
    test.assign_wfn(CIWavefunction(2, 4, memory=10))
    test._cache_fns = {}
    test.wfn.params = np.arange(1, 7)
    wfn_sd_coeff = {0b0101: 1, 0b0110: 2, 0b1100: 3, 0b0011: 4, 0b1001: 5, 0b1010: 6}

    # restricted
    test.params = [np.array([[1, 2], [5, 6]])]
    test.enable_cache()
    # 0b0101 uses [[1, 2, 0, 0],
    #              [0, 0, 1, 2]]
    assert np.isclose(
        test.get_overlap(0b0101),
        (
            (1 * 0 - 2 * 0) * wfn_sd_coeff[0b0011]
            + (1 * 1 - 0 * 0) * wfn_sd_coeff[0b0101]
            + (1 * 2 - 0 * 0) * wfn_sd_coeff[0b1001]
            + (2 * 1 - 0 * 0) * wfn_sd_coeff[0b0110]
            + (2 * 2 - 0 * 0) * wfn_sd_coeff[0b1010]
            + (0 * 2 - 0 * 1) * wfn_sd_coeff[0b1100]
        ),
        rtol=0,
        atol=1e-12,
    )
    # 0b0110 uses [[5, 6, 0, 0],
    #              [0, 0, 1, 2]]
    assert np.isclose(
        test.get_overlap(0b0110, (test, np.array([0]))),
        (5 * wfn_sd_coeff[0b0101] + 6 * wfn_sd_coeff[0b0110] + 0 * wfn_sd_coeff[0b1100]),
        rtol=0,
        atol=1e-12,
    )
    with pytest.raises(ValueError):
        assert test.get_overlap(0b0101, (test, np.array([8]))) == 0
    assert test.get_overlap(0b0101, (test, np.array([2]))) == 0

    # unrestricted
    test.clear_cache()
    test.params = [np.array([[1, 2], [5, 6]]), np.array([[11, 12], [15, 16]])]
    test.enable_cache()
    assert np.isclose(test.get_overlap(0b0101, (test, np.array([6]))), 0, rtol=0, atol=1e-12)

    # generalized
    test.clear_cache()
    test.params = [np.arange(1, 17).reshape(4, 4)]
    test.enable_cache()
    assert np.isclose(test.get_overlap(0b0101, (test, np.array([6]))), 0, rtol=0, atol=1e-12)

    with pytest.raises(TypeError):
        test.get_overlap("1")
    with pytest.raises(TypeError):
        test.get_overlap(0b0101, deriv=np.array([0]))
    with pytest.raises(TypeError):
        test.get_overlap(0b0101, deriv=(test.wfn, np.array([0.0])))
    with pytest.raises(TypeError):
        test.get_overlap(0b0101, deriv=(test.wfn, np.array([[0]])))
    with pytest.raises(ValueError):
        test.get_overlap(0b0101, deriv=(CIWavefunction(2, 4), np.array([0])))
    with pytest.raises(ValueError):
        test.get_overlap(0b0101, deriv=(test, np.array([-1])))
    with pytest.raises(ValueError):
        test.get_overlap(0b0101, deriv=(test, np.array([test.nparams])))
    with pytest.raises(NotImplementedError):
        test.get_overlap(0b0101, deriv=(test.wfn, np.array([0])))


def test_nonorth_energy_unitary_transform_hamiltonian():
    """Test the energy of NonorthWavefunction by comparing it to the transformed Hamiltonian.

    While the NonorthWavefunction does not necessarily need to have unitary wavefunction, this test
    requires that the transformation applied to the Hamiltonian is unitary.
    """
    nelec = 4
    nspin = 8

    sds = sd_list(4, 8, num_limit=None, exc_orders=None)

    # transformed hamiltonian
    transform = np.array(
        [
            [0.707106752870, -0.000004484084, 0.000006172115, -0.707106809462],
            [0.707106809472, -0.000004868924, -0.000006704609, 0.707106752852],
            [0.000004942751, 0.707106849959, 0.707106712365, 0.000006630781],
            [0.000004410256, 0.707106712383, -0.707106849949, -0.000006245943],
        ]
    )

    # NOTE: we need to be a little careful with the hamiltonian construction because the integrals
    #       are stored by reference and using the same hamiltonian while transforming it will cause
    #       some headach
    def get_energy(wfn_type, expectation_type):
        """Get energy of the tranformed wavefunction."""
        doci = CIWavefunction(nelec, nspin, seniority=0)
        # optimized parameters for the transformed hamiltonian
        doci.assign_params(
            np.array(
                [
                    8.50413921e-04,
                    2.01842198e-01,
                    -9.57460494e-01,
                    -4.22775180e-02,
                    2.01842251e-01,
                    8.50414717e-04,
                ]
            )
        )

        ham = RestrictedMolecularHamiltonian(
            np.load(find_datafile("data/data_h4_square_hf_sto6g_oneint.npy")),
            np.load(find_datafile("data/data_h4_square_hf_sto6g_twoint.npy")),
        )

        # rotating hamiltonian using orb_rotate_matrix
        if wfn_type == "doci":
            wfn = doci
            ham.orb_rotate_matrix(transform)
            ham.cache_two_ints()
        # rotating wavefunction as a NonorthWavefunction
        elif wfn_type == "nonorth":
            wfn = NonorthWavefunction(nelec, nspin, doci, memory=doci.memory, params=transform)

        norm = sum(wfn.get_overlap(sd) ** 2 for sd in sds)
        if expectation_type == "ci matrix":
            return (
                sum(
                    wfn.get_overlap(sd1) * ham.integrate_sd_sd(sd1, sd2) * wfn.get_overlap(sd2)
                    for sd1 in sds
                    for sd2 in sds
                )
                / norm
            )
        elif expectation_type == "projected":
            return sum(wfn.get_overlap(sd) * ham.integrate_sd_wfn(sd, wfn) for sd in sds) / norm

    assert np.allclose(get_energy("doci", "ci matrix"), get_energy("nonorth", "ci matrix"))
    assert np.allclose(get_energy("doci", "projected"), get_energy("nonorth", "projected"))
    assert np.allclose(get_energy("doci", "ci matrix"), get_energy("doci", "projected"))
