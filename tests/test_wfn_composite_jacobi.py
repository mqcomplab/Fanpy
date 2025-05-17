"""Test fanpy.wavefunction.composite.jacobi."""
import itertools as it

from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.solver.ci import brute
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.base import BaseWavefunction
from fanpy.wfn.ci.base import CIWavefunction
from fanpy.wfn.ci.doci import DOCI
from fanpy.wfn.composite.jacobi import JacobiWavefunction
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
        """Return spin of the wavefunction."""
        return self._spin

    @property
    def seniority(self):
        """Return seniority of the wavefunction."""
        return self._seniority

    @property
    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction."""
        if params is None:
            params = np.identity(10)
        super().assign_params(params=params, add_noise=add_noise)


def test_jacobi_spin():
    """Test JacobiWavefunction.spin."""
    test = skip_init(JacobiWavefunction)
    test_wfn = TempWavefunction()
    test_wfn._spin = 2
    test.wfn = test_wfn

    # restricted
    test.orbtype = "restricted"
    assert test.spin == 2
    # unrestricted
    test.orbtype = "unrestricted"
    assert test.spin == 2
    # generalized
    test.orbtype = "generalized"
    assert test.spin is None


def test_jacobi_seniority():
    """Test JacobiWavefunction.seniority."""
    test = skip_init(JacobiWavefunction)
    test_wfn = TempWavefunction()
    test_wfn._seniority = 2
    test.wfn = test_wfn

    # restricted
    test.orbtype = "restricted"
    assert test.seniority == 2
    # unrestricted
    test.orbtype = "unrestricted"
    assert test.seniority == 2
    # generalized
    test.orbtype = "generalized"
    assert test.seniority is None


def test_jacobi_jacobi_rotation():
    """Tests JacobiWavefunction.jacobi_rotation."""
    test = skip_init(JacobiWavefunction)
    theta = 2 * np.pi * (np.random.random() - 0.5)
    test.params = np.array(theta)
    test.nspin = 6

    # generalized
    test.orbtype = "generalized"
    test.jacobi_indices = (0, 4)
    answer = np.identity(6)
    answer[0, 0] = np.cos(theta)
    answer[0, 4] = np.sin(theta)
    answer[4, 0] = -np.sin(theta)
    answer[4, 4] = np.cos(theta)
    assert len(test.jacobi_rotation) == 1
    assert np.allclose(test.jacobi_rotation[0], answer)

    # restricted
    test.orbtype = "restricted"
    test.jacobi_indices = (0, 2)
    answer = np.identity(3)
    answer[0, 0] = np.cos(theta)
    answer[0, 2] = np.sin(theta)
    answer[2, 0] = -np.sin(theta)
    answer[2, 2] = np.cos(theta)
    assert len(test.jacobi_rotation) == 1
    assert np.allclose(test.jacobi_rotation[0], answer)

    # unrestricted
    test.orbtype = "unrestricted"
    test.jacobi_indices = (0, 2)
    answer = np.identity(3)
    answer[0, 0] = np.cos(theta)
    answer[0, 2] = np.sin(theta)
    answer[2, 0] = -np.sin(theta)
    answer[2, 2] = np.cos(theta)
    assert len(test.jacobi_rotation) == 2
    assert np.allclose(test.jacobi_rotation[0], answer)
    assert np.allclose(test.jacobi_rotation[1], np.identity(3))

    test.jacobi_indices = (3, 4)
    answer = np.identity(3)
    answer[0, 0] = np.cos(theta)
    answer[0, 1] = np.sin(theta)
    answer[1, 0] = -np.sin(theta)
    answer[1, 1] = np.cos(theta)
    assert len(test.jacobi_rotation) == 2
    assert np.allclose(test.jacobi_rotation[0], np.identity(3))
    assert np.allclose(test.jacobi_rotation[1], answer)


def test_jacobi_assign_params():
    """Test JacobiWavefunction.assign_params."""
    test = skip_init(JacobiWavefunction)
    test.memory = 10
    test._cache_fns = {}
    test.assign_params(0)
    test.enable_cache()
    assert test.params.size == 1
    assert test.params.shape == ()
    assert test.params == 0


def test_jacobi_assign_orbtype():
    """Test JacobiWavefunction.assign_orbtype."""
    test = skip_init(JacobiWavefunction)
    test.nelec = 4
    test.nspin = 10
    test.memory = 10

    test.assign_orbtype(None)
    assert test.orbtype == "restricted"
    test.assign_orbtype("restricted")
    assert test.orbtype == "restricted"
    test.assign_orbtype("unrestricted")
    assert test.orbtype == "unrestricted"
    test.assign_orbtype("generalized")
    assert test.orbtype == "generalized"

    with pytest.raises(ValueError):
        test.assign_orbtype("lkjsadf")


def test_jacobi_assign_jacobi_indices():
    """Test JacobiWavefunction.assign_jacobi_indices."""
    test = skip_init(JacobiWavefunction)
    test.nelec = 4
    test.nspin = 10
    test.memory = 10

    # not tuple or list
    with pytest.raises(TypeError):
        test.assign_jacobi_indices(1)
    with pytest.raises(TypeError):
        test.assign_jacobi_indices({1: 0, 2: 3})
    with pytest.raises(TypeError):
        test.assign_jacobi_indices(set([0, 1]))
    # bad number of entries
    with pytest.raises(TypeError):
        test.assign_jacobi_indices([0])
    with pytest.raises(TypeError):
        test.assign_jacobi_indices((1, 2, 3))
    # bad type of entries
    with pytest.raises(TypeError):
        test.assign_jacobi_indices([0, 1.0])
    with pytest.raises(TypeError):
        test.assign_jacobi_indices(("0", 1))

    # negative entry
    with pytest.raises(ValueError):
        test.assign_jacobi_indices([0, -1])
    # same entries
    with pytest.raises(ValueError):
        test.assign_jacobi_indices([0, 0])
    # bad input for given orbital type
    test.orbtype = "generalized"
    with pytest.raises(ValueError):
        test.assign_jacobi_indices([0, 10])
    test.assign_jacobi_indices([0, 9])
    assert test.jacobi_indices == (0, 9)
    test.orbtype = "restricted"
    with pytest.raises(ValueError):
        test.assign_jacobi_indices([0, 5])
    test.assign_jacobi_indices([0, 4])
    assert test.jacobi_indices == (0, 4)
    test.orbtype = "unrestricted"
    with pytest.raises(ValueError):
        test.assign_jacobi_indices([0, 6])
    with pytest.raises(ValueError):
        test.assign_jacobi_indices([1, 9])
    test.assign_jacobi_indices([0, 1])
    assert test.jacobi_indices == (0, 1)
    test.assign_jacobi_indices([5, 6])
    assert test.jacobi_indices == (5, 6)

    # swap entries
    test.assign_jacobi_indices([4, 1])
    assert test.jacobi_indices == (1, 4)


def test_jacobi_get_overlap():
    """Test JacobiWavefunction.get_overlap."""
    test = skip_init(JacobiWavefunction)
    test.nelec = 2
    test.nspin = 4
    test.memory = 10
    test.assign_wfn(CIWavefunction(2, 4, memory=10))
    test._cache_fns = {}
    test.wfn.params = np.arange(1, 7)
    wfn_sd_coeff = {0b0101: 1, 0b0110: 2, 0b1100: 3, 0b0011: 4, 0b1001: 5, 0b1010: 6}
    test.assign_params(np.array(2 * np.pi * (np.random.random() - 0.5)))
    test.enable_cache()

    # generalized
    test.assign_orbtype("generalized")
    test.assign_jacobi_indices((0, 3))
    # jacobi matrix [[cos, 0, 0, sin],
    #                [0, 1, 0, 0],
    #                [0, 0, 1, 0],
    #                [-sin, 0, 0, cos]]
    # 0b1001 uses [[cos, 0, 0, sin],
    #              [-sin, 0, 0, cos]]
    assert np.isclose(test.get_overlap(0b1001), 1 * wfn_sd_coeff[0b1001])
    # 0b1001 uses [[0, 1, 0, 0],
    #              [0, 0, 1, 0]]
    assert np.isclose(test.get_overlap(0b0110), 1 * wfn_sd_coeff[0b0110])
    # 0b0101 uses [[cos, 0, 0, sin],
    #              [0, 0, 1, 0]]
    assert np.isclose(
        test.get_overlap(0b0101),
        (np.cos(test.params) * wfn_sd_coeff[0b0101] - np.sin(test.params) * wfn_sd_coeff[0b1100]),
    )
    # 0b0011 uses [[cos, 0, 0, sin],
    #              [0, 1, 0, 0]]
    assert np.isclose(
        test.get_overlap(0b0011),
        (np.cos(test.params) * wfn_sd_coeff[0b0011] - np.sin(test.params) * wfn_sd_coeff[0b1010]),
    )
    # 0b0101 uses [[0, 0, 1, 0],
    #              [-sin, 0, 0, cos]]
    assert np.isclose(
        test.get_overlap(0b1100),
        (np.sin(test.params) * wfn_sd_coeff[0b0101] + np.cos(test.params) * wfn_sd_coeff[0b1100]),
    )
    # 0b0011 uses [[0, 1, 0, 0]],
    #              [-sin, 0, 0, cos]]
    assert np.isclose(
        test.get_overlap(0b1010),
        (np.sin(test.params) * wfn_sd_coeff[0b0011] + np.cos(test.params) * wfn_sd_coeff[0b1010]),
    )

    test.assign_jacobi_indices((0, 2))
    # jacobi matrix [[cos, 0, sin, 0],
    #                [0, 1, 0, 0],
    #                [-sin, 0, cos, 0],
    #                [0, 0, 0, 1]]
    # 0b1001 uses [[cos, 0, sin, 0],
    #              [0, 0, 0, 1]]
    assert np.isclose(
        test.get_overlap(0b1001),
        (np.cos(test.params) * wfn_sd_coeff[0b1001] + np.sin(test.params) * wfn_sd_coeff[0b1100]),
    )
    # 0b0011 uses [[cos, 0, sin, 0],
    #              [0, 1, 0, 0]]
    assert np.isclose(
        test.get_overlap(0b0011),
        (np.cos(test.params) * wfn_sd_coeff[0b0011] - np.sin(test.params) * wfn_sd_coeff[0b0110]),
    )

    # unrestricted
    test.assign_orbtype("unrestricted")
    test.assign_jacobi_indices((0, 1))
    test.clear_cache()
    # jacobi matrix [[cos, sin, 0, 0],
    #                [-sin, cos, 0, 0],
    #                [0, 0, 1, 0],
    #                [0, 0, 0, 1]]
    # 0b0011 uses [[cos, sin, 0, 0],
    #              [-sin, cos, 0, 0]]
    assert np.isclose(test.get_overlap(0b0011), 1 * wfn_sd_coeff[0b0011])
    # 0b1100 uses [[0, 0, 1, 0],
    #              [0, 0, 0, 1]]
    assert np.isclose(test.get_overlap(0b1100), 1 * wfn_sd_coeff[0b1100])
    # 0b0101 uses [[cos, sin, 0, 0],
    #              [0, 0, 1, 0]]
    assert np.isclose(
        test.get_overlap(0b0101),
        (np.cos(test.params) * wfn_sd_coeff[0b0101] + np.sin(test.params) * wfn_sd_coeff[0b0110]),
    )
    # 0b0110 uses [[-sin, cos, 0, 0],
    #              [0, 0, 1, 0]]
    assert np.isclose(
        test.get_overlap(0b0110),
        (-np.sin(test.params) * wfn_sd_coeff[0b0101] + np.cos(test.params) * wfn_sd_coeff[0b0110]),
    )

    # restricted
    test.assign_orbtype("restricted")
    test.assign_jacobi_indices((0, 1))
    test.clear_cache()
    # jacobi matrix [[cos, sin, 0, 0],
    #                [-sin, cos, 0, 0],
    #                [0, 0, cos, sin],
    #                [0, 0, -sin, cos]]
    # 0b0011 uses [[cos, sin, 0, 0],
    #              [-sin, cos, 0, 0]]
    assert np.isclose(test.get_overlap(0b0011), 1 * wfn_sd_coeff[0b0011])
    # 0b1100 uses [[0, 0, cos, sin],
    #              [0, 0, -sin, cos]]
    assert np.isclose(test.get_overlap(0b1100), 1 * wfn_sd_coeff[0b1100])
    # 0b0101 uses [[cos, sin, 0, 0],
    #              [0, 0, cos, sin]]
    assert np.isclose(
        test.get_overlap(0b0101),
        (
            np.cos(test.params) ** 2 * wfn_sd_coeff[0b0101]
            + np.sin(test.params) ** 2 * wfn_sd_coeff[0b1010]
            + np.cos(test.params) * np.sin(test.params) * wfn_sd_coeff[0b1001]
            + np.cos(test.params) * np.sin(test.params) * wfn_sd_coeff[0b0110]
        ),
    )
    # 0b1001 uses [[cos, sin, 0, 0],
    #              [0, 0, -sin, cos]]
    assert np.isclose(
        test.get_overlap(0b1001),
        (
            np.cos(test.params) ** 2 * wfn_sd_coeff[0b1001]
            - np.sin(test.params) ** 2 * wfn_sd_coeff[0b0110]
            + np.cos(test.params) * np.sin(test.params) * wfn_sd_coeff[0b1010]
            - np.cos(test.params) * np.sin(test.params) * wfn_sd_coeff[0b0101]
        ),
    )
    # 0b0110 uses [[-sin, cos, 0, 0],
    #              [0, 0, cos, sin]]
    assert np.isclose(
        test.get_overlap(0b0110),
        (
            np.cos(test.params) ** 2 * wfn_sd_coeff[0b0110]
            - np.sin(test.params) ** 2 * wfn_sd_coeff[0b1001]
            - np.cos(test.params) * np.sin(test.params) * wfn_sd_coeff[0b0101]
            + np.cos(test.params) * np.sin(test.params) * wfn_sd_coeff[0b1010]
        ),
    )

    with pytest.raises(TypeError):
        test.get_overlap("1")
    with pytest.raises(TypeError):
        test.get_overlap("1", (test.wfn, np.array([0])))
    with pytest.raises(TypeError):
        test.get_overlap(0b0101, deriv=np.array([0]))
    with pytest.raises(TypeError):
        test.get_overlap(0b0101, deriv=(test.wfn, np.array([0.0])))
    with pytest.raises(TypeError):
        test.get_overlap(0b0101, deriv=(test.wfn, np.array([[0]])))
    with pytest.raises(ValueError):
        test.get_overlap(0b0101, deriv=(DOCI(2, 4), np.array([0])))
    with pytest.raises(ValueError):
        test.get_overlap(0b0101, deriv=(test, np.array([-1])))
    with pytest.raises(ValueError):
        test.get_overlap(0b0101, deriv=(test, np.array([test.nparams])))
    with pytest.raises(NotImplementedError):
        test.get_overlap(0b0101, deriv=(test.wfn, np.array([0])))


def test_jacobi_get_overlap_restricted():
    """Test JacobiWavefunction.get_overlap for a larger restricted case."""
    test = skip_init(JacobiWavefunction)
    test.nelec = 4
    test.nspin = 8
    test.memory = 10
    test.assign_wfn(CIWavefunction(4, 8, memory=10))
    test._cache_fns = {}
    test.wfn.params = np.arange(1, test.wfn.nparams + 1)
    wfn_sd_coeff = {sd: test.wfn.params[index] for sd, index in test.wfn.dict_sd_index.items()}
    test.assign_params(np.array(2 * np.pi * (np.random.random() - 0.5)))
    test.enable_cache()

    sin = np.sin(test.params)
    cos = np.cos(test.params)

    test.assign_orbtype("restricted")
    test.assign_jacobi_indices((0, 2))
    # jacobi matrix [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                [0, 1, 0, 0, 0, 0, 0, 0],
    #                [-sin, 0, cos, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 1, 0, 0, 0, 0],
    #                [0, 0, 0, 0, cos, 0, sin, 0, ],
    #                [0, 0, 0, 0, 0, 1, 0, 0],
    #                [0, 0, 0, 0, -sin, 0, cos, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 1, ]]
    # 0b00001111 uses [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 0, 0, 0, 0]]
    assert np.isclose(test.get_overlap(0b00001111), 1 * wfn_sd_coeff[0b00001111])
    # 0b00100111 uses [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0]]
    assert np.isclose(test.get_overlap(0b00100111), 1 * wfn_sd_coeff[0b00100111])
    # 0b01010101 uses [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, cos, 0, sin, 0],
    #                  [0, 0, 0, 0, -sin, 0, cos, 0]]
    assert np.isclose(test.get_overlap(0b01010101), 1 * wfn_sd_coeff[0b01010101])
    # 0b00100111 uses [[cos, 0, sin, 0, 0, 0, 0, 0]
    #                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, cos, 0, sin, 0]]
    assert np.isclose(
        test.get_overlap(0b00010111),
        (cos * wfn_sd_coeff[0b00010111] + sin * wfn_sd_coeff[0b01000111]),
    )
    # 0b01101010 uses [[0, 1, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0],
    #                  [0, 0, 0, 0, -sin, 0, cos, 0]]
    assert np.isclose(
        test.get_overlap(0b01101010),
        (sin * wfn_sd_coeff[0b00111010] + cos * wfn_sd_coeff[0b01101010]),
    )
    # 0b00101011 uses [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0]]
    assert np.isclose(
        test.get_overlap(0b00101011),
        (cos * wfn_sd_coeff[0b00101011] - sin * wfn_sd_coeff[0b00101110]),
    )
    # 0b00110011 uses [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, cos, 0, sin, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0]]
    assert np.isclose(
        test.get_overlap(0b00110011),
        (
            cos ** 2 * wfn_sd_coeff[0b00110011]
            - cos * sin * wfn_sd_coeff[0b01100011]
            - cos * sin * wfn_sd_coeff[0b00110110]
            + sin ** 2 * wfn_sd_coeff[0b01100110]
        ),
    )
    # 0b01100011 uses [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0],
    #                  [0, 0, 0, 0, -sin, 0, cos, 0]]
    assert np.isclose(
        test.get_overlap(0b01100011),
        (
            cos ** 2 * wfn_sd_coeff[0b01100011]
            + cos * sin * wfn_sd_coeff[0b00110011]
            - cos * sin * wfn_sd_coeff[0b01100110]
            - sin ** 2 * wfn_sd_coeff[0b00110110]
        ),
    )
    # 0b00101110 uses [[0, 1, 0, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0]]
    assert np.isclose(
        test.get_overlap(0b00101110),
        (cos * wfn_sd_coeff[0b00101110] + sin * wfn_sd_coeff[0b00101011]),
    )
    # 0b00110110 uses [[0, 1, 0, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, cos, 0, sin, 0]]
    #                  [0, 0, 0, 0, 0, 1, 0, 0],
    assert np.isclose(
        test.get_overlap(0b00110110),
        (
            cos * cos * wfn_sd_coeff[0b00110110]
            - cos * sin * wfn_sd_coeff[0b01100110]
            + cos * sin * wfn_sd_coeff[0b00110011]
            - sin * sin * wfn_sd_coeff[0b01100011]
        ),
    )
    # 0b01100110 uses [[0, 1, 0, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0],
    #                  [0, 0, 0, 0, -sin, 0, cos, 0]]
    assert np.isclose(
        test.get_overlap(0b01100110),
        (
            cos * cos * wfn_sd_coeff[0b01100110]
            + cos * sin * wfn_sd_coeff[0b00110110]
            + cos * sin * wfn_sd_coeff[0b01100011]
            + sin * sin * wfn_sd_coeff[0b00110011]
        ),
    )


def test_jacobi_get_overlap_der():
    """Test JacobiWavefunction.get_overlap with derivatization."""
    test = skip_init(JacobiWavefunction)
    test.nelec = 2
    test.nspin = 4
    test.memory = 10
    test.assign_wfn(CIWavefunction(2, 4, memory=10))
    test._cache_fns = {}
    test.wfn.params = np.arange(1, 7)
    wfn_sd_coeff = {0b0101: 1, 0b0110: 2, 0b1100: 3, 0b0011: 4, 0b1001: 5, 0b1010: 6}
    test.assign_params(np.array(2 * np.pi * (np.random.random() - 0.5)))
    test.enable_cache()
    sin = np.sin(test.params)
    cos = np.cos(test.params)

    # generalized
    test.assign_orbtype("generalized")
    test.assign_jacobi_indices((0, 3))
    # jacobi matrix [[cos, 0, 0, sin],
    #                [0, 1, 0, 0],
    #                [0, 0, 1, 0],
    #                [-sin, 0, 0, cos]]

    # 0b1001 uses [[cos, 0, 0, sin],
    #              [-sin, 0, 0, cos]]
    assert np.isclose(test.get_overlap(0b1001, deriv=(test, np.array([0]))), 0.0)
    # 0b1001 uses [[0, 1, 0, 0],
    #              [0, 0, 1, 0]]
    assert np.isclose(test.get_overlap(0b0110, deriv=(test, np.array([0]))), 0.0)
    # 0b0101 uses [[cos, 0, 0, sin],
    #              [0, 0, 1, 0]]
    assert np.isclose(
        test.get_overlap(0b0101, deriv=(test, np.array([0]))),
        (-sin * wfn_sd_coeff[0b0101] - cos * wfn_sd_coeff[0b1100]),
    )
    # 0b0011 uses [[cos, 0, 0, sin],
    #              [0, 1, 0, 0]]
    assert np.isclose(
        test.get_overlap(0b0011, deriv=(test, np.array([0]))),
        (-sin * wfn_sd_coeff[0b0011] - cos * wfn_sd_coeff[0b1010]),
    )
    # 0b0101 uses [[0, 0, 1, 0],
    #              [-sin, 0, 0, cos]]
    assert np.isclose(
        test.get_overlap(0b1100, deriv=(test, np.array([0]))),
        (cos * wfn_sd_coeff[0b0101] - sin * wfn_sd_coeff[0b1100]),
    )
    # 0b0011 uses [[0, 1, 0, 0]],
    #              [-sin, 0, 0, cos]]
    assert np.isclose(
        test.get_overlap(0b1010, deriv=(test, np.array([0]))),
        (cos * wfn_sd_coeff[0b0011] - sin * wfn_sd_coeff[0b1010]),
    )

    test.assign_jacobi_indices((0, 2))
    # jacobi matrix [[cos, 0, sin, 0],
    #                [0, 1, 0, 0],
    #                [-sin, 0, cos, 0],
    #                [0, 0, 0, 1]]
    # 0b1001 uses [[cos, 0, sin, 0],
    #              [0, 0, 0, 1]]
    assert np.isclose(
        test.get_overlap(0b1001, deriv=(test, np.array([0]))),
        (-sin * wfn_sd_coeff[0b1001] + cos * wfn_sd_coeff[0b1100]),
    )
    # 0b0011 uses [[cos, 0, sin, 0],
    #              [0, 1, 0, 0]]
    assert np.isclose(
        test.get_overlap(0b0011, deriv=(test, np.array([0]))),
        (-sin * wfn_sd_coeff[0b0011] - cos * wfn_sd_coeff[0b0110]),
    )

    # unrestricted
    test.assign_orbtype("unrestricted")
    test.assign_jacobi_indices((0, 1))
    test.clear_cache()
    # jacobi matrix [[cos, sin, 0, 0],
    #                [-sin, cos, 0, 0],
    #                [0, 0, 1, 0],
    #                [0, 0, 0, 1]]
    # 0b0011 uses [[cos, sin, 0, 0],
    #              [-sin, cos, 0, 0]]
    assert np.isclose(test.get_overlap(0b0011, deriv=(test, np.array([0]))), 0.0)
    # 0b1100 uses [[0, 0, 1, 0],
    #              [0, 0, 0, 1]]
    assert np.isclose(test.get_overlap(0b1100, deriv=(test, np.array([0]))), 0.0)
    # 0b0101 uses [[cos, sin, 0, 0],
    #              [0, 0, 1, 0]]
    assert np.isclose(
        test.get_overlap(0b0101, deriv=(test, np.array([0]))),
        (-sin * wfn_sd_coeff[0b0101] + cos * wfn_sd_coeff[0b0110]),
    )
    # 0b0110 uses [[-sin, cos, 0, 0],
    #              [0, 0, 1, 0]]
    assert np.isclose(
        test.get_overlap(0b0110, deriv=(test, np.array([0]))),
        (-cos * wfn_sd_coeff[0b0101] - sin * wfn_sd_coeff[0b0110]),
    )

    # restricted
    test.assign_orbtype("restricted")
    test.assign_jacobi_indices((0, 1))
    test.clear_cache()
    # jacobi matrix [[cos, sin, 0, 0],
    #                [-sin, cos, 0, 0],
    #                [0, 0, cos, sin],
    #                [0, 0, -sin, cos]]
    # 0b0011 uses [[cos, sin, 0, 0],
    #              [-sin, cos, 0, 0]]
    assert np.isclose(test.get_overlap(0b0011, deriv=(test, np.array([0]))), 0.0)
    # 0b1100 uses [[0, 0, cos, sin],
    #              [0, 0, -sin, cos]]
    assert np.isclose(test.get_overlap(0b1100, deriv=(test, np.array([0]))), 0.0)
    # 0b0101 uses [[cos, sin, 0, 0],
    #              [0, 0, cos, sin]]
    assert np.isclose(
        test.get_overlap(0b0101, deriv=(test, np.array([0]))),
        (
            -2 * cos * sin * wfn_sd_coeff[0b0101]
            + 2 * sin * cos * wfn_sd_coeff[0b1010]
            + (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b1001]
            + (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b0110]
        ),
    )
    # 0b1001 uses [[cos, sin, 0, 0],
    #              [0, 0, -sin, cos]]
    assert np.isclose(
        test.get_overlap(0b1001, deriv=(test, np.array([0]))),
        (
            -2 * cos * sin * wfn_sd_coeff[0b1001]
            - 2 * sin * cos * wfn_sd_coeff[0b0110]
            + (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b1010]
            - (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b0101]
        ),
    )
    # 0b0110 uses [[-sin, cos, 0, 0],
    #              [0, 0, cos, sin]]
    assert np.isclose(
        test.get_overlap(0b0110, deriv=(test, np.array([0]))),
        (
            -2 * cos * sin * wfn_sd_coeff[0b0110]
            - 2 * sin * cos * wfn_sd_coeff[0b1001]
            - (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b0101]
            + (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b1010]
        ),
    )


def test_jacobi_get_overlap_restricted_der():
    """Test JacobiWavefunction.get_overlap for a larger restricted case with derivatization."""
    test = skip_init(JacobiWavefunction)
    test.nelec = 4
    test.nspin = 8
    test.memory = 10
    test.assign_wfn(CIWavefunction(4, 8, memory=10))
    test._cache_fns = {}
    test.wfn.params = np.arange(1, test.wfn.nparams + 1)
    wfn_sd_coeff = {sd: test.wfn.params[index] for sd, index in test.wfn.dict_sd_index.items()}
    test.assign_params(np.array(2 * np.pi * (np.random.random() - 0.5)))
    test.enable_cache()

    sin = np.sin(test.params)
    cos = np.cos(test.params)

    test.assign_orbtype("restricted")
    test.assign_jacobi_indices((0, 2))
    # jacobi matrix [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                [0, 1, 0, 0, 0, 0, 0, 0],
    #                [-sin, 0, cos, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 1, 0, 0, 0, 0],
    #                [0, 0, 0, 0, cos, 0, sin, 0, ],
    #                [0, 0, 0, 0, 0, 1, 0, 0],
    #                [0, 0, 0, 0, -sin, 0, cos, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 1, ]]
    # 0b00001111 uses [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 0, 0, 0, 0]]
    assert np.isclose(test.get_overlap(0b00001111, deriv=(test, np.array([0]))), 0.0)
    # 0b00100111 uses [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0]]
    assert np.isclose(test.get_overlap(0b00100111, deriv=(test, np.array([0]))), 0.0)
    # 0b01010101 uses [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, cos, 0, sin, 0],
    #                  [0, 0, 0, 0, -sin, 0, cos, 0]]
    assert np.isclose(test.get_overlap(0b01010101, deriv=(test, np.array([0]))), 0.0)
    # 0b00100111 uses [[cos, 0, sin, 0, 0, 0, 0, 0]
    #                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, cos, 0, sin, 0]]
    assert np.isclose(
        test.get_overlap(0b00010111, deriv=(test, np.array([0]))),
        (-sin * wfn_sd_coeff[0b00010111] + cos * wfn_sd_coeff[0b01000111]),
    )
    # 0b01101010 uses [[0, 1, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0],
    #                  [0, 0, 0, 0, -sin, 0, cos, 0]]
    assert np.isclose(
        test.get_overlap(0b01101010, deriv=(test, np.array([0]))),
        (cos * wfn_sd_coeff[0b00111010] - sin * wfn_sd_coeff[0b01101010]),
    )
    # 0b00101011 uses [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0]]
    assert np.isclose(
        test.get_overlap(0b00101011, deriv=(test, np.array([0]))),
        (-sin * wfn_sd_coeff[0b00101011] - cos * wfn_sd_coeff[0b00101110]),
    )
    # 0b00110011 uses [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, cos, 0, sin, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0]]
    assert np.isclose(
        test.get_overlap(0b00110011, deriv=(test, np.array([0]))),
        (
            -2 * cos * sin * wfn_sd_coeff[0b00110011]
            - (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b01100011]
            - (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b00110110]
            + 2 * sin * cos * wfn_sd_coeff[0b01100110]
        ),
    )
    # 0b01100011 uses [[cos, 0, sin, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0],
    #                  [0, 0, 0, 0, -sin, 0, cos, 0]]
    assert np.isclose(
        test.get_overlap(0b01100011, deriv=(test, np.array([0]))),
        (
            -2 * cos * sin * wfn_sd_coeff[0b01100011]
            + (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b00110011]
            - (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b01100110]
            - 2 * sin * cos * wfn_sd_coeff[0b00110110]
        ),
    )
    # 0b00101110 uses [[0, 1, 0, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0]]
    assert np.isclose(
        test.get_overlap(0b00101110, deriv=(test, np.array([0]))),
        (-sin * wfn_sd_coeff[0b00101110] + cos * wfn_sd_coeff[0b00101011]),
    )
    # 0b00110110 uses [[0, 1, 0, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, cos, 0, sin, 0]]
    #                  [0, 0, 0, 0, 0, 1, 0, 0],
    assert np.isclose(
        test.get_overlap(0b00110110, deriv=(test, np.array([0]))),
        (
            -2 * cos * sin * wfn_sd_coeff[0b00110110]
            - (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b01100110]
            + (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b00110011]
            - 2 * sin * cos * wfn_sd_coeff[0b01100011]
        ),
    )
    # 0b01100110 uses [[0, 1, 0, 0, 0, 0, 0, 0],
    #                  [-sin, 0, cos, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0],
    #                  [0, 0, 0, 0, -sin, 0, cos, 0]]
    assert np.isclose(
        test.get_overlap(0b01100110, deriv=(test, np.array([0]))),
        (
            -2 * cos * sin * wfn_sd_coeff[0b01100110]
            + (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b00110110]
            + (cos ** 2 - sin ** 2) * wfn_sd_coeff[0b01100011]
            + 2 * sin * cos * wfn_sd_coeff[0b00110011]
        ),
    )


def test_jacobi_compare_nonorth():
    """Test overlap of Jacobi rotated wavefunction with that of nonorthonormal wavefunction."""
    nelec = 4
    nspin = 8
    doci = DOCI(nelec, nspin)
    doci.assign_params(
        np.array(
            [
                7.06555071e-01,
                -7.06555071e-01,
                -4.16333634e-15,
                -3.88578059e-15,
                -2.79272538e-02,
                2.79272538e-02,
            ]
        )
    )

    # one rotation
    jacobi = JacobiWavefunction(
        nelec, nspin, doci, memory=doci.memory, orbtype="restricted", jacobi_indices=(0, 1)
    )
    nonorth = NonorthWavefunction(nelec, nspin, doci, memory=doci.memory)

    sds = sd_list(4, 8, num_limit=None, exc_orders=None)

    for sd in sds:
        for theta in np.linspace(-np.pi, np.pi, 100):
            nonorth.assign_params(
                (
                    np.array(
                        [
                            [np.cos(theta), np.sin(theta), 0, 0],
                            [-np.sin(theta), np.cos(theta), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ]
                    ),
                )
            )
            nonorth.clear_cache()
            jacobi.assign_params(np.array(theta))
            jacobi.clear_cache()
            assert np.isclose(nonorth.get_overlap(sd), jacobi.get_overlap(sd))

    # two rotations
    nonorth = NonorthWavefunction(nelec, nspin, memory=doci.memory, wfn=doci)
    for sd in sds:
        for theta_one in np.linspace(-np.pi, np.pi, 10):
            jacobi_one = JacobiWavefunction(
                nelec,
                nspin,
                memory=doci.memory,
                wfn=doci,
                orbtype="restricted",
                jacobi_indices=(0, 1),
            )
            jacobi_one.assign_params(np.array(theta_one))
            jacobi_one.clear_cache()
            for theta_two in np.linspace(-np.pi, np.pi, 10):
                jacobi_two = JacobiWavefunction(
                    nelec,
                    nspin,
                    memory=doci.memory,
                    wfn=jacobi_one,
                    orbtype="restricted",
                    jacobi_indices=(2, 3),
                )
                jacobi_two.assign_params(np.array(theta_two))
                jacobi_two.clear_cache()

                nonorth.assign_params(
                    (
                        np.array(
                            [
                                [np.cos(theta_one), np.sin(theta_one), 0, 0],
                                [-np.sin(theta_one), np.cos(theta_one), 0, 0],
                                [0, 0, np.cos(theta_two), np.sin(theta_two)],
                                [0, 0, -np.sin(theta_two), np.cos(theta_two)],
                            ]
                        ),
                    )
                )
                nonorth.clear_cache()

                assert np.isclose(nonorth.get_overlap(sd), jacobi_two.get_overlap(sd))


def test_jacobi_energy():
    """Test energy of Jacobi rotated wavefunction with that of rotated Hamiltonian."""
    nelec = 4
    nspin = 8
    sds = sd_list(4, 8, num_limit=None, exc_orders=None)

    # NOTE: we need to be a little careful with the hamiltonian construction because the integrals
    #       are stored by reference and using the same hamiltonian while transforming it will cause
    #       some headache
    def get_energy(theta, orbpair, wfn_type, expectation_type):
        """Get energy that correspond to the rotation of the given orbitals."""
        doci = DOCI(nelec, nspin)
        ham = RestrictedMolecularHamiltonian(
            np.load(find_datafile("data/data_h4_square_hf_sto6g_oneint.npy")),
            np.load(find_datafile("data/data_h4_square_hf_sto6g_twoint.npy")),
        )
        results = brute(doci, ham)
        coeffs = results["eigvec"]
        doci.assign_params(coeffs[:, 0].flatten())
        jacobi = JacobiWavefunction(
            nelec,
            nspin,
            memory=doci.memory,
            wfn=doci,
            orbtype="restricted",
            jacobi_indices=orbpair,
            params=np.array(theta),
        )

        # check that rotation is unitary
        rotation = jacobi.jacobi_rotation[0]
        assert np.allclose(rotation.dot(rotation.T), np.identity(4))
        assert np.allclose(rotation.T.dot(rotation), np.identity(4))

        # rotating hamiltonian using orb_rotate_jacobi
        if wfn_type == "doci":
            wfn = doci
            ham.orb_rotate_jacobi(orbpair, theta)
            ham.cache_two_ints()
        # rotating hamiltonian using orb_rotate_matrix
        elif wfn_type == "doci_full":
            wfn = doci
            ham.orb_rotate_matrix(jacobi.jacobi_rotation[0])
            ham.cache_two_ints()
        # rotating wavefunction as a JacobiWavefunction
        elif wfn_type == "jacobi":
            wfn = jacobi
        # rotating wavefunction as a NonorthWavefunction
        elif wfn_type == "nonorth":
            wfn = NonorthWavefunction(
                nelec, nspin, doci, memory=doci.memory, params=jacobi.jacobi_rotation
            )
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

    for orbpair in it.combinations(range(4), 2):
        theta = np.pi * (np.random.random() - 0.5)
        assert np.allclose(
            get_energy(theta, orbpair, "doci", "ci matrix"),
            get_energy(theta, orbpair, "jacobi", "ci matrix"),
        )
        assert np.allclose(
            get_energy(theta, orbpair, "doci", "ci matrix"),
            get_energy(theta, orbpair, "nonorth", "ci matrix"),
        )
        assert np.allclose(
            get_energy(theta, orbpair, "doci", "projected"),
            get_energy(theta, orbpair, "jacobi", "projected"),
        )
        assert np.allclose(
            get_energy(theta, orbpair, "doci", "projected"),
            get_energy(theta, orbpair, "nonorth", "projected"),
        )
        assert np.allclose(
            get_energy(theta, orbpair, "doci", "ci matrix"),
            get_energy(theta, orbpair, "doci", "projected"),
        )
