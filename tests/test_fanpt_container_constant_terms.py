import numpy as np
import pytest

from fanpy.fanpt.containers.base import FANPTContainer
from fanpy.fanpt.containers.constant_terms import FANPTConstantTerms


class DummyContainer(FANPTContainer):
    r"""Minimal FANPT container mock for testing constant-term generation.

    This dummy container stores precomputed derivative arrays required by
    ``FANPTConstantTerms`` without constructing a full FanCI/PyCI backend.
    """

    def __init__(
        self,
        nequation,
        nactive,
        active_energy,
        d_g_lambda=None,
        d2_g_lambda_wfnparams=None,
        d2_g_e_wfnparams=None,
        d2_g_wfnparams2=None,
        d3_g_e_wfnparams2=None,
        d3_g_lambda_wfnparams2=None,
    ):
        self._nequation = nequation
        self._nactive = nactive
        self.active_energy = active_energy

        self.d_g_lambda = self._as_array(d_g_lambda)
        self.d2_g_lambda_wfnparams = self._as_array(d2_g_lambda_wfnparams)
        self.d2_g_e_wfnparams = self._as_array(d2_g_e_wfnparams)
        self.d2_g_wfnparams2 = self._as_array(d2_g_wfnparams2)
        self.d3_g_e_wfnparams2 = self._as_array(d3_g_e_wfnparams2)
        self.d3_g_lambda_wfnparams2 = self._as_array(d3_g_lambda_wfnparams2)

    @staticmethod
    def _as_array(value):
        """Convert a value to a float array unless it is None."""
        if value is None:
            return None
        return np.array(value, dtype=float)

    @property
    def nequation(self):
        return self._nequation

    @property
    def nactive(self):
        return self._nactive

    # Abstract methods implemented as no-ops for test instantiation.
    def der_g_lambda(self):
        pass

    def der2_g_lambda_wfnparams(self):
        pass

    def gen_coeff_matrix(self):
        pass


# ---------- Validation tests ----------


def test_assign_fanpt_container_rejects_non_child():
    class NotAContainer:
        pass

    with pytest.raises(TypeError, match="fanpt_container must be a child of FANPTContainer"):
        FANPTConstantTerms(fanpt_container=NotAContainer(), order=1)


def test_assign_order_type_and_value():
    cont = DummyContainer(
        nequation=3,
        nactive=3,
        active_energy=False,
        d_g_lambda=[1, 2, 3],
    )

    with pytest.raises(TypeError, match="order must be an integer"):
        FANPTConstantTerms(cont, order=1.5)

    with pytest.raises(ValueError, match="order must be non-negative"):
        FANPTConstantTerms(cont, order=-1)


def test_assign_previous_responses_shape_and_type_checks():
    nequation = 3
    nactive = 4

    cont = DummyContainer(
        nequation=nequation,
        nactive=nactive,
        active_energy=False,
        d_g_lambda=[0, 0, 0],
    )

    with pytest.raises(TypeError, match="previous_responses must be a numpy array"):
        FANPTConstantTerms(cont, order=2, previous_responses="not an array")

    bad_responses = np.array([1, 2], dtype=object)
    with pytest.raises(TypeError, match="elements of previous_responses must be numpy arrays"):
        FANPTConstantTerms(cont, order=2, previous_responses=bad_responses)

    good_rows = np.array([np.zeros(nactive), np.zeros(nactive)])
    wrong_shape = np.array([good_rows])

    with pytest.raises(
        ValueError,
        match=r"shape of previous_responses must be \(1, {}\)".format(nactive),
    ):
        FANPTConstantTerms(cont, order=2, previous_responses=wrong_shape)

    valid_responses = np.array([np.zeros(nactive)])
    cont.d2_g_lambda_wfnparams = np.zeros((nequation, nactive))

    FANPTConstantTerms(cont, order=2, previous_responses=valid_responses)


def test_assign_quasi_approximation_order_rejects_invalid_value():
    cont = DummyContainer(
        nequation=3,
        nactive=3,
        active_energy=False,
        d_g_lambda=[1, 2, 3],
    )

    with pytest.raises(ValueError, match="quasi_approximation_order must be 2 or 3"):
        FANPTConstantTerms(cont, order=1, quasi_approximation_order=4)


# ---------- Numerical behavior tests ----------


def test_gen_constant_terms_order1_negates_d_g_lambda():
    d_g_lambda = np.array([1.0, 2.0, 3.0])

    cont = DummyContainer(
        nequation=3,
        nactive=3,
        active_energy=False,
        d_g_lambda=d_g_lambda,
    )

    ct = FANPTConstantTerms(cont, order=1)

    np.testing.assert_allclose(ct.constant_terms, -d_g_lambda)


def test_gen_constant_terms_efree_order2():
    r"""Test energy-free second-order constant terms.

    For active_energy=False,

        constant_terms = -N * d2_g_lambda_wfnparams @ previous_response

    with N = 2.
    """
    nequation = 3
    nactive = 3

    d2_g_lambda_wfnparams = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=float,
    )

    previous_responses = np.array([[1.0, 2.0, 3.0]])

    cont = DummyContainer(
        nequation=nequation,
        nactive=nactive,
        active_energy=False,
        d_g_lambda=[0, 0, 0],
        d2_g_lambda_wfnparams=d2_g_lambda_wfnparams,
    )

    ct = FANPTConstantTerms(
        cont,
        order=2,
        previous_responses=previous_responses,
    )

    expected = -2.0 * d2_g_lambda_wfnparams.dot(previous_responses[-1])

    np.testing.assert_allclose(ct.constant_terms, expected)


def test_gen_constant_terms_eparam_order2_qao3_with_tensors():
    r"""Test active-energy second-order QAO3 constant terms.

    For order 2 and active energy,

        r_vec = 2 * E1 * p1

        constant_terms =
            -2 * M @ p1
            - Me @ r_vec
            - einsum(d2_g_wfnparams2, p1, p1)
    """
    nequation = 3
    nactive = 3

    d2_g_lambda_wfnparams = np.array(
        [
            [10, 11],
            [12, 13],
            [14, 15],
        ],
        dtype=float,
    )

    d2_g_e_wfnparams = np.array(
        [
            [2, 3],
            [4, 5],
            [6, 7],
        ],
        dtype=float,
    )

    d2_g_wfnparams2 = np.full((nequation, 2, 2), 0.5, dtype=float)

    response_1 = np.array([1.0, 2.0, 0.5])
    previous_responses = np.array([response_1])

    cont = DummyContainer(
        nequation=nequation,
        nactive=nactive,
        active_energy=True,
        d_g_lambda=[0, 0, 0],
        d2_g_lambda_wfnparams=d2_g_lambda_wfnparams,
        d2_g_e_wfnparams=d2_g_e_wfnparams,
        d2_g_wfnparams2=d2_g_wfnparams2,
    )

    ct = FANPTConstantTerms(
        cont,
        order=2,
        previous_responses=previous_responses,
        quasi_approximation_order=3,
    )

    wfn_response_1 = response_1[:-1]
    energy_response_1 = response_1[-1]

    r_vector = 2.0 * energy_response_1 * wfn_response_1

    expected = (
        -2.0 * d2_g_lambda_wfnparams.dot(wfn_response_1)
        - d2_g_e_wfnparams.dot(r_vector)
        - np.einsum(
            "mkl,k,l->m",
            d2_g_wfnparams2,
            wfn_response_1,
            wfn_response_1,
        )
    )

    np.testing.assert_allclose(ct.constant_terms, expected)


def test_gen_constant_terms_eparam_order3_qao3_with_tensors():
    r"""Test active-energy third-order QAO3 constant terms.

    For order 3 and active energy,

        r_vec = 3 * E1 * p2 + 3 * E2 * p1

        constant_terms =
            -3 * M @ p2
            - Me @ r_vec
            - 3 * einsum(d2_g_wfnparams2, p2, p1)
            - 3 * E1 * einsum(d3_g_e_wfnparams2, p2, p1)
            - 3 * einsum(d3_g_lambda_wfnparams2, p1, p1)
    """
    nequation = 3
    nactive = 3

    d2_g_lambda_wfnparams = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=float,
    )

    d2_g_e_wfnparams = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ],
        dtype=float,
    )

    d2_g_wfnparams2 = np.ones((nequation, 2, 2), dtype=float) * 0.2
    d3_g_e_wfnparams2 = np.ones((nequation, 2, 2), dtype=float) * 0.1
    d3_g_lambda_wfnparams2 = np.ones((nequation, 2, 2), dtype=float) * 0.05

    response_1 = np.array([1.0, 2.0, 0.5])
    response_2 = np.array([3.0, 4.0, 0.7])

    previous_responses = np.array([response_1, response_2])

    cont = DummyContainer(
        nequation=nequation,
        nactive=nactive,
        active_energy=True,
        d_g_lambda=[0, 0, 0],
        d2_g_lambda_wfnparams=d2_g_lambda_wfnparams,
        d2_g_e_wfnparams=d2_g_e_wfnparams,
        d2_g_wfnparams2=d2_g_wfnparams2,
        d3_g_e_wfnparams2=d3_g_e_wfnparams2,
        d3_g_lambda_wfnparams2=d3_g_lambda_wfnparams2,
    )

    ct = FANPTConstantTerms(
        cont,
        order=3,
        previous_responses=previous_responses,
        quasi_approximation_order=3,
    )

    wfn_response_1 = response_1[:-1]
    wfn_response_2 = response_2[:-1]

    energy_response_1 = response_1[-1]
    energy_response_2 = response_2[-1]

    r_vector = (
        3.0 * energy_response_1 * wfn_response_2
        + 3.0 * energy_response_2 * wfn_response_1
    )

    expected = (
        -3.0 * d2_g_lambda_wfnparams.dot(wfn_response_2)
        - d2_g_e_wfnparams.dot(r_vector)
        - 3.0
        * np.einsum(
            "mkl,k,l->m",
            d2_g_wfnparams2,
            wfn_response_2,
            wfn_response_1,
        )
        - 3.0
        * energy_response_1
        * np.einsum(
            "mkl,k,l->m",
            d3_g_e_wfnparams2,
            wfn_response_1,
            wfn_response_1,
        )
        - 3.0
        * np.einsum(
            "mkl,k,l->m",
            d3_g_lambda_wfnparams2,
            wfn_response_1,
            wfn_response_1,
        )
    )

    np.testing.assert_allclose(ct.constant_terms, expected)
