import numpy as np
import pytest

from fanpy.fanpt.containers.constant_terms import FANPTConstantTerms
from fanpy.fanpt.containers.base import FANPTContainer


class DummyContainer(FANPTContainer):
    r"""
    Minimal mock implementation of a FANPT container used for unit testing.

    This class provides just enough structure for FANPT routines—such as
    ``FANPTConstantTerms``—to read precomputed derivative data without requiring
    a full FanCI or PyCI backend. It bypasses Hamiltonian construction, overlap
    evaluation, and wavefunction logic, making it suitable for isolated and fast
    unit tests.

    Attributes
    ----------
    _nequation : int
        Number of FANPT equations.
    _nactive : int
        Number of active wavefunction parameters.
    active_energy : bool
        Whether the energy is treated as an active parameter in the perturbation
        expansion.
    d_g_lambda : np.ndarray
        First derivative ∂G/∂λ of the FANPT equations with respect to λ.
        Shape: ``(nequation,)``.
    d2_g_lambda_wfnparams : np.ndarray
        Mixed second derivatives ∂²G/∂λ∂pₖ.
        Shape: ``(nequation, nactive)``.
    d2_g_e_wfnparams : np.ndarray
        Mixed second derivatives ∂²G/∂E∂pₖ.
        Shape: ``(nequation, nactive-1)`` if ``active_energy=True``.
    d2_g_wfnparams2 : np.ndarray
        Pure second derivatives ∂²G/∂pₖ∂pₗ.
        Shape: ``(nequation, nactive, nactive)``.
    d3_g_e_wfnparams2 : np.ndarray
        Third derivatives ∂³G/∂E∂pₖ∂pₗ.
        Shape: ``(nequation, nactive-1, nactive-1)`` if ``active_energy=True``.
    d3_g_lambda_wfnparams2 : np.ndarray
        Third derivatives ∂³G/∂λ∂pₖ∂pₗ.
        Shape: ``(nequation, nactive, nactive)``.
    """

    def __init__(self,
                 nequation,
                 nactive,
                 active_energy,
                 d_g_lambda=None,
                 d2_g_lambda_wfnparams=None,
                 d2_g_e_wfnparams=None,
                 d2_g_wfnparams2=None,
                 d3_g_e_wfnparams2=None,
                 d3_g_lambda_wfnparams2=None):
        self._nequation = nequation
        self._nactive = nactive
        self.active_energy = active_energy
        self.d_g_lambda = np.array(d_g_lambda, dtype=float) 
        self.d2_g_lambda_wfnparams = (np.array(d2_g_lambda_wfnparams, dtype=float))
        self.d2_g_e_wfnparams = (np.array(d2_g_e_wfnparams, dtype=float))
        self.d2_g_wfnparams2 = (np.array(d2_g_wfnparams2, dtype=float))
        self.d3_g_e_wfnparams2 = (np.array(d3_g_e_wfnparams2, dtype=float))
        self.d3_g_lambda_wfnparams2 = (np.array(d3_g_lambda_wfnparams2, dtype=float))
                                       

    # read-only properties like real base
    @property
    def nequation(self):
        return self._nequation

    @property
    def nactive(self):
        return self._nactive

    # --- abstract methods implemented as no-ops for test instantiation ---
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
    cont = DummyContainer(nequation=3, nactive=3, active_energy=False, d_g_lambda=[1, 2, 3])

    with pytest.raises(TypeError, match="order must be an integer"):
        FANPTConstantTerms(cont, order=1.5)

    with pytest.raises(ValueError, match="order must be non-negative"):
        FANPTConstantTerms(cont, order=-1)

def test_assign_previous_responses_shape_and_type_checks():
    nequation, nactive = 3, 4
    cont = DummyContainer(nequation=nequation, nactive=nactive, active_energy=False, d_g_lambda=[0, 0, 0])

    with pytest.raises(TypeError, match="previous_responses must be a numpy array"):
        FANPTConstantTerms(cont, order=2, previous_responses="not an array")

    bad = np.array([1, 2], dtype=object)
    with pytest.raises(TypeError, match="elements of previous_responses must be numpy arrays"):
        FANPTConstantTerms(cont, order=2, previous_responses=bad)

    good_rows = np.array([np.zeros(nactive), np.zeros(nactive)])
    wrong_shape = np.array([good_rows])  # (1, 2, nactive)
    with pytest.raises(ValueError, match=r"shape of previous_responses must be \(1, {}\)".format(nactive)):
        FANPTConstantTerms(cont, order=2, previous_responses=wrong_shape)

    #correct shape passes (no exception) — also provide the needed d2 matrix
    ok = np.array([np.zeros(nactive)])
    cont.d2_g_lambda_wfnparams = np.zeros((nequation, nactive))
    FANPTConstantTerms(cont, order=2, previous_responses=ok)


# ---------- Numerical behavior tests ----------

def test_gen_constant_terms_order1_negates_d_g_lambda():
    cont = DummyContainer(nequation=3, nactive=3, active_energy=False, d_g_lambda=[1, 2, 3])
    ct = FANPTConstantTerms(cont, order=1)
    np.testing.assert_allclose(ct.constant_terms, -np.array([1, 2, 3], dtype=float))


def test_gen_constant_terms_efree_order2():
    """
    E-free branch (active_energy=False):
    constant_terms = -N * (d2_g_lambda_wfnparams @ prev[-1]), N=2
    """
    nequation, nactive = 3, 3
    d2 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]], dtype=float)
    prev = np.array([[1., 2., 3.]])  # shape (order-1, nactive)

    cont = DummyContainer(nequation=nequation, nactive=nactive, active_energy=False,
                          d2_g_lambda_wfnparams=d2, d_g_lambda=[0, 0, 0])

    ct = FANPTConstantTerms(cont, order=2, previous_responses=prev)
    expected = -2.0 * d2.dot(prev[-1])
    np.testing.assert_allclose(ct.constant_terms, expected)


def test_gen_constant_terms_eparam_order2_with_tensors():
    """
    Active energy (nactive=3 -> 2 wfn params + 1 energy).
    For order=2:
      r_vec = 2 * E1 * p1
      const = -2 * (M @ p1) - (Me @ r_vec) - einsum(d2, p0, p0)
    """
    nequation, nactive = 3, 3  # 2 params + energy
    M = np.array([[10, 11],
                  [12, 13],
                  [14, 15]], dtype=float)                 # d2_g_lambda_wfnparams
    Me = np.array([[2, 3],
                   [4, 5],
                   [6, 7]], dtype=float)                  # d2_g_e_wfnparams
    T2 = np.full((nequation, 2, 2), 0.5, dtype=float)     # d2_g_wfnparams2 (simple)

    # previous responses: (order-1, nactive)
    p0 = np.array([1.0, 2.0, 0.5])   # [k0, k1, E0]
    prev = np.array([p0])

    cont = DummyContainer(
        nequation=nequation, nactive=nactive, active_energy=True,
        d2_g_lambda_wfnparams=M, d2_g_e_wfnparams=Me, d2_g_wfnparams2=T2, d_g_lambda=[0, 0, 0]
    )

    ct = FANPTConstantTerms(cont, order=2, previous_responses=prev)

    p1 = prev[-1][:-1]                      # same as p0 here
    E1 = prev[-1][-1]
    r_vec = 2.0 * E1 * p1
    expected = -2.0 * M.dot(p1) - Me.dot(r_vec) - np.einsum("mkl,k,l->m", T2, p0[:-1], p0[:-1])
    np.testing.assert_allclose(ct.constant_terms, expected)


def test_gen_constant_terms_eparam_order3_with_tensors():
    """
    Active energy, order=3:
      r_vec = C(3,1)*E0*p1 + C(3,2)*E1*p0 = 3*E0*p1 + 3*E1*p0
      const = -3*(M @ p1) - (Me @ r_vec)
              - 3*einsum(d2, p1, p0)
              - 3*E0 * einsum(d3_e, p1, p0)
              - 3*einsum(d3_lambda, p0, p0)
    """
    nequation, nactive = 3, 3
    M = np.array([[1, 0],
                  [0, 1],
                  [1, 1]], dtype=float)
    Me = np.array([[1, 2],
                   [3, 4],
                   [5, 6]], dtype=float)
    T2 = np.ones((nequation, 2, 2), dtype=float) * 0.2
    T3e = np.ones((nequation, 2, 2), dtype=float) * 0.1
    T3l = np.ones((nequation, 2, 2), dtype=float) * 0.05

    p0 = np.array([1.0, 2.0, 0.5])   # [k0, k1, E0]
    p1 = np.array([3.0, 4.0, 0.7])   # [k0, k1, E1]
    prev = np.array([p0, p1])        # shape (2, 3)

    cont = DummyContainer(
        nequation=nequation, nactive=nactive, active_energy=True,
        d2_g_lambda_wfnparams=M, d2_g_e_wfnparams=Me,
        d2_g_wfnparams2=T2, d3_g_e_wfnparams2=T3e, d3_g_lambda_wfnparams2=T3l,
        d_g_lambda=[0, 0, 0]
    )

    ct = FANPTConstantTerms(cont, order=3, previous_responses=prev)

    k0 = p0[:-1]; k1 = p1[:-1]
    E0 = p0[-1];  E1 = p1[-1]
    r_vec = 3.0 * E0 * k1 + 3.0 * E1 * k0

    expected = (
        -3.0 * M.dot(k1)
        - Me.dot(r_vec)
        - 3.0 * np.einsum("mkl,k,l->m", T2, k1, k0)
        - 3.0 * E0 * np.einsum("mkl,k,l->m", T3e, k1, k0)
        - 3.0 * np.einsum("mkl,k,l->m", T3l, k0, k0)
    )
    np.testing.assert_allclose(ct.constant_terms, expected)
