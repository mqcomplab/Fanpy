import numpy as np
import pytest

from fanpy.fanpt.containers.energy_free import FANPTContainerEFree


# ---- tiny helpers ----
def make_scaling_op(factor: float):
    """Return an op(x, out=None) that multiplies by `factor` (mimics PyCI op)."""
    def op(x, out=None):
        if out is None:
            return factor * x
        out[...] = factor * x
    return op


def make_jacobian_mock(matrix: np.ndarray):
    """Return an object exposing compute_jacobian + call tracking."""
    class _JacMock:
        def __init__(self, ret):
            self._ret = ret
            self.calls = 0
            self.last_params = None

        def compute_jacobian(self, params):
            self.calls += 1
            self.last_params = params
            return self._ret
    return _JacMock(matrix)


# ---- Simple interface ----
class FakeObjective:
    """Mock objective.mask to test 'energy cannot be active' check."""
    def __init__(self, mask_last_active=False):
        # mask[-1] == True would indicate energy active (invalid for EFree)
        self.mask = np.array([False, False, False, mask_last_active], dtype=bool)


class FakeFanCIInterface:
    """Mock interface exposing .objective.mask."""
    def __init__(self, mask_last_active=False):
        self.objective = FakeObjective(mask_last_active)


# ---- Lightweight dummy container ----
class DummyEFree(FANPTContainerEFree):
    """Minimal EFree container for tests; wires only what the methods need."""
    def __init__(self, *,
                 nequation=6,
                 nproj=4,
                 nactive=3,
                 ref_sd=1,
                 inorm=False,
                 energy_value=1.25):
        # store “read-only” counters and expose via properties
        self._nequation = int(nequation)
        self._nproj     = int(nproj)
        self._nactive   = int(nactive)

        # E-free: energy must not be active
        self.active_energy = False

        # normalization choices
        self.inorm  = bool(inorm)
        self.ref_sd = int(ref_sd)

        # operators: V scales by 2, H scales by 3
        self.f_pot_ci_op = make_scaling_op(2.0)
        self.ham_ci_op   = make_scaling_op(3.0)

        # overlaps / derivatives (Fortran order where used that way)
        self.ovlp_s = np.arange(1, self._nproj + 1, dtype=float)  # [1,2,3,...]
        self.d_ovlp_s = np.arange(self._nproj * self._nactive, dtype=float)\
                            .reshape(self._nproj, self._nactive, order="F")

        # jacobian provider: starts at 100 everywhere
        J = np.full((self._nequation, self._nactive), 100.0, dtype=float)
        self.fanci_objective = make_jacobian_mock(J)

        # params & energy (energy used in inorm=False branch of gen_coeff_matrix)
        self.params = np.zeros(self._nactive, dtype=float)  # not used directly by these tests
        self.energy = float(energy_value)

    @property
    def nequation(self): return self._nequation
    @property
    def nproj(self):     return self._nproj
    @property
    def nactive(self):   return self._nactive


# ---- __init__ guard: energy cannot be active in EFree ----
def test_init_rejects_active_energy_flag():
    iface_bad = FakeFanCIInterface(mask_last_active=True)
    with pytest.raises(TypeError, match="energy cannot be an active parameter"):
        FANPTContainerEFree(
            fanci_interface=iface_bad, params=None, ham0=None, ham1=None
        )



# ---- der_g_lambda ----
@pytest.mark.parametrize("inorm", [True, False])
def test_der_g_lambda_efree_adjustment(inorm):
    """
    super(): d_g_lambda[:nproj] = 2 * ovlp_s
    EFree:
      if inorm=True:   subtract d_ref * ovlp_s = (2*ovlp_ref)*ovlp_s
      if inorm=False:  subtract (d_ref * ovlp_s / ovlp_ref) = 2*ovlp_s  -> zeros
    """
    inst = DummyEFree(inorm=inorm, nproj=4, nactive=3, nequation=6, ref_sd=1)

    inst.der_g_lambda()

    super_val = 2.0 * inst.ovlp_s
    d_ref = super_val[inst.ref_sd]
    if inorm:
        expected = super_val - d_ref * inst.ovlp_s
    else:
        expected = super_val - (d_ref * inst.ovlp_s / inst.ovlp_s[inst.ref_sd])

    np.testing.assert_allclose(inst.super_d_g_lambda[:inst.nproj], super_val)
    np.testing.assert_allclose(inst.d_g_lambda[:inst.nproj], expected)
    np.testing.assert_allclose(inst.d_g_lambda[inst.nproj:], 0.0)


# ---- der2_g_lambda_wfnparams ----
@pytest.mark.parametrize("inorm", [True, False])
def test_der2_g_lambda_wfnparams_efree(inorm):
    inst = DummyEFree(inorm=inorm, nproj=4, nactive=3, nequation=6, ref_sd=2)

    # Deterministic d_ovlp grid
    inst.d_ovlp_s = np.arange(inst.nproj * inst.nactive, dtype=float).reshape(
        inst.nproj, inst.nactive, order="F"
    )
    inst.der_g_lambda()
    inst.der2_g_lambda_wfnparams()

    super_mat = 2.0 * inst.d_ovlp_s               # (nproj, nactive)
    super_ref = inst.super_d_g_lambda[inst.ref_sd] # = 2*ovlp_ref

    if inorm:
        expected = super_mat.copy()
        # subtract row_ref(super) * ovlp_s (outer product over columns)
        expected -= super_mat[inst.ref_sd][None, :] * inst.ovlp_s[:inst.nproj, None]
        # subtract super_ref * d_ovlp_s
        expected -= super_ref * inst.d_ovlp_s
        np.testing.assert_allclose(inst.d2_g_lambda_wfnparams[:inst.nproj], expected)
    else:
        # Should collapse to zeros in projected block
        np.testing.assert_allclose(inst.d2_g_lambda_wfnparams[:inst.nproj], 0.0)

    np.testing.assert_allclose(inst.d2_g_lambda_wfnparams[inst.nproj:], 0.0)


# ---- gen_coeff_matrix ----
@pytest.mark.parametrize("inorm", [True, False])
def test_gen_coeff_matrix_efree_adjustment(inorm):
    nproj, nactive, nequation = 4, 3, 6
    ref_sd = 1
    E = 1.25

    inst = DummyEFree(
        inorm=inorm, nproj=nproj, nactive=nactive, nequation=nequation,
        ref_sd=ref_sd, energy_value=E
    )

    inst.d_ovlp_s = np.arange(nproj * nactive, dtype=float).reshape(nproj, nactive, order="F")

    inst.gen_coeff_matrix()

    expected = np.full((nequation, nactive), 100.0)
    f_proj = 3.0 * inst.d_ovlp_s
    ovlp_ref = inst.ovlp_s[ref_sd]

    if inorm:
        expected[:nproj] -= f_proj[ref_sd][None, :] * inst.ovlp_s[:nproj, None]
    else:
        expected[:nproj] -= (
            (f_proj[ref_sd][None, :] - E * inst.d_ovlp_s[ref_sd][None, :])
            * inst.ovlp_s[:nproj, None] / ovlp_ref
        )

    np.testing.assert_allclose(inst.c_matrix, expected)
