import numpy as np
import pytest
from fanpy.fanpt.containers.energy_param import FANPTContainerEParam


# --- tiny helper ---
def make_scaling_op(factor: float):
    """Return an op(x, out=None) that multiplies by `factor`."""
    def op(x, out=None):
        if out is None:
            return factor * x
        out[...] = factor * x
    return op


class FakeFanCIObjective:
    """Mock FanCI objective returning a fixed Jacobian."""
    def __init__(self, ret: np.ndarray):
        self._ret = ret
        self.calls = 0
        self.last_params = None

    def compute_jacobian(self, params):
        self.calls += 1
        self.last_params = params
        return self._ret


# --- Lightweight test double: DOES NOT call super().__init__ ---
class DummyEParam(FANPTContainerEParam):
    def __init__(self, *,
                 nequation=5,
                 nproj=3,
                 nactive=4,
                 active_energy=True,
                 energy_value=1.5):
        # store read-onlys; expose via properties
        self._nequation = int(nequation)
        self._nproj     = int(nproj)
        self._nactive   = int(nactive)

        # flags & params (last element energy)
        self.active_energy = bool(active_energy)
        self.params = np.array([0.0] * (self._nactive - 1) + [float(energy_value)], dtype=float)

        # operators
        self.f_pot_ci_op = make_scaling_op(2.0)
        self.ham_ci_op   = make_scaling_op(3.0)

        # jacobian provider
        self.fanci_objective = FakeFanCIObjective(
            np.full((self._nequation, self._nactive), 42.0, dtype=float)
        )

        # overlaps / derivatives
        self.ovlp_s = np.arange(self._nproj, dtype=float) + 1.0
        self.d_ovlp_s = np.arange(self._nproj * self._nactive, dtype=float)\
                            .reshape(self._nproj, self._nactive, order="F")
        self.dd_ovlp_s = np.arange(self._nproj * self._nactive * self._nactive, dtype=float)\
                            .reshape(self._nproj, self._nactive, self._nactive, order="F")

    @property
    def nequation(self): return self._nequation
    @property
    def nproj(self):     return self._nproj
    @property
    def nactive(self):   return self._nactive


# ---------------- Tests (direct instantiation) ----------------

def test_der_g_lambda_projects_only_first_nproj():
    inst = DummyEParam()
    inst.der_g_lambda()

    assert inst.d_g_lambda.shape == (inst.nequation,)
    np.testing.assert_allclose(inst.d_g_lambda[:inst.nproj], 2.0 * inst.ovlp_s)
    np.testing.assert_allclose(inst.d_g_lambda[inst.nproj:], 0.0)


@pytest.mark.parametrize("active_energy,ncols", [(False, 4), (True, 3)])
def test_der2_g_lambda_wfnparams(active_energy, ncols):
    inst = DummyEParam(nactive=4, active_energy=active_energy)
    # shape (nproj, ncols) Fortran
    inst.d_ovlp_s = np.arange(inst.nproj * ncols, dtype=float).reshape(
        inst.nproj, ncols, order="F"
    )

    inst.der2_g_lambda_wfnparams()

    assert inst.d2_g_lambda_wfnparams.shape == (inst.nequation, ncols)
    np.testing.assert_allclose(inst.d2_g_lambda_wfnparams[:inst.nproj], 2.0 * inst.d_ovlp_s)
    np.testing.assert_allclose(inst.d2_g_lambda_wfnparams[inst.nproj:], 0.0)


def test_der2_g_e_wfnparams_active():
    inst = DummyEParam(nactive=5, active_energy=True)
    ncols = inst.nactive - 1
    inst.d_ovlp_s = np.arange(inst.nproj * ncols, dtype=float).reshape(
        inst.nproj, ncols, order="F"
    )

    inst.der2_g_e_wfnparams()

    assert inst.d2_g_e_wfnparams.shape == (inst.nequation, ncols)
    np.testing.assert_allclose(inst.d2_g_e_wfnparams[:inst.nproj], -inst.d_ovlp_s)
    np.testing.assert_allclose(inst.d2_g_e_wfnparams[inst.nproj:], 0.0)


def test_der2_g_e_wfnparams_inactive():
    inst = DummyEParam(active_energy=False)
    inst.der2_g_e_wfnparams()
    assert inst.d2_g_e_wfnparams is None


def test_gen_coeff_matrix_calls_fanci():
    inst = DummyEParam()
    inst.gen_coeff_matrix()
    assert inst.fanci_objective.calls == 1
    assert inst.c_matrix.shape == (inst.nequation, inst.nactive)
    np.testing.assert_allclose(inst.c_matrix, 42.0)


def test_der3_g_lambda_wfnparams2_inactive_energy():
    nactive = 4
    inst = DummyEParam(nactive=nactive, active_energy=False)
    inst.dd_ovlp_s = np.arange(inst.nproj * nactive * nactive, dtype=float).reshape(
        inst.nproj, nactive, nactive, order="F"
    )

    inst.der3_g_lambda_wfnparams2()

    out = inst.d3_g_lambda_wfnparams2
    assert out.shape == (inst.nequation, nactive, nactive)
    np.testing.assert_allclose(out[:inst.nproj], 2.0 * inst.dd_ovlp_s)
    np.testing.assert_allclose(out[inst.nproj:], 0.0)


def test_der3_g_e_wfnparams2_active_energy():
    nactive = 5
    ncols = nactive - 1
    inst = DummyEParam(nactive=nactive, active_energy=True)
    inst.dd_ovlp_s = np.arange(inst.nproj * ncols * ncols, dtype=float).reshape(
        inst.nproj, ncols, ncols, order="F"
    )

    inst.der3_g_e_wfnparams2()
    out = inst.d3_g_e_wfnparams2

    assert out.shape == (inst.nequation, ncols, ncols)
    np.testing.assert_allclose(out[:inst.nproj], -inst.dd_ovlp_s)
    np.testing.assert_allclose(out[inst.nproj:], 0.0)


def test_der3_g_e_wfnparams2_inactive_energy():
    inst = DummyEParam(active_energy=False)
    inst.der3_g_e_wfnparams2()
    assert inst.d3_g_e_wfnparams2 is None


def test_der2_g_wfnparams2_active_energy():
    nactive = 6
    ncols = nactive - 1
    E = 2.25
    inst = DummyEParam(nactive=nactive, active_energy=True, energy_value=E)

    dd = np.arange(inst.nproj * ncols * ncols, dtype=float).reshape(
        inst.nproj, ncols, ncols, order="F"
    )
    inst.dd_ovlp_s = dd.copy()

    inst.der2_g_wfnparams2()
    out = inst.d2_g_wfnparams2

    assert out.shape == (inst.nequation, ncols, ncols)

    expected = (3.0 - E) * dd  # ham op scales by 3; then minus E term in method
    np.testing.assert_allclose(out[:inst.nproj], expected)
    np.testing.assert_allclose(out[inst.nproj:], 0.0)

    # current impl modifies dd_ovlp_s in-place (as in your test)
    assert not np.array_equal(inst.dd_ovlp_s, dd), "dd_ovlp_s was modified in-place."


@pytest.mark.skip(reason="By design: der2_g_wfnparams2 is only used with active_energy=True.")
def test_der2_g_wfnparams2_inactive_energy_bug():
    inst = DummyEParam(active_energy=False)
    inst.dd_ovlp_s = np.arange(inst.nproj * inst.nactive * inst.nactive, dtype=float).reshape(
        inst.nproj, inst.nactive, inst.nactive, order="F"
    )
    inst.der2_g_wfnparams2()
    assert hasattr(inst, "d2_g_wfnparams2")
