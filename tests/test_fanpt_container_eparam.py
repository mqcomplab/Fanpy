import numpy as np
import pytest


from fanpy.fanpt.containers.energy_param import FANPTContainerEParam


# --- Test helpers ---
class FakeLinearOp:
    """Mock for f_pot_ci_op: doubles the input vector."""
    def __call__(self, x, out=None):
        if out is None:
            return 2.0 * x
        out[...] = 2.0 * x


class FakeFanCIObjective:
    """Mock for fanci_objective.compute_jacobian"""
    def __init__(self, ret):
        self._ret = ret
        self.calls = 0
        self.last_params = None

    def compute_jacobian(self, params):
        self.calls += 1
        self.last_params = params
        return self._ret


@pytest.fixture
def make_instance():
    """
    Create a dummy subclass so we can override read-only @property attributes
    like nequation, nproj, nactive.
    """
    def _factory(*, nequation=5, nproj=3, nactive=4, active_energy=False):
        class Dummy(FANPTContainerEParam):
            @property
            def nequation(self):
                return nequation

            @property
            def nproj(self):
                return nproj

            @property
            def nactive(self):
                return nactive

        inst = object.__new__(Dummy)  # Don’t call real __init__
        inst.active_energy = active_energy
        inst.ovlp_s = np.arange(nproj, dtype=float) + 1  # [1, 2, 3...]
        inst.d_ovlp_s = np.arange(nproj * nactive, dtype=float).reshape(nproj, nactive, order="F")
        inst.f_pot_ci_op = FakeLinearOp()
        inst.fanci_objective = FakeFanCIObjective(
            ret=np.full((nequation, nactive), fill_value=42.0)
        )
        inst.params = np.linspace(0.1, 1.0, 7)
        return inst
    return _factory


# --- Tests ---

def test_der_g_lambda_projects_only_first_nproj(make_instance):
    inst = make_instance()
    inst.der_g_lambda()

    assert inst.d_g_lambda.shape == (inst.nequation,)
    np.testing.assert_allclose(inst.d_g_lambda[:inst.nproj], 2 * inst.ovlp_s)
    np.testing.assert_allclose(inst.d_g_lambda[inst.nproj:], 0.0)


@pytest.mark.parametrize("active_energy,ncols", [(False, 4), (True, 3)])
def test_der2_g_lambda_wfnparams(make_instance, active_energy, ncols):
    inst = make_instance(nactive=4, active_energy=active_energy)
    inst.d_ovlp_s = np.arange(inst.nproj * ncols).reshape(inst.nproj, ncols, order="F")

    inst.der2_g_lambda_wfnparams()

    assert inst.d2_g_lambda_wfnparams.shape == (inst.nequation, ncols)
    np.testing.assert_allclose(inst.d2_g_lambda_wfnparams[:inst.nproj], 2 * inst.d_ovlp_s)
    np.testing.assert_allclose(inst.d2_g_lambda_wfnparams[inst.nproj:], 0.0)


def test_der2_g_e_wfnparams_active(make_instance):
    inst = make_instance(nactive=5, active_energy=True)
    ncols = inst.nactive - 1
    inst.d_ovlp_s = np.arange(inst.nproj * ncols).reshape(inst.nproj, ncols, order="F")

    inst.der2_g_e_wfnparams()

    assert inst.d2_g_e_wfnparams.shape == (inst.nequation, ncols)
    np.testing.assert_allclose(inst.d2_g_e_wfnparams[:inst.nproj], -inst.d_ovlp_s)
    np.testing.assert_allclose(inst.d2_g_e_wfnparams[inst.nproj:], 0)


def test_der2_g_e_wfnparams_inactive(make_instance):
    inst = make_instance(active_energy=False)
    inst.der2_g_e_wfnparams()
    assert inst.d2_g_e_wfnparams is None


def test_gen_coeff_matrix_calls_fanci(make_instance):
    inst = make_instance()
    inst.gen_coeff_matrix()

    assert inst.fanci_objective.calls == 1
    assert inst.c_matrix.shape == (inst.nequation, inst.nactive)
    np.testing.assert_allclose(inst.c_matrix, 42)
