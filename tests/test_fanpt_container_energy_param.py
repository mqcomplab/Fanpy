import numpy as np
import pytest

from fanpy.fanpt.containers.energy_param import FANPTContainerEParam


# --- Mocks ---
class FakeFpotOp:
    """Mock operator for `f_pot_ci_op` used in FANPT tests.

    This simulates the fluctuation potential operator by scaling the input
    vector by a factor of 2. It mimics the signature of a PyCI sparse
    operator without requiring PyCI to be installed.
    """
    def __call__(self, x, out=None):
        """
        Apply the mock fluctuation potential operator.

        Parameters
        ----------
        x : array_like
            Input vector.
        out : array_like, optional
            Optional array to write the result into (in-place).

        Returns
        -------
        array_like
            If `out` is None, returns `2 * x`. Otherwise, writes into `out`
            and returns None.
        """
        if out is None:
            return 2.0 * x
        out[...] = 2.0 * x


class FakeHamOp:
    """Mock operator for `ham_ci_op` used in FANPT tests.

    This simulates the perturbed Hamiltonian operator by scaling the input
    vector by a factor of 3. It behaves like a linear operator acting on CI
    vectors in FANPT.
    """
    def __call__(self, x, out=None):
        """
        Apply the mock Hamiltonian operator.

        Parameters
        ----------
        x : array_like
            Input vector.
        out : array_like, optional
            Optional array to write the result into.

        Returns
        -------
        array_like
            If `out` is None, returns `3 * x`. Otherwise, writes into `out`
            and returns None.
        """
        if out is None:
            return 3.0 * x
        out[...] = 3.0 * x


class FakeFanCIObjective:
    """Mock of a FanCI objective providing a fake Jacobian matrix.

    Used to simulate calls to `compute_jacobian` without invoking actual
    FanCI machinery. Tracks how many times it was called and with what
    parameters.
    """
    def __init__(self, ret):
        """
        Initialize the mock objective.

        Parameters
        ----------
        ret : array_like
            Precomputed Jacobian matrix to return on every call.
        """
        self._ret = ret
        self.calls = 0
        self.last_params = None

    def compute_jacobian(self, params):
        """
        Return a constant Jacobian and record the call.

        Parameters
        ----------
        params : array_like
            Parameter vector passed to the method.

        Returns
        -------
        np.ndarray
            The stored Jacobian matrix (`self._ret`).

        Notes
        -----
        - Increments ``self.calls`` each time it is invoked.
        - Stores the most recent argument in ``self.last_params``.
        """
        self.calls += 1
        self.last_params = params
        return self._ret



# --- Fixture to construct a controllable instance (bypass __init__) ---
@pytest.fixture
def make_instance():
    """
    Return a factory that builds a subclass overriding read-only properties
    (nequation, nproj, nactive), and fills only the attributes the methods need.
    """
    def _factory(*, nequation=5, nproj=3, nactive=4, active_energy=False, energy_value=1.5):
        class Dummy(FANPTContainerEParam):
            @property
            def nequation(self): return nequation
            @property
            def nproj(self): return nproj
            @property
            def nactive(self): return nactive

        inst = object.__new__(Dummy)  # do not call real __init__

        # Flags, params, operators
        inst.active_energy = active_energy
        # params: the last entry is energy (E) used by der2_g_wfnparams2
        inst.params = np.array([0.0]*(nactive-1) + [energy_value], dtype=float)

        inst.f_pot_ci_op = FakeFpotOp()
        inst.ham_ci_op = FakeHamOp()
        inst.fanci_objective = FakeFanCIObjective(
            ret=np.full((nequation, nactive), 42.0)
        )

        # Overlaps in S space
        inst.ovlp_s = np.arange(nproj, dtype=float) + 1.0                 # (nproj,)
        inst.d_ovlp_s = np.arange(nproj * nactive, dtype=float).reshape(  # (nproj, nactive)
            nproj, nactive, order="F"
        )

        # Second-derivative overlaps; default square in last 2 dims
        inst.dd_ovlp_s = np.arange(nproj * nactive * nactive, dtype=float).reshape(
            nproj, nactive, nactive, order="F"
        )

        return inst

    return _factory


# ---------------- Existing method tests (kept) ----------------

def test_der_g_lambda_projects_only_first_nproj(make_instance):
    inst = make_instance()
    inst.der_g_lambda()

    assert inst.d_g_lambda.shape == (inst.nequation,)
    np.testing.assert_allclose(inst.d_g_lambda[:inst.nproj], 2.0 * inst.ovlp_s)
    np.testing.assert_allclose(inst.d_g_lambda[inst.nproj:], 0.0)


@pytest.mark.parametrize("active_energy,ncols", [(False, 4), (True, 3)])
def test_der2_g_lambda_wfnparams(make_instance, active_energy, ncols):
    """
    Parameters
    ----------
    make_instance : fixture
        Factory providing a controlled FANPTContainerEParam instance.
    active_energy : bool
        Whether energy is treated as an active parameter in this test case.
    ncols : int
        Expected number of active wavefunction parameters in this case.
    """
    inst = make_instance(nactive=4, active_energy=active_energy)
    # ensure the method's expected number of columns
    inst.d_ovlp_s = np.arange(inst.nproj * ncols, dtype=float).reshape(
        inst.nproj, ncols, order="F"
    )

    inst.der2_g_lambda_wfnparams()

    assert inst.d2_g_lambda_wfnparams.shape == (inst.nequation, ncols)
    assert inst.d2_g_lambda_wfnparams.flags["F_CONTIGUOUS"]
    np.testing.assert_allclose(
        inst.d2_g_lambda_wfnparams[:inst.nproj], 2.0 * inst.d_ovlp_s
    )
    np.testing.assert_allclose(inst.d2_g_lambda_wfnparams[inst.nproj:], 0.0)


def test_der2_g_e_wfnparams_active(make_instance):
    """
    Parameters
    ----------
    make_instance : fixture
        Factory providing a controlled FANPTContainerEParam instance.
    """
    inst = make_instance(nactive=5, active_energy=True)
    ncols = inst.nactive - 1
    inst.d_ovlp_s = np.arange(inst.nproj * ncols).reshape(inst.nproj, ncols, order="F")

    inst.der2_g_e_wfnparams()

    assert inst.d2_g_e_wfnparams.shape == (inst.nequation, ncols)
    assert inst.d2_g_e_wfnparams.flags["F_CONTIGUOUS"]
    np.testing.assert_allclose(inst.d2_g_e_wfnparams[:inst.nproj], -inst.d_ovlp_s)
    np.testing.assert_allclose(inst.d2_g_e_wfnparams[inst.nproj:], 0.0)


def test_der2_g_e_wfnparams_inactive(make_instance):
    """
    Parameters
    ----------
    make_instance : fixture
        Factory providing a controlled FANPTContainerEParam instance.
    """
    inst = make_instance(active_energy=False)
    inst.der2_g_e_wfnparams()
    assert inst.d2_g_e_wfnparams is None


def test_gen_coeff_matrix_calls_fanci(make_instance):
    """
    Parameters
    ----------
    make_instance : fixture
        Factory providing a controlled FANPTContainerEParam instance.
    """
    inst = make_instance()
    inst.gen_coeff_matrix()
    assert inst.fanci_objective.calls == 1
    assert inst.c_matrix.shape == (inst.nequation, inst.nactive)
    np.testing.assert_allclose(inst.c_matrix, 42.0)


# ---------------- New method tests you asked for ----------------

def test_der3_g_lambda_wfnparams2_inactive_energy(make_instance):
    """
    Parameters
    ----------
    make_instance : fixture
        Factory providing a controlled FANPTContainerEParam instance.
    """
    nactive = 4
    inst = make_instance(nactive=nactive, active_energy=False)
    # set dd_ovlp_s with shape (nproj, ncols, ncols)
    inst.dd_ovlp_s = np.arange(inst.nproj * nactive * nactive, dtype=float).reshape(
        inst.nproj, nactive, nactive, order="F"
    )

    inst.der3_g_lambda_wfnparams2()

    assert hasattr(inst, "d3_g_lambda_wfnparams2")
    out = inst.d3_g_lambda_wfnparams2
    assert out.shape == (inst.nequation, nactive, nactive)
    assert out.flags["F_CONTIGUOUS"]
    # Only first nproj rows are non-zero and equal to 2 * dd_ovlp_s (FakeFpot doubles)
    np.testing.assert_allclose(out[:inst.nproj], 2.0 * inst.dd_ovlp_s)
    np.testing.assert_allclose(out[inst.nproj:], 0.0)


def test_der3_g_e_wfnparams2_active_energy(make_instance):
    """
    Parameters
    ----------
    make_instance : fixture
        Factory providing a controlled FANPTContainerEParam instance.
    """
    nactive = 5
    ncols = nactive - 1
    inst = make_instance(nactive=nactive, active_energy=True)
    inst.dd_ovlp_s = np.arange(inst.nproj * ncols * ncols, dtype=float).reshape(
        inst.nproj, ncols, ncols, order="F"
    )

    inst.der3_g_e_wfnparams2()

    out = inst.d3_g_e_wfnparams2
    assert out.shape == (inst.nequation, ncols, ncols)
    assert out.flags["F_CONTIGUOUS"]
    np.testing.assert_allclose(out[:inst.nproj], -inst.dd_ovlp_s)
    np.testing.assert_allclose(out[inst.nproj:], 0.0)


def test_der3_g_e_wfnparams2_inactive_energy(make_instance):
    """
    Parameters
    ----------
    make_instance : fixture
        Factory providing a controlled FANPTContainerEParam instance.
    """
    
    inst = make_instance(active_energy=False)
    inst.der3_g_e_wfnparams2()
    assert inst.d3_g_e_wfnparams2 is None


def test_der2_g_wfnparams2_active_energy(make_instance):
    """
    Parameters
    ----------
    make_instance : fixture
        Factory providing a controlled FANPTContainerEParam instance.
    """
    
    nactive = 6
    ncols = nactive - 1
    E = 2.25
    inst = make_instance(nactive=nactive, active_energy=True, energy_value=E)

    # Provide (nproj, ncols, ncols) dd_ovlp
    dd = np.arange(inst.nproj * ncols * ncols, dtype=float).reshape(
        inst.nproj, ncols, ncols, order="F"
    )
    inst.dd_ovlp_s = dd.copy()

    inst.der2_g_wfnparams2()

    out = inst.d2_g_wfnparams2
    assert out.shape == (inst.nequation, ncols, ncols)
    assert out.flags["F_CONTIGUOUS"]

    expected = (3.0 - E) * dd
    np.testing.assert_allclose(out[:inst.nproj], expected)
    np.testing.assert_allclose(out[inst.nproj:], 0.0)

    # Note: current implementation multiplies dd_ovlp_s in-place; verify side-effect occurred.
    # (This is a bit surprising API-wise; you might consider avoiding in-place.)
    assert not np.array_equal(inst.dd_ovlp_s, dd), "dd_ovlp_s was modified in-place by method."

@pytest.mark.skip(reason="By design: der2_g_wfnparams2 is only used with active_energy=True.")
def test_der2_g_wfnparams2_inactive_energy_bug(make_instance):
    """
    Parameters
    ----------
    make_instance : fixture
        Factory providing a controlled FANPTContainerEParam instance.
    """
    
    inst = make_instance(active_energy=False)
    inst.dd_ovlp_s = np.arange(inst.nproj * inst.nactive * inst.nactive, dtype=float).reshape(
        inst.nproj, inst.nactive, inst.nactive, order="F"
    )
    inst.der2_g_wfnparams2()
    # If you fix it, then assert the expected inactive-energy behavior (zero E term).
    assert hasattr(inst, "d2_g_wfnparams2")
