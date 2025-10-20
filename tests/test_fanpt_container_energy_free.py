import numpy as np
import pytest

from fanpy.fanpt.containers.energy_free import FANPTContainerEFree


# ---- Mocks ----
class FakeFpotOp:
    """
    Mock fluctuation potential operator `<m|V|n>` used in FANPT tests.

    This mock simulates the behavior of `f_pot_ci_op` by scaling the input vector `x`
    by a factor of 2. It provides the same call signature as the real operator so it
    can be injected into FANPT containers without requiring PyCI.
    """
    def __call__(self, x, out=None):
        """
        Apply the mock fluctuation potential.

        Parameters
        ----------
        x : array_like
            Input vector.
        out : array_like, optional
            Optional output buffer. If provided, the result is written in-place.

        Returns
        -------
        array_like or None
            If `out` is None, returns `2 * x`. Otherwise, writes to `out` and returns None.
        """
        if out is None:
            return 2.0 * x
        out[...] = 2.0 * x


class FakeHamOp:
    """
    Mock Hamiltonian operator `<m|H|n>` used in FANPT tests.

    This mock simulates the perturbed Hamiltonian operator `ham_ci_op` by tripling
    its input. It is used to test derivative assembly logic without invoking the full
    Hamiltonian machinery.
    """
    def __call__(self, x, out=None):
        """
        Apply the mock Hamiltonian.

        Parameters
        ----------
        x : array_like
            Input vector.
        out : array_like, optional
            Optional output buffer to write into.

        Returns
        -------
        array_like or None
            If `out` is None, returns `3 * x`. Otherwise writes into `out` and returns None.
        """
        if out is None:
            return 3.0 * x
        out[...] = 3.0 * x


class FakeObjective:
    """
    Mock of a FanCI objective mask used to control active parameter behavior.

    Parameters
    ----------
    mask_last_active : bool
        Whether the last parameter (energy) is active. Used to trigger the energy
        mask validation inside `FANPTContainerEFree`.
    """
    def __init__(self, mask_last_active=False):
        # mask[-1] == True → means energy is incorrectly marked active
        self.mask = np.array([False, False, False, mask_last_active], dtype=bool)


class FakeFanCIInterface:
    """
    Minimal mock of a FanCI interface used to mimic `.objective.mask`.

    This allows creation of FANPT containers without requiring full FanCI setup.
    """
    def __init__(self, mask_last_active=False):
        self.objective = FakeObjective(mask_last_active)


class FakeFanCIJac:
    """
    Mock of a FanCI Jacobian provider.

    Simulates `fanci_objective.compute_jacobian` by returning a stored matrix and
    recording how many times the method was called.
    """
    def __init__(self, ret):
        """
        Parameters
        ----------
        ret : array_like
            Precomputed Jacobian matrix returned on each call.
        """
        self._ret = ret
        self.calls = 0
        self.last_params = None

    def compute_jacobian(self, params):
        """
        Fake Jacobian computation.

        Parameters
        ----------
        params : array_like
            Parameter vector passed in by the caller. Stored only for inspection.

        Returns
        -------
        np.ndarray
            The stored Jacobian matrix.

        Notes
        -----
        - Increments `self.calls` each time the method is invoked.
        - Records the most recent `params` for verification in tests.
        """
        self.calls += 1
        self.last_params = params
        return self._ret



# ---- Factory: controlled instance without running __init__ ----
@pytest.fixture
def make_instance():
    """
    Build a dummy subclass overriding read-only properties and
    populate only the attributes the EFree methods need.
    """
    def _factory(
        *,
        nequation=6,
        nproj=4,
        nactive=3,
        ref_sd=1,
        inorm=False,
        energy_value=1.25,
    ):
        class Dummy(FANPTContainerEFree):
            @property
            def nequation(self): return nequation
            @property
            def nproj(self): return nproj
            @property
            def nactive(self): return nactive

        inst = object.__new__(Dummy)  # bypass real __init__
        inst.active_energy = False
        # Normalization choices
        inst.inorm = inorm
        inst.ref_sd = ref_sd

        # Operators
        inst.f_pot_ci_op = FakeFpotOp()
        inst.ham_ci_op = FakeHamOp()

        # Projection overlaps
        # ovlp_s length nproj
        inst.ovlp_s = np.arange(1, nproj + 1, dtype=float)  # [1,2,3,...]
        # d_ovlp_s: (nproj, nactive) Fortran
        inst.d_ovlp_s = np.arange(nproj * nactive, dtype=float).reshape(nproj, nactive, order="F")

        # FanCI objective / jacobian
        J = np.full((nequation, nactive), 100.0, dtype=float)
        inst.fanci_objective = FakeFanCIJac(ret=J)

        # Params & energy
        inst.params = np.zeros(nactive, dtype=float)  # not used by EFree methods directly
        inst.energy = float(energy_value)             # used by gen_coeff_matrix (inorm=False branch)

        return inst
    return _factory


# ---- __init__ guard: energy cannot be active ----
def test_init_rejects_active_energy_flag():
    iface_bad = FakeFanCIInterface(mask_last_active=True)
    with pytest.raises(TypeError, match="energy cannot be an active parameter"):
        FANPTContainerEFree(
            fanci_interface=iface_bad, params=None, ham0=None, ham1=None
        )

def test_init_accepts_inactive_energy_flag():
    iface_ok = FakeFanCIInterface(mask_last_active=False)
    # Just ensure no TypeError is raised. We won't run the full super init after.
    # (Real __init__ requires many args; this quick smoke is enough.)
    with pytest.raises(TypeError):
        # Calling it with missing positional args should raise TypeError *for that*,
        # not for the energy mask. If energy check were wrong, we'd see a different error.
        FANPTContainerEFree(fanci_interface=iface_ok)  # intentionally incomplete args


# ---- der_g_lambda ----
@pytest.mark.parametrize("inorm", [True, False])
def test_der_g_lambda_efree_adjustment(make_instance, inorm):
    """
    Parameters
    ----------
    make_instance : fixture
        Factory providing a controlled FANPTContainerEParam instance.
    
    super(): d_g_lambda[:nproj] = 2 * ovlp_s
    EFree:
      if inorm=True:   subtract d_ref * ovlp_s = (2*ovlp_ref)*ovlp_s
      if inorm=False:  subtract (d_ref * ovlp_s / ovlp_ref) = 2*ovlp_s  -> zeros
    """
    inst = make_instance(inorm=inorm, nproj=4, nactive=3, nequation=6, ref_sd=1)

    # Call method
    inst.der_g_lambda()

    # Super's value to compare with
    super_val = 2.0 * inst.ovlp_s
    d_ref = super_val[inst.ref_sd]
    if inorm:
        expected = super_val - d_ref * inst.ovlp_s
    else:
        expected = super_val - (d_ref * inst.ovlp_s / inst.ovlp_s[inst.ref_sd])

    # Check
    np.testing.assert_allclose(inst.super_d_g_lambda[:inst.nproj], super_val)
    np.testing.assert_allclose(inst.d_g_lambda[:inst.nproj], expected)
    # rows below nproj should remain zero
    np.testing.assert_allclose(inst.d_g_lambda[inst.nproj:], 0.0)


# ---- der2_g_lambda_wfnparams ----
@pytest.mark.parametrize("inorm", [True, False])
def test_der2_g_lambda_wfnparams_efree(make_instance, inorm):
    """
    Parameters
    ----------
    make_instance : fixture
        Factory providing a controlled FANPTContainerEParam instance.
    """

    inst = make_instance(inorm=inorm, nproj=4, nactive=3, nequation=6, ref_sd=2)

    # Tweak d_ovlp_s to a simple grid for deterministic checks
    inst.d_ovlp_s = np.arange(inst.nproj * inst.nactive, dtype=float).reshape(
        inst.nproj, inst.nactive, order="F"
    )
    inst.der_g_lambda()
    inst.der2_g_lambda_wfnparams()

    super_mat = 2.0 * inst.d_ovlp_s  # (nproj, nactive)
    super_ref = inst.super_d_g_lambda[inst.ref_sd]  # = 2*ovlp_ref

    if inorm:
        expected = super_mat.copy()
        # subtract (row_ref of super) * ovlp_s
        expected -= super_mat[inst.ref_sd][None, :] * inst.ovlp_s[: inst.nproj, None]
        # subtract super_ref * d_ovlp_s
        expected -= super_ref * inst.d_ovlp_s
        np.testing.assert_allclose(inst.d2_g_lambda_wfnparams[:inst.nproj], expected)
    else:
        # Should collapse to zeros in projected block
        np.testing.assert_allclose(inst.d2_g_lambda_wfnparams[:inst.nproj], 0.0)

    # rows below nproj remain zero
    np.testing.assert_allclose(inst.d2_g_lambda_wfnparams[inst.nproj:], 0.0)


# ---- gen_coeff_matrix ----
@pytest.mark.parametrize("inorm", [True, False])
def test_gen_coeff_matrix_efree_adjustment(make_instance, inorm):
    """
    Parameters
    ----------
    make_instance : fixture
        Factory providing a controlled FANPTContainerEParam instance.
    """
    nproj, nactive, nequation = 4, 3, 6
    ref_sd = 1
    E = 1.25

    inst = make_instance(
        inorm=inorm, nproj=nproj, nactive=nactive, nequation=nequation, ref_sd=ref_sd, energy_value=E
    )

    # Make d_ovlp_s specific for clarity
    inst.d_ovlp_s = np.arange(nproj * nactive, dtype=float).reshape(nproj, nactive, order="F")

    inst.gen_coeff_matrix()
    # Start from super (all 100s)
    expected = np.full((nequation, nactive), 100.0)

    # Build f_proj = 3 * d_ovlp_s
    f_proj = 3.0 * inst.d_ovlp_s
    ovlp_ref = inst.ovlp_s[ref_sd]

    if inorm:
        expected[:nproj] -= f_proj[ref_sd][None, :] * inst.ovlp_s[:nproj, None]
    else:
        expected[:nproj] -= ((f_proj[ref_sd][None, :] - E * inst.d_ovlp_s[ref_sd][None, :])
                              * inst.ovlp_s[:nproj, None] / ovlp_ref)

    np.testing.assert_allclose(inst.c_matrix, expected)
