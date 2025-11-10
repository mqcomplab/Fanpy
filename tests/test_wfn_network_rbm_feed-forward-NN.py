
#----------------
import numpy as np
import pytest
from fanpy.wfn.network.rbm import RestrictedBoltzmannMachine
from fanpy.tools import slater

@pytest.fixture
def rbm_default():
    return RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=3, num_layers=1, orders=(1,))


# --- Initialization tests ---
@pytest.mark.parametrize("nspin,nbath,orders,num_layers", [
    (4, 2, (1,), 1),
    (4, 2, (1, 2), 2),
    (8, 3, (1, 2, 3), 3),
    (4, 0, (), 1),  # zero bath edge case
])
def test_rbm_initialization(nspin, nbath, orders, num_layers):
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=nspin, nbath=nbath, orders=orders, num_layers=num_layers)
    assert rbm.nspin == nspin
    assert rbm.nbath == nbath
    assert rbm.num_layers == num_layers
    total_size = sum(np.prod(shape) for shape in rbm.params_shape)
    assert rbm.nparams == total_size
    for p, shape in zip(rbm._params, rbm.params_shape):
        assert p.shape == shape
    repr(rbm)  # cover __repr__


# --- Activation tests ---
@pytest.mark.parametrize("x", [
    np.array([-100, 0, 100]),
    np.array([-1e2, -1e-2, 0, 1e-2, 1e2]),
    np.array([-np.inf, 0, np.inf]),
])
def test_activation_functions(x):
    y = RestrictedBoltzmannMachine.activation(x)
    dy = RestrictedBoltzmannMachine.activation_deriv(x)
    assert np.all(np.isfinite(y[np.isfinite(x)]))
    assert np.all(np.isfinite(dy[np.isfinite(x)]))


# --- Forward and overlap tests ---
@pytest.mark.parametrize("sd", [0b0011, 0b0101, 0b1100, 0b1001])
def test_olp_helper_and_scalar_overlap(sd):
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2, orders=(1, 2), num_layers=2)
    val_no_cache = rbm._olp_helper(sd, cache=False)
    val_cache = rbm._olp_helper(sd, cache=True)
    np.testing.assert_allclose(val_no_cache, val_cache)
    assert len(rbm.forward_cache_act) > 0
    scalar_val = rbm._olp(sd)
    assert np.isfinite(scalar_val)
    assert isinstance(scalar_val, float)

@pytest.mark.parametrize("sds", [
    [0b1010, 0b1100],
    [0b0011, 0b0101, 0b0110],
])
def test_get_overlaps_vectorized_and_derivative(sds):
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2, orders=(1,))
    rbm.assign_params(np.random.normal(scale=0.1, size=rbm.nparams))
    overlaps = rbm.get_overlaps(sds)
    assert overlaps.shape == (len(sds),)

    # Derivatives w.r.t multiple params
    for deriv_index in [0, rbm.nparams - 1]:
        derivs = rbm.get_overlaps(sds, deriv=deriv_index)
        assert derivs.shape == (len(sds),)
        sd = sds[0]
        eps = 1e-6
        base_flat = rbm.params.copy()
        plus = base_flat.copy(); plus[deriv_index] += eps
        minus = base_flat.copy(); minus[deriv_index] -= eps
        rbm.assign_params(plus); f_plus = rbm.get_overlap(sd)
        rbm.assign_params(minus); f_minus = rbm.get_overlap(sd)
        numeric = (f_plus - f_minus) / (2 * eps)
        rbm.assign_params(base_flat)
        assert np.isclose(derivs[0], numeric, rtol=1e-3, atol=1e-6)


def test_get_overlaps_empty_input():
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2)
    overlaps = rbm.get_overlaps([])
    assert overlaps.size == 0


# ensure forward loop over extra layers in get_overlaps and that forward_cache_lin
# is appended for those layers 
def test_get_overlaps_appends_forward_cache_for_dense_layers():
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=3, num_layers=3, orders=(1,))
    rng = np.random.default_rng(7)
    rbm.assign_params(rng.normal(scale=0.05, size=rbm.nparams))

    sds = [0b0011, 0b1100, 0b1010]
    _ = rbm.get_overlaps(sds)  # run forward pass
    # there should be one forward_cache_lin for each layer (input orders -> stored earlier),
    # plus extra entries for each dense layer; ensure length matches expectation
    n_input_layers = len(rbm.orders)
    n_dense_layers = max(0, rbm.num_layers - 1)
    # forward_cache_lin is populated once per layer activation step in the implementation
    assert len(rbm.forward_cache_lin) >= n_input_layers + n_dense_layers


# --- Template overlaps ---
@pytest.mark.parametrize("sd", [0b0, 0b1, 0b1010])
def test_template_overlap_is_one(sd):
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=8, nbath=3, orders=(1,))
    olp = rbm.get_overlap(sd)
    assert pytest.approx(olp, rel=1e-12) == 1.0


# 1) cover the get_overlap return-derivative line
def test_get_overlap_returns_single_derivative():
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2, num_layers=1, orders=(1,))
    rng = np.random.default_rng(42)
    flat = rng.normal(scale=0.1, size=rbm.nparams)
    rbm.assign_params(flat.copy())

    sd = 0b1010
    # ask for a single parameter derivative via the scalar API
    deriv_index = 0
    deriv_from_get = rbm.get_overlap(sd, deriv=deriv_index)
    # compare with direct _olp_deriv element
    deriv_direct = rbm._olp_deriv(sd)[deriv_index]
    assert np.allclose(deriv_from_get, deriv_direct, rtol=1e-8, atol=1e-12)


# --- Normalization ---
@pytest.mark.parametrize("pspace", [
    [0b1100, 0b1010],
    [0b0101, 0b1010],
    [0b0101, 0b0101],  # repeated SD edge case
])
def test_normalize_output_scale(pspace):
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2, orders=(1,))
    rbm.assign_template_params()
    rbm.normalize(pspace=pspace)
    overlaps_after = rbm.get_overlaps([0b0101, 0b1100])
    norm_after = np.sum(overlaps_after**2)**(-0.5 / rbm.nbath)
    assert np.isfinite(norm_after) and norm_after > 0


@pytest.mark.xfail(reason="pspace=None not supported yet")
def test_normalize_none():
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2)
    rbm.normalize(pspace=None)


# --- Parameter assignment ---
def test_assign_template_params_and_flatten_shapes():
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2, orders=(1, 2), num_layers=2)
    rbm.assign_template_params()
    assert len(rbm._params) == len(rbm.params_shape)
    for p, shape in zip(rbm._params, rbm.params_shape):
        assert p.shape == shape


def test_assign_params_flat_and_influences_overlap():
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2, orders=(1,))
    flat = np.zeros(rbm.nparams); flat[0] = 0.2
    rbm.assign_params(flat)
    olp0 = rbm.get_overlap(0b0)
    olp1 = rbm.get_overlap(0b1)
    assert olp0 == pytest.approx(1.0)
    assert not np.isclose(olp1, 1.0)


def test_assign_params_invalid_length():
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2)
    original = rbm.params.copy()
    longer = np.zeros(rbm.nparams + 1)
    rbm.assign_params(longer)  # should not crash
    # Parameters should still have expected shape and size
    assert rbm.params.shape == original.shape
    assert np.isfinite(rbm.params).all()

# # --- Cache clearing ---
# @pytest.mark.skip(reason="clear_cache is not functioning as expected, " \
# "as it relies on _cache_fns which is not yet implemented ")
# def test_clear_cache_safe(rbm_default):
#     rbm = rbm_default
#     rbm.forward_cache_act = [np.array([1.0])]
#     rbm.clear_cache()
#     assert rbm.forward_cache_act == []


def test_template_params_multi_layer():
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2, num_layers=2, orders=(1,2))
    rbm.assign_template_params()
    assert len(rbm._template_params) == len(rbm.params_shape)


def test_get_overlaps_derivative_shape():
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2)
    sds = [0b101, 0b110]
    derivs = rbm.get_overlaps(sds, deriv=0)
    assert derivs.shape == (len(sds),)


def finite_diff_grad(rbm, sd, eps=1e-6):
    base = rbm.params.copy()
    grad = np.zeros_like(base)
    for i in range(len(base)):
        plus, minus = base.copy(), base.copy()
        plus[i] += eps
        minus[i] -= eps
        rbm.assign_params(plus)
        f_plus = rbm.get_overlap(sd)
        rbm.assign_params(minus)
        f_minus = rbm.get_overlap(sd)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    rbm.assign_params(base)
    return grad


def test_analytic_derivative_matches_numeric():
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2)
    rng = np.random.default_rng(123)
    rbm.assign_params(rng.normal(scale=0.1, size=rbm.nparams))
    sd = 0b1010
    analytic = rbm._olp_deriv(sd)
    numeric = finite_diff_grad(rbm, sd)
    assert np.allclose(analytic, numeric, rtol=5e-3, atol=1e-6)


# force the _olp_deriv branch where i >= len(self.orders) (multi-layer network)
def test__olp_deriv_with_dense_second_layer_matches_fd():
    # set num_layers > len(orders) so there is at least one dense layer after input
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=3, num_layers=2, orders=(1,))
    rng = np.random.default_rng(12345)
    rbm.assign_params(rng.normal(scale=0.05, size=rbm.nparams))

    sd = 0b1010
    analytic = rbm._olp_deriv(sd)
    numeric = finite_diff_grad(rbm, sd, eps=1e-6)
    assert np.allclose(analytic, numeric, rtol=5e-3, atol=1e-6)

#  vectorized derivative path that uses tensordot + swapaxes in get_overlaps
def test_get_overlaps_vectorized_derivative_tensordot_branch_matches_fd():
    # create vectorized case with multiple sds and at least one dense layer
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=3, num_layers=2, orders=(1,))
    rng = np.random.default_rng(99)
    rbm.assign_params(rng.normal(scale=0.05, size=rbm.nparams))

    sds = [0b0011, 0b1100]
    # pick a parameter index to compare
    deriv_index = 5 if rbm.nparams > 6 else 0
    vec_derivs = rbm.get_overlaps(sds, deriv=deriv_index)
    assert vec_derivs.shape == (len(sds),)

    # numeric check for first sd in vector (finite difference on scalar get_overlap)
    sd0 = sds[0]
    eps = 1e-6
    base_flat = rbm.params.copy()
    plus = base_flat.copy(); plus[deriv_index] += eps
    minus = base_flat.copy(); minus[deriv_index] -= eps
    rbm.assign_params(plus); f_plus = rbm.get_overlap(sd0)
    rbm.assign_params(minus); f_minus = rbm.get_overlap(sd0)
    numeric = (f_plus - f_minus) / (2 * eps)
    rbm.assign_params(base_flat)

    assert np.isclose(vec_derivs[0], numeric, rtol=1e-3, atol=1e-6)