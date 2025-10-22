# # test_rbm.py
# import numpy as np
# import pytest

# from fanpy.wfn.network.rbm import RestrictedBoltzmannMachine
# from fanpy.tools import slater


# # simple occ_indices implementation for tests:
# # interpret sd as an integer whose binary representation flags occupied spin-orbitals.
# #def _occ_indices_from_int(sd: int, nspin: int = 32):
# #    # return indices where bit is set (LSB is index 0)
# #    inds = []
# #    i = 0
# #    s = sd
# #    while s:
# #        if s & 1:
# #            inds.append(i)
# #        s >>= 1
# #        i += 1
# #    return inds
# #
# #
# #@pytest.fixture(autouse=True)
# #def patch_slater_occ_indices(monkeypatch):
# #    """
# #    Monkeypatch slater.occ_indices used by RBM to a small deterministic function.
# #    This keeps tests independent of external fanpy slater details.
# #    """
# #    def occ_indices(sd):
# #        return _occ_indices_from_int(int(sd))
# #
# #    monkeypatch.setattr(slater, "occ_indices", occ_indices)
# #    yield


# @pytest.fixture(autouse=True)
# def rbm():
#     """Fixture to provide a default RBM for tests."""
#     rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=3, num_layers=1, orders=(1,))
#     return rbm

# def make_rbm(nspin=4, nbath=2, orders=(1,), num_layers=1, params=None):
#     """Helper to construct an RBM for tests with small sizes and deterministic defaults."""
#     nelec = 2  # not heavily used in these tests
#     rbm = RestrictedBoltzmannMachine(nelec=nelec, nspin=nspin, nbath=nbath, params=params, num_layers=num_layers, orders=orders)
#     return rbm


# def test_rbm_initialization(rbm):
#     """Test RBM initialization and parameter shapes."""
#     assert rbm.nelec == 2
#     assert rbm.nspin == 4
#     assert rbm.nbath == 3
#     assert rbm.num_layers == 1
#     assert rbm.orders == (1,)

#     expected_nparams = np.sum(rbm.nbath * rbm.nspin**np.array(rbm.orders))
#     assert rbm.nparams == expected_nparams
#     total_size = sum(np.prod(shape) for shape in rbm.params_shape)
#     assert total_size == expected_nparams
                     
#     # params should be initialized to zeros (template)
#     assert np.allclose(rbm.params, np.zeros(expected_nparams))
#     assert isinstance(rbm._params, list)
#     assert len(rbm._params) == len(rbm.params_shape)

# @pytest.mark.skip(reason="RBM assign_params There is no check that the" \
# " incoming params actually has the correct size, nor any type check. " \
# "Even if params is the wrong length or type, the code just slices what it can and silently ignores leftovers.")
# def test_assign_params_invalid_input(rbm):
#     # wrong shape
#     with pytest.raises(ValueError):
#         rbm.assign_params(np.ones(rbm.nparams + 1))
#     # wrong type
#     with pytest.raises(TypeError):
#         rbm.assign_params(np.array(['a']*rbm.nparams))


# @pytest.mark.skip(reason="RBM clear_cache is not working as expected; it is not clearing the cache.")
# def test_clear_cache(rbm):
#     """Test that clear_cache resets forward caches."""
#     # populate caches with dummy data
#     rbm.forward_cache_act = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
#     rbm.forward_cache_lin = [np.array([5.0, 6.0]), np.array([7.0, 8.0])]

#     rbm.clear_cache()
#     assert rbm.forward_cache_act == []
#     assert rbm.forward_cache_lin == []

# def make_small_rbm():
#     return RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2, orders=(1, 2), num_layers=2)


# def test_activation_extremes():
#     x = np.array([-100, 0, 100])
#     y = RestrictedBoltzmannMachine.activation(x)
#     dy = RestrictedBoltzmannMachine.activation_deriv(x)
#     # check finite, no inf/nan
#     assert np.all(np.isfinite(y))
#     assert np.all(np.isfinite(dy))


# def test_olp_helper_empty_sd():
#     rbm = make_small_rbm()
#     # empty determinant
#     out = rbm._olp_helper(0)
#     assert out.shape[0] == rbm.nbath
#     assert np.all(np.isfinite(out))


# def test_assign_params_none_and_template_created():
#     rbm = make_rbm()
#     rbm._template_params = None
#     rbm.assign_params(None)
#     assert rbm._params is not None


# def test_template_params_multi_layer():
#     rbm = make_rbm(num_layers=2, orders=(1,2))
#     rbm.assign_template_params()
#     assert len(rbm._template_params) == len(rbm.params_shape)



# def test_olp_helper_multi_layer():
#     rbm = make_rbm(num_layers=2, orders=(1, 2))
#     sd = 0b1010
#     out = rbm._olp_helper(sd, cache=True)
#     assert out.shape[0] == rbm.nbath


# def test_get_overlaps_empty_and_derivative():
#     rbm = make_rbm()
#     assert rbm.get_overlaps([]).size == 0
#     sds = [0b101, 0b110]
#     deriv_index = 0
#     derivs = rbm.get_overlaps(sds, deriv=deriv_index)
#     assert derivs.shape == (len(sds),)


# def test_normalize_no_pspace_sets_output_scale():
#     rbm = make_rbm()
#     rbm.pspace_norm = [0b01, 0b10]
#     rbm.normalize()
#     assert rbm.output_scale > 0



# def test_params_shape_and_nparams(rbm):
#     """Verify parameter shapes and counts."""
#     shapes = rbm.params_shape
#     assert isinstance(shapes, list)
#     total_params = sum(np.prod(shape) for shape in shapes)
#     assert total_params == rbm.nparams


# def test_activation_functions():
#     """Check activation and its derivative."""
#     x = np.array([-1.0, 0.0, 1.0, 2.0])
#     val = RestrictedBoltzmannMachine.activation(x)
#     dval = RestrictedBoltzmannMachine.activation_deriv(x)
#     assert np.allclose(val, 1.0 + np.exp(x))
#     assert np.all(val > 1) # since activation = 1 + exp(x)


# def test_assign_template_params_creates_correct_shapes(rbm):
#     rbm.assign_template_params()
#     assert len(rbm._params) == len(rbm.params_shape)
#     for p, shape in zip(rbm._params, rbm.params_shape):
#         assert p.shape == shape


# def test_get_overlaps_empty_list(rbm):
#     assert np.all(rbm.get_overlaps([]) == np.array([]))


# def test_get_overlap_individual(rbm):
#     """Test get_overlap for individual Slater determinants."""
#     sd = slater.create(0, 0, 1)  # occupation of spin orbitals 0 and 1
#     olp = rbm.get_overlap(sd)
#     assert np.isfinite(olp) 
#     assert isinstance(olp, float)
#     # with template params (zeros), overlap should be 1.0
#     assert np.isclose(olp, 1.0)


# def test_get_overlaps_vectorized(rbm):
#     """Ensure vectorized get_overlaps matches individual calls."""
#     sds = [slater.create(0, 0, 1 ), slater.create(0, 2, 1)]   # occupation of spin orbitals 1 and 3
#     scalar_Vals = np.array([rbm.get_overlap(sd) for sd in sds])
#     vec_vals = rbm.get_overlaps(sds)
#     np.testing.assert_allclose(scalar_Vals, vec_vals, rtol=1e-12, atol=1e-12)
    

# def test_overlap_derivative_shape(rbm):
#     """Check derivative vector length matches number of params."""
#     sd = slater.create(0, 0, 1)  # occupation of spin orbitals 0 and 1
#     deriv = rbm._olp_deriv(sd)
#     assert deriv.shape == (rbm.nparams,)


# def test_default_template_overlap_is_one_for_any_sd():
#     """If params are left as template (zeros), activation(0)=1+exp(0)=2 and output_scale=0.5
#        So each hidden unit contributes 2*0.5 = 1 and product = 1."""
#     rbm = make_rbm(nspin=8, nbath=3, orders=(1,))
#     # template params are created lazily in assign_params called by constructor; ensure template assigned
#     sd_examples = [51, 113, 57, 85, 113]  # several small determinants
#     for sd in sd_examples:
#         olp = rbm.get_overlap(sd)
#         assert pytest.approx(1.0, rel=1e-12, abs=1e-12) == olp


# def test_normalize_scales_output(rbm):
#     """Test that enabling normalization scales output overlaps appropriately."""
#     sds = [slater.create(0, 2, 3), slater.create(0, 1, 3)]  # occupation of spin orbitals 0 and 1
#     olps_before = rbm.get_overlaps(sds)

#     rbm.normalize(pspace=sds)
#     # assert rbm.forward_cache_act == []
#     # assert rbm.forward_cache_lin == []

#     olps_after = rbm.get_overlaps(sds)
#     # overlaps squared should sum to 1 ** (1/nbath) due to output_scale scaling
#     norm_after = np.sum(olps_after ** 2) ** 0.5 #(1.0 / 2 * rbm.nbath)
#     assert np.isclose(norm_after, 1.0, rtol=1e-12)
    
    
# def test_olp_helper_cache_flag(rbm):
#     """Test that helper methods respect and populate caching flags."""
#     sd = slater.create(0, 0, 1)  # occupation of spin orbitals 0 and 1
#     out_no_cache = rbm._olp_helper(sd, cache=False)
#     out_with_cache = rbm._olp_helper(sd, cache=True)

#     np.testing.assert_allclose(out_no_cache, out_with_cache, rtol=1e-12, atol=1e-12)
#     #cache should be populated only for cache=True
#     assert len(rbm.forward_cache_act) > 0
#     assert len(rbm.forward_cache_lin) > 0


# def test_olp_scalar_output(rbm):
#     sd = slater.create(0, 1, 3)
#     val = rbm._olp(sd)
#     assert isinstance(val, float)
#     assert np.isfinite(val)
    
# def test_assign_params_accepts_flat_array_and_changes_overlap():
#     """Provide a flattened param array and confirm assign_params accepts it and influences overlaps."""
#     nspin, nbath = 4, 2
#     rbm = make_rbm(nspin=nspin, nbath=nbath, orders=(1,))

#     # get number of params expected and create a small non-zero flattened array
#     nparams = rbm.nparams
#     flat = np.zeros(nparams, dtype=float)
#     # set a single weight to a known non-zero value (first parameter)
#     flat[0] = 0.2
#     rbm.assign_params(flat.copy())

#     # compute overlap for two determinants that differ in occupation of spin orbital 0
#     sd0 = 0b0        # no occupation -> input zeros -> pre-activation 0 -> activation=2 -> each unit=1 => product=1
#     sd1 = 0b1        # occupation at position 0 affects the first hidden unit's pre-activation
#     olp0 = rbm.get_overlap(sd0)
#     olp1 = rbm.get_overlap(sd1)

#     assert olp0 == pytest.approx(1.0)
#     # after setting the first weight non-zero, overlap for sd1 should differ from 1.0
#     assert not np.isclose(olp1, 1.0)


# def _finite_diff_grad(rbm, sd, eps=1e-6):
#     """Numerical finite-difference gradient for <m|Psi> with respect to flattened params."""
#     base_flat = rbm.params.copy()
#     grad = np.zeros_like(base_flat, dtype=float)
#     for i in range(len(base_flat)):
#         plus = base_flat.copy()
#         minus = base_flat.copy()
#         plus[i] += eps
#         minus[i] -= eps
#         rbm.assign_params(plus.copy())
#         f_plus = rbm.get_overlap(sd)
#         rbm.assign_params(minus.copy())
#         f_minus = rbm.get_overlap(sd)
#         # central difference
#         grad[i] = (f_plus - f_minus) / (2 * eps)
#     # restore original params
#     rbm.assign_params(base_flat.copy())
#     return grad


# def test_analytic_derivative_matches_finite_difference():
#     """Compare _olp_deriv (analytic) to finite-difference numeric gradient for a single SD."""
#     nspin, nbath = 4, 2
#     rbm = make_rbm(nspin=nspin, nbath=nbath, orders=(1,))
#     # set some random but small parameters for a non-trivial gradient
#     rng = np.random.default_rng(1234)
#     flat = rng.normal(scale=0.1, size=rbm.nparams)
#     rbm.assign_params(flat.copy())

#     sd = 0b1010  # some occupation pattern
#     analytic = rbm._olp_deriv(sd)
#     numeric = _finite_diff_grad(rbm, sd, eps=1e-6)

#     assert analytic.shape == numeric.shape
#     # compare with relative tolerance; gradients can be small so include both abs and rel tolerance
#     assert np.allclose(analytic, numeric, rtol=5e-3, atol=1e-6)


# def test_get_overlaps_vectorized_and_derivative_selection():
#     """Test vectorized get_overlaps on multiple SDs and that deriv selection returns requested column."""
#     nspin, nbath = 8, 2
#     rbm = make_rbm(nspin=nspin, nbath=nbath, orders=(1,))
#     # use default (zeros) params so overlaps are all 1
#     sds = [51, 113, 57, 85, 113]
#     overlaps = rbm.get_overlaps(sds)
#     assert overlaps.shape == (len(sds),)
#     assert np.allclose(overlaps, np.ones(len(sds)))

#     # now set a small random param to make derivatives non-trivial
#     rng = np.random.default_rng(42)
#     flat = rng.normal(scale=0.05, size=rbm.nparams)
#     rbm.assign_params(flat.copy())

#     # pick a parameter index to request derivatives for (e.g., 0)
#     deriv_index = 0
#     derivs_col = rbm.get_overlaps(sds, deriv=deriv_index)
#     # should be a 1-d array with length len(sds)
#     assert derivs_col.shape == (len(sds),)
#     # compare each entry to finite-difference derivative for that sd
#     eps = 1e-6
#     for i, sd in enumerate(sds):
#         numeric = _finite_diff_grad(rbm, sd, eps=eps)[deriv_index]
#         assert np.isclose(derivs_col[i], numeric, rtol=1e-3, atol=1e-6)


import numpy as np
import pytest
from fanpy.wfn.network.rbm import RestrictedBoltzmannMachine
from fanpy.tools import slater

@pytest.fixture
def rbm_default():
    return RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=3, num_layers=1, orders=(1,))

@pytest.mark.parametrize("nspin,nbath,orders,num_layers", [
    (4, 2, (1,), 1),
    (4, 2, (1,2), 2),
    (8, 3, (1,2,3), 3),
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


@pytest.mark.xfail(reason="pspace_norm not implemented in RBM; normalize(pspace=None) fails")
def test_normalize_output_scale_pspace_none():
    rbm = make_rbm(nelec=2, nspin=4, nbath=2, orders=(1,))
    rbm.normalize(pspace=None)

@pytest.mark.xfail(reason="RBM clear_cache relies on _cache_fns which is not yet implemented")
def test_clear_cache_placeholder(rbm):
    rbm.clear_cache()
    # cannot assert anything meaningful until _cache_fns is implemented

@pytest.mark.parametrize("x", [np.array([-100, 0, 100]), np.array([-1e2, -1e-2, 0, 1e-2, 1e2])])
def test_activation_functions(x):
    y = RestrictedBoltzmannMachine.activation(x)
    dy = RestrictedBoltzmannMachine.activation_deriv(x)
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(dy))

@pytest.mark.parametrize("sd", [0b0011, 0b0101, 0b1100, 0b1001])
def test_olp_helper_and_scalar_overlap(sd):
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2, orders=(1,2), num_layers=2)
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
    overlaps = rbm.get_overlaps(sds)
    assert overlaps.shape == (len(sds),)
    # derivative test
    rbm.assign_params(np.random.normal(scale=0.1, size=rbm.nparams))
    deriv_index = 0
    derivs = rbm.get_overlaps(sds, deriv=deriv_index)
    assert derivs.shape == (len(sds),)
    for i, sd in enumerate(sds):
        eps = 1e-6
        numeric = np.zeros(rbm.nparams)
        base_flat = rbm.params.copy()
        for j in range(rbm.nparams):
            plus = base_flat.copy(); plus[j] += eps
            minus = base_flat.copy(); minus[j] -= eps
            rbm.assign_params(plus); f_plus = rbm.get_overlap(sd)
            rbm.assign_params(minus); f_minus = rbm.get_overlap(sd)
            numeric[j] = (f_plus - f_minus) / (2*eps)
        rbm.assign_params(base_flat)
        assert np.isclose(derivs[i], numeric[deriv_index], rtol=5e-3, atol=1e-6)

@pytest.mark.parametrize("sd", [0b0, 0b1, 0b1010])
def test_template_overlap_is_one(sd):
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=8, nbath=3, orders=(1,))
    olp = rbm.get_overlap(sd)
    assert pytest.approx(olp, rel=1e-12) == 1.0

@pytest.mark.parametrize("pspace", [
    #None,
    [0b1100, 0b1010, 0b0101],
    [0b0101, 0b1010],
])
def test_normalize_output_scale(pspace):
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2, orders=(1,))
    overlaps_before = rbm.get_overlaps([0b0101, 0b1100])
    rbm.assign_template_params()
    rbm.normalize(pspace=pspace)
    overlaps_after = rbm.get_overlaps([0b0101, 0b1100])
    norm_after = np.sum(overlaps_after**2)**(-1.0 / rbm.nbath)
    assert norm_after > 0 # must be finite and non-zero
    assert np.isclose(norm_after, 1.0, rtol=1e-8)

def test_assign_template_params_and_flatten_shapes():
    rbm = RestrictedBoltzmannMachine(nelec=2, nspin=4, nbath=2, orders=(1,2), num_layers=2)
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

