"""Tests for the fanpy.wfn.network.rbm."""
from fanpy.wfn.network.rbm import RestrictedBoltzmannMachine
from fanpy.tools.sd_list import sd_list
from fanpy.tools.slater import occ_indices

import numdifftools as nd

import numpy as np

import pytest

from utils import skip_init


def test_init():
    """Test RestrictedBoltzmannMachine.__init__."""
    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, ), num_layers=2)
    assert test.nelec == 4
    assert test.nspin == 10
    assert test.nbath == 5
    assert test.orders == (1, )
    assert test.num_layers == 2
    assert np.allclose(test._template_params[0], np.zeros((5, 10)))
    assert np.allclose(test._template_params[1], np.zeros((5, 5)))
    assert np.allclose(test.params, np.hstack((np.zeros((5, 10)).flat, np.zeros((5, 5)).flat)))
    assert test.forward_cache_lin == []
    assert test.forward_cache_act == []


def test_params():
    """Test RestrictedBoltzmannMachine.params."""
    params = np.random.rand(75)
    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, ), num_layers=2, params=params)
    assert np.allclose(test.params, params)


def test_nparams():
    """Test RestrictedBoltzmannMachine.nparams."""
    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, ), num_layers=2)
    assert test.params.size == test.nparams
    test = RestrictedBoltzmannMachine(7, 14, 21, orders=(1, 2), num_layers=3)
    assert test.params.size == test.nparams


def test_params_shape():
    """Test RestrictedBoltzmannMachine.params_shape."""
    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, ), num_layers=2)
    assert test.params_shape[0] == (5, 10)
    assert test.params_shape[1] == (5, 5)

    test = RestrictedBoltzmannMachine(7, 14, 21, orders=(1, 2), num_layers=3)
    assert test.params_shape[0] == (21, 14)
    assert test.params_shape[1] == (21, 14, 14)
    assert test.params_shape[2] == (21, 21)
    assert test.params_shape[3] == (21, 21)


def test_template_params():
    """Test RestrictedBoltzmannMachine.template_params."""
    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, ), num_layers=2)
    test._template_params = 23123
    assert np.allclose(test.template_params, test._template_params)


def test_assign_template_params():
    """Test RestrictedBoltzmannMachine.assign_template_params."""
    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, ), num_layers=2)
    del test._template_params
    assert not hasattr(test, "_template_params")
    test.assign_template_params()
    assert np.allclose(test._template_params[0], np.zeros((5, 10)))
    assert np.allclose(test._template_params[1], np.zeros((5, 5)) * 0.5)

    test = RestrictedBoltzmannMachine(7, 14, 21, orders=(1, 2), num_layers=3)
    del test._template_params
    assert not hasattr(test, "_template_params")
    test.assign_template_params()
    assert np.allclose(test._template_params[0], np.zeros((21, 14)))
    assert np.allclose(test._template_params[1], np.zeros((21, 14, 14)))
    assert np.allclose(test._template_params[2], np.zeros((21, 21)) * 0.5)
    assert np.allclose(test._template_params[3], np.zeros((21, 21)) * 0.5)


def test_assign_params():
    """Test RestrictedBoltzmannMachine.assign_params."""
    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, ), num_layers=2)
    del test._params
    assert not hasattr(test, "_params")
    test.assign_params()
    assert np.allclose(test._params[0], np.zeros((5, 10)))
    assert np.allclose(test._params[1], np.zeros((5, 5)) * 0.5)

    params = np.random.rand(test.nparams)
    test.assign_params(params)
    assert np.allclose(test._params[0], params[:50].reshape(5, 10))
    assert np.allclose(test._params[1], params[50:].reshape(5, 5))


def test_activation():
    """Test RestrictedBoltzmannMachine.activation."""
    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, ), num_layers=2)
    assert test.activation(10) == np.exp(10) + 1
    assert test.activation(0.87) == np.exp(0.87) + 1


def test_activation_deriv():
    """Test RestrictedBoltzmannMachine.activation_deriv."""
    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, ), num_layers=2)
    grad = nd.Gradient(test.activation)(10)
    assert np.allclose(test.activation_deriv(10), grad)
    grad = nd.Gradient(test.activation)(0.87)
    assert np.allclose(test.activation_deriv(0.87), grad)


def test_olp():
    """Test RestrictedBoltzmannMachine._olp."""
    test = RestrictedBoltzmannMachine(4, 10, 4, orders=(1, ), num_layers=1)
    for sd in sd_list(4, 10):
        assert test._olp(sd) == 1

    test.assign_params(np.random.rand(test.nparams) * 0.001)
    for sd in sd_list(4, 10):
        input_layer = np.zeros(10)
        input_layer[np.array(occ_indices(sd))] = 1
        assert np.allclose(
            test._olp(sd),
            np.prod(0.5 * test.activation(test._params[0].dot(input_layer)))
        )

    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, 2), num_layers=1)
    for sd in sd_list(4, 10):
        assert test._olp(sd) == 1

    test.assign_params(np.random.rand(test.nparams))
    for sd in sd_list(4, 10):
        indices = np.array(occ_indices(sd))

        input_layer1 = np.zeros(10)
        input_layer1[indices] = 1
        network1 = test._params[0].dot(input_layer1)

        input_layer2 = np.zeros((10, 10))
        input_layer2[indices[:, None], indices[None, :]] = 1
        network2 = np.einsum('ijk,jk->i', test._params[1], input_layer2)

        assert np.allclose(test._olp(sd), np.prod(0.5 * test.activation(network1 + network2)))

    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, 2), num_layers=2)
    for sd in sd_list(4, 10):
        assert test._olp(sd) == 1


    test.assign_params(np.random.rand(test.nparams) * 0.001)
    for sd in sd_list(4, 10):
        indices = np.array(occ_indices(sd))

        input_layer1 = np.zeros(10)
        input_layer1[indices] = 1
        network1 = test._params[0].dot(input_layer1)

        input_layer2 = np.zeros((10, 10))
        input_layer2[indices[:, None], indices[None, :]] = 1
        network2 = np.einsum('ijk,jk->i', test._params[1], input_layer2)

        assert np.allclose(
            test._olp(sd),
            np.prod(
                0.5 * test.activation(test._params[2].dot(test.activation(network1 + network2)))
            ),
        )


def test_olp_deriv():
    """Test RestrictedBoltzmannMachine._olp_deriv."""
    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, ), num_layers=1)
    test.assign_params(np.random.rand(test.nparams) * 0.001)
    for sd in sd_list(4, 10):
        def func(x):
            test.assign_params(x)
            return test._olp(sd)

        grad = nd.Gradient(func, step=1e-8)(test.params)
        assert np.allclose(test._olp_deriv(sd), grad)

    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, 2), num_layers=1)
    test.assign_params(np.random.rand(test.nparams) * 0.001)
    for sd in sd_list(4, 10):
        def func(x):
            test.assign_params(x)
            return test._olp(sd)

        grad = nd.Gradient(func, step=1e-8)(test.params)
        assert np.allclose(test._olp_deriv(sd), grad, atol=1e-7)

    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, 2), num_layers=2)
    test.assign_params(np.random.rand(test.nparams) * 0.001)
    for sd in sd_list(4, 10):
        def func(x):
            test.assign_params(x)
            return test._olp(sd)

        grad = nd.Gradient(func, step=1e-8)(test.params)
        assert np.allclose(test._olp_deriv(sd), grad, atol=1e-7)


def test_get_overlaps():
    """Test RestrictedBoltzmannMachine.get_overlaps."""
    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, ), num_layers=1)
    test.assign_params(np.random.rand(test.nparams) * 0.001)
    sds = sd_list(4, 10)
    assert np.allclose(test.get_overlaps(sds), [test._olp(sd) for sd in sds])

    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, 2), num_layers=2)
    test.assign_params(np.random.rand(test.nparams) * 0.001)
    assert np.allclose(test.get_overlaps(sds), [test._olp(sd) for sd in sds])

    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, 2), num_layers=2)
    assert np.allclose(test.get_overlaps(sds), [test._olp(sd) for sd in sds])


def test_get_overlaps_deriv():
    """Test RestrictedBoltzmannMachine.get_overlaps."""
    sds = sd_list(4, 10)

    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, ), num_layers=1)
    test.assign_params(np.random.rand(test.nparams) * 0.001)

    def func(x):
        test.assign_params(x)
        return test.get_overlaps(sds)

    grad = nd.Gradient(func, step=1e-8)(test.params)
    assert np.allclose(test.get_overlaps(sds, deriv=True), grad)

    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, 2), num_layers=1)
    test.assign_params(np.random.rand(test.nparams) * 0.001)

    def func(x):
        test.assign_params(x)
        return test.get_overlaps(sds)


    grad = nd.Gradient(func, step=1e-8)(test.params)
    assert np.allclose(test.get_overlaps(sds, deriv=True), grad)

    test = RestrictedBoltzmannMachine(4, 10, 5, orders=(1, 2), num_layers=2)
    test.assign_params(np.random.rand(test.nparams) * 0.001)

    def func(x):
        test.assign_params(x)
        return test.get_overlaps(sds)

    grad = nd.Gradient(func, step=1e-8)(test.params)
    assert np.allclose(test.get_overlaps(sds, deriv=True), grad, atol=1e-7)
