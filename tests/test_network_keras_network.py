"""Tests for fanpy.wfn.network.keras_network.KerasNetwork."""
import fanpy.tools.slater as slater
from fanpy.tools.sd_list import sd_list
from fanpy.wfn.network.keras_network import KerasNetwork

import numpy as np

import pytest

from tensorflow.python import keras

from utils import skip_init


keras.backend.set_floatx("float64")

@pytest.mark.skip(reason="This test fails and is being worked on (PR 36).")
def test_keras_assign_model():
    """Test KerasNetwork.assign_model."""
    test = skip_init(KerasNetwork)
    test.nspin = 10
    test.num_layers = 2
    # default
    test.assign_model()
    assert isinstance(test.model, keras.engine.training.Model)
    assert len(test.model.inputs) == 1
    assert test.model.inputs[0].shape[1] == 10
    assert len(test.model.layers[0].weights) == 1
    assert test.model.layers[0].weights[0].shape == (10, 10)
    assert len(test.model.layers[1].weights) == 1
    assert test.model.layers[1].weights[0].shape == (10, 10)
    assert len(test.model.layers[2].weights) == 1
    assert test.model.layers[2].weights[0].shape == (10, 1)
    assert len(test.model.outputs) == 1
    assert test.model.outputs[0].shape[1] == 1
    # custom model
    test.model = None
    model = keras.engine.sequential.Sequential()
    model.add(keras.layers.core.Dense(100, input_dim=10, use_bias=True))
    model.add(keras.layers.core.Dense(1, input_dim=100, use_bias=True))
    test.assign_model(model)
    assert isinstance(test.model, keras.engine.training.Model)
    assert len(test.model.inputs) == 1
    assert test.model.inputs[0].shape[1] == 10
    assert len(test.model.layers[0].weights) == 2
    assert test.model.layers[0].weights[0].shape == (10, 100)
    assert test.model.layers[0].weights[1].shape == (100,)
    assert len(test.model.layers[1].weights) == 2
    assert test.model.layers[1].weights[0].shape == (100, 1)
    assert test.model.layers[1].weights[1].shape == (1,)
    assert len(test.model.outputs) == 1
    assert test.model.outputs[0].shape[1] == 1
    # raises
    with pytest.raises(TypeError):
        test.assign_model(1)
    with pytest.raises(TypeError):
        test.assign_model(keras.engine.network.Network())
    #  more than one set of inputs
    model_input = keras.layers.Input(shape=(10,))
    model = keras.models.Model(
        inputs=[model_input, keras.layers.Input(shape=(10,))],
        outputs=keras.layers.Dense(10)(model_input),
    )
    with pytest.raises(ValueError):
        test.assign_model(model)
    #  bad number of input nodes
    model = keras.engine.sequential.Sequential()
    model.add(keras.layers.core.Dense(1, input_dim=9, use_bias=True))
    with pytest.raises(ValueError):
        test.assign_model(model)
    #  more than one set of inputs
    model_input = keras.layers.Input(shape=(10,))
    model = keras.models.Model(
        inputs=model_input,
        outputs=[keras.layers.Dense(10)(model_input), keras.layers.Dense(10)(model_input)],
    )
    with pytest.raises(ValueError):
        test.assign_model(model)
    # bad number of output nodes
    model = keras.engine.sequential.Sequential()
    model.add(keras.layers.core.Dense(2, input_dim=10, use_bias=True))
    with pytest.raises(ValueError):
        test.assign_model(model)


def test_keras_nparams():
    """Test KerasNetwork.nparams."""
    test = skip_init(KerasNetwork)
    test.nspin = 10
    model = keras.engine.sequential.Sequential()
    model.add(keras.layers.core.Dense(100, input_dim=10, use_bias=True))
    model.add(keras.layers.core.Dense(1, input_dim=100, use_bias=True))
    test.assign_model(model)
    assert test.nparams == 1100 + 101


def test_keras_assign_default_params():
    """Test KerasNetwork.assign_default_params."""
    test = skip_init(KerasNetwork)
    test.nelec = 4
    test.nspin = 10
    test.num_layers = 2
    test.assign_model()
    test.assign_default_params()
    assert np.allclose(np.identity(10), test._default_params[:100].reshape(10, 10))
    assert np.allclose(np.identity(10), test._default_params[100:200].reshape(10, 10))
    # FIXME: need a better test to check the quality of the default params/initial guess
    hidden_sds = [slater.occ_indices(sd) for sd in sd_list(4, 10, exc_orders=[1, 2])]
    hidden_units = np.zeros((len(hidden_sds), 10))
    for i, hidden_sd in enumerate(hidden_sds):
        hidden_units[i, hidden_sd] = 1
    assert all(
        test._default_params[200:].dot(z) < test._default_params[200:].dot(hidden_units[0])
        for z in hidden_units[1:]
    )
    # bad model
    model = keras.engine.sequential.Sequential()
    model.add(keras.layers.core.Dense(100, input_dim=10, use_bias=True))
    model.add(keras.layers.core.Dense(1, input_dim=100, use_bias=True))
    test.assign_model(model)
    test.nelec = 0
    with pytest.raises(ValueError):
        test.assign_default_params()
    # FIXME: need to test Keras network with a layer with more than one variable for weights


def test_keras_assign_params():
    """Test KerasNetwork.assign_params."""
    test = skip_init(KerasNetwork)
    test.nspin = 10
    test.num_layers = 2
    test.assign_model()
    # default
    test._default_params = np.ones(test.nparams)
    test.assign_params()
    weights = test.model.get_weights()
    assert np.allclose(weights[0], np.ones(100).reshape(10, 10))
    assert np.allclose(weights[1], np.ones(100).reshape(10, 10))
    assert np.allclose(weights[2], np.ones(10).reshape(10, 1))
    # user specified
    answer = np.random.rand(210)
    test.assign_params(answer)
    weights = test.model.get_weights()
    assert np.allclose(weights[0], answer[:100].reshape(10, 10))
    assert np.allclose(weights[1], answer[100:200].reshape(10, 10))
    assert np.allclose(weights[2], answer[200:].reshape(10, 1))


def test_keras_init():
    """Test KerasNetwork.__init__."""
    test = KerasNetwork(4, 10)
    # check model
    assert isinstance(test.model, keras.engine.training.Model)
    assert len(test.model.inputs) == 1
    assert test.model.inputs[0].shape[1] == 10
    assert len(test.model.layers[0].weights) == 1
    assert test.model.layers[0].weights[0].shape == (10, 10)
    assert len(test.model.layers[1].weights) == 1
    assert test.model.layers[1].weights[0].shape == (10, 10)
    assert len(test.model.layers[2].weights) == 1
    assert test.model.layers[2].weights[0].shape == (10, 1)
    assert len(test.model.outputs) == 1
    assert test.model.outputs[0].shape[1] == 1
    # check default_params
    assert np.allclose(np.identity(10), test._default_params[:100].reshape(10, 10))
    assert np.allclose(np.identity(10), test._default_params[100:200].reshape(10, 10))
    weights = test.model.get_weights()
    assert np.allclose(weights[0], test._default_params[:100].reshape(10, 10))
    assert np.allclose(weights[1], test._default_params[100:200].reshape(10, 10))
    assert np.allclose(weights[2], test._default_params[200:].reshape(10, 1))


@pytest.mark.skip(reason="This test fails and is being worked on (PR 36).")
def test_keras_get_overlap():
    """Test KerasNetwork.get_overlap."""
    nd = pytest.importorskip("numdifftools")
    test = skip_init(KerasNetwork)
    test.nspin = 4
    model = keras.engine.sequential.Sequential()
    model.add(
        keras.layers.core.Dense(4, input_dim=4, activation=keras.activations.relu, use_bias=False)
    )
    model.add(keras.layers.core.Dense(1, input_dim=4, activation=None, use_bias=False))
    test.assign_model(model)
    matrix1 = np.random.rand(4, 4)
    matrix2 = np.random.rand(4)
    test.assign_params(np.hstack([matrix1.flatten(), matrix2]))
    # overlap
    input_vec = np.zeros(4)
    input_vec[[0, 2]] = 1
    hidden_units = input_vec[None, :].dot(matrix1)
    hidden_units[hidden_units < 0] = 0
    assert np.allclose(test.get_overlap(0b0101), hidden_units.dot(matrix2))
    input_vec = np.zeros(4)
    input_vec[[1, 2, 3]] = 1
    hidden_units = input_vec[None, :].dot(matrix1)
    hidden_units[hidden_units < 0] = 0
    assert np.allclose(test.get_overlap(0b1110), hidden_units.dot(matrix2))
    # derivative
    new_wfn = KerasNetwork(2, 4, num_layers=1)

    def overlap(params):
        new_wfn.assign_params(params)
        return new_wfn.get_overlap(0b0101)

    grad = nd.Gradient(overlap)(test.params)
    assert np.allclose(grad, test.get_overlap(0b0101, np.arange(test.nparams)))

    with pytest.raises(TypeError):
        test.get_overlap("1")
