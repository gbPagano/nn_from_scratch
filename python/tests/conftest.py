import numpy as np
import pytest

from nn_from_scratch import NeuralNetwork
from nn_from_scratch.layers import Dense, Sigmoid, TanH


@pytest.fixture
def simple_layers_a():
    # defining layers and initial weights
    layer_a = Dense(neurons=(2, 2))
    layer_a.weights = np.array(
        [
            [0.15, 0.2],
            [0.25, 0.3],
        ]
    )
    layer_a.bias = np.array(
        [
            [0.35],
            [0.35],
        ]
    )
    layer_b = Dense(neurons=(2, 2))
    layer_b.weights = np.array(
        [
            [0.4, 0.45],
            [0.5, 0.55],
        ]
    )
    layer_b.bias = np.array(
        [
            [0.6],
            [0.6],
        ]
    )
    # defining input and desired output
    inputs = np.array([[0.05], [0.1]])
    desired = np.array([[0.01], [0.99]])

    activate_fn = Sigmoid

    return layer_a, layer_b, activate_fn, inputs, desired


@pytest.fixture
def simple_nn():
    # defining layers and initial weights
    nn = NeuralNetwork(
        Dense(neurons=(2, 2)),
        Sigmoid(),
        Dense(neurons=(2, 2)),
        Sigmoid(),
    )
    nn.layers[0].weights = np.array(
        [
            [0.15, 0.2],
            [0.25, 0.3],
        ]
    )
    nn.layers[0].bias = np.array(
        [
            [0.35],
            [0.35],
        ]
    )
    nn.layers[2].weights = np.array(
        [
            [0.4, 0.45],
            [0.5, 0.55],
        ]
    )
    nn.layers[2].bias = np.array(
        [
            [0.6],
            [0.6],
        ]
    )
    # defining input and desired output
    inputs = np.array([[0.05], [0.1]])
    desired = np.array([[0.01], [0.99]])

    return nn, inputs, desired


@pytest.fixture
def simple_layers_b():
    # defining layers and initial weights
    layer_a = Dense(neurons=(2, 3))
    layer_a.weights = np.array(
        [
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.3],
        ]
    )
    layer_a.bias = np.array(
        [
            [-0.2],
            [-0.3],
            [-0.4],
        ]
    )
    layer_b = Dense(neurons=(3, 2))
    layer_b.weights = np.array(
        [
            [0.6, 0.2, 0.7],
            [0.7, 0.2, 0.8],
        ]
    )
    layer_b.bias = np.array(
        [
            [0.7],
            [0.3],
        ]
    )
    layer_c = Dense(neurons=(2, 1))
    layer_c.weights = np.array(
        [
            [0.8, 0.5],
        ]
    )
    layer_c.bias = np.array(
        [
            [-0.1],
        ]
    )
    # defining input
    inputs = np.array([[0.3], [0.7]])

    activate_fn = TanH

    return layer_a, layer_b, layer_c, activate_fn, inputs
