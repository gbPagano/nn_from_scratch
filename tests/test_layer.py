import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from src.main import Layer, Sigmoid

@pytest.fixture
def simple_layers():
    # defining layers and initial weights
    layer_a = Layer(neurons=(2, 2), function=Sigmoid())
    layer_a.weights = np.array([
        [0.15, 0.2],
        [0.25, 0.3],
    ])
    layer_a.bias = np.array([
        0.35, .35,
    ])
    layer_b = Layer(neurons=(2, 2), function=Sigmoid())
    layer_b.weights = np.array([
        [0.4, 0.45],
        [0.5, 0.55],
    ])
    layer_b.bias = np.array([
        0.6, .6,
    ])
    # defining input and desired output
    inputs = np.array([
        0.05, .1
    ])
    desired = np.array([
        0.01, 0.99
    ])

    return layer_a, layer_b, inputs, desired


def test_forward(simple_layers):
    layer_a, layer_b, inputs, _ = simple_layers

    layer_a.forward(inputs)
    layer_b.forward(layer_a.output)

    # np asserts
    assert_array_almost_equal(layer_a.net, np.array([0.3775, 0.3925]))
    assert_array_almost_equal(layer_a.output, np.array([0.59326999, 0.59688438]))
    assert_array_almost_equal(layer_b.net, np.array([1.10590597, 1.2249214]))
    assert_array_almost_equal(layer_b.output, np.array([0.75136507, 0.77292847]))


def test_last_layer_backward(simple_layers):
    layer_a, layer_b, inputs, desired = simple_layers
    
    layer_a.forward(inputs)
    layer_b.forward(layer_a.output)

    output_err = desired - layer_b.output
    layer_b.backward(0.5, output_err)

    expected_weights = np.array([
        [.35891648, .408666186],
        [.511301270, .561370121],
    ])

    # np asserts
    assert_array_almost_equal(layer_b.weights, expected_weights)
