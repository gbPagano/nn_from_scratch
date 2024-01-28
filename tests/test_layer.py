import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from src.functions import Sigmoid, TanH
from src.main import Layer


@pytest.fixture
def simple_layers_a():
    # defining layers and initial weights
    layer_a = Layer(neurons=(2, 2), function=Sigmoid())
    layer_a.weights = np.array(
        [
            [0.15, 0.2],
            [0.25, 0.3],
        ]
    )
    layer_a.bias = np.array(
        [
            0.35,
            0.35,
        ]
    )
    layer_b = Layer(neurons=(2, 2), function=Sigmoid())
    layer_b.weights = np.array(
        [
            [0.4, 0.45],
            [0.5, 0.55],
        ]
    )
    layer_b.bias = np.array(
        [
            0.6,
            0.6,
        ]
    )
    # defining input and desired output
    inputs = np.array([0.05, 0.1])
    desired = np.array([0.01, 0.99])

    return layer_a, layer_b, inputs, desired


@pytest.fixture
def simple_layers_b():
    # defining layers and initial weights
    layer_a = Layer(neurons=(2, 3), function=TanH())
    layer_a.weights = np.array(
        [
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.3],
        ]
    )
    layer_a.bias = np.array(
        [
            -0.2,
            -0.3,
            -0.4,
        ]
    )
    layer_b = Layer(neurons=(3, 2), function=TanH())
    layer_b.weights = np.array(
        [
            [0.6, 0.2, 0.7],
            [0.7, 0.2, 0.8],
        ]
    )
    layer_b.bias = np.array(
        [
            0.7,
            0.3,
        ]
    )
    layer_c = Layer(neurons=(2, 1), function=TanH())
    layer_c.weights = np.array(
        [
            [0.8, 0.5],
        ]
    )
    layer_c.bias = np.array(
        [
            -0.1,
        ]
    )
    # defining input
    inputs = np.array([0.3, 0.7])

    return layer_a, layer_b, layer_c, inputs


def test_forward_a(simple_layers_a):
    # example from https://ww2.inf.ufg.br/~anderson/deeplearning/20181/Aula%202%20-%20Multilayer%20Perceptron.pdf
    layer_a, layer_b, inputs, _ = simple_layers_a

    layer_a.forward(inputs)
    layer_b.forward(layer_a.output)

    # np asserts
    assert_array_almost_equal(layer_a.net, np.array([0.3775, 0.3925]))
    assert_array_almost_equal(layer_a.output, np.array([0.59326999, 0.59688438]))
    assert_array_almost_equal(layer_b.net, np.array([1.10590597, 1.2249214]))
    assert_array_almost_equal(layer_b.output, np.array([0.75136507, 0.77292847]))


def test_forward_linked_layers(simple_layers_a):
    # example from https://ww2.inf.ufg.br/~anderson/deeplearning/20181/Aula%202%20-%20Multilayer%20Perceptron.pdf
    layer_a, layer_b, inputs, _ = simple_layers_a

    layer_a.next_layer = layer_b
    layer_b.previous_layer = layer_a
    layer_a.forward(inputs)

    # np asserts
    assert_array_almost_equal(layer_a.net, np.array([0.3775, 0.3925]))
    assert_array_almost_equal(layer_a.output, np.array([0.59326999, 0.59688438]))
    assert_array_almost_equal(layer_b.net, np.array([1.10590597, 1.2249214]))
    assert_array_almost_equal(layer_b.output, np.array([0.75136507, 0.77292847]))


def test_forward_b(simple_layers_b):
    # example from book "Redes Neurais Artificiais Para Engenharia e Ciências Aplicadas.
    # Fundamentos Teóricos e Aspectos Práticos (Ivan Nunes, Danilo Hernane, Rogério Andrade)"
    layer_a, layer_b, layer_c, inputs = simple_layers_b

    layer_a.forward(inputs)
    layer_b.forward(layer_a.output)
    layer_c.forward(layer_b.output)

    # np asserts
    assert_array_almost_equal(layer_a.net, np.array([0.27, 0.37, 0.05]), decimal=2)
    assert_array_almost_equal(layer_a.output, np.array([0.26, 0.35, 0.05]), decimal=2)
    assert_array_almost_equal(layer_b.net, np.array([0.96, 0.59]), decimal=2)
    assert_array_almost_equal(layer_b.output, np.array([0.74, 0.53]), decimal=2)
    assert_array_almost_equal(layer_c.net, np.array([0.76]), decimal=2)
    assert_array_almost_equal(layer_c.output, np.array([0.64]), decimal=2)


def test_last_layer_backward(simple_layers_a):
    # example from https://ww2.inf.ufg.br/~anderson/deeplearning/20181/Aula%202%20-%20Multilayer%20Perceptron.pdf
    layer_a, layer_b, inputs, desired = simple_layers_a

    layer_a.forward(inputs)
    layer_b.forward(layer_a.output)

    output_err = desired - layer_b.output
    layer_b.backward(0.5, output_err)

    expected_weights = np.array(
        [
            [0.35891648, 0.408666186],
            [0.511301270, 0.561370121],
        ]
    )

    # np asserts
    assert_array_almost_equal(layer_b.weights, expected_weights)


def test_middle_layer_backward(simple_layers_a):
    # example from https://ww2.inf.ufg.br/~anderson/deeplearning/20181/Aula%202%20-%20Multilayer%20Perceptron.pdf
    layer_a, layer_b, inputs, desired = simple_layers_a

    layer_a.forward(inputs)
    layer_b.forward(layer_a.output)
    layer_a.next_layer = layer_b

    output_err = desired - layer_b.output
    layer_b.backward(0.5, output_err)
    layer_a.backward(0.5)

    expected_weights = np.array(
        [
            [0.14978072, 0.19956143],
            [0.24975114, 0.29950229],
        ]
    )

    # np asserts
    assert_array_almost_equal(layer_a.weights, expected_weights)


def test_backward_linked_layers(simple_layers_a):
    # example from https://ww2.inf.ufg.br/~anderson/deeplearning/20181/Aula%202%20-%20Multilayer%20Perceptron.pdf
    layer_a, layer_b, inputs, desired = simple_layers_a

    layer_a.forward(inputs)
    layer_b.forward(layer_a.output)
    layer_a.next_layer = layer_b
    layer_b.previous_layer = layer_a

    output_err = desired - layer_b.output
    layer_b.backward(0.5, output_err)

    expected_weights_a = np.array(
        [
            [0.14978072, 0.19956143],
            [0.24975114, 0.29950229],
        ]
    )
    expected_weights_b = np.array(
        [
            [0.35891648, 0.408666186],
            [0.511301270, 0.561370121],
        ]
    )

    # np asserts
    assert_array_almost_equal(layer_a.weights, expected_weights_a)
    assert_array_almost_equal(layer_b.weights, expected_weights_b)
