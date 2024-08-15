import numpy as np
from numpy.testing import assert_array_almost_equal


def test_forward_a(simple_layers_a):
    # example from https://ww2.inf.ufg.br/~anderson/deeplearning/20181/Aula%202%20-%20Multilayer%20Perceptron.pdf
    layer_a, layer_b, activate_fn, inputs, _ = simple_layers_a
    activate = activate_fn()

    out_1 = layer_a.forward(inputs)
    out_2 = activate.forward(out_1)
    out_3 = layer_b.forward(out_2)
    out_4 = activate.forward(out_3)

    # np asserts
    assert_array_almost_equal(out_1, np.array([[0.3775], [0.3925]]))
    assert_array_almost_equal(out_2, np.array([[0.59326999], [0.59688438]]))
    assert_array_almost_equal(out_3, np.array([[1.10590597], [1.2249214]]))
    assert_array_almost_equal(out_4, np.array([[0.75136507], [0.77292847]]))


def test_forward_b(simple_layers_b):
    # example from book "Redes Neurais Artificiais Para Engenharia e Ciências Aplicadas.
    # Fundamentos Teóricos e Aspectos Práticos (Ivan Nunes, Danilo Hernane, Rogério Andrade)"
    layer_a, layer_b, layer_c, activate_fn, inputs = simple_layers_b
    activate = activate_fn()

    out_1 = layer_a.forward(inputs)
    out_2 = activate.forward(out_1)
    out_3 = layer_b.forward(out_2)
    out_4 = activate.forward(out_3)
    out_5 = layer_c.forward(out_4)
    out_6 = activate.forward(out_5)

    # np asserts
    assert_array_almost_equal(out_1, np.array([[0.27], [0.37], [0.05]]), decimal=2)
    assert_array_almost_equal(out_2, np.array([[0.26], [0.35], [0.05]]), decimal=2)
    assert_array_almost_equal(out_3, np.array([[0.96], [0.59]]), decimal=2)
    assert_array_almost_equal(out_4, np.array([[0.74], [0.53]]), decimal=2)
    assert_array_almost_equal(out_5, np.array([[0.76]]), decimal=2)
    assert_array_almost_equal(out_6, np.array([[0.64]]), decimal=2)


def test_last_layer_backward(simple_layers_a):
    # example from https://ww2.inf.ufg.br/~anderson/deeplearning/20181/Aula%202%20-%20Multilayer%20Perceptron.pdf
    layer_a, layer_b, activate_fn, inputs, desired = simple_layers_a
    activate_a = activate_fn()
    activate_b = activate_fn()

    out_1 = layer_a.forward(inputs)
    out_2 = activate_a.forward(out_1)
    out_3 = layer_b.forward(out_2)
    out_4 = activate_b.forward(out_3)

    # TODO: document cost function used here
    # output_err = desired - out
    # output_err = 2 * (out - desired) / np.size(desired)
    output_err = -(desired - out_4)
    grad = activate_b.backward(output_err, 0.5, 1)
    layer_b.backward(grad, 0.5, 1)

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
    layer_a, layer_b, activate_fn, inputs, desired = simple_layers_a
    activate_a = activate_fn()
    activate_b = activate_fn()

    out_1 = layer_a.forward(inputs)
    out_2 = activate_a.forward(out_1)
    out_3 = layer_b.forward(out_2)
    out_4 = activate_b.forward(out_3)

    output_err = -(desired - out_4)
    grad = activate_b.backward(output_err, 0.5, 1)
    grad = layer_b.backward(grad, 0.5, 1)
    grad = activate_a.backward(grad, 0.5, 1)
    grad = layer_a.backward(grad, 0.5, 1)

    expected_weights = np.array(
        [
            [0.14978072, 0.19956143],
            [0.24975114, 0.29950229],
        ]
    )

    # np asserts
    assert_array_almost_equal(layer_a.weights, expected_weights)
