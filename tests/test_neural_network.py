import numpy as np
from numpy.testing import assert_array_almost_equal


def test_forward_nn(simple_nn):
    nn, inputs, _ = simple_nn

    nn._forward(inputs)

    # np asserts
    assert_array_almost_equal(nn.layers[0].net, np.array([0.3775, 0.3925]))
    assert_array_almost_equal(nn.layers[0].output, np.array([0.59326999, 0.59688438]))
    assert_array_almost_equal(nn.layers[1].net, np.array([1.10590597, 1.2249214]))
    assert_array_almost_equal(nn.layers[1].output, np.array([0.75136507, 0.77292847]))


def test_backward_nn(simple_nn):
    nn, inputs, desired = simple_nn

    output = nn._forward(inputs)
    output_err = desired - output
    nn._backward(0.5, output_err)


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
    assert_array_almost_equal(nn.layers[0].weights, expected_weights_a)
    assert_array_almost_equal(nn.layers[1].weights, expected_weights_b)