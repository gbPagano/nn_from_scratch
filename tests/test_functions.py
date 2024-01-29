from src.functions import ErrorFunction


def test_error_function(simple_nn):
    nn, inputs, desired_output = simple_nn

    assert nn.layers[0].next_layer is not None
    assert nn.layers[0].previous_layer is None
    assert nn.layers[1].next_layer is None
    assert nn.layers[1].previous_layer is not None

    nn._forward(inputs)

    error = ErrorFunction().get_mse(desired_output, nn.output)

    assert round(error, 9) == 0.298371109