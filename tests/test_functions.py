from nn_from_scratch.functions import ErrorFunction


def test_error_function(simple_nn):
    nn, inputs, desired_output = simple_nn

    nn._forward(inputs)

    error = ErrorFunction().get_mse([desired_output], [nn.output])

    assert round(error, 9) == 0.298371109
