from nn_from_scratch.loss import MSE


def test_loss_function(simple_nn):
    nn, inputs, desired_output = simple_nn
    nn._forward(inputs)

    error = MSE().loss(desired_output, nn.output)

    assert round(error, 9) == 0.298371109
