import numpy as np

from .layer import Layer


class Dense(Layer):
    def __init__(self, neurons: tuple[int]):
        input_size, output_size = neurons
        self.weights = np.random.uniform(-0.5, 0.5, size=(output_size, input_size))
        self.bias = np.random.uniform(-0.5, 0.5, size=(output_size, 1))
        self.curr_batch = 0

        # to be defined later
        self.delta = None
        self.input = None
        self.net = None
        self.output = None

    def forward(self, input: np.ndarray):
        self.input = input
        self.output = (self.weights @ self.input) + self.bias
        return self.output

    def backward(self, output_gradient: np.ndarray, alpha: float, batch_size: int):
        weights_gradient = output_gradient @ self.input.T
        input_gradient = self.weights.T @ output_gradient

        if self.curr_batch == 0:
            self.weights_gradient = weights_gradient
            self.bias_gradient = output_gradient
        else:
            self.weights_gradient += weights_gradient
            self.bias_gradient += output_gradient

        self.curr_batch += 1
        if self.curr_batch == batch_size:
            self.weights -= alpha * (self.weights_gradient / batch_size)
            self.bias -= alpha * (self.bias_gradient / batch_size)
            self.curr_batch = 0

        return input_gradient
