import numpy as np

from src.functions import ActivationFunction


class Layer:
    def __init__(self, neurons: tuple[int], function: ActivationFunction):
        self.weights = np.random.uniform(-0.5, 0.5, size=neurons)
        self.bias = np.random.uniform(-0.5, 0.5, size=(neurons[1]))
        self.function = function
        self.is_last_layer = False

    def forward(self, layer_input):
        self.input = layer_input
        self.net = self.input.dot(self.weights.T) + self.bias
        self.output = self.function.activate(self.net)
        return self.output

    def backward(self, alpha, error):
        self.delta = error * self.function.derivative(self.net)
        self.weights += np.array([self.delta]).T * np.array([self.input]) * alpha
