from typing import Optional

import numpy as np

from src.functions import ActivationFunction


class Layer:
    def __init__(self, neurons: tuple[int], function: ActivationFunction):
        self.weights = np.random.uniform(-0.5, 0.5, size=neurons)
        self.bias = np.random.uniform(-0.5, 0.5, size=(neurons[1]))
        self.function = function

        # to be defined later
        self.previous_layer: Optional[Layer] = None
        self.next_layer: Optional[Layer] = None
        self.delta = None
        self.old_weights = None
        self.input = None
        self.net = None
        self.output = None

    def forward(self, initial_input: Optional[np.ndarray] = None):
        self.input = (
            self.previous_layer.output if initial_input is None else initial_input
        )
        self.net = self.input.dot(self.weights.T) + self.bias
        self.output = self.function.activate(self.net)
        if self.next_layer is None:
            return self.output
        return self.next_layer.forward()

    def backward(self, alpha: float, error: Optional[float] = None):
        if self.next_layer is None:  # is last layer
            self.delta = error * self.function.derivative(self.net)
        else:
            self.delta = (
                self.next_layer.delta @ self.next_layer.old_weights
            ) * self.function.derivative(self.net)

        self.old_weights = self.weights.copy()
        self.weights += np.array([self.delta]).T @ np.array([self.input]) * alpha

        if self.previous_layer is not None:
            self.previous_layer.backward(alpha)


class NeuralNetwork:
    def __init__(self, *layers: Layer):
        self.layers = layers

        # linking the layers
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                layer.previous_layer = self.layers[idx]
            if idx + 1 < len(self.layers):
                layer.next_layer = self.layers[idx + 1]

        self.first_layer = self.layers[0]
        self.last_layer = self.layers[-1]

    def _forward(self, x_input: np.ndarray):
        self.first_layer.forward(x_input)

    def _backward(self, alpha: float, error: float):
        self.last_layer.backward(alpha, error)
