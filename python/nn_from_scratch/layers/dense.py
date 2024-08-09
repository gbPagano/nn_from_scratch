from typing import Optional

import numpy as np

from nn_from_scratch.functions import ActivationFunction
from .layer import Layer


class Dense(Layer):
    def __init__(self, neurons: tuple[int], function: ActivationFunction):
        input_size, output_size = neurons
        self.weights = np.random.uniform(-0.5, 0.5, size=(output_size, input_size))
        self.bias = np.random.uniform(-0.5, 0.5, size=(output_size))
        self.function = function

        # to be defined later
        self.previous_layer: Optional[Layer] = None
        self.next_layer: Optional[Layer] = None
        self.delta = None
        self.input = None
        self.net = None
        self.output = None

    def forward(self, initial_input: Optional[np.ndarray] = None):
        self.input = (
            self.previous_layer.output if initial_input is None else initial_input
        )
        self.net = (self.input @ self.weights.T) + self.bias
        self.output = self.function.activate(self.net)
        if self.next_layer is None:
            return self.output
        return self.next_layer.forward()

    def gradient_descent(self, error: Optional[np.ndarray] = None):
        if self.next_layer is None:  # is last layer
            self.delta = error * self.function.derivative(self.net)
        else:
            self.delta = (
                self.next_layer.delta @ self.next_layer.weights
            ) * self.function.derivative(self.net)

        gradient_descent = np.array([self.delta]).T @ np.array([self.input])
        return gradient_descent

    def update_weights(self, alpha: float, gradient_descent: np.ndarray):
        self.weights += gradient_descent * alpha
