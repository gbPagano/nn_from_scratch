from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC): # pragma: no cover
    @abstractmethod
    def activate(self, x) -> float:
        ...

    @abstractmethod
    def derivative(self, x) -> float:
        ...


class Tanh(ActivationFunction):
    def activate(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - self.activate(x) ** 2

class Sigmoid(ActivationFunction):
    def activate(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        return self.activate(x) * (1 - self.activate(x))

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


class ErrorFunc:
    def get_error(self, desired, output):
        return .5 * (desired - output)**2




layer_a = Layer(neurons=(2, 2), function=Sigmoid())
layer_a.weights = np.array([
    [0.15, 0.2],
    [0.25, 0.3],
])
layer_a.bias = np.array([
    0.35, .35,
])
layer_b = Layer(neurons=(2, 2), function=Sigmoid())
layer_b.weights = np.array([
    [0.4, 0.45],
    [0.5, 0.55],
])
layer_b.bias = np.array([
    0.6, .6,
])


inputs = np.array([
    0.05, .1
])
desired = np.array([
    0.01, 0.99
])



out_a = layer_a.forward(inputs)

out_b = layer_b.forward(out_a)

err = ErrorFunc().get_error(desired, out_b)
err_total = sum(err)