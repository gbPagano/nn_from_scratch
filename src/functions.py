from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, x) -> float:
        ...

    @abstractmethod
    def derivative(self, x) -> float:
        ...


class TanH(ActivationFunction):
    def activate(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - self.activate(x) ** 2


class Sigmoid(ActivationFunction):
    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.activate(x) * (1 - self.activate(x))


class ReLU(ActivationFunction):
    def activate(self, x):
        return max(0, x)

    def derivative(self, x):
        return 1 if x >= 0 else 0


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.a = alpha

    def activate(self, x):
        return max(self.a * x, x)

    def derivative(self, x):
        return 1 if x >= 0 else self.a


class ELU(ActivationFunction):
    def __init__(self, alpha=1.0):
        self.a = alpha

    def activate(self, x):
        return x if x >= 0 else self.a * (np.exp(x) - 1)

    def derivative(self, x):
        return 1 if x >= 0 else self.activate(x) + self.a


class ErrorFunction:
    def get_error(self, desired, output):
        return 0.5 * (desired - output) ** 2
