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
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.a = alpha

    def activate(self, x):
        return np.maximum(self.a * x, x)

    def derivative(self, x):
        return np.where(x >= 0, 1, self.a)


class ELU(ActivationFunction):
    def __init__(self, alpha=1.0):
        self.a = alpha

    def activate(self, x):
        return np.where(x >= 0, x, self.a * (np.exp(x) - 1))

    def derivative(self, x):
        return np.where(x >= 0, 1, self.activate(x) + self.a)


class ErrorFunction:
    def get_mse(self, desired, output):
        errors = []
        for y, out in zip(desired, output):
            err = 0.5 * sum((y - out) ** 2)
            errors.append(err)
        
        return sum(errors) / len(errors)
