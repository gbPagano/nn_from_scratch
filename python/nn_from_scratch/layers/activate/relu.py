import numpy as np

from .activate import ActivateFunction


class ReLU(ActivateFunction):
    def activate(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class LeakyReLU(ActivateFunction):
    def __init__(self, alpha=0.01):
        self.a = alpha

    def activate(self, x):
        return np.maximum(self.a * x, x)

    def derivative(self, x):
        return np.where(x >= 0, 1, self.a)


class ELU(ActivateFunction):
    def __init__(self, alpha=1.0):
        self.a = alpha

    def activate(self, x):
        return np.where(x >= 0, x, self.a * (np.exp(x) - 1))

    def derivative(self, x):
        return np.where(x >= 0, 1, self.activate(x) + self.a)
