import numpy as np

from .activate import ActivateFunction


class Sigmoid(ActivateFunction):
    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.activate(x) * (1 - self.activate(x))
