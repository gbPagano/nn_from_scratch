import numpy as np

from .activate import ActivateFunction


class TanH(ActivateFunction):
    def activate(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - self.activate(x) ** 2
