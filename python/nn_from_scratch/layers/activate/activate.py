from abc import ABC, abstractmethod

import numpy as np

from ..layer import Layer


class ActivateFunction(Layer, ABC):
    def forward(self, input):
        self.input = input
        self.output = self.activate(self.input)
        return self.output

    def backward(self, output_gradient: np.ndarray, alpha: float, batch_size: int):
        return output_gradient * self.derivative(self.input)

    @abstractmethod
    def activate(self, x) -> float:
        ...

    @abstractmethod
    def derivative(self, x) -> float:
        ...
