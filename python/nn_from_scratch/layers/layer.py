from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward():
        ...

    @abstractmethod
    def gradient_descent():
        ...

    @abstractmethod
    def update_weights():
        ...
