from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def loss(self, real, predicted) -> float:
        ...

    @abstractmethod
    def gradient(self, real, predicted):
        ...
