from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward():
        ...

    @abstractmethod
    def backward():
        ...
