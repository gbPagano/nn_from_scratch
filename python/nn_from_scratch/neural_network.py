import numpy as np

from nn_from_scratch.layers import Layer
from nn_from_scratch.loss import MSE, Loss
from nn_from_scratch.utils import progress


class NeuralNetwork:
    def __init__(self, *layers: Layer):
        self.layers = layers

    def _forward(self, x_input: np.ndarray):
        self.output = x_input
        for layer in self.layers:
            self.output = layer.forward(self.output)
        return self.output

    def _backward(self, error, alpha: float, batch_size):
        for layer in reversed(self.layers):
            error = layer.backward(error, alpha, batch_size)

    def fit(
        self,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        alpha: float,
        batch_size: int,
        evaluate_step: int = 5,
        loss_function: Loss = MSE(),
    ):
        with progress:
            for epoch in progress.track(
                range(1, epochs + 1), description="Training..."
            ):
                # shuffling the data
                indexes = np.random.permutation(len(x_train))
                x_train, y_train = x_train[indexes], y_train[indexes]

                loss = 0
                for x, y in zip(x_train, y_train):
                    out = self._forward(x)
                    loss += loss_function.loss(y, out)
                    grad = loss_function.gradient(y, out)
                    self._backward(grad, alpha, batch_size)

                loss /= len(x_train)
                if not (epoch % evaluate_step):
                    padding = " " * (len(str(epochs)) - len(str(epoch)))
                    progress.console.print(f"Epoch: {padding}{epoch} | Loss: {loss}")
