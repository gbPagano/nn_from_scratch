import numpy as np
from rich.progress import track

from nn_from_scratch.functions import ErrorFunction
from nn_from_scratch.layers import Layer


class NeuralNetwork:
    def __init__(self, *layers: Layer):
        self.layers = layers

        # linking the layers
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                layer.previous_layer = self.layers[idx - 1]
            if idx + 1 < len(self.layers):
                layer.next_layer = self.layers[idx + 1]

        self.first_layer = self.layers[0]
        self.last_layer = self.layers[-1]

    def _forward(self, x_input: np.ndarray):
        self.output = self.first_layer.forward(x_input)
        return self.output

    def _backward(self, alpha: float, error: float):
        self.last_layer.backward(alpha, error)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        alpha: float,
        batch_size: int,
    ):
        batch_errors = []
        for epoch in track(range(1, epochs + 1), description="Processing..."):
            outputs = []

            # shuffling the data
            indexes = np.random.permutation(len(x_train))
            x_train, y_train = x_train[indexes], y_train[indexes]

            for x, y in zip(x_train, y_train):
                out = self._forward(x)
                outputs.append(out)
                error = y - out
                batch_errors.append(error)
                if len(batch_errors) == batch_size:
                    batch_error = sum(batch_errors) / batch_size
                    self._backward(alpha, batch_error)
                    batch_errors = []

            mse = ErrorFunction().get_mse(y_train, outputs)
            if not epoch % 5:
                print(f"EPOCH: {epoch} MSE: {mse}")
