import numpy as np

from nn_from_scratch.functions import ErrorFunction
from nn_from_scratch.layers import Layer
from nn_from_scratch.utils import progress


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
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        alpha: float,
        batch_size: int,
        evaluate_step: int = 5,
    ):
        assert (
            len(y_train) % batch_size == 0
        ), "The dataset must be divisible by the batch size"
        with progress:
            for epoch in progress.track(
                range(1, epochs + 1), description="Training..."
            ):
                batch_errors = []
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

                if not epoch % evaluate_step:
                    mse = ErrorFunction().get_mse(y_train, outputs)
                    padding = " " * (len(str(epochs)) - len(str(epoch)))
                    progress.console.print(f"Epoch: {padding}{epoch} | MSE: {mse}")
