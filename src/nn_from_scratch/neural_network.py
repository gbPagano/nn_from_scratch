from functools import reduce

import numpy as np

from nn_from_scratch.functions import ErrorFunction
from nn_from_scratch.layers import Layer
from nn_from_scratch.utils import chunks, progress


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

    def _get_all_layers_gradient_descent(self, error: np.ndarray):
        gradients = []
        for layer in reversed(self.layers):
            gradients.append(layer.calc_gradient_descent(error))

        return gradients

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
        with progress:
            for epoch in progress.track(
                range(1, epochs + 1), description="Training..."
            ):
                # shuffling the data
                indexes = np.random.permutation(len(x_train))
                x_train, y_train = x_train[indexes], y_train[indexes]

                outputs = []
                for chunk in chunks(zip(x_train, y_train), batch_size):
                    batch_errors = []
                    batch_gradients = []
                    for x, y in chunk:
                        out = self._forward(x)
                        outputs.append(out)
                        error = y - out
                        batch_errors.append(error)
                        gradients = self._get_all_layers_gradient_descent(error)
                        batch_gradients.append(gradients)

                    mean_gradients = [
                        np.mean(layer_grad, axis=0)
                        for layer_grad in zip(*batch_gradients)
                    ]
                    for layer, gradient in zip(reversed(self.layers), mean_gradients):
                        layer.update_weights(alpha, gradient)

                if not epoch % evaluate_step:
                    mse = ErrorFunction().get_mse(y_train, outputs)
                    padding = " " * (len(str(epochs)) - len(str(epoch)))
                    progress.console.print(f"Epoch: {padding}{epoch} | MSE: {mse}")
