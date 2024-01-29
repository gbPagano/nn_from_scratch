from typing import Optional

import numpy as np
from rich.progress import track

from src.functions import ActivationFunction, ErrorFunction


class Layer:
    def __init__(self, neurons: tuple[int], function: ActivationFunction):
        input_size, output_size = neurons
        self.weights = np.random.uniform(-0.5, 0.5, size=(output_size, input_size))
        self.bias = np.random.uniform(-0.5, 0.5, size=(output_size))
        self.function = function

        # to be defined later
        self.previous_layer: Optional[Layer] = None
        self.next_layer: Optional[Layer] = None
        self.delta = None
        self.old_weights = None
        self.input = None
        self.net = None
        self.output = None

    def forward(self, initial_input: Optional[np.ndarray] = None):
        self.input = (
            self.previous_layer.output if initial_input is None else initial_input
        )
        self.net = (self.input @ self.weights.T) + self.bias
        self.output = self.function.activate(self.net)
        if self.next_layer is None:
            return self.output
        return self.next_layer.forward()

    def backward(self, alpha: float, error: Optional[float] = None):
        if self.next_layer is None:  # is last layer
            self.delta = error * self.function.derivative(self.net)
        else:
            self.delta = (
                self.next_layer.delta @ self.next_layer.old_weights
            ) * self.function.derivative(self.net)

        self.old_weights = self.weights.copy()
        self.weights += np.array([self.delta]).T @ np.array([self.input]) * alpha

        if self.previous_layer is not None:
            self.previous_layer.backward(alpha)


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
        for epoch in track(range(1, epochs+1), description="Processing..."):
            outputs = []
            data = list(zip(x_train,y_train))
            np.random.shuffle(data)
            x_train,y_train = zip(*data)
            x_train,y_train = np.array(x_train), np.array(y_train)
            for x, y in zip(x_train, y_train):
                out = self._forward(x)
                outputs.append(out)
                error = (y - out)
                batch_errors.append(error)
                if len(batch_errors) == batch_size:
                    batch_error = sum(batch_errors) / batch_size
                    self._backward(alpha, batch_error)
                    batch_errors = []

            
            mse = ErrorFunction().get_mse(y_train, outputs)
            if not epoch % 5:
                print(f"EPOCH: {epoch} MSE: {mse}")


if __name__ == "__main__":
    import polars as pl
    def number_to_neurons(n):
        res = [-1] * 10
        res[int(n)] = 1
        return res

    data_train = pl.read_csv("train.csv")

    y_train = np.array(data_train.drop_in_place("label"))
    y_train = np.array([number_to_neurons(y) for y in y_train])

    x_train = np.array([row for row in data_train.rows()]) / 255


    from src.functions import ReLU, TanH

 

    nn = NeuralNetwork(
        Layer(neurons=(784, 10), function=ReLU()),
        # Layer(neurons=(10, 10), function=ReLU()),
        Layer(neurons=(10, 10), function=TanH()),
    )

    nn.fit(
        x_train=x_train,
        y_train=y_train,
        epochs=500,
        alpha=0.05,
        batch_size=15,
    )

    def kaggle_predict(rede, n):
        data_test = pl.read_csv("test.csv")
        x_test = np.array([row for row in data_test.rows()]) / 255
        kaggle_df = pl.read_csv("sample_submission.csv")
        predicts = []
        for idx in range(28_000):
            predict = np.argmax(rede._forward(x_test[idx]))
            predicts.append(predict)

        df_predicts = pl.DataFrame({
            "Label": predicts
        })

        submission = kaggle_df.update(df_predicts)
        submission.write_csv(f"predicts_kaggle_{n}.csv")

    kaggle_predict(nn, "28/01 22:19")