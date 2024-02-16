from datetime import datetime

import numpy as np
import polars as pl

from nn_from_scratch import Layer, NeuralNetwork
from nn_from_scratch.functions import ELU, TanH


def number_to_neurons(
    n: int, *, negative_output: int = -1, positive_output: int = 1
) -> list[int]:
    """Creates an output vector to represent the outputs of the last layer of neurons."""
    res = [negative_output] * 10
    res[n] = positive_output
    return res


def kaggle_submission(nn: NeuralNetwork, filename: str):
    data_test = pl.read_csv("datasets/kaggle_mnist/test.csv")
    x_test = np.array([row for row in data_test.rows()]) / 255

    predicts = [np.argmax(nn._forward(x)) for x in x_test]

    kaggle_sample_df = pl.read_csv("datasets/kaggle_mnist/sample_submission.csv")
    df_predicts = pl.DataFrame({"Label": predicts})
    submission = kaggle_sample_df.update(df_predicts)
    submission.write_csv(f"{filename}.csv")


def main():
    data_train = pl.read_csv("datasets/kaggle_mnist/train.csv")

    y_train = np.array(data_train.drop_in_place("label"))
    y_train = np.array([number_to_neurons(y) for y in y_train])

    x_train = np.array([row for row in data_train.rows()]) / 255

    nn = NeuralNetwork(
        Layer(neurons=(784, 28), function=ELU()),
        Layer(neurons=(28, 19), function=ELU()),
        Layer(neurons=(19, 10), function=TanH()),
    )
    nn.fit(
        x_train=x_train,
        y_train=y_train,
        epochs=100,
        alpha=0.02,
        batch_size=8,
        evaluate_step=5,
    )

    filename = f"kaggle_submission_{datetime.now().strftime('%Y-%m-%d_%H:%M')}"
    # kaggle_submission(nn, filename)


if __name__ == "__main__":
    main()
