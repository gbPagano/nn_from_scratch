import numpy as np

from .loss import Loss


class MSE(Loss):
    def loss(self, real, predicted):
        return np.mean(np.power(real - predicted, 2))

    def gradient(self, real, predicted):
        return 2 * (predicted - real)


class HalfMSE(Loss):
    def loss(self, real, predicted):
        return np.mean(np.power(real - predicted, 2)) / 2

    def gradient(self, real, predicted):
        return predicted - real
