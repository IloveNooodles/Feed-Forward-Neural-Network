import math

import numpy as np


class Activation:
    LINEAR = 0
    RELU = 1
    SIGMOID = 2
    SOFTMAX = 3

    def __init__(self, mode, weights, input_neuron, attribute) -> None:
        self.mode = mode
        self.num_attribute = attribute
        self.weights = weights
        self.input_neuron = input_neuron

    def __linear(self, res):
        return res

    def __sigmoid(self, res):
        return [(1 / (1 + pow(math.e, -x))) for x in res]

    def __relu(self, res):
        res[res < 0] = 0
        return res

    def __softmax(self, res):
        numerator = np.array([pow(math.e, x) for x in res])
        denominator = np.sum([pow(math.e, x) for x in res])
        return numerator / denominator

    def calculate(self, x, w, b):
        res = np.matmul(x, w)
        res = np.add(res, b)
        if self.mode == Activation.LINEAR:
            return self.__linear(res)
        elif self.mode == Activation.RELU:
            return self.__relu(res)
        elif self.mode == Activation.SIGMOID:
            return self.__sigmoid(res)
        elif self.mode == Activation.SOFTMAX:
            return self.__softmax(res)
        else:
            raise Exception(
                "Mode is not implemented, please select correct mode")


if __name__ == "__main__":

    # x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # w1 = np.array([[1, 1], [1, 1]])
    # b1 = np.array([0, -1])
    # res = np.matmul(x, w1)
    # res = np.add(res, b1)
    # print(res)
    # res[res < 0] = 0
    # print(res)

    # w = np.array([1, -2])
    # res = np.matmul(res, w)
    # Max 0

    # numerator = np.array([pow(math.e, x) for x in [3, 4, 1]])
    # denominator = np.sum([pow(math.e, x) for x in [3, 4, 1]])
    # print(numerator / denominator)
