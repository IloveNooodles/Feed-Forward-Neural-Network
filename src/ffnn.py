from typing import List

import numpy as np

from activation import Activation
from reader import Reader


class FFNN:
    def __init__(self, model, epochs=1, batch_size=1) -> None:
        self.layers = model['layers']
        self.activation_functions = np.array(model['activation_functions'])
        self.neurons = np.array(model['neurons'])
        self.weights = model['weights']
        self.rows = model['rows']
        self.data_names = np.array(model['data_names'])
        self.data = np.array(model['data'])
        self.target = np.array(model['target'])
        self.target_names = np.array(model['target_names'])
        self.epochs = epochs
        self.batch_size = batch_size
        self.output = None
        pass

    def __str__(self) -> str:
        return f"\
  Layers: {self.layers}\n\
  Activations: {self.activation_functions}\n\
  Neurons: {self.neurons}\n\
  Weights: {self.weights}\n\
  Rows: {self.rows}\n\
  Data: {self.data}\n\
  Data_names: {self.data_names}\n\
  target: {self.target}\n\
  target_names: {self.target_names}\n\
  epochs: {self.epochs}\n\
  batch_size: {self.batch_size}\n"

    # Will return output functions
    def compute(self):
        res = self.data
        for i in range(self.layers - 1):
            activation_function = Activation(self.activation_functions[i])
            transposed_weights = np.transpose(np.array(self.weights[i]))
            weights, bias = self._separate_bias(transposed_weights)
            res = activation_function.calculate(res, weights, bias)

        self.output = res
        return res

    def _add_bias(self, data):
        temp_data = np.ones((data.shape[0], data.shape[0] + 1))
        temp_data[:, 1:] = data
        return temp_data

    def _separate_bias(self, data):
        bias = data[0, :]
        weight = data[1:, :]
        print(weight, bias)
        return weight, bias

    def predict(self, instances):
        activation_function = Activation(self.activation_functions[-1])
        res = activation_function.calculate_one(instances)
        print(res)

    def predict_classes(self):
        pass


if __name__ == "__main__":
    model = Reader.read_ffnn("./test/test.json")
    a = FFNN(model=model)
    output_function = a.compute()
    
    # print(type(a.data))
    # print(a)
