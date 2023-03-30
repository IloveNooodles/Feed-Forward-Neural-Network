import numpy as np

from activation import Activation
from reader import Reader


class FFNN:
    def __init__(self, model) -> None:
        self.layers = model['layers']
        self.activation_functions = np.array(model['activation_functions'])
        self.neurons = np.array(model['neurons'])
        self.weights = model['weights']
        self.rows = model['rows']
        self.data_names = np.array(model['data_names'])
        self.data = np.array(model['data'])
        self.target = np.array(model['target'])
        self.target_names = np.array(model['target_names'])
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
  target_names: {self.target_names}\n"

    # Will return output functions
    def compute(self):
        res = self.data
        for i in range(self.layers - 1):
            activation_function = Activation(self.activation_functions[i])
            transposed_weights = np.transpose(np.array(self.weights[i]))
            weights, bias = self._separate_bias(transposed_weights)
            res = activation_function.calculate(res, weights, bias)
            # print(res)

        self.output = res
        return res

    def _add_bias(self, data):
        temp_data = np.ones((data.shape[0], data.shape[0] + 1))
        temp_data[:, 1:] = data
        return temp_data

    def _separate_bias(self, data):
        bias = data[0, :]
        weight = data[1:, :]
        return weight, bias

    def predict(self):
        self.compute()
        A = Activation(self.activation_functions[-1])
        res = A.predict(self.output)
        print(f"\
  Data Names: {self.data_names}\n\
  Data: {self.data}\n\
  Target Names: {self.target_names}\n\
  Target: {self.target}\n\
  Predictions: {np.transpose(res)}\n")
  # Predictions: {np.transpose(self.res)}\n")


if __name__ == "__main__":
    model = Reader.read_ffnn("./test/xor_linear_relu.json")
    a = FFNN(model=model)
    a.compute()
    a.predict()
    # print(type(a.data))
    # print(a)
