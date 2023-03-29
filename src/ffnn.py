from typing import List

from activation import Activation
from reader import Reader


class FFNN:
    def __init__(self, model) -> None:
        self.layers = model['layers']
        self.activation_functions = model['activation_functions']
        self.neurons = model['neurons']
        self.weights = model['weights']
        self.rows = model['rows']
        self.data_names = model['data_names']
        self.data = model['data']
        self.target = model['target']
        self.target_names = model['target_names']
        self.epochs = None
        self.batch_size = None
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

    def set_data():
        pass

    def compute():
        pass

    def predict():
        pass

    def predict_classes():
        pass


if __name__ == "__main__":
    model = Reader.read_ffnn("./test/test.json")
    try:
        a = FFNN(model=model)
        print(a)
    except:
        raise Exception("Invalid formats")
