from typing import List

from activation import Activation
from reader import Reader


class FFNN:
    def __init__(self, layers: int, activation_functions: List[int], neurons: List[int], weights: List[List[int]]) -> None:
        self.layers = layers
        self.activation_functions = activation_functions
        self.neurons = neurons
        self.weights = weights
        self.epochs = None
        self.batch_size = None
        pass

    def __str__(self) -> str:
        return f"Layers: {self.layers}\nActivations: {self.activation_functions}\nNeurons: {self.neurons}\nWeights: {self.weights}"

    def set_data():
        pass

    def compute():
        pass

    def predict():
        pass

    def predict_classes():
        pass


if __name__ == "__main__":
    layer, activations, neurons, weights = Reader.read_ffnn("./test/test.json")
    try:
        a = FFNN(layers=layer, activation_functions=activations,
                 neurons=neurons, weights=weights)
        print(a)
    except:
        raise ("Invalid formats")
