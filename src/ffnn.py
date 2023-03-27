from typing import List

from activation import Activation
from reader import Reader


class FFNN:
    def __init__(self, layers: int, activation_functions: List[int], neurons: List[int], weights: List[List[int]]) -> None:
        self.layers = layers
        self.activation_functions = activation_functions
        self.neurons = neurons
        self.weights = weights
        pass

    def __str__(self) -> str:
        return f"Layers: {self.layers}\nActivations: {self.activation_functions}\nNeurons: {self.neurons}\nWeights: {self.weights}"

    def compute():
        pass

    def predict_one():
        pass
    
    def predict_many():
        pass


if __name__ == "__main__":
    layer, activations, neurons, weights = Reader.read_ffnn("./test/test.json")
    try:
        a = FFNN(layers=layer, activation_functions=activations,
                 neurons=neurons, weights=weights)
        print(a)
    except:
        raise ("Invalid formats")
