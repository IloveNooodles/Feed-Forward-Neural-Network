import json
import os

import pandas as pd
from sklearn import datasets

from activation import Activation

ACTIVATION_LIST = [Activation.LINEAR, Activation.RELU,
                   Activation.SIGMOID, Activation.SOFTMAX]


class Reader:
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_data_excel(self, filepath: str):
        self.filepath = filepath
        df = pd.read_excel(filepath, index_col=False)
        if not df['target']:
            raise ("Dataset must have target attribute")
        return df

    """ 
    FFNN models are json like
    """
    @staticmethod
    def read_ffnn(filepath):
        try:
            with open(filepath, "rb") as f:
                json_file = json.loads(f.read())
                # Return models
                if validate_data(json_file):
                    return json_file["layers"], json_file['activation_functions'], json_file['neurons'], json_file['weights']
                return None
        except OSError as e:
            print("File not found")
            os._exit(-1)


def validate_data(json_data) -> bool:
    # Validate layers
    layers = json_data['layers']
    activation_functions = json_data['activation_functions']
    neurons = json_data['neurons']
    weights = json_data['weights']
    if not isinstance(json_data['layers'], int):
        return False

    # Validate activation function per layers
    if len(activation_functions) != layers:
        return False

    for function in activation_functions:
        if not isinstance(function, int):
            return False

        if function not in ACTIVATION_LIST:
            return False

    # Validate neurons
    if len(neurons) != layers:
        return False

    for neuron in neurons:
        if not isinstance(neuron, int):
            return False

    # Validate weights, weight len must be neuron + 1
    if len(weights) != layers:
        return False

    for index, weight in enumerate(weights):
        if len(weight) != neurons[index] + 1:
            return False

        for num in weight:
            if not isinstance(num, int):
                return False

    return True


if __name__ == "__main__":
    data = Reader.read_ffnn("./test/test.json")
    data2 = datasets.load_breast_cancer()
    print(data)
    print(data2)
