class Activation:
    LINEAR = 0
    RELU = 1
    SIGMOID = 2
    SOFTMAX = 3

    def __init__(self, mode) -> None:
        self.mode = mode
        pass

    def __linear(self):
        pass

    def __sigmoid(self):
        pass

    def __relu(self):
        pass

    def __softmax(self):
        pass

    def calculate(self):
        if self.mode == Activation.LINEAR:
            return self.__linear()
        elif self.mode == Activation.RELU:
            return self.__relu()
        elif self.mode == Activation.SIGMOID:
            return self.__sigmoid()
        elif self.mode == Activation.SOFTMAX:
            return self.__softmax()
        else:
            raise ("Mode is not implemented, please select correct mode")
