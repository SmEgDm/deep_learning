from abc import abstractclassmethod

import numpy as np

from .module import Module


class Activation(Module):
    def __init__(self):
        super().__init__()

    @abstractclassmethod
    def forward(self, input):
        ...

    @abstractclassmethod
    def backward(self, input):
        ...


class Tanh(Activation):
    def forward(self, input):
        return np.tanh(input)

    def backward(self, input):
        return (1 - self.forward_output**2)[::-1] * input


class ReLU(Activation):
    def forward(self, input):
        return input * (input > 0)

    def backward(self, input):
        return np.heaviside(self.forward_output, 0)[::-1] * input


class Softmax(Activation):
    def forward(self, input):
        return np.exp(input) / np.sum(np.exp(input))

    def backward(self, input):
        s = self.forward_output

        return (np.diag(s.T[0]) - s @ s.T).T @ input
