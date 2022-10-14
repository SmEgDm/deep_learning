from abc import abstractclassmethod

import numpy as np


class Optimizer:
    def __init__(self, parameters, learning_rate=10e-2, weight_decay=0):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    @abstractclassmethod
    def step(self):
        ...

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.grad)
