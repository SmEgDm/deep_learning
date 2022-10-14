from abc import abstractclassmethod

import numpy as np


class Loss:
    def __init__(self, model):
        self.model = model

    @abstractclassmethod
    def forward(self, prediction, label):
        ...

    @abstractclassmethod
    def backward(self):
        ...

    def __call__(self, prediction, label):
        self.prediction = prediction
        self.label = label

        return self.forward(prediction, label)


class MSELoss(Loss):
    def forward(self, prediction, label):
        self.n = len(label)

        return np.sum((prediction - label) ** 2) / self.n

    def backward(self):
        return self.model.backward(2 * (self.prediction - self.label) / self.n)


class CrossEntropyLoss(Loss):
    def forward(self, prediction, label):
        self.s = np.exp(prediction) / np.sum(np.exp(prediction))

        return -np.sum(label * np.log(self.s))

    def backward(self):
        return self.model.backward(self.s - self.label)
