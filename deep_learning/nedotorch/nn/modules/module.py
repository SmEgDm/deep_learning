from abc import ABC, abstractclassmethod


class Module(ABC):
    def __init__(self, *paramenters):
        self.parameters = paramenters

    @abstractclassmethod
    def forward(self, input):
        ...

    @abstractclassmethod
    def backward(self, input):
        ...

    def __call__(self, input):
        self.forward_input = input.copy()
        self.forward_output = self.forward(input)

        return self.forward_output
