from .optimizer import Optimizer


class SGD(Optimizer):
    def step(self):
        for parameter in self.parameters:
            parameter.value -= self.learning_rate * parameter.grad
