from nedotorch.nn.parameter import Parameter

from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.w = Parameter(out_features, in_features)
        self.b = Parameter(out_features, 1)

        super().__init__(self.w, self.b)

    def forward(self, input):
        return self.w.value @ input + self.b.value

    def backward(self, input):
        self.w.grad = input @ self.forward_input.T
        self.b.grad = input.copy()

        return self.w.value.T @ input
