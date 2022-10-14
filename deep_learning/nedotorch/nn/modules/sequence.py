from functools import reduce

from .module import Module


class Sequence(Module):
    def __init__(self, *sequence):
        self.sequence = sequence

        super().__init__(
            *reduce(
                lambda result, module: result + module.parameters,
                self.sequence,
                (),
            )
        )

    def forward(self, input):
        return reduce(
            lambda x, f: f(x),
            self.sequence,
            input,
        )

    def backward(self, output):
        return reduce(
            lambda x, f: f.backward(x),
            self.sequence[::-1],
            output,
        )
