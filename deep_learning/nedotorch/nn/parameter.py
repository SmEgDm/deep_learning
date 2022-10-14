import numpy as np


class Parameter:
    def __init__(self, *shape):
        self.value = np.random.uniform(-0.5, 0.5, shape)
        self.grad = None
