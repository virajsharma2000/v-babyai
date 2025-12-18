
import numpy as np

class BabyModel:
    def __init__(self, input_size):
        # Baby is born knowing nothing
        # Random weights = random thoughts
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
