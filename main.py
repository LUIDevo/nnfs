import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output=np.dot(inputs, self.weights)+self.biases

x, y = spiral_data(samples=100, classes=3)
dense1=Layer_Dense(2,3)
# creating the layer with 3 neurons, 2 inputs

dense1.forward(x)

print(dense1.output[:5])
