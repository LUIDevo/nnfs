import numpy as np
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

class LayerDense:
    def __init__(self, feautres, neurons):
        self.weights=0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases=np.zeros((1, neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases



# program setup
X, y = vertical_data(samples=100, classes=3)
dense1=
# program loop
