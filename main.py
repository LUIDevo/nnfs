import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output=np.dot(inputs, self.weights)+self.biases

x, y = spiral_data(samples=100, classes=3)
dense1=Layer_Dense(2,3)
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)
# sample data                                            
dense1 = Layer_Dense(2, 3)
# creating a layer
activation1 = Activation_ReLU()
# creating an activation function
dense2 = Layer_Dense(3, 3)
# creating a second layer
activation2 = Activation_Softmax()
# Softmax output function
dense1.forward(X)
# putting input forward
activation1.forward(dense1.output)
# normalising the output
dense2.forward(activation1.output)
# putting layer1 into layer 2
activation2.forward(dense2.output)
# softmax output 

print(activation2.output[:5])
