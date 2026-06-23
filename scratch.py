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

class Activation_Softmax:
    def forward(self, inputs):
       exp_values=np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
       probabilities=exp_values/inputs.sum(exp_values, axis=1, keepdims=True)
       self.output=probabilities

class Loss_CategoricalCrossentropy:

class Activation_Softmax_Loss:
    def __init__(self):
        self.activation=Activation_Softmax()
        self.loss=Loss_CategoricalCrossentropy()
    def forward(self, inputs,y_true):
        self.activation.forward(inputs)
        self.loss.calculate(inputs, y_true)

# program setup
X, y = vertical_data(samples=100, classes=3)
dense1=LayerDense(2,3)
output=Activation_Softmax_Loss()

# program loop
dense1.forward(X)
loss=output.forward(dense1.output, y)
print(loss, dense1.weights)
output.backward(output.output, y)
dense1.backward(output.dinputs)
print(dense1.weights)
