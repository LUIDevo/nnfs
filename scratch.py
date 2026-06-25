import numpy as np
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

class LayerDense:
    def __init__(self, features, neurons):
        self.weights=0.01 * np.random.randn(features, neurons)
        self.biases=np.zeros((1, neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_Softmax:
    def forward(self, inputs):
       exp_values=np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
       probabilities=exp_values/np.sum(exp_values, axis=1, keepdims=True)
       self.output=probabilities
       return self.output

class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y):
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred, 1e-7, 1 - 1e-7) # clip to small value between (0, 1)
        correct_confidences=y_pred_clipped[range(samples), y] # take the correct prediction confidence for all rows
        return -np.log(correct_confidences) # return the negative log of the above values

    def calculate(self, inputs, y):
        sample_losses=self.forward(inputs, y)
        data_losses=np.mean(sample_losses)
        return data_losses

class Activation_Softmax_Loss:
    def __init__(self):
        self.activation=Activation_Softmax()
        self.loss=Loss_CategoricalCrossentropy()
    def forward(self, inputs,y_true):
        self.output=self.activation.forward(inputs)
        self.loss.calculate(self.output, y_true)
        return self.output

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
