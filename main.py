import numpy as np
import nnfs
from nnfs.datasets import vertical_data
nnfs.init()

class Optimizer_SGD:
    def __init__(self, lr=0.01, momentum=0.9, max_lr=0.1, warmup_steps=1000, warmdown_steps=9000, lr_growth=0.001, lr_decay=0.01):
        self.lr = lr
        self.current_lr = lr
        self.momentum = momentum
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.warmdown_steps = warmdown_steps
        self.lr_growth = lr_growth
        self.lr_decay = lr_decay
        self.iterations = 0
    def pre_update(self):
        if self.iterations < self.warmup_steps:
            self.current_lr = min(self.current_lr + self.lr_growth * self.current_lr, self.max_lr)
        elif self.iterations > self.warmdown_steps:
            self.current_lr = max(self.current_lr - self.lr_decay * self.current_lr, 0)
    def update_params(self, layer):
        layer.v_weights = self.momentum * layer.v_weights - self.current_lr * layer.dweights
        layer.v_biases  = self.momentum * layer.v_biases  - self.current_lr * layer.dbiases
        layer.weights += layer.v_weights
        layer.biases  += layer.v_biases
    def post_update(self):
        self.iterations += 1

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # Creates a shape of n_inputs x n_neurons of numbers of the normal distribution, can be negative
        self.biases = np.zeros((1, n_neurons)) # creates an array of len n_neurons of 0's. Why not random nums as well?
        self.v_weights = np.zeros_like(self.weights) # 0's for velocity (because initially they dont change)
        self.v_biases = np.zeros_like(self.biases) # same thing
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    # class contains inputs, outputs, dinputs
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities # turns the inputs into rows summing to 1, 100 times.

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred) # prediction length
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # we clip values to an extremely small value or close to 1 (presumably to remove 0's) to prevent 0 becoming infinity in the log step
        if len(y_true.shape) == 1: # why do we differentaite between shapes? When would it be 2, and why do we only have 2 if statements? Are there other shapes that can happen?
            correct_confidences = y_pred_clipped[range(samples), y_true] # we set correct_confidences to items of y_pre_clipped to a range? between the size of samples to not overflow and y_true? Really not sure what that means
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1) # we multiple y_pred_clipped and y_true, then sum them? Im not sure what this accomplishes
        negative_log_likelihoods = -np.log(correct_confidences) # then this becomes the loss. We do this to make sure a high confidence is a small loss, and a low confidence becomes a very high loss
        return negative_log_likelihoods

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true) # goes backward to Loss.calculate
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

X, y = vertical_data(samples=100, classes=3) # X takes coord data (a,b) and y becomes class
dense1 = Layer_Dense(2, 3) # creates a layer of 2 inputs and 3 neurons
activation1 = Activation_ReLU() # forward applies relu, backward also applies relu?
dense2 = Layer_Dense(3, 3) # creates a layer of 3 inputs and 3 neurons
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy() # sets up activation function, and loss function
optimizer = Optimizer_SGD(lr=0.01, momentum=0.9) # sets up optimizer | what is role of optimimizer? Is it an abstraction of some sort?

for iteration in range(10001):
    optimizer.pre_update() # changes learning rate dependant on iteration count (warmup/warmdown)
    # forward
    dense1.forward(X) # multiplying inputs and weights and adding biases to self.output, the entirety of X everytime
    activation1.forward(dense1.output) # applies relu
    dense2.forward(activation1.output) # applies forward activation to dense 2
    loss = loss_activation.forward(dense2.output, y) # Softmax categorical loss -> Activation_softmax forward ->

    predictions = np.argmax(loss_activation.output, axis=1) # collapses 300x3 array matrix into vector of class predictions
    accuracy = np.mean(predictions == y) # finds the average of correct vs incorrect, based on how many prediction classes equal the real class

    if not iteration % 1000:
        print(f'iteration: {iteration}, loss: {loss:.4f}, acc: {accuracy:.3f}')

    # backward
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # gradient descent update
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update()
