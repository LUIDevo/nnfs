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
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.v_weights = np.zeros_like(self.weights)
        self.v_biases = np.zeros_like(self.biases)
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
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
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

X, y = vertical_data(samples=100, classes=3) # X takes coord data (a,b) and y becomes class
dense1 = Layer_Dense(2, 3) # creates a layer of 2 inputs and 3 neurons
activation1 = Activation_ReLU() # forward applies relu
dense2 = Layer_Dense(3, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(lr=0.01, momentum=0.9)

for iteration in range(10001):
    optimizer.pre_update()
    # forward
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

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
