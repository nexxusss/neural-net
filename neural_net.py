#import dependencies
import numpy as np
import nnfs
#see generate_data.py
from nnfs.datasets import spiral_data, vertical_data
from activation_fc import ReLU, SoftMax
from loss import Loss_CategoricalCrossEntropy

nnfs.init()



# using a batch of inputs
# batches are there to feed the neural network in order to ensure better 
# generalisation of the sample data. In practice a common sample size is between 32 to 64(rarely 128)
#for the sake of simplicity we will only use a batch size of n = 3

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # we want the values as small as possible
        # if you upload a model then you just load the weights and biases of that model
        # but in this case we set the weights randomly so they are as small as possible
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

X, y = vertical_data(100, 3)


# n_inputs is 2 because spiral_data gives data in batches of array of 2(x and y)
dense1 = Layer_Dense(2, 3) # is the num of inputs/features 
#in layer 2 size of inputs should be num of neurons/outputs from layer1
activation1 = ReLU()
dense2 =  Layer_Dense(3, 3) # 3 in the numb of neurons because our data has 3 classes we are trying to predict
activation2 = SoftMax()


#Randomly optimizing the weights and biases
lowest_loss = 999999
best_weight1 = dense1.weights.copy()
best_bias1 = dense1.biases.copy()
best_weight2 = dense2.weights.copy()
best_bias2 = dense2.biases.copy()

for i in range(100000000):

    #Update weights with some small random values 
    dense1.weights += 0.01 * np.random.randn(2, 3) 
    dense1.biases += 0.01 * np.random.randn(1, 3)
    dense2.weights += 0.01 * np.random.randn(3, 3) 
    dense2.biases += 0.01 * np.random.randn(1, 3)

    # passing the data through the neural network
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    #computing the loss
    loss_function = Loss_CategoricalCrossEntropy()
    loss = loss_function.calculate(activation2.output, y)

    #calculate accuracy of prediction from output of activation2 and target
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        lowest_loss = loss
        print("New set of parameters found: Iteration: {}, Loss: {}, acc: {}".format(i, loss, accuracy))

        best_weight1 = dense1.weights.copy()
        best_bias1 = dense1.biases.copy()
        best_weight2 = dense2.weights.copy()
        best_bias2 = dense2.biases.copy()

    else:
        dense1.weights = best_weight1.copy()
        dense2.weights = best_weight2.copy()
        dense1.biases = best_bias1.copy()
        dense2.biases = best_bias2.copy()

