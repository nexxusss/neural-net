import numpy as np
import math


#rectified linear function
class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Input -> Exponentiate -> Normalize -> Output
# Gives a robability distribution for outputs 
# Exponentiation + Normalization = Softmax activation function 
#to protect the code from an overflow error due to the nature of ex(values can get big easily)
# we will subtract the values of output by the largest value in that output
# v = u - max(u)
class SoftMax():
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # sum of each row in the batch in the same orientation as exp values
        self.output = probabilities