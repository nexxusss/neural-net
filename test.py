import numpy as np
import math

# to explain backpropagation we will use a simple case of backpropagating
# a single neuron and we suppose we are intending to minimize the output of this neurons

#single neuron with three input/weights, and a bias
inputs = [1.0, -2.0, 3.0]
weights = [-3.0, -1.0, 2.0]
bias = 1

#computing the output

xw0 = inputs[0]*weights[0]
xw1 = inputs[1]*weights[1]
xw2 = inputs[2]*weights[2]

print(xw0, xw1, xw2, bias)

z = xw0 + xw1 + xw2 + bias
print(z)

#ReLU activation
y = max(z, 0)

print("Neuron output: ", y)

