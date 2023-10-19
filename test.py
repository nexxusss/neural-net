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


################# Backward pass ###################
# see reference in book for the function above
# and for explanation of the chain rule


#derivative value from the next form the next layer
dvalue = 1.0

#derivative of ReLU 
relu_dz = (1 if z > 0 else 0)

#derivative of ReLU and the chain rule 
drelu_dz = dvalue * relu_dz


print(drelu_dz)

#Partial derivatives of the multiplication, the chain rule

# dsum_dxw0: the partial derivative of the sum with respect to the 
# x(input), weighted, for the 0th pair of inputs and weights
# the value of dsum_dxw0 is multiplied, using chain rule, with the
# with the derivative subsequent function, which is the ReLU function
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1

drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db

print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# Partial derivative of the multiplication(weight[i]*input[i]), and chain rule
# partiasl derivative of f(x, y) = x * y, df(x, y)/dx = y and df(x, y)/dy = x

dmul_dx0 = weights[0]
dmul_dx1 = weights[1]
dmul_dx2 = weights[2]

dmul_dw0 = inputs[0]
dmul_dw1 = inputs[1]
dmul_dw2 = inputs[2]


drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2

print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)
