import numpy as np

np.random.seed(0)
DIMENSION = 2

# https://cs231n.github.io/neural-networks-case-study/
def create_data(points, classes):
    X = np.zeros((points*classes,DIMENSION)) # data matrix (each row = single example)
    y = np.zeros(points*classes, dtype='uint8') # class labels
    for j in range(classes):
        ix = range(points*j,points*(j+1))
        r = np.linspace(0.0,1,points) # radius
        t = np.linspace(j*4,(j+1)*4,points) + np.random.randn(points)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = j

    return X, y
# lets visualize the data:
import matplotlib.pyplot as plt

X, y = create_data(100, 3)

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()
    




# print('here')
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()