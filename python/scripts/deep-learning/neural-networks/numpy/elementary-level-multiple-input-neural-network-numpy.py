# Created By: Xavier De Carvalho
# Created On: 04/10/2021
# Updated By:
# Updated On:
# Version: NPMINN0.0.01
# Reference: https://www.manning.com/books/grokking-deep-learning

# Import Package
import numpy as np

# Neural Network
def neural_network(inputs, weights):
    pred = inputs.dot(weights)      # Weighted Sum
    return pred

# Weights
w = np.array([0.1, 0.2, 0])

# Example Input Vectors
a = np.array([8.5, 9.5, 9.9, 9.0])
b = np.array([0.65, 0.8, 0.8, 0.9])
c = np.array([1.2, 1.3, 0.5, 1.0])

# Example Inputs
i = np.array([a[0], b[0], c[0]])

# Prediction
pred = neural_network(i, w)
print(pred)

# EXAMPLE OUTPUT
# 0.9800000000000001