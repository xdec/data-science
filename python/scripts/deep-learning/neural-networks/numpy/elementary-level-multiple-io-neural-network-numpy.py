# Created By: Xavier De Carvalho
# Created On: 04/10/2021
# Updated By:
# Updated On:
# Version: NPMIONN0.0.01
# Reference: https://www.manning.com/books/grokking-deep-learning

# Import Package
import numpy as np

# Neural Network Function
def neural_network(inputs, weights):
    '''
    This function automatically does
    the weighted sum and the vector
    matrix multiplication using the
    `.dot` method.
    '''
    hidden_layer = inputs.dot(weights[0])
    prediction = hidden_layer.dot(weights[1])
    return prediction

# Example Weights
#Col1 Col2 Col3
ih_wgt = np.array([
                   [0.1, 0.2, -0.1],  # Hidden_Layer 1 hid[0]
                   [-0.1, 0.1, 0.9 ], # Hidden_Layer 2 hid[1]
                   [0.1, 0.4, 0.1]    # Hidden_Layer 3 hid[2]
                  ]).T

# hid[0] hid[1] hid[2]
hp_wgt = np.array([
                   [0.3, 1.1, -0.3],  # prediction_variable_1
                   [0.1, 0.2, 0.0 ],  # prediction_variable_2
                   [0.0, 1.3, 0.1]    # prediction_variable_3
                  ]).T

# Weights are two stored matrices
w = [ih_wgt, hp_wgt]

# Example Inputs
a = np.array([8.5, 9.5, 9.9, 9.0])
b = np.array([0.65, 0.8, 0.8, 0.9])
c = np.array([1.2, 1.3, 0.5, 1.0])

i = np.array([a[0], b[0], c[0]])

# Predictions
pred = neural_network(i, w)
print(pred)

# EXAMPLE OUTPUT
# [0.2135 0.145  0.5065]