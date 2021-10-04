# Created By: Xavier De Carvalho
# Created On: 04/10/2021
# Updated By:
# Updated On:
# Version: SPMIONN0.0.01
# Reference: https://www.manning.com/books/grokking-deep-learning

# Weighted Sum
def w_sum(a,b):
    '''
    This is what NumPy `dot` function is doing under the hood.
    '''
    assert(len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output

# Vector Matrix Multiplication
def vect_mat_mul(vector, matrix):
    '''
    Take a vector and perform a
    dot product with every row
    in a matrix.
    '''
    assert(len(vect) == len(matrix))
    output = [0,0,0]
    for i in range(len(vector)):
        output [i] = w_sum(vector, matrix[i])
    return output

# Prediction Function
def neural_network(inputs, weights):
    hidden_layer = vect_mat_mul(inputs, weights[0])         # Use default input and Matrix[0]
    prediction = vect_mat_mul(hidden_layer, weights[1])     # Input is output of hidden_layer and Matrix[1]
    return prediction

# Example Weights
          #Col1 Col2 Col3
ih_wgt = [[0.1, 0.2, -0.1], # Hidden_Layer 1 hid[0]
          [-0.1, 0.1, 0.9], # Hidden_Layer 2 hid[1]
          [0.1, 0.4, 0.1]]  # Hidden_Layer 3 hid[2]

          #hid[0] hid[1] hid[2]
hp_wgt = [[0.3, 1.1, -0.3], # prediction_variable_1
          [0.1, 0.2, 0.0],  # prediction_variable_2
          [0.0, 1.3, 0.1]]  # prediction_variable_3

# Weights are two stored matrices
w = [ih_wgt, hp_wgt]

# Example Inputs
a = [8.5, 9.5, 9.9, 9.0]
b = [0.65, 0.8, 0.8, 0.9]
c = [1.2, 1.3, 0.5, 1.0]

i = [a[0], b[0], c[0]]

# Predictions
pred = neural_network(i, w)
print(pred)

# EXAMPLE OUTPUT
# [0.21350000000000002, 0.14500000000000002, 0.5065]