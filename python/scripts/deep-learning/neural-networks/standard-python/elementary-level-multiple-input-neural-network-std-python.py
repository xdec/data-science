# Created By: Xavier De Carvalho
# Created On: 04/10/2021
# Updated By:
# Updated On:
# Version: SPMINN0.0.01
# Reference: https://www.manning.com/books/grokking-deep-learning

# Weighted Sum
def w_sum(a,b):
    '''
    This is what NumPy `dot` function is doing under the hood.
    '''
    assert(len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output+= (a[i] * b[i])
    return output

# Neural Network Function
def neural_network(inputs, weights):
    '''
    This function takes multiple inputs and gives a single output.
    '''
    pred = w_sum(inputs, weights)
    return pred

# Example Input Vectors
a = [8.5, 9.5, 9.9, 9.0]
b = [0.65, 0.8, 0.8, 0.9]
c = [1.2, 1.3, 0.5, 1.0]

# Example Weights
w = [0.1, 0.2, 0]

# Example Inputs
i = [a[0], b[0], c[0]]

# Prediction
pred = neural_network(i, w)
print(pred)

# EXAMPLE OUTPUT
# 0.9800000000000001