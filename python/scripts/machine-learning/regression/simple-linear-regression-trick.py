# Created By: Xavier De Carvalho
# Created On: 04/10/2021
# Updated By:
# Updated On:
# Version: SLRT0.0.01
# Reference: https://www.manning.com/books/grokking-machine-learning

# Coding The Square Trick
def square_trick(bias, weight, X, Y, learning_rate):
    '''
    This is an algorithm that helps calculate whether the line needs to translate
    positively or negatively, and also whether it needs to rotate positively or
    negatively.
    '''
    # y = mx + b
    prediction = bias + weight * X
    # eta(Test-prediction)+bias
    bias += learning_rate * (Test - prediction)
    # eta * X(Test-prediction)+weight
    weight += learning_rate * X * (Test-prediction)
    return weight, bias

# EXAMPLE OUTPUT
#
# square_trick(100,30,5,190,0.015)
#
# (25.5, 99.1)