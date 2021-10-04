# Created By: Xavier De Carvalho
# Created On: 04/10/2021
# Updated By:
# Updated On:
# Version: EMLT0.0.01
# Reference: https://www.manning.com/books/grokking-deep-learning

def elementWise_multiplication(vec_a, vec_b):
    assert(len(vec_a) == len(vec_b))         # Both vectors must be of the same size
    output = 0
    for i in range(len(vec_a)):
        output += (vec_a[i] * vec_b[i])
    return output

# EXAMPLE OUTPUT
#
# a = [1,2,3]
# b = [4,5,6]
#
# elementWise_mutiplication(a,b)
#
# 32