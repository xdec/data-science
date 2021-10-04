# Created By: Xavier De Carvalho
# Created On: 04/10/2021
# Updated By:
# Updated On:
# Version: VAVG0.0.01
# Reference: https://www.manning.com/books/grokking-deep-learning

def vector_average(vec_a):
    assert(len(vec_a) > 0)         # Vector length must be 1 or more
    output = 0
    for i in range(len(vec_a)):
        output += vec_a[i] / len(vec_a)
    return output

# EXAMPLE OUTPUT
#
# a = [1,2,3]
#
# vector_average(a,b)
#
# 2