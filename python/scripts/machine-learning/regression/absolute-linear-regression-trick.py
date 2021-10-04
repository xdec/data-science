# Created By: Xavier De Carvalho
# Created On: 04/10/2021
# Updated By:
# Updated On:
# Version: ALRT0.0.01
# Reference: https://www.manning.com/books/grokking-machine-learning

# Coding The Absolute Trick
def absolute_trick(bias, weight, X, Y, learning_rate):
    '''
    This is an algorithm that helps fit the regression line to the data. It
    translates and rotates the line using the following logic:

    IF PRICE > PREDICTION
    {
        # Translate Line Up
        Y_INTERCEPT += ETA
        # Rotate Line
        SLOPE += ETA * X_AXIS_VALUE
    }
    ELSE IF PRICE < PREDICTION
    {
        # Translate Line Down
        Y_INTERCEPT -= ETA
        # Rotate Line
        SLOPE -= ETA * X_AXIS_VALUE
    }
    RETURN line using `Prediction = Slope * X_Axis_Value + Bias`
    '''
    # y=mx+b
    prediction = bias + weight * X
    if Test > prediction:
        weight += learning_rate * X     # Rotate Line Anti-Clockwise
        bias += learning_rate           # Translate Line Up
    else:
        weight -= learning_rate * X     # Rotate Line Clockwise
        bias -= learning_rate           # Translate Line Down
    return weight, bias