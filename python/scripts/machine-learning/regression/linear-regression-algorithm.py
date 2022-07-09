# Created By: Xavier De Carvalho
# Created On: 09/07/2022
# Updated By:
# Updated On:
# Version: LRA0.0.01
# Reference: https://www.manning.com/books/grokking-machine-learning

# Packages
import random

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
    if Y > prediction:
        weight += learning_rate * X     # Rotate Line Anti-Clockwise
        bias += learning_rate           # Translate Line Up
    else:
        weight -= learning_rate * X     # Rotate Line Clockwise
        bias -= learning_rate           # Translate Line Down
    return weight, bias

# Root Mean Squared Error (RMSE)
def rmse(labels, predictions):
    n = len(labels)                                                 # Length of labels OR x
    differences = np.subtract(labels, predictions)                  # Difference between labels and predictions
    RMSE = np.sqrt(1.0/n * (np.dot(differences, differences)))      # The Root Mean Squared Error
    return RMSE

# Repeat the absolute or square trick many times to move the line closer to each point
def linear_regression(features, labels, learning_rate=0.01, epochs=1000):
    '''
    Inputs:
        A dataset with 2 dimensions (features,labels)
    Outputs:
        Model weights: weights AS (labels/features), bias
    Process:
        1. Start with a random value for the slope and bias
        2. Repeat many times:
            a. Pick a random data point
            b. Update the slope and bias using the absolute or square trick
    '''
    weights = random.random()
    bias = random.random()

    # Create empty lists to append our labels and predictions to
    l_list, y_pred_list = [], []

    # Train the model
    for epoch in range(epochs):
        i = random.randint(0, len(features)-1)
        x = features[i] # Features
        y = labels[i] # Labels

        # Get predictions using the absolute trick
        X_PRED, Y_PRED = absolute_trick(
            y,
            x,
            bias,
            weights,
            learning_rate=learning_rate
        )

        # Append results to the lists we created earlier
        l_list.append(y)
        y_pred_list.append(Y_PRED)

    return round(X_PRED), round(Y_PRED), rmse(l_list, y_pred_list)

# Test Dataset
import numpy as np
features = np.array([1,2,3,5,6,7])
labels = np.array([155,197,244,356,407,448])

print(linear_regression(features, labels, learning_rate=0.001, epochs = 10000))

# Example output
# (2, 197, 0.00099)