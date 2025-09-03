# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(len(X1)), X1]

    # Initialize theta with zeros
    theta = np.zeros((X.shape[1],)).reshape(-1,1)

    # Perform gradient descent
    for _ in range(num_iters):
        # Calculate predictions
        predictions = (X).dot(theta).reshape(-1, 1)

        # Calculate errors
        errors = (predictions - y).reshape(-1,1)

        # Update theta using gradient descent
        theta = theta - learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta

print('Name : Junjar U')
print('Register No.: 212224230110 ')
data = pd.read_csv('/50_Startups.csv', header=None)
print(data.head())
print()
# Assuming the last column is your target variable 'y' and the preceding columns are your features 'X'
X = (data.iloc[1:, :-2].values)
print(X)
X1 = X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:, -1].values).reshape(-1,1)
print()
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)

print(X1_Scaled)
print()
print(Y1_Scaled)

# Learn model parameters
theta = linear_regression(X1_Scaled, Y1_Scaled)

# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_data = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_data), theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print()
print(f"Predicted Value: {pre}")
```

## Output:
<img width="547" height="157" alt="image" src="https://github.com/user-attachments/assets/fdbf7ff9-39c2-48c5-8b69-1976e49f0800" />

<img width="339" height="877" alt="image" src="https://github.com/user-attachments/assets/52271883-b2b6-4710-8611-71294e107bd3" />

<img width="429" height="877" alt="image" src="https://github.com/user-attachments/assets/06bddd4c-ca2e-4c0e-aaca-58355ce295b7" />

<img width="145" height="888" alt="image" src="https://github.com/user-attachments/assets/83444f6d-e235-42e7-b60e-7fadc71068bf" />

<img width="298" height="32" alt="image" src="https://github.com/user-attachments/assets/c587a8e1-81be-4678-ab9c-b525c0caf3b4" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
