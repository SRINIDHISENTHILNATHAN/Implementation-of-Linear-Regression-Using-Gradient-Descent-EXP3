# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load and Inspect Data: Import required libraries, load the dataset (50_Startups.csv), and inspect initial rows.

2.Separate Features and Target: Extract X (features) and y (target variable) from the dataset.

3.Scale Data: Use StandardScaler to normalize X and y to ensure uniform feature scaling.

4.Define Gradient Descent Function: Create a linear_regression function to initialize theta, calculate predictions, errors, and update theta iteratively.

5.Train the Model: Apply linear_regression to the scaled data (X1_scaled and Y1_scaled) to find optimal theta.

6.Predict for New Data: Scale new input data, apply the model, and inverse-transform to get predictions in the original scale.

7.Output Results: Display the predicted values for interpretability in the original data units.

## Program:
### Head
```
Program to implement the linear regression using gradient descent.
Developed by: SRINIDHI SENTHIL
RegisterNumber:  212222230148
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv",header=None)
data.head()
```
### X Data
```
X=(data.iloc[1:,:-2].values)
print(X)
```
### Y Data
```
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
```
### X1_scaled
### Y1_scaled
```
X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X1_scaled)
print(Y1_scaled)
theta=linear_regression(X1_scaled,Y1_scaled)
```
### predicted output
```
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"predicted values:{pre}")
```
## Output
### Head
![image](https://github.com/user-attachments/assets/404e59f4-7c17-4659-94e4-6bdecff09640)
### X Data
![image](https://github.com/user-attachments/assets/ee77f167-4800-4575-a353-114868356425)
### Y Data
![image](https://github.com/user-attachments/assets/87c2e744-ccd3-41c9-911a-1842b16f4c93)
### X1_scaled
![image](https://github.com/user-attachments/assets/4e6c3aad-3b6b-4159-b4af-e1763476179f)
### Y1_scaled
![image](https://github.com/user-attachments/assets/310052f1-54dc-4cc6-bbdb-c4ae6cce3f8f)
### predicted output
![image](https://github.com/user-attachments/assets/85810f99-58f0-4120-aa83-39e0e25e258d)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
