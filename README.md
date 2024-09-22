# Project Summary:

## Aim:
To predict the housing prices using Linear Regression

## 1. Dataset Creation:
We created a sample dataset related to housing prices. The dataset includes features such as the number of rooms (Rooms), area in square feet (Area), and age of the house in years (Age). The target variable is the price of the house (Price).
```
import pandas as pd
```

### Sample dataset
```
data = {
    'Rooms': [3, 2, 4, 3, 5, 4, 3, 2, 4, 5],
    'Area': [1500, 800, 2000, 1200, 2500, 1800, 1300, 900, 2100, 2700],
    'Age': [10, 15, 8, 12, 5, 7, 14, 20, 6, 4],
    'Price': [300000, 200000, 400000, 250000, 500000, 450000, 270000, 180000, 420000, 520000]
}
```
### Creating DataFrame
```
df = pd.DataFrame(data)
```

### Display the DataFrame
```
print(df)
```
## 2. Data Splitting:
We split the dataset into features (X) and target (y), and then further split it into training and testing sets using train_test_split from sklearn.model_selection.

from sklearn.model_selection import train_test_split

### Splitting the data into features (X) and target (y)
```
X = df[['Rooms', 'Area', 'Age']]
y = df['Price']
```
### Splitting the data into training and testing sets
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## 3. Model Training:

We created and trained a linear regression model using the training data.


from sklearn.linear_model import LinearRegression

### Creating and training the model
```
model = LinearRegression()
model.fit(X_train, y_train)
```
## 4. Making Predictions:
We made predictions on the test data using the trained model.

### Making predictions
```
y_pred = model.predict(X_test)
```
## 5. Model Evaluation:
We evaluated the model's performance using Mean Squared Error (MSE) and R-squared (R²) metrics.

from sklearn.metrics import mean_squared_error, r2_score

### Evaluating the model

```
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
```
## 6. Visualizing Results:
We created a scatter plot to visualize the actual vs. predicted values.

import matplotlib.pyplot as plt

### Scatter plot of actual vs. predicted values

```
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()
```
## 7.Output:
![image](https://github.com/user-attachments/assets/614eae16-fac4-484f-8528-a863a3402097)

```
Actual values (y_test):
8    420000
1    200000
Name: Price, dtype: int64
Predicted values (y_pred):
[418037.02695344 182210.72692622]
```

## 8. Result:
Mean Squared Error (MSE): Measures the average of the squares of the errors. Lower values indicate better performance.
R-squared (R²): Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. Values closer to 1 indicate better performance.
Review the MSE and R² values to understand the model's performance.


