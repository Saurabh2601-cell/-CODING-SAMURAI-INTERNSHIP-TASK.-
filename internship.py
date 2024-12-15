#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Sample data (Advertising spend in thousands, Sales in dollars)
X = np.array([1000, 2000, 3000, 4000, 5000]).reshape(-1, 1)  # Independent variable
Y = np.array([15000, 25000, 35000, 45000, 55000])  # Dependent variable

# Create a linear regression model
model = LinearRegression()

# Train the model using all the data
model.fit(X, Y)

# Make predictions using the same data
Y_pred = model.predict(X)

# Calculate Mean Squared Error (MSE) and R-squared
mse = metrics.mean_squared_error(Y, Y_pred)
r2 = metrics.r2_score(Y, Y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the training data and the regression line
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression line')

# Labels and title
plt.xlabel('Advertising Spend ($)')
plt.ylabel('Sales ($)')
plt.title('Linear Regression: Sales vs Advertising Spend')

plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




