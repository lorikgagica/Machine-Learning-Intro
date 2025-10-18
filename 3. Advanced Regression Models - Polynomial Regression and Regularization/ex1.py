import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

# Load the California Housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Select feature (Median Income) and traget (Median House Value)
X = df[['MedInc']]
y = df[['MedHouseVal']]

# Transform feature to polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Ridge Regression
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)

# Evaluate Ridge Regression
ridge_mse = mean_squared_error(y_test, ridge_predictions)
print("Ridge Regression MSE:", ridge_mse)


# Evaluate Lasso Regression
lasso_mse = mean_squared_error(y_test, lasso_predictions)
print("Lasso Regression MSE:", lasso_mse)

# Visualize Ridge vs Lasso predictions
plt.figure(figsize=(10,6))
plt.scatter(X_test[:, 0], y_test, color="blue", label="Actual Data", alpha=0.5)
plt.scatter(X_test[:, 0], ridge_predictions, color="green", label="Ridge Predictions", alpha=0.5)
plt.scatter(X_test[:, 0], lasso_predictions, color="orange", label="Lasso Predictions", alpha=0.5)
plt.title("Ridge vs Lasso Regression")
plt.xlabel("Median Income (Transformed)")
plt.ylabel("Median House Value in California")
plt.legend()
plt.show()

# # Fit polynomial regression model
# model = LinearRegression()
# model.fit(X_poly, y)

# # Make Predictions
# y_pred = model.predict(X_poly)

# # Plot actual vs predicted values
# plt.figure(figsize=(10,6))
# plt.scatter(X, y, color="blue", label="Actual Data", alpha=0.5)
# plt.scatter(X, y_pred, color="red", label="Predicted Curve", alpha=0.5)
# plt.title("Polynomial Regression")
# plt.xlabel("Median Income in California")
# plt.ylabel("Median House Value in California")
# plt.legend()
# plt.show()

# # Evaluate model performance
# mse = mean_squared_error(y, y_pred)
# print("Mean Squared Error (MSE): ", mse)


# ADDITIONAL PRACTICE/EXPERIMENTS
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
data = fetch_california_housing(as_frame=True)
df = data.frame

# 1. Vary Regularization Parameters
alphas = [0.1, 1, 10]
ridge_results, lasso_results = [], []
X = df[['MedInc']]
y = df['MedHouseVal']
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_results.append((alpha, ridge.coef_, mean_squared_error(y_test, ridge_pred)))
    
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_results.append((alpha, lasso.coef_, mean_squared_error(y_test, lasso_pred)))

print("Ridge Results (alpha, coefficients, MSE):", ridge_results)
print("Lasso Results (alpha, coefficients, MSE):", lasso_results)

# 2. Use Multiple Features
features = ['MedInc', 'HouseAge', 'AveRooms']
X_multi = df[features]
poly_multi = PolynomialFeatures(degree=2, include_bias=False)
X_poly_multi = poly_multi.fit_transform(X_multi)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_poly_multi, y, test_size=0.2, random_state=42)

ridge_m = Ridge(alpha=1)
ridge_m.fit(X_train_m, y_train_m)
ridge_multi_mse = mean_squared_error(y_test_m, ridge_m.predict(X_test_m))

lasso_m = Lasso(alpha=0.1, max_iter=10000)
lasso_m.fit(X_train_m, y_train_m)
lasso_multi_mse = mean_squared_error(y_test_m, lasso_m.predict(X_test_m))

print("Ridge MSE (multiple features):", ridge_multi_mse)
print("Lasso MSE (multiple features):", lasso_multi_mse)

# 3. Feature Importance with Lasso
feature_names = poly_multi.get_feature_names_out(features)
lasso_feat_imp = pd.Series(lasso_m.coef_, index=feature_names)
most_relevant_features = lasso_feat_imp.abs().sort_values(ascending=False).head(10)
print("Top 10 most relevant features (Lasso):")
print(most_relevant_features)