import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
# https://www.kaggle.com/datasets/prokshitha/home-value-insights
df = pd.read_csv("house_price_regression_dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

# Data preprocessing
print(df.isnull().sum())
# No missing values in this dataset

plt.figure(figsize=(10, 6))
sns.histplot(df["House_Price"], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
# plt.show()
sns.boxplot(x=df["House_Price"])
plt.title("Boxplot of House Prices")
plt.xlabel("Price")
# plt.show()
sns.pairplot(df, diag_kind="kde", height=2.5)
# plt.show()

X = df.drop("House_Price", axis=1)
y = df["House_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning

from sklearn.model_selection import GridSearchCV

param_grid = {
    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    "splitter": ["best", "random"],
    "max_depth": [None, 5, 8, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10, 15, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": [None, "sqrt", "log2"],
}
dtr = DecisionTreeRegressor(random_state=7)
grid_search = GridSearchCV(
    estimator=dtr, param_grid=param_grid, cv=5, verbose=3, n_jobs=-1
)
import warnings

warnings.filterwarnings("ignore")
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)



# criterion='squared_error', max_depth=10, max_features=None, min_samples_leaf=2, min_samples_split=2, splitter='best'

# Train model with best hyperparameters
dtr_best = DecisionTreeRegressor(
    criterion='squared_error', max_depth=10, max_features=None,
    min_samples_leaf=2, min_samples_split=2, splitter='best', random_state=7
)
dtr_best.fit(X_train, y_train)
y_pred = dtr_best.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
# %98

# Lazy prediction
from lazypredict.Supervised import LazyRegressor
lazy_reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = lazy_reg.fit(X_train, X_test, y_train, y_test)
print(models)

# Linear Regression for comparison
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression Mean Squared Error: {mse_lr}")
print(f"Linear Regression R^2 Score: {r2_lr}")
# %100
