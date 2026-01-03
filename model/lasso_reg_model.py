import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv("data/housing.csv")

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# One-hot encode categorical column
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Split features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# --------------------------------------------------
# Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Pipeline: Scaling + Lasso
# --------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=10000))
])

# --------------------------------------------------
# Hyperparameter grid
# (Lasso needs small alphas)
# --------------------------------------------------
param_grid = {
    "lasso__alpha": [0.0001, 0.001, 0.01, 0.1, 1]
}

# --------------------------------------------------
# GridSearchCV
# --------------------------------------------------
grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,
)

# --------------------------------------------------
# Fit model
# --------------------------------------------------
grid.fit(X_train, y_train)

# --------------------------------------------------
# Best results
# --------------------------------------------------
print("Best alpha:", grid.best_params_["lasso__alpha"])
print("Best CV RMSE:", -grid.best_score_)

# --------------------------------------------------
# Evaluate on test data
# --------------------------------------------------
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", test_rmse)

# --------------------------------------------------
# Plot Actual vs Predicted
# --------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Lasso Regression (Tuned) – Actual vs Predicted")
plt.grid(True)
plt.show()

# --------------------------------------------------
# Feature selection insight (VERY IMPORTANT)
# --------------------------------------------------
lasso_model = best_model.named_steps["lasso"]
coef = lasso_model.coef_

feature_names = X.columns
selected_features = feature_names[coef != 0]

print("\nNumber of features before Lasso:", len(feature_names))
print("Number of features after Lasso:", len(selected_features))

print("\nSelected features:")
for f in selected_features:
    print(f)
