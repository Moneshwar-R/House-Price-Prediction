import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor, plot_importance

# ----------------------------------
# 1. LOAD DATA
# ----------------------------------

df = pd.read_csv("data/housing.csv")

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# One-hot encode categorical column
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# ----------------------------------
# 2. TRAIN-TEST SPLIT
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# 3. TRAIN XGBOOST REGRESSOR
# ----------------------------------

model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------
# 4. PREDICTION
# ----------------------------------

y_pred = model.predict(X_test)

# ----------------------------------
# 5. EVALUATION
# ----------------------------------

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# ----------------------------------
# 6. ACTUAL vs PREDICTED PLOT
# ----------------------------------

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("XGBoost Regression: Actual vs Predicted")
plt.grid(True)
plt.show()

# ----------------------------------
# 7. FEATURE IMPORTANCE
# ----------------------------------

plt.figure(figsize=(10,6))
plot_importance(model, max_num_features=10)
plt.title("Top Feature Importances (XGBoost Regressor)")
plt.show()
