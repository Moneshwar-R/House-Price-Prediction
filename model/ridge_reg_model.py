import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/housing.csv")

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# One-hot encode categorical column
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

model=Ridge(alpha=1.0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse_scores = cross_val_score(
    model,
    X,
    y,
    scoring="neg_mean_squared_error",
    cv=5
)

rmse = np.sqrt(-mse_scores)

print("RMSE per fold:", rmse)
print("Average RMSE:", rmse.mean())

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()