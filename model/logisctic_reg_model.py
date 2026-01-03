import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv("data/breast_cancer.csv")

# --------------------------------------------------
# Clean dataset
# --------------------------------------------------
# Drop ID column
df.drop(columns=["id"], inplace=True)

# Drop fully empty columns
df.dropna(axis=1, how="all", inplace=True)

# Encode target
y = df["diagnosis"].map({"M": 1, "B": 0})
X = df.drop("diagnosis", axis=1)

# --------------------------------------------------
# Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# Pipeline: Impute → Scale → Logistic
# --------------------------------------------------
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=2000))
])

# --------------------------------------------------
# Hyperparameter grid (modern sklearn compatible)
# --------------------------------------------------
param_grid = {
    "logreg__C": [0.01, 0.1, 1, 10, 100]
}

# --------------------------------------------------
# GridSearchCV
# --------------------------------------------------
grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="accuracy",
    cv=5,
)

# --------------------------------------------------
# Train
# --------------------------------------------------
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

# --------------------------------------------------
# Test evaluation
# --------------------------------------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------------------------------
# ROC Curve
# --------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression – Breast Cancer")
plt.legend()
plt.grid(True)
plt.show()
