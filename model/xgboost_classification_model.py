import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from xgboost import XGBClassifier, plot_importance

# ----------------------------------
# 1. LOAD DATA
# ----------------------------------

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target   # 0 = malignant, 1 = benign

# ----------------------------------
# 2. TRAIN-TEST SPLIT
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# 3. TRAIN XGBOOST CLASSIFIER
# ----------------------------------

model = XGBClassifier(
    n_estimators=100,        # number of trees
    max_depth=3,             # shallow trees
    learning_rate=0.1,       # controls correction strength
    subsample=0.8,           # row sampling
    colsample_bytree=0.8,    # feature sampling
    eval_metric="logloss",
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

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------
# 6. FEATURE IMPORTANCE
# ----------------------------------

plt.figure(figsize=(10,6))
plot_importance(model, max_num_features=10)
plt.title("Top Feature Importances (XGBoost Classifier)")
plt.show()
