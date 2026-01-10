import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# ----------------------------------
# 1. LOAD DATA
# ----------------------------------
df = pd.read_csv("data/breast_cancer.csv")

df["diagnosis"] = df["diagnosis"].map({"M": 0, "B": 1})

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ----------------------------------
# 2. HANDLE MISSING VALUES
# ----------------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# ----------------------------------
# 3. TRAIN-TEST SPLIT
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# 4. ADABOOST MODEL
# ----------------------------------
base_estimator = DecisionTreeClassifier(max_depth=1)

model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------
# 5. EVALUATION
# ----------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
