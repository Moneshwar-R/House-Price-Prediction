from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------
# 1. LOAD DATA (CSV)
# ----------------------------------
df = pd.read_csv("data/breast_cancer.csv")
print(df.columns)

# ----------------------------------
# 2. SPLIT FEATURES & TARGET
# ----------------------------------
X = df.drop("diagnosis", axis=1)   # all feature columns
y = df["diagnosis"]                # 0 = malignant, 1 = benign

# ----------------------------------
# 3. TRAIN-TEST SPLIT
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# 4. TRAIN DECISION TREE
# ----------------------------------
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------
# 5. PREDICTION
# ----------------------------------
y_pred = model.predict(X_test)

# ----------------------------------
# 6. EVALUATION
# ----------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------
# 7. VISUALIZE TREE
# ----------------------------------
plt.figure(figsize=(18, 8))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Malignant", "Benign"],
    filled=True
)
plt.show()
