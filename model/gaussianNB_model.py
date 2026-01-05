import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Example dataset
data = {
    "Study_Hours": [2, 4, 6, 8, 10, 1, 3, 7],
    "Attendance": [60, 70, 80, 90, 95, 50, 65, 85],
    "Pass": [0, 0, 1, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
