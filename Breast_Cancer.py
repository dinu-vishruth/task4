

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)
from sklearn.impute import SimpleImputer

# 1) Load dataset
# Change filename if your CSV is named differently (e.g., "data.csv")
df = pd.read_csv("breast_cancer_wisconsin.csv")

# 2) Clean dataset
if 'id' in df.columns:
    df = df.drop(columns=['id'])   # drop ID column if present

# Encode target: M = malignant (1), B = benign (0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Check missing values
print("Missing values per column:\n", X.isna().sum())

# 3) Handle missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# 4) Train-Test split + Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 5) Train Logistic Regression
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_s, y_train)

# 6) Predictions
y_pred = clf.predict(X_test_s)
y_proba = clf.predict_proba(X_test_s)[:, 1]

print("\n--- Default Threshold (0.5) ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# 7) Threshold Tuning (Max F1)
thresholds = np.linspace(0, 1, 101)
f1s = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(f1s)]
print("\nBest Threshold (Max F1):", best_t)

y_pred_best = (y_proba >= best_t).astype(int)
print("Confusion Matrix (best F1):\n", confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# 8) ROC Curve
fpr, tpr, thr = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label="ROC curve (AUC = %.3f)" % roc_auc_score(y_test, y_proba))
plt.plot([0, 1], [0, 1], '--', color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 9) Precision-Recall Curve
prec, rec, thr = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# 10) Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 200)
plt.figure()
plt.plot(z, sigmoid(z))
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("σ(z)")
plt.grid(True)
plt.show()

# 11) Save Model + Scaler
joblib.dump(clf, "logistic_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("\n✅ Model and scaler saved successfully!")
