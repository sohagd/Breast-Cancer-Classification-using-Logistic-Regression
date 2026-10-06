import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

df = pd.read_csv("data.csv")
print("Dataset Shape:", df.shape)
df = df.drop(
    columns=["id", "Unnamed: 32"],
    errors="ignore"
)
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})
print("\nMissing Values:")
print(df.isnull().sum().sum())

X = df.drop(
    columns=["diagnosis"]
)
y = df["diagnosis"]
print("\nNumber of Features:", X.shape[1])

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
model.fit(
    X_train_scaled,
    y_train
)

validation_probability = model.predict_proba(
    X_validation_scaled
)[:, 1]

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)
threshold_results = []
for threshold in thresholds:
    y_validation_pred = (
        validation_probability >= threshold
    ).astype(int)
    precision = precision_score(
        y_validation,
        y_validation_pred,
        zero_division=0
    )
    recall = recall_score(
        y_validation,
        y_validation_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_validation,
        y_validation_pred,
        zero_division=0
    )
    threshold_results.append([
        threshold,
        precision,
        recall,
        f1
    ])
threshold_results = pd.DataFrame(
    threshold_results,
    columns=[
        "Threshold",
        "Precision",
        "Recall",
        "F1-Score"
    ]
)

best_row = threshold_results.loc[
    threshold_results["F1-Score"].idxmax()
]
best_threshold = best_row["Threshold"]
best_validation_precision = best_row["Precision"]
best_validation_recall = best_row["Recall"]
best_validation_f1 = best_row["F1-Score"]

print("THRESHOLD TUNING RESULTS")
print(
    f"Best Threshold : {best_threshold:.2f}"
)
print(
    f"Precision      : {best_validation_precision:.4f}"
)
print(
    f"Recall         : {best_validation_recall:.4f}"
)
print(
    f"F1-Score       : {best_validation_f1:.4f}"
)

plt.figure(figsize=(8, 5))
plt.plot(
    threshold_results["Threshold"],
    threshold_results["Precision"],
    label="Precision"
)
plt.plot(
    threshold_results["Threshold"],
    threshold_results["Recall"],
    label="Recall"
)
plt.plot(
    threshold_results["Threshold"],
    threshold_results["F1-Score"],
    label="F1-Score"
)
plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best Threshold = {best_threshold:.2f}"
)
plt.xlabel("Classification Threshold")
plt.ylabel("Score")

plt.title(
    "Threshold Tuning - Precision, Recall and F1-Score"
)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "threshold_tuning.png",
    dpi=300
)

plt.show()

test_probability = model.predict_proba(
    X_test_scaled
)[:, 1]
y_test_pred = (
    test_probability >= best_threshold
).astype(int)

test_precision = precision_score(
    y_test,
    y_test_pred
)
test_recall = recall_score(
    y_test,
    y_test_pred
)
test_f1 = f1_score(
    y_test,
    y_test_pred
)
test_roc_auc = roc_auc_score(
    y_test,
    test_probability
)

print(
    f"Optimized Threshold : {best_threshold:.2f}"
)
print(
    f"Precision           : {test_precision:.4f}"
)
print(
    f"Recall              : {test_recall:.4f}"
)
print(
    f"F1-Score            : {test_f1:.4f}"
)
print(
    f"ROC-AUC             : {test_roc_auc:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=[
            "Benign",
            "Malignant"
        ]
    )
)

cm = confusion_matrix(
    y_test,
    y_test_pred
)
print("\nConfusion Matrix:")
print(cm)
plt.figure(figsize=(6, 5))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Benign",
        "Malignant"
    ]
)
display.plot()
plt.title(
    "Confusion Matrix"
)
plt.tight_layout()
plt.savefig(
    "confusion_matrix.png",
    dpi=300
)
plt.show()

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    test_probability
)
plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {test_roc_auc:.4f}"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)
plt.xlabel(
    "False Positive Rate"
)
plt.ylabel(
    "True Positive Rate"
)
plt.title(
    "ROC Curve - Logistic Regression"
)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "roc_curve.png",
    dpi=300
)
plt.show()

z = np.linspace(
    -10,
    10,
    200
)
sigmoid = 1 / (
    1 + np.exp(-z)
)
plt.figure(figsize=(7, 5))
plt.plot(
    z,
    sigmoid
)
plt.axhline(
    0.5,
    linestyle="--",
    label="Probability = 0.5"
)
plt.axvline(
    0,
    linestyle="--"
)
plt.xlabel(
    "Input (z)"
)
plt.ylabel(
    "Probability"
)
plt.title(
    "Sigmoid Function"
)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    "sigmoid_curve.png",
    dpi=300
)
plt.show()

default_prediction = (
    test_probability >= 0.50
).astype(int)
default_precision = precision_score(
    y_test,
    default_prediction
)
default_recall = recall_score(
    y_test,
    default_prediction
)
default_f1 = f1_score(
    y_test,
    default_prediction
)
print("\nDefault Threshold (0.50)")
print(
    f"Precision : {default_precision:.4f}"
)
print(
    f"Recall    : {default_recall:.4f}"
)
print(
    f"F1-Score  : {default_f1:.4f}"
)
print(
    f"\nOptimized Threshold ({best_threshold:.2f})"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1-Score  : {test_f1:.4f}"
)

threshold_results.to_csv(
    "threshold_results.csv",
    index=False
)