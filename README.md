# Breast Cancer Classification using Logistic Regression

## Project Overview

The objective of this project is to understand and implement **Logistic Regression** for a real-world binary classification problem.
The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**, which contains medical measurements of breast cell characteristics. The target variable is **diagnosis**, which indicates whether a tumor is benign or malignant.
The Logistic Regression model was trained to classify tumors into two categories:
1. **Benign (B)** – represented as `0`
2. **Malignant (M)** – represented as `1`

## Objectives

- Import and preprocess the breast cancer dataset.
- Inspect the dataset and handle unnecessary columns.
- Convert the categorical target variable into numerical values.
- Split the dataset into training, validation and testing sets.
- Standardize the numerical features.
- Implement a **Logistic Regression** model.
- Evaluate the model using:
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix
- Plot the ROC curve.
- Visualize the sigmoid function.
- Tune the classification threshold using the validation dataset.
- Compare different classification thresholds.
- Evaluate the final model using the optimized threshold.

## Technologies Used

- **Python**
- **Pandas** – data loading and preprocessing
- **NumPy** – numerical operations
- **Matplotlib** – visualization
- **Scikit-learn** – preprocessing, Logistic Regression and model evaluation

## Dataset

The dataset used is the **Breast Cancer Wisconsin Diagnostic Dataset**.
The dataset contains **569 records and 32 relevant features**, along with the target variable.

### Target Variable

`diagnosis`
The target variable represents the diagnosis of the tumor.

| Value | Meaning | Encoded Value |
|---|---|---:|
| `B` | Benign | 0 |
| `M` | Malignant | 1 |

### Features

The dataset contains measurements related to breast cell nuclei, including:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension
Each measurement is provided as mean, standard error and worst-value measurements.

## Working

- Loaded and explored the Breast Cancer Wisconsin dataset.
- Removed unnecessary columns such as `id` and `Unnamed: 32`.
- Converted the `diagnosis` column from categorical values into numerical values.
- Used **Benign = 0** and **Malignant = 1**.
- Split the dataset into **60% training, 20% validation and 20% testing data**.
- Standardized the numerical features using **StandardScaler**.
- Built a **Logistic Regression** classification model.
- Generated class probabilities using `predict_proba()`.
- Tested classification thresholds from **0.10 to 0.90**.
- Calculated Precision, Recall and F1-Score for each threshold.
- Selected the threshold with the highest **F1-Score** on the validation dataset.
- Used the optimized threshold to generate final predictions on the untouched test dataset.
- Evaluated the final model using Precision, Recall, F1-Score and ROC-AUC.
- Created a confusion matrix to analyze correct and incorrect classifications.
- Created an ROC curve to evaluate classification performance across different thresholds.
- Visualized the sigmoid function used by Logistic Regression.
- Generated a threshold tuning graph showing how Precision, Recall and F1-Score change with the classification threshold.

## Logistic Regression

Logistic Regression is a classification algorithm that estimates the probability of an observation belonging to a particular class.
The model uses the **sigmoid function** to convert its output into a probability between 0 and 1.
The sigmoid function is:

              1
Sigmoid(z) = ─────────
             1 + e⁻ᶻ

A probability close to `1` indicates a higher likelihood of the tumor being classified as malignant, while a probability close to `0` indicates a higher likelihood of being benign.

## Threshold Tuning

The default classification threshold in Logistic Regression is usually **0.50**.
In this project, instead of directly accepting the default threshold, multiple thresholds between **0.10 and 0.90** were evaluated.
For every threshold, the following metrics were calculated:
- Precision
- Recall
- F1-Score
The threshold producing the highest **F1-Score on the validation dataset** was selected as the optimized threshold. This approach provides a better balance between precision and recall than simply assuming that `0.50` is always the best threshold.

## Model Evaluation

The final model was evaluated on the **test dataset**, which was kept separate during threshold tuning.

The following metrics were used:

### Precision

Precision measures how many of the observations predicted as malignant were actually malignant.

### Recall

Recall measures how many of the actual malignant cases were correctly identified by the model.

### F1-Score

F1-Score provides a balance between precision and recall.

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between benign and malignant cases across different classification thresholds.

### Confusion Matrix

The confusion matrix shows:
- True Negatives
- False Positives
- False Negatives
- True Positives

## Outputs

The project generates:
- **Confusion matrix**
- **ROC curve**
- **Sigmoid function plot**
- **Threshold tuning plot**
- **Threshold results CSV**
- **Classification report**
- **Precision, Recall and F1-Score results**
- **ROC-AUC score**
- **Optimized classification threshold**