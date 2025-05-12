# Loan Approval Classification Using Neural Networks

## Overview
This project builds and evaluates deep learning models to predict loan approval decisions using applicant demographic, financial, and loan-related data. Multiple neural network architectures were tested and optimized, and the best-performing model was explained using SHAP for transparency and fairness.

## Dataset
- Source: Kaggle Loan Approval Dataset
- Records: Approximately 45,000
- Features: Demographic, financial, and loan application variables
- Target variable: `loan_status` (1 = Approved, 0 = Rejected)

Note: Please update the dataset file path in the Python script to match your local environment before execution.

## Environment Setup

### Install Required Libraries
Run the following command to install all necessary Python packages:

```bash
pip install pandas numpy scikit-learn tensorflow keras keras-tuner shap matplotlib seaborn
````

## Running the Project

Execute the Python script (e.g., `loan_approval_model.py`) using the command:

```bash
python loan_approval_model.py
```

### The script will:

* Load and preprocess the dataset (handle outliers, apply log transformation, encode categorical features, scale numeric variables)
* Split the data into training, validation, and test sets
* Train multiple neural network models with regularization and early stopping
* Tune hyperparameters using Keras Tuner (Hyperband strategy)
* Evaluate the best model using accuracy, AUC, F1 score, and confusion matrix
* Generate performance plots and SHAP-based explainability results

## Outputs

* Training vs Validation Accuracy and Loss Plots
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* SHAP Summary Plot (Feature importance and model interpretability)

## License

This project is for academic and educational purposes only.
