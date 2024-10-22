

# HealthRiskPredictor


**HealthRiskPredictor** is a machine learning-based tool designed to predict the likelihood of cancer and heart disease. Utilizing advanced classification models like Random Forest, SVM, and Neural Networks, this tool aims to provide accurate predictions and enhance early diagnosis. 

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Contributing](#contributing)


## Features

- **Error Handling**: Robust mechanisms to ensure smooth data loading and preprocessing.
- **Batch Prediction**: Predict outcomes for multiple records at once and save results to a CSV file.
- **Model Tuning**: Implements GridSearchCV for hyperparameter optimization.
- **Visualization**: ROC curve plotting to evaluate model performance.
- **Feature Importance**: Analyze key features that impact predictions, especially for Random Forest models.

## Installation

### Prerequisites

Make sure you have Python installed. It's recommended to use a virtual environment for package management. You will need the following libraries:

```bash
pip install pandas joblib matplotlib scikit-learn tkinter
```

### Clone the Repository

```bash
git clone https://github.com/KyleBrian/HealthModel.git
cd HealthPredictor
```

## Usage

### Running the Program

To run the program, execute the following command:

```bash
HealthModel.py
```

### Step-by-Step Guide

1. **Select Mode**: Choose whether to train a new model or load an existing one.
2. **File Selection**: 
   - If training a new model, select a CSV file containing patient data.
   - If loading a model, select a previously saved model file.
3. **Batch Predictions**: After model training or loading, you can select another CSV file for batch predictions. Results will be saved to `predictions_output.csv`.

## Code Explanation

### 1. File Selection

The `select_file()` function opens a dialog for users to select a CSV or model file. It handles errors gracefully, ensuring that only valid files are processed.

```python
def select_file(file_type="csv"):
    # Code to select file
```

### 2. Data Loading and Preprocessing

The `load_data()` function loads the CSV file, while `preprocess_data()` prepares the dataset by handling missing values and scaling features.

```python
def load_data(file_path: str):
    # Load data from CSV file
```

### 3. Model Training

`cross_validate_models()` function performs hyperparameter tuning using GridSearchCV to identify the best model parameters.

```python
def cross_validate_models(X_train, y_train):
    # Hyperparameter tuning code
```

### 4. Model Evaluation

The `evaluate_model()` function assesses the model's performance using metrics like accuracy, confusion matrix, and classification report.

```python
def evaluate_model(model, X_test, y_test):
    # Evaluate model performance
```

### 5. Batch Prediction

`batch_predict()` allows the user to run predictions on a new dataset and saves the results.

```python
def batch_predict(model, csv_file):
    # Predict outcomes for a batch of data
```

## Contributing

We welcome contributions! If you have suggestions or improvements, please fork the repository and submit a pull request.



---

