# 🌟 HealthRiskPredictor

**HealthRiskPredictor** is a machine learning-based tool for predicting the likelihood of health risks like cancer, heart disease, diabetes, and more. By utilizing advanced models like Random Forest, SVM, and Neural Networks, this tool aims to provide accurate predictions, aiding in early diagnosis and healthcare insights.

---

## 📖 Table of Contents
- [✨ Features](#-features)
- [🔧 Installation](#-installation)
- [🚀 Usage](#-usage)
- [🧩 Code Explanation](#-code-explanation)
- [🤝 Contributing](#-contributing)

---

## ✨ Features

- **Interactive Mode**: No need to specify command-line arguments—just run the script and follow prompts to select files, target columns, and more.
- **Error Handling**: Robust checks ensure smooth data loading and preprocessing.
- **Batch Prediction**: Make predictions for multiple records at once and save results to a CSV file.
- **Model Tuning**: Leverages `GridSearchCV` and genetic algorithms for hyperparameter and feature optimization.
- **Visualization**: ROC curve plotting to evaluate model performance.
- **Feature Importance**: Analyze key features influencing predictions, particularly for Random Forest models.
  
---

## 🔧 Installation

### Prerequisites
Ensure Python is installed. A virtual environment is recommended for package management. Install the required packages with:

```bash
pip install pandas joblib matplotlib scikit-learn tabulate deap
```

### Clone the Repository
```bash
git clone https://github.com/KyleBrian/HealthForecasting-with-Machine-learning.git
cd HealthForecasting-with-Machine-learning
```

---

## 🚀 Usage

### Running the Program
To launch the program, simply use the command:

```bash
python HealthModel.py
```

The program will then prompt you to enter:

1. **Path to the Dataset**: Enter the path to your health data CSV file (e.g., `health_data.csv`).
2. **Target Column**: Select the column name to predict, such as `HeartDisease` or `Diabetes`.

Afterward, the program will proceed to train multiple machine learning models, evaluate them, and display a comparison of each model’s performance.

### Example Usage

Let’s go through an example step-by-step:

1. **Run the Script**:
   ```bash
   python HealthModel.py
   ```

2. **File Selection Prompt**:
   - The program will prompt:  
     ```
     Enter the path to the CSV file containing health data (e.g., 'health_data.csv'):
     ```
   - Example input:  
     ```
     health_data.csv
     ```

3. **Target Column Selection Prompt**:
   - The program will display the columns in the dataset and prompt:  
     ```
     Columns available for prediction: ['Age', 'BloodPressure', 'HeartDisease']
     Enter the name of the target column (the column to predict):
     ```
   - Example input:  
     ```
     HeartDisease
     ```

After these inputs, the program will proceed to load data, preprocess it, and train and evaluate each model, displaying a summary of model performances (accuracy, precision, recall, F1-score) in the terminal.

---

## 🧩 Code Explanation

### 1. **File Selection**
   The `select_file()` function allows users to select a CSV or model file interactively, ensuring only valid files are processed.

   ```python
   def select_file():
       # Code to select file
   ```

### 2. **Data Loading and Preprocessing**
   - `load_data()`: Loads the CSV file.
   - `preprocess_data()`: Handles missing values, scales features, and splits data for training/testing.

   ```python
   def load_data(file_path: str):
       # Load data from CSV file
   ```

### 3. **Model Training**
   - `cross_validate_models()`: Performs hyperparameter tuning using GridSearchCV.
   - `genetic_algorithm_optimization()`: Evolves model configurations and feature subsets using genetic algorithms.

   ```python
   def cross_validate_models(X_train, y_train):
       # Hyperparameter tuning code
   ```

### 4. **Model Evaluation**
   The `evaluate_model()` function assesses performance with metrics like accuracy, confusion matrix, and classification report.

   ```python
   def evaluate_model(model, X_test, y_test):
       # Evaluate model performance
   ```

### 5. **Batch Prediction**
   `batch_predict()` enables predictions on new data in bulk and saves the results to a CSV file.

   ```python
   def batch_predict(model, csv_file):
       # Predict outcomes for a batch of data
   ```

---

## 🤝 Contributing

We welcome contributions! If you have ideas or improvements, please fork the repository and submit a pull request. Together, let's make healthcare predictions more accessible and powerful! 💡
