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

- **Interactive Mode**: Start the program, and interactively select files for training and batch predictions using file dialogs.
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

Make sure `tkinter` is installed as well, as it is used for file selection dialogs. It typically comes pre-installed with Python on many systems. On Ubuntu, you can install it with:

```bash
sudo apt-get install python3-tk
```

### Clone the Repository
```bash
git clone https://github.com/KyleBrian/HealthForecasting-with-Machine-learning.git
cd HealthForecasting-with-Machine-learning
```

---

## 🚀 Usage

To launch the program, simply use the command:

```bash
python HealthModel.py
```

The program will prompt you to:

1. **Choose Mode**: Select whether to train a new model or load an existing one for batch predictions.
2. **File Selection**:
   - For training a model, select a CSV file containing health data.
   - For loading a model, select a saved model file (`.pkl`).
3. **Batch Predictions**: After loading a model, you can select another CSV file for batch predictions. Results are saved to `predictions_output.csv`.

### Example Workflow

Let’s go through an example step-by-step:

1. **Run the Script**:
   ```bash
   python HealthModel.py
   ```

2. **Choose Mode**:
   - You will be prompted:  
     ```
     Would you like to (1) Train a new model or (2) Load an existing model? Enter 1 or 2:
     ```
   - Enter `1` to train a new model, or `2` to load a saved model for batch predictions.

3. **File Selection with Dialogs**:
   - When prompted, a file dialog will open:
     - **For training**: Select the CSV file with patient data.
     - **For batch predictions**: Select the CSV file with batch data to analyze after model loading.

4. **Model Saving with Dialogs**:
   - After training, a dialog will prompt you to save each model with a specified file name and location.

---

## 🧩 Code Explanation

### 1. **File Selection and Model Saving with `tkinter`**
   The `select_file()` function uses a `tkinter` file dialog to allow users to select CSV or model files for input.
   
   ```python
   def select_file(file_type="csv"):
       # Code to open file dialog
   ```

   Similarly, `save_model_dialog()` opens a save dialog to let users specify a file path for each trained model.

### 2. **Data Loading and Preprocessing**
   - `load_data()`: Loads the CSV file.
   - `preprocess_data()`: Handles missing values, scales features, and splits data for training/testing.

   ```python
   def load_data(file_path: str):
       # Load data from CSV file
   ```

### 3. **Model Training**
   - `cross_validate_models()`: Performs hyperparameter tuning using `GridSearchCV`.
   - Saves each model to a file using the `tkinter` save dialog.

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
