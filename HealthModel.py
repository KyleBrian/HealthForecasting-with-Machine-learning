# File: ml_cancer_heart_predictor_with_error_handling_and_batch_prediction.py

import pandas as pd
import joblib
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import os

# Open a file dialog to select a file (CSV or model)
def select_file(file_type="csv"):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    try:
        if file_type == "csv":
            file_path = filedialog.askopenfilename(
                title="Select the CSV file for cancer/heart disease prediction",
                filetypes=[("CSV files", "*.csv")]
            )
        else:
            file_path = filedialog.askopenfilename(
                title="Select a saved model file",
                filetypes=[("Model files", "*.pkl")]
            )
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError("No file selected or file does not exist.")
        return file_path
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

# Load the dataset with error handling
def load_data(file_path: str):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

# Preprocess the data with error handling
def preprocess_data(df):
    try:
        X = df.iloc[:, :-1]  # Features
        y = df.iloc[:, -1]   # Target (assumed binary classification)
        X.fillna(X.mean(), inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, df.columns[:-1]
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return None, None, None, None, None

# Perform hyperparameter tuning using cross-validation
def cross_validate_models(X_train, y_train):
    try:
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC(probability=True),
            'Neural Network': MLPClassifier(max_iter=1000)
        }
        param_grids = {
            'Logistic Regression': {'C': [0.1, 1, 10], 'penalty': ['l2']},
            'Decision Tree': {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]},
            'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]},
            'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'Neural Network': {'hidden_layer_sizes': [(50,), (100,), (100, 50)], 'activation': ['relu', 'tanh']}
        }
        best_models = {}
        for name, model in models.items():
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_models[name] = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
        return best_models
    except Exception as e:
        print(f"Error during cross-validation: {e}")
        return {}

# Evaluate a model
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)
    except Exception as e:
        print(f"Error during model evaluation: {e}")

# Feature Importance for RandomForest
def feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        sorted_indices = importance.argsort()[::-1]
        print("Feature Importances for Random Forest:")
        for i in sorted_indices:
            print(f"{feature_names[i]}: {importance[i]:.4f}")
    else:
        print("Feature importance is not available for this model.")

# Plot ROC curve for binary classification models
def plot_roc_curve(models, X_test, y_test):
    try:
        plt.figure(figsize=(10, 8))
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
    except Exception as e:
        print(f"Error during ROC curve plotting: {e}")

# Save models to disk
def save_models(models):
    try:
        for name, model in models.items():
            filename = f"{name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, filename)
            print(f"Model saved: {filename}")
    except Exception as e:
        print(f"Error saving models: {e}")

# Load model from disk
def load_model(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Batch predictions and save to CSV
def batch_predict(model, csv_file):
    try:
        data = load_data(csv_file)
        if data is not None:
            X_train, X_test, y_train, y_test, feature_names = preprocess_data(data)
            predictions = model.predict(X_test)
            results = pd.DataFrame({"Predictions": predictions})
            output_file = "predictions_output.csv"
            results.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
        else:
            print("No data to predict on.")
    except Exception as e:
        print(f"Error during batch prediction: {e}")

# Main function
if __name__ == "__main__":
    choice = input("Would you like to (1) Train a new model or (2) Load an existing model? Enter 1 or 2: ").strip()
    
    if choice == "1":
        file_path = select_file("csv")
        if file_path:
            confirm = input(f"Proceed with the selected file '{file_path}'? (yes/no): ").lower()
            if confirm == "yes":
                data = load_data(file_path)
                if data is not None:
                    X_train, X_test, y_train, y_test, feature_names = preprocess_data(data)
                    best_models = cross_validate_models(X_train, y_train)
                    for name, model in best_models.items():
                        print(f"\nEvaluating {name}:")
                        evaluate_model(model, X_test, y_test)
                        if name == 'Random Forest':
                            feature_importance(model, feature_names)
                    plot_roc_curve(best_models, X_test, y_test)
                    save_models(best_models)
            else:
                print("Operation canceled.")
        else:
            print("No file selected.")
    
    elif choice == "2":
        model_file_path = select_file("model")
        if model_file_path:
            model = load_model(model_file_path)
            if model:
                file_path = select_file("csv")
                if file_path:
                    batch_predict(model, file_path)
                else:
                    print("No CSV file selected for batch prediction.")
            else:
                print("Model loading failed.")
        else:
            print("No model file selected.")
    
    else:
        print("Invalid choice.")
