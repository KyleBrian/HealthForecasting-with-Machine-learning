import pandas as pd
import joblib
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt


# File selection with tkinter dialog
def select_file(file_type="csv"):
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    try:
        if file_type == "csv":
            file_path = filedialog.askopenfilename(
                title="Select the CSV file for data",
                filetypes=[("CSV files", "*.csv")]
            )
        elif file_type == "model":
            file_path = filedialog.askopenfilename(
                title="Select a saved model file",
                filetypes=[("Model files", "*.pkl")]
            )
        if not file_path or not os.path.exists(file_path):
            print("No file selected or file does not exist.")
            return None
        return file_path
    except Exception as e:
        print(f"Error during file selection: {e}")
        return None


# Save model with tkinter dialog
def save_model_dialog(model, model_name):
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    try:
        file_path = filedialog.asksaveasfilename(
            title=f"Save {model_name} model",
            defaultextension=".pkl",
            filetypes=[("Model files", "*.pkl")]
        )
        if file_path:
            joblib.dump(model, file_path)
            print(f"Model saved as: {file_path}")
        else:
            print("Model save canceled.")
    except Exception as e:
        print(f"Error saving model: {e}")


# Load data
def load_data(file_path: str):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


# Data preprocessing
def preprocess_data(df):
    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X.fillna(X.mean(), inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, df.columns[:-1]
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return None, None, None, None, None


# Train and optimize models
def cross_validate_models(X_train, y_train):
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
        print(f"Training and optimizing {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
    return best_models


# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)


# Plot ROC Curve
def plot_roc_curve(models, X_test, y_test):
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


# Load model
def load_model(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Batch prediction
def batch_predict(model, csv_file):
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


if __name__ == "__main__":
    choice = input("Would you like to (1) Train a new model or (2) Load an existing model? Enter 1 or 2: ").strip()

    if choice == "1":
        # Training mode
        file_path = select_file("csv")
        if file_path:
            data = load_data(file_path)
            if data is not None:
                X_train, X_test, y_train, y_test, feature_names = preprocess_data(data)
                best_models = cross_validate_models(X_train, y_train)
                for name, model in best_models.items():
                    print(f"\nEvaluating {name}:")
                    evaluate_model(model, X_test, y_test)
                plot_roc_curve(best_models, X_test, y_test)

                # Save each model
                for name, model in best_models.items():
                    save_model_dialog(model, name)
            else:
                print("Error: Could not load data.")
        else:
            print("No file selected for training.")

    elif choice == "2":
        # Batch prediction mode
        model_file_path = select_file("model")
        if model_file_path:
            model = load_model(model_file_path)
            if model:
                csv_file = select_file("csv")
                if csv_file:
                    batch_predict(model, csv_file)
                else:
                    print("No CSV file selected for batch prediction.")
            else:
                print("Failed to load model.")
        else:
            print("No model file selected.")

    else:
        print("Invalid choice.")
