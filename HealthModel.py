import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tabulate import tabulate

MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Neural Network': MLPClassifier(max_iter=1000)
}

def select_file():
    file_path = input("Enter the path to the CSV file containing health data (e.g., 'health_data.csv'): ").strip()
    if not os.path.exists(file_path):
        print("Error: File not found. Please try again.")
        return select_file()
    return file_path

def get_target_column(df):
    print("\nColumns available for prediction:")
    print(list(df.columns))
    target_column = input("Enter the name of the target column (the column to predict): ").strip()
    if target_column not in df.columns:
        print("Error: Column not found. Please enter a valid column name.")
        return get_target_column(df)
    return target_column

def load_and_prepare_data(file_path, target_col):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X.fillna(X.mean(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report

def main():
    print("Welcome to the HealthRiskPredictor!")

    # Step 1: Select file interactively
    file_path = select_file()
    df = pd.read_csv(file_path)

    # Step 2: Get target column interactively
    target_col = get_target_column(df)

    # Step 3: Data preparation
    X_train, X_test, y_train, y_test = load_and_prepare_data(file_path, target_col)

    results = []
    for model_name, model in MODELS.items():
        print(f"\nTraining {model_name}...")
        accuracy, report = train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test)
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": report["1"]["precision"],
            "Recall": report["1"]["recall"],
            "F1-Score": report["1"]["f1-score"]
        })

    # Print summary table
    print("\nModel Comparison:")
    print(tabulate(results, headers="keys", tablefmt="pretty"))

if __name__ == "__main__":
    main()

