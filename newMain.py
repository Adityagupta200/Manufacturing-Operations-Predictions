# Copyright (c) [2025] [ADITYA GUPTA]
# Licensed under the MIT License. See LICENSE for details.

from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import numpy as np
import optuna

app = Flask(__name__)

# Directory to save uploaded files and model
UPLOAD_FOLDER = "uploads"
MODEL_PATH = os.path.join(UPLOAD_FOLDER, "machine_downtime_model.joblib")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def objective(trial, balanced_features_train, balanced_labels_train):
    """Optimize Logistic Regression hyperparameters using Optuna."""
    # Define hyperparameter search space
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])
    penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet', None])

    # Handle incompatible solver-penalty combinations
    if solver in ['lbfgs', 'newton-cg', 'sag'] and penalty not in ['l2', None]:
        raise optuna.exceptions.TrialPruned()
    elif solver == "liblinear" and penalty not in ['l1', 'l2']:
        raise optuna.exceptions.TrialPruned()
    elif solver == "saga" and penalty not in ['l1', 'l2', 'elasticnet', None]:
        raise optuna.exceptions.TrialPruned()

    # Add l1_ratio only if penalty is elasticnet
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0) if penalty == 'elasticnet' else None

    C = trial.suggest_float('C', 1e-4, 10, log=True)
    max_iter = trial.suggest_int('max_iter', 50, 500)

    # Initialize and evaluate model
    model = LogisticRegression(
        solver=solver,
        penalty=penalty if penalty else None,
        C=C,
        max_iter=max_iter,
        random_state=42,
        l1_ratio=l1_ratio
    )
    scores = cross_val_score(model, balanced_features_train, balanced_labels_train, cv=5, scoring='f1')
    return scores.mean()

def train_model(file_path):
    """Train a Logistic Regression model on the dataset."""
    # Load and validate dataset
    data = pd.read_csv(file_path, encoding="ISO-8859-1")

    if not all(col in data.columns for col in ["Downtime flag", "Temperature", "Run time"]):
        raise ValueError("Dataset must include 'Downtime flag', 'Temperature', and 'Run time' columns.")

    # Extract features and labels
    downtime_label = data["Downtime flag"]
    features = data[["Temperature", "Run time"]]

    # Balance dataset using SMOTE
    smote = SMOTE(random_state=42)
    balanced_features, balanced_labels = smote.fit_resample(features, downtime_label)

    # Split data into training and testing sets
    balanced_features_train, balanced_features_test, balanced_labels_train, balanced_labels_test = train_test_split(
        balanced_features, balanced_labels, test_size=0.2, random_state=42
    )

    # Optimize model using Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, balanced_features_train, balanced_labels_train), n_trials=10)

    # Train final model with best parameters
    best_params = study.best_params
    model = LogisticRegression(
        solver=best_params['solver'],
        penalty=best_params['penalty'],
        C=best_params['C'],
        max_iter=best_params['max_iter'],
        random_state=42,
        l1_ratio=best_params.get('l1_ratio', None)
    )
    model.fit(balanced_features_train, balanced_labels_train)

    # Evaluate model
    predictions = model.predict(balanced_features_test)
    metrics = {
        "accuracy": accuracy_score(balanced_labels_test, predictions),
        "f1_score": f1_score(balanced_labels_test, predictions)
    }

    # Save trained model
    joblib.dump(model, MODEL_PATH)
    return metrics

@app.route('/upload', methods=['POST'])
def upload_data():
    """Upload a CSV file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return jsonify({"message": f"File {file.filename} uploaded successfully", "path": file_path})

@app.route('/train', methods=['POST'])
def train():
    """Train the model on the uploaded dataset."""
    files = os.listdir(UPLOAD_FOLDER)
    if not files:
        return jsonify({"error": "No uploaded file found"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, files[0])
    if not file_path.endswith(".csv"):
        return jsonify({"error": "Uploaded file must be a CSV."}), 400

    try:
        metrics = train_model(file_path)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    return jsonify({"message": "Model trained successfully", "metrics": metrics})

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the trained model."""
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet."}), 400

    input_data = request.get_json()
    if not input_data or not all(k in input_data for k in ["temperature", "run_time"]):
        return jsonify({"error": "Invalid input data. Provide 'temperature' and 'run_time'."}), 400

    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([{ "Temperature": input_data["temperature"], "Run time": input_data["run_time"] }])

    prediction = model.predict(input_df)
    confidence = max(model.predict_proba(input_df)[0])

    return jsonify({
        "Downtime": "Yes" if prediction[0] == 1 else "No",
        "Confidence": round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
