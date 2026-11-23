
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime, timezone

from sklearn.base import clone
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

from data import (
    load_data,
    prepare_features,
    split_data,
    preprocess_data,
    RANDOM_STATE
)
from models import create_logistic_regression


# Setting up directories for experiment results and model checkpoints
ROOT_DIR = Path(__file__).resolve().parent.parent

EXPERIMENT_DIR = ROOT_DIR / "experiments" / "logistic_regression"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# Evaluates the model using precision, recall, F1, average precision and confusion matrix
def evaluate_model(model, X, y):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "precision": float(precision_score(
            y, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y, predictions, zero_division=0
        )),
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist()
    }

    return metrics


# Generates a learning curve to compare training and validation performance
def generate_learning_curve(X_train, y_train, X_val, y_val, preprocessor):
    train_sizes = [0.10, 0.25, 0.50, 0.75, 1.0]

    training_f1 = []
    validation_f1 = []
    sample_counts = []

    for fraction in train_sizes:

        # Select a portion of training data while maintaining class distribution
        if fraction == 1.0:
            X_subset = X_train
            y_subset = y_train
        else:
            X_subset, _, y_subset, _ = train_test_split(
                X_train,
                y_train,
                train_size=fraction,
                random_state=RANDOM_STATE,
                stratify=y_train
            )

        # Create and fit a new preprocessor using the selected training data
        subset_preprocessor = clone(preprocessor)

        X_subset_processed = (
            subset_preprocessor.fit_transform(X_subset)
        )

        X_val_processed = (
            subset_preprocessor.transform(X_val)
        )

        # Train the logistic regression model using the selected data
        model = create_logistic_regression()
        model.fit(X_subset_processed, y_subset)

        # Calculate F1 scores for training and validation
        train_pred = model.predict(X_subset_processed)
        val_pred = model.predict(X_val_processed)

        train_f1 = f1_score(
            y_subset, train_pred, zero_division=0
        )

        val_f1 = f1_score(
            y_val, val_pred, zero_division=0
        )

        sample_counts.append(len(X_subset))
        training_f1.append(float(train_f1))
        validation_f1.append(float(val_f1))

        print(
            f"Samples: {len(X_subset)} | "
            f"Train F1: {train_f1:.4f} | "
            f"Validation F1: {val_f1:.4f}"
        )

    # Plot the training and validation F1 scores
    plt.figure(figsize=(8, 5))

    plt.plot(
        sample_counts,
        training_f1,
        marker="o",
        label="Training F1"
    )

    plt.plot(
        sample_counts,
        validation_f1,
        marker="o",
        label="Validation F1"
    )

    plt.xlabel("Number of Training Samples")
    plt.ylabel("F1 Score")
    plt.title("Logistic Regression Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the learning curve as an image
    plt.savefig(
        EXPERIMENT_DIR / "learning_curve.png",
        dpi=300
    )

    plt.close()

    return {
        "sample_counts": sample_counts,
        "training_f1": training_f1,
        "validation_f1": validation_f1
    }


# Saves the trained model, preprocessor, configurations and experiment results
def save_experiment(model, preprocessor, metrics, curve):

    # Save the trained model and preprocessor
    joblib.dump(
        model,
        CHECKPOINT_DIR / "logistic_regression.joblib"
    )

    joblib.dump(
        preprocessor,
        CHECKPOINT_DIR / "preprocessor.joblib"
    )

    # Store the model configurations
    config = {
        "model": "LogisticRegression",
        "random_state": RANDOM_STATE,
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "max_iter": model.max_iter,
        "class_weight": model.class_weight,
        "classification_threshold": 0.5,
        "n_features": int(model.n_features_in_)
    }

    # Save configurations to JSON
    with open(EXPERIMENT_DIR / "params.json", "w") as f:
        json.dump(config, f, indent=4)

    # Save validation metrics to JSON
    with open(EXPERIMENT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save learning curve values to JSON
    with open(EXPERIMENT_DIR / "learning_curve.json", "w") as f:
        json.dump(curve, f, indent=4)

    # Record experiment details and results in a log file
    with open(EXPERIMENT_DIR / "train.log", "w") as f:
        f.write(
            f"Run time (UTC): "
            f"{datetime.now(timezone.utc).isoformat()}\n"
        )
        f.write("Model: Logistic Regression\n")
        f.write("Evaluation split: Validation\n")
        f.write(f"Seed: {RANDOM_STATE}\n")
        f.write(f"Precision: {metrics['precision']:.6f}\n")
        f.write(f"Recall: {metrics['recall']:.6f}\n")
        f.write(f"F1: {metrics['f1']:.6f}\n")
        f.write(
            f"Average Precision: "
            f"{metrics['average_precision']:.6f}\n"
        )
        f.write(
            f"Confusion Matrix: "
            f"{metrics['confusion_matrix']}\n"
        )

    print("\nExperiment results saved successfully!")


# Runs the full training and evaluation process
def main():

    # Load dataset and separate inputs from target
    df = load_data()
    X, y = prepare_features(df)

    # Split dataset into training, validation and testing
    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(X, y)
    )

    # Apply preprocessing using only training data for fitting
    (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        preprocessor
    ) = preprocess_data(X_train, X_val, X_test)

    # Create and train logistic regression baseline
    model = create_logistic_regression()
    model.fit(X_train_processed, y_train)

    # Evaluate the model using validation data
    metrics = evaluate_model(
        model, X_val_processed, y_val
    )

    print("\nLogistic Regression Validation Results")
    print("--------------------------------------")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    # Generate learning curve using different training set sizes
    print("\nGenerating learning curve...")

    curve = generate_learning_curve(
        X_train, y_train,
        X_val, y_val,
        preprocessor
    )

    # Save the model, preprocessor and experiment results
    save_experiment(
        model, preprocessor, metrics, curve
    )


if __name__ == "__main__":
    main()
