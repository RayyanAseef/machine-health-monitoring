
import joblib
import json
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from data import (
    load_data,
    prepare_features,
    split_data
)


# Setting up directories for model checkpoints and experiment results
ROOT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
EXPERIMENT_DIR = ROOT_DIR / "experiments" / "logistic_regression"


# Loads the saved logistic regression model and preprocessing pipeline
def load_model():

    model = joblib.load(
        CHECKPOINT_DIR / "logistic_regression.joblib"
    )

    preprocessor = joblib.load(
        CHECKPOINT_DIR / "preprocessor.joblib"
    )

    return model, preprocessor


# Evaluates the model using precision, recall, F1 and average precision
def evaluate(model, X, y):

    # Generate predictions and failure probabilities
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y, probabilities)
        )
    }

    # Display validation results
    print("\nValidation Results")
    print("------------------------------")

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    # Display classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y, predictions, zero_division=0
        )
    )

    # Generate confusion matrix
    cm = confusion_matrix(y, predictions, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Failure"]
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title("Logistic Regression - Validation Confusion Matrix")
    plt.tight_layout()

    # Save confusion matrix as an image
    plt.savefig(
        EXPERIMENT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    return metrics


# Runs the evaluation process using the saved model and validation data
def main():

    # Load saved model and preprocessor
    model, preprocessor = load_model()

    # Load dataset and separate inputs from target
    df = load_data()
    X, y = prepare_features(df)

    # Recreate the original training, validation and testing splits
    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_data(X, y)
    )

    # Apply saved preprocessing to validation data
    X_val_processed = preprocessor.transform(X_val)

    # Evaluate the model using validation data
    metrics = evaluate(
        model, X_val_processed, y_val
    )

    # Save evaluation metrics to JSON
    with open(
        EXPERIMENT_DIR / "evaluation.json", "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
