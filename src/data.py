
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Setting the dataset path
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "ai4i2020.csv"
)

RANDOM_STATE = 42

# Loads the dataset
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Inspects the data and displays dataset properties
def explore_data(df):
    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nDataset information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nMachine failure counts:")
    print(df["Machine failure"].value_counts())

    print("\nMachine failure percentages:")
    print(df["Machine failure"].value_counts(normalize=True) * 100)

    print("\nNumerical feature statistics:")
    print(df.describe())

# Handle features: Remove unnecessary features and seperate input and ouput features
def prepare_features(df):
    # Remove identifiers, target, and leakage columns
    columns_to_drop = [
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]

    X = df.drop(columns=columns_to_drop)
    y = df["Machine failure"]

    print("\nInput features:")
    print(X.columns.tolist())

    return X, y

# Splits the data (Training/Validation/Testing): 70/15/15
def split_data(X, y):
    # 70% training, 30% remaining
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Split remaining data equally:
    # 15% validation and 15% testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    # Display new split dataset sizes
    print("\nDataset splits:")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    # Displaying class distribution in each
    print("\nFailure counts in each split:")
    print("Training:", y_train.value_counts().to_dict())
    print("Validation:", y_val.value_counts().to_dict())
    print("Testing:", y_test.value_counts().to_dict())

    return X_train, X_val, X_test, y_train, y_val, y_test

# Preprocessing data: Normalizes numerical features and hot encodes catergorical features
def preprocess_data(X_train, X_val, X_test):
    numerical_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    categorical_features = ["Type"]

    # Standardize numerical features
    # and one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ]
    )

    # Fit preprocessing using ONLY training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply the same transformations to other splits
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print("\nPreprocessed dataset shapes:")
    print("Training:", X_train_processed.shape)
    print("Validation:", X_val_processed.shape)
    print("Testing:", X_test_processed.shape)

    print("\nProcessed feature names:")
    print(preprocessor.get_feature_names_out())

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        preprocessor
    )

def main():
    # Load dataset
    df = load_data()

    # Explore dataset
    explore_data(df)

    # Separate inputs and target
    X, y = prepare_features(df)

    # Create reproducible splits
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Apply encoding and standardization
    X_train_processed, X_val_processed, X_test_processed, preprocessor = (preprocess_data(X_train, X_val, X_test))

if __name__ == "__main__":
    main()
