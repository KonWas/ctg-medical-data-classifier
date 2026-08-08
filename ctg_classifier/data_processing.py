"""Data loading, missing-value handling and train/test splitting."""

import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split


def load_data(data_path, target_column="CLASS"):
    """Load the CTG dataset and split it into features (X) and target (y)."""
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    print(f"Data shape: {data.shape}")
    print(f"Number of classes: {y.nunique()}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print("\nData description:")
    print(data.describe())

    return data, X, y


def impute_missing_values(X, y, method="mean"):
    """Handle missing values in the feature matrix.

    Supported methods: 'mean', 'median', 'knn', 'drop'.
    'drop' removes any row that has at least one missing value (from both X and y).
    """
    print(f"Handling missing values using method: {method}")

    if method == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif method == "median":
        imputer = SimpleImputer(strategy="median")
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=5)
    elif method == "drop":
        mask = ~X.isnull().any(axis=1)
        return X[mask], y[mask]
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X_imputed, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split features/target into stratified train and test sets."""
    print(f"Splitting data ({int((1 - test_size) * 100)}% train, {int(test_size * 100)}% test)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test
