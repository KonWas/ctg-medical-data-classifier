"""Feature preprocessing: scaling, dimensionality reduction and feature selection."""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


def standardize_features(X):
    """Zero-mean, unit-variance scaling (StandardScaler)."""
    print("Preprocessing: standardization (StandardScaler)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)


def normalize_features(X):
    """Min-max scaling to the [0, 1] range."""
    print("Preprocessing: normalization (MinMaxScaler)")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)


def apply_pca(X, variance_threshold=0.95):
    """Reduce dimensionality with PCA, keeping enough components to explain
    at least `variance_threshold` of the variance."""
    print("Preprocessing: PCA")
    pca = PCA(n_components=variance_threshold)
    X_reduced = pca.fit_transform(X)
    print(f"PCA: reduced from {X.shape[1]} to {X_reduced.shape[1]} components "
          f"(explained variance: {pca.explained_variance_ratio_.sum():.4f})")
    columns = [f"PC{i + 1}" for i in range(X_reduced.shape[1])]
    return pd.DataFrame(X_reduced, columns=columns)


def select_k_best_features(X, y, k=15):
    """Select the k features most correlated with the target using ANOVA F-value."""
    print(f"Preprocessing: feature selection (SelectKBest, k={k})")
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Selected features: {list(selected_features)}")
    return pd.DataFrame(X_selected, columns=selected_features)
