"""Model training, evaluation and the overfitting-prevention experiment."""

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_naive_bayes(X_train, y_train, var_smoothing_values=(1e-9, 1e-8, 1e-7)):
    """Train Gaussian Naive Bayes over a small grid of var_smoothing values."""
    print("Training Gaussian Naive Bayes...")
    results = {}

    for vs in var_smoothing_values:
        print(f"  var_smoothing = {vs}")
        model = GaussianNB(var_smoothing=vs)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)

        results[f"NB_vs_{vs}"] = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "hyperparams": {"var_smoothing": vs},
        }
        print(f"    CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    return results


def train_decision_tree(X_train, y_train,
                         max_depths=(None, 5, 10, 15),
                         min_samples_splits=(2, 5, 10),
                         criteria=("gini", "entropy")):
    """Train a Decision Tree over a grid of criterion / max_depth / min_samples_split."""
    print("Training Decision Tree...")
    results = {}

    for criterion in criteria:
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                print(f"  criterion={criterion}, max_depth={max_depth}, "
                      f"min_samples_split={min_samples_split}")

                model = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                )

                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
                model.fit(X_train, y_train)

                key = f"DT_{criterion}_{max_depth}_{min_samples_split}"
                results[key] = {
                    "model": model,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "hyperparams": {
                        "criterion": criterion,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                    },
                }
                print(f"    CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    return results


def train_bonus_models(X_train, y_train):
    """Train the bonus algorithms: Random Forest and SVM, each over a small config grid."""
    print("Training bonus algorithms...")
    results = {}

    print("  Random Forest...")
    rf_configs = [
        {"n_estimators": 50, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 15},
        {"n_estimators": 200, "max_depth": None},
    ]
    for i, params in enumerate(rf_configs):
        model = RandomForestClassifier(random_state=42, **params)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)

        results[f"RF_config_{i + 1}"] = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "hyperparams": params,
        }
        print(f"    Config {i + 1}: CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print("  Support Vector Machine...")
    svm_configs = [
        {"C": 1.0, "kernel": "rbf"},
        {"C": 10.0, "kernel": "rbf"},
        {"C": 1.0, "kernel": "linear"},
    ]
    for i, params in enumerate(svm_configs):
        model = SVC(random_state=42, **params)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)

        results[f"SVM_config_{i + 1}"] = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "hyperparams": params,
        }
        print(f"    Config {i + 1}: CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    return results


def evaluate_models(models_dict, X_test, y_test):
    """Evaluate a dict of trained models on the held-out test set."""
    print("Evaluating models on the test set...")
    evaluation = {}

    for model_name, model_info in models_dict.items():
        model = model_info["model"]
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        evaluation[model_name] = {
            "cv_accuracy": model_info["cv_mean"],
            "cv_std": model_info["cv_std"],
            "test_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "hyperparams": model_info["hyperparams"],
        }
        print(f"  {model_name}: Test Accuracy = {accuracy:.4f}")

    return evaluation


def run_overfitting_experiment(X_train, y_train, X_test, y_test):
    """Compare an unconstrained Decision Tree against a regularized one to
    illustrate the effect of regularization on overfitting."""
    print("Running overfitting-prevention experiment...")

    dt_overfit = DecisionTreeClassifier(random_state=42)
    dt_regulated = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    )

    results = {}
    for name, model in [("DT_Overfit", dt_overfit), ("DT_Regulated", dt_regulated)]:
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        results[name] = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "cv_accuracy": cv_scores.mean(),
            "overfitting_gap": train_acc - test_acc,
        }

        print(f"  {name}:")
        print(f"    Train Accuracy: {train_acc:.4f}")
        print(f"    Test Accuracy: {test_acc:.4f}")
        print(f"    CV Accuracy: {cv_scores.mean():.4f}")
        print(f"    Overfitting Gap: {train_acc - test_acc:.4f}")

    return results
