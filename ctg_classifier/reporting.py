"""Textual summaries of experiment results."""

import pandas as pd


def build_results_summary(all_evaluations):
    """Build (and print) a summary table across all evaluated models,
    sorted by test accuracy (descending)."""
    print("Building results summary...")

    summary_data = [
        {
            "Model": model_name,
            "CV_Accuracy": f"{metrics['cv_accuracy']:.4f} +/- {metrics['cv_std']:.4f}",
            "Test_Accuracy": metrics["test_accuracy"],
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1_Score": f"{metrics['f1_score']:.4f}",
        }
        for model_name, metrics in all_evaluations.items()
    ]

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Test_Accuracy", ascending=False)
    summary_df["Test_Accuracy"] = summary_df["Test_Accuracy"].map(lambda v: f"{v:.4f}")

    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    return summary_df


def print_best_results(results):
    """Print the overall best model, the best model per preprocessing
    category, and an overfitting-gap readout."""
    print("\n" + "=" * 20)
    print("BEST EXPERIMENT RESULTS")
    print("=" * 20)

    best_model, best_score = None, 0.0
    for exp_name in ("raw", "standardized", "pca"):
        for model_name, metrics in results.get(exp_name, {}).items():
            if metrics["test_accuracy"] > best_score:
                best_score = metrics["test_accuracy"]
                best_model = f"{exp_name.upper()}_{model_name}"

    print(f"BEST MODEL: {best_model}")
    print(f"ACCURACY: {best_score:.4f} ({best_score * 100:.2f}%)")

    print("\nBEST RESULT PER CATEGORY:")
    for exp_name in ("raw", "standardized", "pca"):
        exp_results = results.get(exp_name)
        if exp_results:
            model_name, metrics = max(exp_results.items(), key=lambda kv: kv[1]["test_accuracy"])
            print(f"  {exp_name.upper()}: {model_name} = {metrics['test_accuracy'] * 100:.2f}%")

    if "overfitting" in results:
        print("\nOVERFITTING ANALYSIS:")
        for model_name, metrics in results["overfitting"].items():
            gap = metrics["overfitting_gap"]
            status = "OK" if gap < 0.20 else "OVERFITTING"
            print(f"  {model_name}: Gap = {gap * 100:.1f}% ({status})")
