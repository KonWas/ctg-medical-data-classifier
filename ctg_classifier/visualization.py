"""Plotting utilities: exploratory data analysis and experiment result charts."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def configure_plot_style():
    """Apply a consistent visual style to all plots."""
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10


def plot_missing_values_analysis(data, save_path=None):
    """Plot the count and percentage of missing values per feature."""
    print("Generating missing values analysis...")

    missing_counts = data.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if len(missing_counts) == 0:
        print("No missing values in the data!")
        return

    total_samples = len(data)
    missing_percent = (missing_counts / total_samples) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(missing_counts)))
    bars1 = ax1.bar(range(len(missing_counts)), missing_counts.values,
                     color=colors, alpha=0.8, edgecolor="black", linewidth=1)
    for bar, count in zip(bars1, missing_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 5,
                  f"{int(count)}", ha="center", va="bottom", fontweight="bold")

    ax1.set_xlabel("Feature", fontweight="bold")
    ax1.set_ylabel("Number of missing values", fontweight="bold")
    ax1.set_title("Missing value count per feature", fontweight="bold", pad=15)
    ax1.set_xticks(range(len(missing_counts)))
    ax1.set_xticklabels(missing_counts.index, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    colors2 = plt.cm.Oranges(np.linspace(0.4, 0.8, len(missing_percent)))
    bars2 = ax2.bar(range(len(missing_percent)), missing_percent.values,
                     color=colors2, alpha=0.8, edgecolor="black", linewidth=1)
    for bar, percent in zip(bars2, missing_percent.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
                  f"{percent:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax2.set_xlabel("Feature", fontweight="bold")
    ax2.set_ylabel("Missing values (%)", fontweight="bold")
    ax2.set_title("Missing value percentage per feature", fontweight="bold", pad=15)
    ax2.set_xticks(range(len(missing_percent)))
    ax2.set_xticklabels(missing_percent.index, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(y=10, color="red", linestyle="--", alpha=0.7, label="Warning threshold (10%)")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nMISSING VALUES STATISTICS:")
    print(f"  Total samples: {total_samples}")
    print(f"  Features with missing values: {len(missing_counts)}")
    print(f"  Total missing values: {missing_counts.sum()}")
    print(f"  Average missing percentage: {missing_percent.mean():.2f}%")
    print(f"  Highest missing percentage: {missing_percent.max():.2f}% ({missing_percent.idxmax()})")


def plot_class_distribution(y, save_path=None):
    """Plot the class distribution as a bar chart and a pie chart."""
    print("Generating class distribution analysis...")

    class_counts = y.value_counts().sort_index()
    class_percentages = (class_counts / len(y)) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
    bars1 = ax1.bar(class_counts.index, class_counts.values,
                     color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    for bar, count in zip(bars1, class_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 5,
                  f"{int(count)}", ha="center", va="bottom", fontweight="bold")

    ax1.set_xlabel("Class", fontweight="bold")
    ax1.set_ylabel("Number of samples", fontweight="bold")
    ax1.set_title("Number of samples per class", fontweight="bold", pad=15)
    ax1.set_xticks(class_counts.index)
    ax1.grid(True, alpha=0.3, axis="y")

    mean_count = class_counts.mean()
    ax1.axhline(y=mean_count, color="red", linestyle="--", alpha=0.7,
                label=f"Mean: {mean_count:.1f}")
    ax1.legend()

    wedges, texts, autotexts = ax2.pie(
        class_percentages.values,
        labels=[f"Class {i}" for i in class_counts.index],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=[0.05 if count < mean_count else 0 for count in class_counts.values],
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)

    ax2.set_title("Class distribution (%)", fontweight="bold", pad=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nCLASS DISTRIBUTION STATISTICS:")
    print(f"  Total samples: {len(y)}")
    print(f"  Number of classes: {len(class_counts)}")
    print(f"  Classes: {list(class_counts.index)}")

    print("\nDETAILED DISTRIBUTION:")
    for class_id in class_counts.index:
        count = class_counts[class_id]
        percentage = class_percentages[class_id]
        print(f"  Class {class_id}: {count:3d} samples ({percentage:5.2f}%)")

    print("\nCLASS BALANCE STATISTICS:")
    print(f"  Fewest samples: {class_counts.min()} (Class {class_counts.idxmin()})")
    print(f"  Most samples: {class_counts.max()} (Class {class_counts.idxmax()})")
    print(f"  Mean: {class_counts.mean():.1f}")
    print(f"  Standard deviation: {class_counts.std():.1f}")

    ratio = class_counts.max() / class_counts.min()
    if ratio <= 2:
        balance_status = "Well balanced"
    elif ratio <= 5:
        balance_status = "Moderately imbalanced"
    else:
        balance_status = "Heavily imbalanced"
    print(f"  Max/min ratio: {ratio:.2f} - {balance_status}")


def _best_test_accuracy(results, experiment_key, algorithm_prefix):
    """Return the best test accuracy (%) for a given algorithm prefix within
    an experiment's results, or None if that algorithm was not run there."""
    experiment_results = results.get(experiment_key, {})
    matches = [v for k, v in experiment_results.items() if k.startswith(algorithm_prefix)]
    if not matches:
        return None
    return max(m["test_accuracy"] for m in matches) * 100


def prepare_comparison_data(results):
    """Derive the datasets used by the comparison plots from the raw
    experiment results (`CTGClassifier.results`)."""
    if not results:
        raise ValueError("No results available. Run run_full_experiment() first.")

    algorithms = ["NB", "DT", "RF", "SVM"]
    preprocessing_data = {"Algorithm": [], "Raw": [], "Standardized": [], "PCA": []}
    for alg in algorithms:
        preprocessing_data["Algorithm"].append(alg)
        preprocessing_data["Raw"].append(_best_test_accuracy(results, "raw", alg) or 0)
        preprocessing_data["Standardized"].append(_best_test_accuracy(results, "standardized", alg) or 0)
        preprocessing_data["PCA"].append(_best_test_accuracy(results, "pca", alg))

    all_models = []
    for exp_name in ("raw", "standardized", "pca"):
        for model_name, metrics in results.get(exp_name, {}).items():
            all_models.append({
                "Model": f"{exp_name.upper()}_{model_name}",
                "Test_Accuracy": metrics["test_accuracy"] * 100,
                "CV_Accuracy": metrics["cv_accuracy"] * 100,
                "CV_Std": metrics["cv_std"] * 100,
            })
    all_models.sort(key=lambda m: m["Test_Accuracy"], reverse=True)
    top_models = all_models[:6]

    top_models_data = {
        "Model": [m["Model"] for m in top_models],
        "Test_Accuracy": [m["Test_Accuracy"] for m in top_models],
        "CV_Accuracy": [m["CV_Accuracy"] for m in top_models],
        "CV_Std": [m["CV_Std"] for m in top_models],
    }

    overfitting = results.get("overfitting", {})
    overfitting_data = {
        "Model": list(overfitting.keys()),
        "Train_Accuracy": [v["train_accuracy"] * 100 for v in overfitting.values()],
        "Test_Accuracy": [v["test_accuracy"] * 100 for v in overfitting.values()],
        "CV_Accuracy": [v["cv_accuracy"] * 100 for v in overfitting.values()],
        "Overfitting_Gap": [v["overfitting_gap"] * 100 for v in overfitting.values()],
    }

    return {
        "top_models": top_models_data,
        "preprocessing_comparison": preprocessing_data,
        "overfitting_analysis": overfitting_data,
    }


def plot_top_models_comparison(results, save_path=None):
    """Bar chart comparing the top 6 models overall by test accuracy."""
    print("Generating top models comparison chart...")

    data = prepare_comparison_data(results)["top_models"]

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    bars = ax.bar(data["Model"], data["Test_Accuracy"],
                   color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.errorbar(data["Model"], data["CV_Accuracy"], yerr=data["CV_Std"],
                fmt="o", color="red", capsize=5, capthick=2,
                label="CV Accuracy +/- std", markersize=8)

    for bar, acc in zip(bars, data["Test_Accuracy"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Top CTG classification models\n(Test Accuracy vs CV Accuracy)",
                 fontweight="bold", pad=20)
    ax.set_ylim(0, max(data["Test_Accuracy"]) + 10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_preprocessing_impact(results, save_path=None):
    """Bar chart comparing raw / standardized / PCA preprocessing per algorithm."""
    print("Generating preprocessing impact chart...")

    data = prepare_comparison_data(results)["preprocessing_comparison"]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(data["Algorithm"]))
    width = 0.25

    bars1 = ax.bar(x - width, data["Raw"], width, label="Raw data", color="#FF6B6B", alpha=0.8)
    bars2 = ax.bar(x, data["Standardized"], width, label="Standardized", color="#4ECDC4", alpha=0.8)
    pca_values = [v if v is not None else 0 for v in data["PCA"]]
    bars3 = ax.bar(x + width, pca_values, width, label="PCA", color="#FFA07A", alpha=0.8)

    def add_value_labels(bars, values):
        for bar, val in zip(bars, values):
            if val is not None and val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    add_value_labels(bars1, data["Raw"])
    add_value_labels(bars2, data["Standardized"])
    add_value_labels(bars3, data["PCA"])

    ax.set_ylabel("Test Accuracy (%)", fontweight="bold")
    ax.set_xlabel("Algorithm", fontweight="bold")
    ax.set_title("Effect of preprocessing on algorithm performance", fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(data["Algorithm"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 90)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_overfitting_analysis(results, save_path=None):
    """Two-panel chart: train vs test accuracy, and the overfitting gap."""
    print("Generating overfitting analysis chart...")

    data = prepare_comparison_data(results)["overfitting_analysis"]
    labels = ["Unregularized", "Regularized"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(len(data["Model"]))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, data["Train_Accuracy"], width,
                     label="Train Accuracy", color="#FF6B6B", alpha=0.8)
    bars2 = ax1.bar(x + width / 2, data["Test_Accuracy"], width,
                     label="Test Accuracy", color="#4ECDC4", alpha=0.8)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                      f"{height:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_title("Overfitting analysis\nTrain vs Test Accuracy", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, 110)

    colors = ["#FF4757", "#2ED573"]
    bars = ax2.bar(x, data["Overfitting_Gap"],
                    color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    for bar, gap in zip(bars, data["Overfitting_Gap"]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                  f"{gap:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax2.set_ylabel("Overfitting gap (%)", fontweight="bold")
    ax2.set_title("Overfitting gap\n(Train - Test Accuracy)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(y=20, color="orange", linestyle="--", alpha=0.7, label="Warning threshold (20%)")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_visualization_report(results, save_dir="./ctg_visualizations/"):
    """Generate and save the full set of comparison plots."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("Generating full visualization report...")
    print("=" * 50)

    if not results:
        print("No results available! Run run_full_experiment() first.")
        return

    print("1. Top models comparison...")
    plot_top_models_comparison(results, f"{save_dir}01_top_models.png")

    print("2. Preprocessing impact analysis...")
    plot_preprocessing_impact(results, f"{save_dir}02_preprocessing_impact.png")

    print("3. Overfitting analysis...")
    plot_overfitting_analysis(results, f"{save_dir}03_overfitting_analysis.png")

    print(f"\nReport generated in: {save_dir}")
    print("Files:")
    for i, name in enumerate([
        "01_top_models.png",
        "02_preprocessing_impact.png",
        "03_overfitting_analysis.png",
    ], 1):
        print(f"  {i}. {name}")
