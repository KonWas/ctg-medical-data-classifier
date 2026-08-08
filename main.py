"""Entry point for the CTG medical data classification pipeline.

Loads `cardiotocography_v2.csv`, runs the raw / standardized / PCA
preprocessing experiments plus the overfitting-prevention experiment, and
saves the resulting plots to `ctg_results/`.
"""

import warnings

from ctg_classifier import CTGClassifier

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("CTG CLASSIFIER WITH VISUALIZATIONS")
    print("=" * 60)

    classifier = CTGClassifier("cardiotocography_v2.csv")
    results = classifier.run_full_experiment(
        create_visualizations=True,
        output_dir="ctg_results/",
    )
