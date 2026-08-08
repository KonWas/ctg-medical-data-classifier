"""CTGClassifier: orchestrates the full data loading -> preprocessing ->
training -> evaluation -> reporting -> visualization pipeline.
"""

from . import data_processing as dp
from . import preprocessing as pp
from . import models as md
from . import reporting as rp
from . import visualization as viz


class CTGClassifier:
    """End-to-end CTG classification pipeline.

    The dataset is imputed once into a fixed baseline (`self.X_imputed`);
    every preprocessing variant (raw / standardized / PCA) is derived from
    that same baseline so the three experiments stay comparable and none of
    them accidentally inherits another experiment's transformed data.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_imputed = None
        self.y_imputed = None
        self.results = {}

        viz.configure_plot_style()

    def load_data(self):
        self.data, self.X, self.y = dp.load_data(self.data_path)

    def run_eda(self, output_dir=None):
        """Visualize missing values and class distribution before imputation."""
        missing_path = f"{output_dir}missing_values_analysis.png" if output_dir else None
        class_dist_path = f"{output_dir}class_distribution.png" if output_dir else None

        viz.plot_missing_values_analysis(self.data, missing_path)
        viz.plot_class_distribution(self.y, class_dist_path)

    def handle_missing_values(self, method="mean"):
        self.X_imputed, self.y_imputed = dp.impute_missing_values(self.X, self.y, method=method)

    def _train_and_evaluate(self, X, y, include_bonus_models=True):
        """Split, train the full model set, and evaluate on the test split."""
        X_train, X_test, y_train, y_test = dp.split_data(X, y)

        trained = {}
        trained.update(md.train_naive_bayes(X_train, y_train))
        trained.update(md.train_decision_tree(X_train, y_train))
        if include_bonus_models:
            trained.update(md.train_bonus_models(X_train, y_train))

        return md.evaluate_models(trained, X_test, y_test)

    def run_full_experiment(self, create_visualizations=True, output_dir="ctg_results/"):
        print("STARTING FULL CTG CLASSIFICATION EXPERIMENT")
        print("=" * 60)

        self.load_data()
        self.run_eda(output_dir)
        self.handle_missing_values(method="mean")

        print("\nEXPERIMENT 1: RAW DATA")
        print("-" * 50)
        evaluation_raw = self._train_and_evaluate(self.X_imputed, self.y_imputed)

        print("\nEXPERIMENT 2: STANDARDIZED DATA")
        print("-" * 50)
        X_std = pp.standardize_features(self.X_imputed)
        evaluation_std = self._train_and_evaluate(X_std, self.y_imputed)

        print("\nEXPERIMENT 3: PCA")
        print("-" * 50)
        X_pca = pp.apply_pca(self.X_imputed)
        evaluation_pca = self._train_and_evaluate(X_pca, self.y_imputed, include_bonus_models=False)

        print("\nBONUS EXPERIMENT: OVERFITTING PREVENTION")
        print("-" * 50)
        X_train, X_test, y_train, y_test = dp.split_data(self.X_imputed, self.y_imputed)
        overfitting_results = md.run_overfitting_experiment(X_train, y_train, X_test, y_test)

        print("\nSUMMARY OF ALL EXPERIMENTS")
        print("=" * 60)
        all_evaluations = {
            **{f"RAW_{k}": v for k, v in evaluation_raw.items()},
            **{f"STD_{k}": v for k, v in evaluation_std.items()},
            **{f"PCA_{k}": v for k, v in evaluation_pca.items()},
        }
        summary_df = rp.build_results_summary(all_evaluations)

        self.results = {
            "raw": evaluation_raw,
            "standardized": evaluation_std,
            "pca": evaluation_pca,
            "overfitting": overfitting_results,
            "summary": summary_df,
        }

        if create_visualizations:
            print("\nCREATING VISUALIZATIONS...")
            print("-" * 50)
            try:
                if output_dir:
                    viz.create_visualization_report(self.results, output_dir)
                else:
                    viz.plot_top_models_comparison(self.results)
                    viz.plot_preprocessing_impact(self.results)
                    viz.plot_overfitting_analysis(self.results)
            except Exception as e:
                print(f"Failed to create visualizations: {e}")
                print("Visualizations are optional; the experiment results are still available.")

        print("\nEXPERIMENT COMPLETED SUCCESSFULLY!")
        rp.print_best_results(self.results)

        return self.results
