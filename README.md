# CTG Medical Data Classifier

A Machine Learning pipeline that classifies Cardiotocography (CTG) recordings into their 10 morphological fetal heart rate patterns. The project covers data preprocessing, handling missing clinical values, and benchmarking several classic ML algorithms while explicitly measuring overfitting.

## 📁 Project Structure
```
.
├── main.py                       # Entry point: runs the full experiment
├── cardiotocography_v2.csv       # Dataset (2126 samples, 21 features, target: CLASS)
├── ctg_classifier/
│   ├── data_processing.py        # Loading, missing-value imputation, train/test split
│   ├── preprocessing.py          # Standardization, normalization, PCA, feature selection
│   ├── models.py                 # Model training, evaluation, overfitting experiment
│   ├── reporting.py               # Text summaries of results
│   ├── visualization.py          # All plots (EDA + experiment comparison charts)
│   └── pipeline.py               # CTGClassifier: orchestrates the steps above
└── ctg_results/                  # Generated plots (created by running main.py)
```

Run it with:
```bash
pip install -r requirements.txt
python main.py
```

## 🚀 Pipeline
1. **Load data** and report shape, class count, and missing values.
2. **Exploratory plots**: missing-value counts/percentages per feature, class distribution.
3. **Impute missing values** (mean imputation by default; median/KNN/drop are also available in `preprocessing`/`data_processing` for experimentation).
4. **Three parallel preprocessing experiments**, each trained and evaluated independently from the same imputed baseline:
   * **Raw** — imputed features, no scaling.
   * **Standardized** — `StandardScaler`.
   * **PCA** — dimensionality reduction (95% explained variance) applied directly to the raw imputed features, kept independent of the standardized branch so the three experiments are a clean side-by-side comparison of preprocessing choices, not a chain of transformations.
5. **Models**: Gaussian Naive Bayes and Decision Tree (grid-searched) on all three variants; Random Forest and SVM (bonus, small config grid) on the raw and standardized variants only, to keep runtime reasonable.
6. **Overfitting experiment**: an unconstrained Decision Tree vs. a regularized one (`max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10`), trained on the raw imputed features, comparing train/test accuracy gaps.
7. **Reporting**: a sorted results table across all 50+ trained models, plus a best-model-per-category summary.
8. **Visualizations**, saved to `ctg_results/`: top-6 models, preprocessing impact per algorithm, and overfitting analysis.

## 📊 Dataset notes
* 10-class target (`CLASS` 1–10, fetal heart rate morphology patterns), **imbalanced**: from 53 samples (class 3) to 579 samples (class 2), a ~11x ratio. `plot_class_distribution` flags this as "heavily imbalanced" — worth keeping in mind since accuracy alone can be a misleading metric here; precision/recall/F1 (weighted) are reported alongside it for that reason.
* Every feature has ~4-6% missing values, uniformly — consistent with values having been randomly masked for this exercise rather than a real-world missingness pattern.
* PCA on the raw (unstandardized) features is dominated by the highest-scale columns (first component explains ~57% of variance) and reduces to 6 components; PCA on standardized data spreads variance more evenly across ~15 components. The pipeline intentionally applies PCA to the raw data to compare it on equal footing with the other two branches — in a production setting you'd typically standardize before PCA.

## 📈 Example results
Running the pipeline end-to-end typically finds:
* **Best model**: Random Forest (200 trees) on raw or standardized data, ~84-85% test accuracy — scaling makes almost no difference to tree ensembles, as expected.
* **PCA branch**: consistently the weakest (~48-59% test accuracy), the cost of compressing 21 features into 6 raw-scale-dominated components.
* **Overfitting experiment**: the unconstrained Decision Tree reaches 100% train accuracy with a ~24% train/test gap; the regularized tree closes that gap to under 5% for a small drop in raw accuracy.

Exact numbers vary slightly between environments (they're driven by `random_state=42`, but library versions can shift floating-point results marginally).

## 🛠️ Tech Stack
* Python 3
* scikit-learn
* pandas, NumPy
* Matplotlib, Seaborn
