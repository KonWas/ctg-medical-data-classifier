# CTG Medical Data Classifier

A Machine Learning pipeline designed to classify Cardiotocography (CTG) data to monitor fetal health. The project focuses on data preprocessing, feature engineering, handling missing data, and evaluating multiple classic ML algorithms to find the most robust model while mitigating overfitting.

## 🚀 Features & Pipeline
* **Data Preprocessing & Imputation:** Implemented `SimpleImputer` and `KNNImputer` to robustly handle missing clinical values.
* **Feature Engineering:** Compares performance across raw data, standardized data (`StandardScaler`, `MinMaxScaler`), and dimensionality reduction using **PCA (Principal Component Analysis)**.
* **Feature Selection:** Automated feature selection using ANOVA F-value (`SelectKBest`).
* **Model Benchmarking:** Evaluates and compares multiple classifiers:
  * Gaussian Naive Bayes
  * Decision Tree
  * Random Forest
  * Support Vector Machine (SVM)
* **Overfitting Analysis:** Monitors the gap between training and testing metrics to ensure generalization.

## 📊 Evaluation Metrics
The models are evaluated based on:
* Accuracy
* Precision
* Recall
* F1-Score

## 🛠️ Tech Stack
* Python 3
* Scikit-Learn
* Pandas, NumPy
* Matplotlib, Seaborn (Advanced experiment visualization)

## 📈 Results
The script automatically generates performance plots for each experimental setup (raw vs. standardized vs. PCA) and outputs the best-performing model architecture alongside an overfitting risk assessment.