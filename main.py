import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class CTGClassifier:
    def __init__(self, data_path):
        """Inicjalizacja klasyfikatora CTG z obsługą wizualizacji"""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        
        self._setup_visualization_style()
    
    def _setup_visualization_style(self):
        """Konfiguracja stylu wizualizacji"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10

    
    def load_data(self):
        """Wczytanie i podstawowe przygotowanie danych"""
        print(" Wczytywanie danych...")
        self.data = pd.read_csv(self.data_path)
        
        self.X = self.data.drop('CLASS', axis=1)
        self.y = self.data['CLASS']
        
        print(f" Kształt danych: {self.data.shape}")
        print(f" Liczba klas: {self.y.nunique()}")
        print(f" Brakujące wartości: {self.data.isnull().sum().sum()}")

        print("\n Opis danych:")
        print(self.data.describe())

    def handle_missing_values(self, method='mean'):
        """Obsługa brakujących wartości"""
        print(f" Obsługa brakujących wartości metodą: {method}")
        
        self.plot_missing_values_analysis('.list4/ctg_results/missing_values_analysis.png')
        self.plot_class_distribution('.list4/ctg_results/plot_class_distribution.png')

        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        elif method == 'drop':
            mask = ~self.X.isnull().any(axis=1)
            self.X = self.X[mask]
            self.y = self.y[mask]
            return self.X, self.y
            
        self.X = pd.DataFrame(imputer.fit_transform(self.X), columns=self.X.columns)
        return self.X, self.y

    def preprocess_data(self, method='none'):
        """Przetwarzanie danych"""
        print(f" Przetwarzanie danych metodą: {method}")
        
        if method == 'standardize':
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(self.X)
            return pd.DataFrame(X_processed, columns=self.X.columns)
            
        elif method == 'normalize':
            scaler = MinMaxScaler()
            X_processed = scaler.fit_transform(self.X)
            return pd.DataFrame(X_processed, columns=self.X.columns)
            
        elif method == 'pca':
            pca = PCA(n_components=0.95)
            X_processed = pca.fit_transform(self.X)
            print(f" PCA: zredukowano z {self.X.shape[1]} do {X_processed.shape[1]} cech")
            return pd.DataFrame(X_processed)
            
        elif method == 'feature_selection':
            selector = SelectKBest(f_classif, k=15)
            X_processed = selector.fit_transform(self.X, self.y)
            selected_features = self.X.columns[selector.get_support()]
            print(f" Wybrane cechy: {list(selected_features)}")
            return pd.DataFrame(X_processed, columns=[f'feature_{i}' for i in range(X_processed.shape[1])])
            
        else:
            return self.X.copy()

    def split_data(self, test_size=0.2, random_state=42):
        """Podział danych na zbiór uczący i testowy"""
        print(f" Podział danych ({int((1-test_size)*100)}% trening, {int(test_size*100)}% test)")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )
        
        print(f" Rozmiar zbioru treningowego: {self.X_train.shape[0]}")
        print(f" Rozmiar zbioru testowego: {self.X_test.shape[0]}")

    def train_naive_bayes(self, X_train, y_train, var_smoothing_values=[1e-9, 1e-8, 1e-7]):
        """Trenowanie naiwnego klasyfikatora Bayesa z różnymi hiperparametrami"""
        print(" Trenowanie Naiwnego Klasyfikatora Bayesa...")
        
        nb_results = {}
        
        for vs in var_smoothing_values:
            print(f"    var_smoothing = {vs}")
            nb = GaussianNB(var_smoothing=vs)
            
            cv_scores = cross_val_score(nb, X_train, y_train, cv=5, scoring='accuracy')
            nb.fit(X_train, y_train)
            
            nb_results[f'NB_vs_{vs}'] = {
                'model': nb,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'hyperparams': {'var_smoothing': vs}
            }
            
            print(f"      CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return nb_results

    def train_decision_tree(self, X_train, y_train, 
                          max_depths=[None, 5, 10, 15],
                          min_samples_splits=[2, 5, 10],
                          criterions=['gini', 'entropy']):
        """Trenowanie drzewa decyzyjnego z różnymi hiperparametrami"""
        print(" Trenowanie Drzewa Decyzyjnego...")
        
        dt_results = {}
        
        for criterion in criterions:
            for max_depth in max_depths:
                for min_samples_split in min_samples_splits:
                    print(f"    criterion={criterion}, max_depth={max_depth}, min_samples_split={min_samples_split}")
                    
                    dt = DecisionTreeClassifier(
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42
                    )
                    
                    cv_scores = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy')
                    dt.fit(X_train, y_train)
                    
                    key = f'DT_{criterion}_{max_depth}_{min_samples_split}'
                    dt_results[key] = {
                        'model': dt,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'hyperparams': {
                            'criterion': criterion,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split
                        }
                    }
                    
                    print(f"      CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return dt_results

    def train_bonus_algorithms(self, X_train, y_train):
        """Trenowanie bonusowych algorytmów (Random Forest, SVM)"""
        print(" Trenowanie algorytmów bonusowych...")
        
        bonus_results = {}
        
        # Random Forest
        print("    Random Forest...")
        rf_params = [
            {'n_estimators': 50, 'max_depth': 10},
            {'n_estimators': 100, 'max_depth': 15},
            {'n_estimators': 200, 'max_depth': None}
        ]
        
        for i, params in enumerate(rf_params):
            rf = RandomForestClassifier(random_state=42, **params)
            cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
            rf.fit(X_train, y_train)
            
            bonus_results[f'RF_config_{i+1}'] = {
                'model': rf,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'hyperparams': params
            }
            
            print(f"      Config {i+1}: CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # SVM
        print("    Support Vector Machine...")
        svm_params = [
            {'C': 1.0, 'kernel': 'rbf'},
            {'C': 10.0, 'kernel': 'rbf'},
            {'C': 1.0, 'kernel': 'linear'}
        ]
        
        for i, params in enumerate(svm_params):
            svm = SVC(random_state=42, **params)
            cv_scores = cross_val_score(svm, X_train, y_train, cv=5, scoring='accuracy')
            svm.fit(X_train, y_train)
            
            bonus_results[f'SVM_config_{i+1}'] = {
                'model': svm,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'hyperparams': params
            }
            
            print(f"      Config {i+1}: CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return bonus_results

    def evaluate_models(self, models_dict, X_test, y_test):
        """Ocena modeli na zbiorze testowym"""
        print(" Ocena modeli na zbiorze testowym...")
        
        evaluation_results = {}
        
        for model_name, model_info in models_dict.items():
            model = model_info['model']
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            evaluation_results[model_name] = {
                'cv_accuracy': model_info['cv_mean'],
                'cv_std': model_info['cv_std'],
                'test_accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'hyperparams': model_info['hyperparams']
            }
            
            print(f"   {model_name}: Test Accuracy = {accuracy:.4f}")
        
        return evaluation_results

    def overfitting_prevention_experiment(self, X_train, y_train, X_test, y_test):
        """Eksperyment z przeciwdziałaniem przeuczeniu"""
        print(" Eksperyment przeciwdziałania przeuczeniu...")
        
        dt_overfit = DecisionTreeClassifier(random_state=42)
        dt_regulated = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        results = {}
        
        for name, model in [('DT_Overfit', dt_overfit), ('DT_Regulated', dt_regulated)]:
            model.fit(X_train, y_train)
            
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'cv_accuracy': cv_scores.mean(),
                'overfitting_gap': train_acc - test_acc
            }
            
            print(f"   {name}:")
            print(f"      Train Accuracy: {train_acc:.4f}")
            print(f"      Test Accuracy: {test_acc:.4f}")
            print(f"      CV Accuracy: {cv_scores.mean():.4f}")
            print(f"      Overfitting Gap: {train_acc - test_acc:.4f}")
        
        return results

    def create_results_summary(self, all_results):
        """Utworzenie podsumowania wyników"""
        print(" Tworzenie podsumowania wyników...")
        
        summary_data = []
        
        for model_name, metrics in all_results.items():
            summary_data.append({
                'Model': model_name,
                'CV_Accuracy': f"{metrics['cv_accuracy']:.4f} ± {metrics['cv_std']:.4f}",
                'Test_Accuracy': f"{metrics['test_accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1_Score': f"{metrics['f1_score']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test_Accuracy', ascending=False)
        
        print("\n" + "="*80)
        print("PODSUMOWANIE WYNIKÓW KLASYFIKACJI")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        
        return summary_df

    
    def _prepare_visualization_data(self):
        """Przygotowanie danych z wyników eksperymentu do wizualizacji"""
        if not hasattr(self, 'results') or not self.results:
            raise ValueError(" Najpierw uruchom eksperyment: run_full_experiment()")
        
        # Zbieranie najlepszych modeli z każdego eksperymentu
        top_models_data = []
        preprocessing_data = {'Algorithm': [], 'Raw': [], 'Standardized': [], 'PCA': []}
        
        # Analiza wyników dla różnych algorytmów
        algorithms = ['NB', 'DT', 'RF', 'SVM']
        
        for alg in algorithms:
            raw_scores = []
            std_scores = []
            pca_scores = []
            
            # Raw data results
            if 'raw' in self.results:
                raw_results = [v for k, v in self.results['raw'].items() if alg in k]
                if raw_results:
                    best_raw = max(raw_results, key=lambda x: x['test_accuracy'])
                    raw_scores.append(best_raw['test_accuracy'] * 100)
                else:
                    raw_scores.append(0)
            
            # Standardized results
            if 'standardized' in self.results:
                std_results = [v for k, v in self.results['standardized'].items() if alg in k]
                if std_results:
                    best_std = max(std_results, key=lambda x: x['test_accuracy'])
                    std_scores.append(best_std['test_accuracy'] * 100)
                else:
                    std_scores.append(0)
            
            # PCA results
            if 'pca' in self.results:
                pca_results = [v for k, v in self.results['pca'].items() if alg in k]
                if pca_results:
                    best_pca = max(pca_results, key=lambda x: x['test_accuracy'])
                    pca_scores.append(best_pca['test_accuracy'] * 100)
                else:
                    pca_scores.append(None)
            
            # Dodanie do preprocessing_data
            preprocessing_data['Algorithm'].append(alg)
            preprocessing_data['Raw'].append(raw_scores[0] if raw_scores else 0)
            preprocessing_data['Standardized'].append(std_scores[0] if std_scores else 0)
            preprocessing_data['PCA'].append(pca_scores[0] if pca_scores else None)
        
        # Top 6 modeli ogólnie
        all_models = []
        for exp_name in ['raw', 'standardized', 'pca']:
            if exp_name in self.results:
                for model_name, metrics in self.results[exp_name].items():
                    all_models.append({
                        'Model': f"{exp_name.upper()}_{model_name}",
                        'Test_Accuracy': metrics['test_accuracy'] * 100,
                        'CV_Accuracy': metrics['cv_accuracy'] * 100,
                        'CV_Std': metrics['cv_std'] * 100
                    })
        
        # Sortowanie i wybór top 6
        all_models.sort(key=lambda x: x['Test_Accuracy'], reverse=True)
        top_6 = all_models[:6]
        
        top_models_data = {
            'Model': [m['Model'] for m in top_6],
            'Test_Accuracy': [m['Test_Accuracy'] for m in top_6],
            'CV_Accuracy': [m['CV_Accuracy'] for m in top_6],
            'CV_Std': [m['CV_Std'] for m in top_6]
        }
        
        # Dane przeuczenia
        overfitting_data = {
            'Model': ['DT_Overfit', 'DT_Regulated'],
            'Train_Accuracy': [100.0, 75.65],
            'Test_Accuracy': [60.09, 60.56],
            'CV_Accuracy': [60.41, 58.88],
            'Overfitting_Gap': [39.91, 15.08]
        }
        
        if 'overfitting' in self.results:
            overfitting_data['Model'] = list(self.results['overfitting'].keys())
            overfitting_data['Train_Accuracy'] = [v['train_accuracy'] * 100 for v in self.results['overfitting'].values()]
            overfitting_data['Test_Accuracy'] = [v['test_accuracy'] * 100 for v in self.results['overfitting'].values()]
            overfitting_data['CV_Accuracy'] = [v['cv_accuracy'] * 100 for v in self.results['overfitting'].values()]
            overfitting_data['Overfitting_Gap'] = [v['overfitting_gap'] * 100 for v in self.results['overfitting'].values()]
        
        return {
            'top_models': top_models_data,
            'preprocessing_comparison': preprocessing_data,
            'overfitting_analysis': overfitting_data
        }

    def plot_top_models_comparison(self, save_path=None):
        """Wykres porównania najlepszych modeli"""
        print(" Generowanie wykresu najlepszych modeli...")
        
        viz_data = self._prepare_visualization_data()
        data = viz_data['top_models']
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Kolory dla różnych typów modeli
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        # Wykres słupkowy z error bars
        bars = ax.bar(data['Model'], data['Test_Accuracy'], 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Dodanie error bars dla CV
        ax.errorbar(data['Model'], data['CV_Accuracy'], yerr=data['CV_Std'], 
                   fmt='o', color='red', capsize=5, capthick=2, 
                   label='CV Accuracy ± std', markersize=8)
        
        # Dodanie wartości na słupkach
        for bar, acc in zip(bars, data['Test_Accuracy']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title(' Porównanie najlepszych modeli klasyfikacji CTG\n(Test Accuracy vs CV Accuracy)', 
                    fontweight='bold', pad=20)
        ax.set_ylim(0, max(data['Test_Accuracy']) + 10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_preprocessing_impact(self, save_path=None):
        """Wykres wpływu przetwarzania danych"""
        print(" Generowanie wykresu wpływu przetwarzania...")
        
        viz_data = self._prepare_visualization_data()
        data = viz_data['preprocessing_comparison']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(data['Algorithm']))
        width = 0.25
        
        # Trzy serie słupków
        bars1 = ax.bar(x - width, data['Raw'], width, label='Dane surowe', 
                      color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x, data['Standardized'], width, label='Standaryzacja', 
                      color='#4ECDC4', alpha=0.8)
        
        # PCA może mieć None values
        pca_values = [v if v is not None else 0 for v in data['PCA']]
        bars3 = ax.bar(x + width, pca_values, width, label='PCA', 
                      color='#FFA07A', alpha=0.8)
        
        # Dodawanie wartości na słupkach
        def add_value_labels(bars, values):
            for bar, val in zip(bars, values):
                if val is not None and val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        add_value_labels(bars1, data['Raw'])
        add_value_labels(bars2, data['Standardized'])
        add_value_labels(bars3, data['PCA'])
        
        ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
        ax.set_xlabel('Algorytm', fontweight='bold')
        ax.set_title(' Wpływ przetwarzania danych na wydajność algorytmów', 
                    fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(data['Algorithm'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 90)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_overfitting_analysis(self, save_path=None):
        """Analiza przeuczenia"""
        print(" Generowanie analizy przeuczenia...")
        
        viz_data = self._prepare_visualization_data()
        data = viz_data['overfitting_analysis']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Wykres 1: Train vs Test
        x = np.arange(len(data['Model']))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, data['Train_Accuracy'], width, 
                       label='Train Accuracy', color='#FF6B6B', alpha=0.8)
        bars2 = ax1.bar(x + width/2, data['Test_Accuracy'], width, 
                       label='Test Accuracy', color='#4ECDC4', alpha=0.8)
        
        # Dodanie wartości
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title(' Analiza przeuczenia\nTrain vs Test Accuracy', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Bez regularyzacji', 'Z regularyzacją'])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 110)
        
        # Wykres 2: Gap przeuczenia
        colors = ['#FF4757', '#2ED573']
        bars = ax2.bar(data['Model'], data['Overfitting_Gap'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, gap in zip(bars, data['Overfitting_Gap']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{gap:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_ylabel('Gap przeuczenia (%)', fontweight='bold')
        ax2.set_title(' Gap przeuczenia\n(Train - Test Accuracy)', fontweight='bold')
        ax2.set_xticklabels(['Bez regularyzacji', 'Z regularyzacją'])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Linia ostrzeżenia
        ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.7, 
                   label='Próg ostrzeżenia (20%)')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_missing_values_analysis(self, save_path=None):
        """Analiza brakujących wartości"""
        print(" Generowanie analizy brakujących wartości...")
        
        # Oblicz brakujące wartości dla każdej cechy
        missing_counts = self.data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]  # Tylko cechy z brakującymi wartościami
        
        if len(missing_counts) == 0:
            print(" Brak brakujących wartości w danych!")
            return
        
        # Oblicz procent brakujących wartości
        total_samples = len(self.data)
        missing_percent = (missing_counts / total_samples) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Wykres 1: Liczba brakujących wartości
        colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(missing_counts)))
        bars1 = ax1.bar(range(len(missing_counts)), missing_counts.values, 
                        color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Dodanie wartości na słupkach
        for bar, count in zip(bars1, missing_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(count)}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Cechy', fontweight='bold')
        ax1.set_ylabel('Liczba brakujących wartości', fontweight='bold')
        ax1.set_title(' Liczba brakujących wartości w każdej cesze', fontweight='bold', pad=15)
        ax1.set_xticks(range(len(missing_counts)))
        ax1.set_xticklabels(missing_counts.index, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Wykres 2: Procent brakujących wartości
        colors2 = plt.cm.Oranges(np.linspace(0.4, 0.8, len(missing_percent)))
        bars2 = ax2.bar(range(len(missing_percent)), missing_percent.values, 
                        color=colors2, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Dodanie wartości na słupkach
        for bar, percent in zip(bars2, missing_percent.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{percent:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Cechy', fontweight='bold')
        ax2.set_ylabel('Procent brakujących wartości (%)', fontweight='bold')
        ax2.set_title(' Procent brakujących wartości w każdej cesze', fontweight='bold', pad=15)
        ax2.set_xticks(range(len(missing_percent)))
        ax2.set_xticklabels(missing_percent.index, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Linia ostrzeżenia na 10%
        ax2.axhline(y=10, color='red', linestyle='--', alpha=0.7, 
                label='Próg ostrzeżenia (10%)')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Wydrukowanie statystyk
        print(f"\n STATYSTYKI BRAKUJĄCYCH WARTOŚCI:")
        print(f"   Całkowita liczba próbek: {total_samples}")
        print(f"   Cech z brakującymi wartościami: {len(missing_counts)}")
        print(f"   Całkowita liczba brakujących wartości: {missing_counts.sum()}")
        print(f"   Średni procent brakujących wartości: {missing_percent.mean():.2f}%")
        print(f"   Maksymalny procent brakujących wartości: {missing_percent.max():.2f}% ({missing_percent.idxmax()})")


    def plot_class_distribution(self, save_path=None):
        """Analiza rozkładu klas"""
        print(" Generowanie analizy rozkładu klas...")
        
        # Oblicz rozkład klas
        class_counts = self.y.value_counts().sort_index()
        class_percentages = (class_counts / len(self.y)) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Wykres 1: Liczba próbek w każdej klasie (wykres słupkowy)
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        bars1 = ax1.bar(class_counts.index, class_counts.values, 
                        color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Dodanie wartości na słupkach
        for bar, count in zip(bars1, class_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(count)}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Klasa', fontweight='bold')
        ax1.set_ylabel('Liczba próbek', fontweight='bold')
        ax1.set_title(' Rozkład liczby próbek w każdej klasie', fontweight='bold', pad=15)
        ax1.set_xticks(class_counts.index)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Średnia linia
        mean_count = class_counts.mean()
        ax1.axhline(y=mean_count, color='red', linestyle='--', alpha=0.7, 
                label=f'Średnia: {mean_count:.1f}')
        ax1.legend()
        
        # Wykres 2: Rozkład procentowy (wykres kołowy)
        wedges, texts, autotexts = ax2.pie(class_percentages.values, 
                                        labels=[f'Klasa {i}' for i in class_counts.index],
                                        autopct='%1.1f%%', 
                                        colors=colors, 
                                        startangle=90,
                                        explode=[0.05 if count < mean_count else 0 for count in class_counts.values])
        
        # Ustawienie czcionki dla procentów
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax2.set_title(' Rozkład procentowy klas', fontweight='bold', pad=15)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Wydrukowanie szczegółowych statystyk
        print(f"\n STATYSTYKI ROZKŁADU KLAS:")
        print(f"   Całkowita liczba próbek: {len(self.y)}")
        print(f"   Liczba klas: {len(class_counts)}")
        print(f"   Klasy: {list(class_counts.index)}")
        
        print(f"\n SZCZEGÓŁOWY ROZKŁAD:")
        for class_id in class_counts.index:
            count = class_counts[class_id]
            percentage = class_percentages[class_id]
            print(f"   Klasa {class_id}: {count:3d} próbek ({percentage:5.2f}%)")
        
        print(f"\n STATYSTYKI BALANSOWANIA:")
        print(f"   Najmniej próbek: {class_counts.min()} (Klasa {class_counts.idxmin()})")
        print(f"   Najwięcej próbek: {class_counts.max()} (Klasa {class_counts.idxmax()})")
        print(f"   Średnia: {class_counts.mean():.1f}")
        print(f"   Odchylenie standardowe: {class_counts.std():.1f}")
        
        # Ocena balansowania
        ratio = class_counts.max() / class_counts.min()
        if ratio <= 2:
            balance_status = " Dobrze zbalansowany"
        elif ratio <= 5:
            balance_status = " Umiarkowanie niezbalansowany"
        else:
            balance_status = " Mocno niezbalansowany"
        
        print(f"   Stosunek max/min: {ratio:.2f} - {balance_status}")


    def create_visualization_report(self, save_dir='./ctg_visualizations/'):
        """Tworzenie kompletnego raportu wizualnego"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(" Generowanie kompletnego raportu wizualnego...")
        print("=" * 50)
        
        try:
            # Sprawdzenie czy są wyniki
            if not hasattr(self, 'results') or not self.results:
                print(" Brak wyników! Uruchom najpierw: run_full_experiment()")
                return
            
            print(" 1. Porównanie najlepszych modeli...")
            self.plot_top_models_comparison(f'{save_dir}01_top_models.png')
            
            print(" 2. Analiza przetwarzania danych...")
            self.plot_preprocessing_impact(f'{save_dir}02_preprocessing_impact.png')
            
            print(" 3. Analiza przeuczenia...")
            self.plot_overfitting_analysis(f'{save_dir}03_overfitting_analysis.png')
            
            print(f"\n Raport wygenerowany w folderze: {save_dir}")
            print(" Pliki:")
            for i, name in enumerate([
                "01_top_models.png",
                "02_preprocessing_impact.png", 
                "03_overfitting_analysis.png"
            ], 1):
                print(f"   {i}. {name}")
            
        except Exception as e:
            print(f" Błąd podczas generowania wizualizacji: {e}")
            print(" Upewnij się, że zainstalowane są: matplotlib, seaborn")
    

    def run_full_experiment(self, create_visualizations=True, save_viz_dir=None):
        """Uruchomienie pełnego eksperymentu z opcjonalnymi wizualizacjami"""
        print(" ROZPOCZĘCIE PEŁNEGO EKSPERYMENTU KLASYFIKACJI CTG")
        print("="*60)
        
        # 1. Wczytanie danych
        self.load_data()
        
        # 2. Obsługa brakujących wartości
        self.handle_missing_values(method='mean')
        
        # 3. Eksperyment bez przetwarzania danych
        print("\n EKSPERYMENT 1: BEZ PRZETWARZANIA DANYCH")
        print("-"*50)
        X_raw = self.preprocess_data(method='none')
        self.X = X_raw
        self.split_data()
        
        # Trenowanie modeli
        nb_results_raw = self.train_naive_bayes(self.X_train, self.y_train)
        dt_results_raw = self.train_decision_tree(self.X_train, self.y_train)
        bonus_results_raw = self.train_bonus_algorithms(self.X_train, self.y_train)
        
        # Ocena modeli
        all_models_raw = {**nb_results_raw, **dt_results_raw, **bonus_results_raw}
        evaluation_raw = self.evaluate_models(all_models_raw, self.X_test, self.y_test)
        
        # 4. Eksperyment ze standaryzacją
        print("\n EKSPERYMENT 2: ZE STANDARYZACJĄ")
        print("-"*50)
        X_std = self.preprocess_data(method='standardize')
        self.X = X_std
        self.split_data()
        
        nb_results_std = self.train_naive_bayes(self.X_train, self.y_train)
        dt_results_std = self.train_decision_tree(self.X_train, self.y_train)
        bonus_results_std = self.train_bonus_algorithms(self.X_train, self.y_train)
        
        all_models_std = {**nb_results_std, **dt_results_std, **bonus_results_std}
        evaluation_std = self.evaluate_models(all_models_std, self.X_test, self.y_test)
        
        # 5. Eksperyment z PCA
        print("\n EKSPERYMENT 3: Z PCA")
        print("-"*50)
        self.X = self.preprocess_data(method='none')
        X_pca = self.preprocess_data(method='pca')
        self.X = X_pca
        self.split_data()
        
        nb_results_pca = self.train_naive_bayes(self.X_train, self.y_train)
        dt_results_pca = self.train_decision_tree(self.X_train, self.y_train)
        
        all_models_pca = {**nb_results_pca, **dt_results_pca}
        evaluation_pca = self.evaluate_models(all_models_pca, self.X_test, self.y_test)
        
        # 6. Eksperyment przeciwdziałania przeuczeniu
        print("\n EKSPERYMENT BONUSOWY: PRZECIWDZIAŁANIE PRZEUCZENIU")
        print("-"*50)
        self.X = self.preprocess_data(method='none')
        self.split_data()
        overfitting_results = self.overfitting_prevention_experiment(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        # 7. Podsumowanie wszystkich wyników
        print("\n PODSUMOWANIE WSZYSTKICH EKSPERYMENTÓW")
        print("="*60)
        
        evaluation_raw_prefixed = {f"RAW_{k}": v for k, v in evaluation_raw.items()}
        evaluation_std_prefixed = {f"STD_{k}": v for k, v in evaluation_std.items()}
        evaluation_pca_prefixed = {f"PCA_{k}": v for k, v in evaluation_pca.items()}
        
        all_evaluations = {**evaluation_raw_prefixed, **evaluation_std_prefixed, **evaluation_pca_prefixed}
        summary_df = self.create_results_summary(all_evaluations)
        
        # Zapisanie wyników
        self.results = {
            'raw': evaluation_raw,
            'standardized': evaluation_std,
            'pca': evaluation_pca,
            'overfitting': overfitting_results,
            'summary': summary_df
        }
        
        # 8. NOWE: Tworzenie wizualizacji
        if create_visualizations:
            print("\n TWORZENIE WIZUALIZACJI...")
            print("-"*50)
            try:
                if save_viz_dir:
                    self.create_visualization_report(save_viz_dir)
                else:
                    print(" Generowanie wykresów na ekranie...")
                    self.plot_top_models_comparison()
                    self.plot_preprocessing_impact()
                    self.plot_overfitting_analysis()
                    
            except Exception as e:
                print(f" Nie udało się utworzyć wizualizacji: {e}")
                print(" Wizualizacje są opcjonalne, wyniki eksperymentu są gotowe!")
        
        print("\n EKSPERYMENT ZAKOŃCZONY POMYŚLNIE!")
        
        # Podsumowanie najlepszych wyników
        self._print_experiment_summary()
        
        return self.results
    
    def _print_experiment_summary(self):
        """Wydrukowanie podsumowania najlepszych wyników"""
        print("\n" + "=" * 20)
        print("NAJLEPSZE WYNIKI EKSPERYMENTU")
        print("=" * 20)
        
        # Znajdź najlepszy model ogólnie
        best_model = None
        best_score = 0
        
        for exp_name, exp_results in self.results.items():
            if exp_name in ['raw', 'standardized', 'pca']:
                for model_name, metrics in exp_results.items():
                    if metrics['test_accuracy'] > best_score:
                        best_score = metrics['test_accuracy']
                        best_model = f"{exp_name.upper()}_{model_name}"
        
        print(f" NAJLEPSZY MODEL: {best_model}")
        print(f" DOKŁADNOŚĆ: {best_score:.4f} ({best_score*100:.2f}%)")
        
        # Najlepszy z każdej kategorii
        print(f"\n NAJLEPSZE WYNIKI WG KATEGORII:")
        for exp_name in ['raw', 'standardized', 'pca']:
            if exp_name in self.results:
                exp_results = self.results[exp_name]
                if exp_results:
                    best_in_category = max(exp_results.items(), key=lambda x: x[1]['test_accuracy'])
                    model_name, metrics = best_in_category
                    print(f"   {exp_name.upper()}: {model_name} = {metrics['test_accuracy']*100:.2f}%")
        
        # Analiza przeuczenia
        if 'overfitting' in self.results:
            print(f"\n ANALIZA PRZEUCZENIA:")
            for model_name, metrics in self.results['overfitting'].items():
                gap = metrics['overfitting_gap']
                status = " Dobry" if gap < 0.20 else " Przeuczenie"
                print(f"   {model_name}: Gap = {gap*100:.1f}% ({status})")


if __name__ == "__main__":
    print(" KLASYFIKATOR CTG Z WIZUALIZACJAMI")
    print("="*60)
    
    classifier = CTGClassifier('list4/cardiotocography_v2.csv')
    results = classifier.run_full_experiment(
        create_visualizations=True,        
        save_viz_dir='.list4/ctg_results/' 
    )