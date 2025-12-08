import json
from typing import List, Dict
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import seaborn as sns


class BaseClassifier:
    """Classe base per tutti i classificatori"""

    def __init__(self, name: str):
        self.name = name
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 class_names: List[str]) -> Dict[str, float]:
        """Valuta il modello e ritorna le metriche"""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n=== {self.name} - RISULTATI TEST ===")
        print(f"Accuracy: {acc:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names,zero_division=0))

        return {
            'accuracy': acc,
            'y_pred': y_pred,
            'y_test': y_test
        }


class SVMFaceClassifier(BaseClassifier):
    """SVM con kernel RBF o lineare"""

    def __init__(self, C: float = 10.0, kernel: str = "rbf", gamma: str = "scale"):
        super().__init__(name=f"SVM ({kernel})")
        self.model = SVC(C=C, kernel=kernel, gamma=gamma)


class KNNFaceClassifier(BaseClassifier):
    """K-Nearest Neighbors"""

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        super().__init__(name=f"KNN (k={n_neighbors})")
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=-1
        )


class RandomForestFaceClassifier(BaseClassifier):
    """Random Forest"""

    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        super().__init__(name="Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )


class LogisticRegressionFaceClassifier(BaseClassifier):
    """Logistic Regression"""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        super().__init__(name="Logistic Regression")
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=42,
            n_jobs=-1
        )


class NaiveBayesFaceClassifier(BaseClassifier):
    """Naive Bayes Gaussiano"""

    def __init__(self):
        super().__init__(name="Naive Bayes")
        self.model = GaussianNB()


def compare_classifiers(classifiers: List[BaseClassifier],
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        class_names: List[str]) -> Dict:
    """
    Confronta più classificatori e genera grafici comparativi.

    Returns:
        Dict con risultati di ogni classificatore
    """
    results = {}

    for clf in classifiers:
        print(f"\n{'=' * 60}")
        print(f"Training {clf.name}...")
        print('=' * 60)

        clf.fit(X_train, y_train)
        metrics = clf.evaluate(X_test, y_test, class_names)

        results[clf.name] = {
            'accuracy': metrics['accuracy'],
            'classifier': clf,
            'y_pred': metrics['y_pred'],
            'y_test': metrics['y_test']
        }

    # Grafico comparativo delle accuracy
    _plot_accuracy_comparison(results)

    # Salva risultati in JSON
    _save_results(results)

    return results


def _plot_accuracy_comparison(results: Dict):
    """Genera grafico a barre delle accuracy"""
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, accuracies, color='steelblue', edgecolor='black')

    # Aggiungi valori sopra le barre
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{acc:.3f}',
                 ha='center', va='bottom', fontsize=10)

    plt.xlabel('Classificatore', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Confronto Accuracy tra Classificatori', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('classifiers_comparison.png', dpi=300)
    print("\n[INFO] Grafico salvato in 'classifiers_comparison.png'")


def _save_results(results: Dict):
    """Salva risultati in formato JSON"""
    summary = {
        name: {
            'accuracy': float(data['accuracy']),
        }
        for name, data in results.items()
    }

    with open('classifiers_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("[INFO] Risultati salvati in 'classifiers_results.json'")
