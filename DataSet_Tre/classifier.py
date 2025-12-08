from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import seaborn as sns

class SVMFaceClassifier:
    """
    Classificatore SVM per face recognition sulle feature PCA/SVD.
    Di default kernel RBF, ma puoi passare kernel="linear" per
    una versione più classica stile eigenfaces.
    """

    def __init__(self, C: float = 10.0, kernel: str = "rbf", gamma: str = "scale"):
        self.model = SVC(C=C, kernel=kernel, gamma=gamma)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SVMFaceClassifier":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 class_names: List[str]) -> float:
        """
        Stampa metriche e ritorna l'accuracy.
        """
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("\n=== RISULTATI TEST ===")
        print(f"Accuracy: {acc:.4f}\n")

        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        return acc
