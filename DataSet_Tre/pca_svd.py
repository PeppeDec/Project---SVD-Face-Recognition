from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from config import N_COMPONENTS


@dataclass
class PCABasedProjector:
    """
    Estrazione feature con PCA (eigenfaces):
    - centriamo i dati
    - calcoliamo i primi N_COMPONENTS autovettori (eigenfaces)
    - proiettiamo ogni immagine nello spazio delle componenti principali
    """
    n_components: int = N_COMPONENTS
    whiten: bool = True

    def __post_init__(self):
        self.pca: Optional[PCA] = None

    def fit(self, X_train: np.ndarray) -> "PCABasedProjector":
        """
        Adatta la PCA sui dati di training.
        """
        self.pca = PCA(
            n_components=self.n_components,
            svd_solver="randomized",
            whiten=self.whiten,
            random_state=0
        )
        self.pca.fit(X_train)
        # Nel metodo fit() di PCABasedProjector
        explained_var = np.cumsum(self.pca.explained_variance_ratio_)
        print(f"[INFO] Varianza spiegata con {self.n_components} componenti: {explained_var[-1]:.2%}")

        print(f"[INFO] PCA addestrata con {self.n_components} componenti.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Proietta i dati nello spazio PCA.
        """
        if self.pca is None:
            raise RuntimeError("PCA non è stata ancora addestrata. Chiama `.fit()` prima di `.transform()`.")
        return self.pca.transform(X)


@dataclass
class SVDBasedProjector:
    """
    Versione "manuale" con SVD:
    X_centered = X - mean
    U, S, Vt = svd(X_centered)
    Le colonne di V (righe di Vt) sono gli autovettori (eigenfaces).
    Proiezione: (X - mean) @ V_k^T
    """
    n_components: int = N_COMPONENTS

    def __post_init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None  # shape: (n_components, n_features)

    def fit(self, X_train: np.ndarray) -> "SVDBasedProjector":
        # Calcolo della media
        self.mean_ = X_train.mean(axis=0)

        # Centro i dati
        X_centered = X_train - self.mean_

        # SVD "full_matrices=False" per usare solo le parti utili
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Prendo i primi n_components autovettori (righe di Vt)
        self.components_ = Vt[: self.n_components, :]

        print(f"[INFO] SVD calcolata, tenuti {self.n_components} autovettori.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.components_ is None:
            raise RuntimeError("SVD non è stata ancora calcolata. Chiama `.fit()` prima di `.transform()`.")

        X_centered = X - self.mean_
        # Proiezione nello spazio delle componenti
        return X_centered @ self.components_.T

    # Piccolo fix: property per non dimenticare underscore
    @property
    def mean(self) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("La media non è stata inizializzata (chiama fit).")
        return self.mean_
