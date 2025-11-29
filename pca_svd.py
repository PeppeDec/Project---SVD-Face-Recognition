# pca_svd.py
from dataclasses import dataclass
import numpy as np



@dataclass
class PCAModel:
    """Semplice modello PCA fatto con SVD."""
    mean_: np.ndarray                    # (n_features,)
    components_: np.ndarray              # (n_components, n_features)
    singular_values_: np.ndarray         # (n_components,)
    explained_variance_: np.ndarray      # (n_components,)
    explained_variance_ratio_: np.ndarray  # (n_components,)


def fit_pca_svd(X: np.ndarray, n_components: int | None = None) -> PCAModel:
    """
    Esegue PCA su X usando SVD.

    Parametri
    ---------
    X : array (n_samples, n_features)
        Matrice dei dati (ogni riga = immagine flattenata).
    n_components : int o None
        Numero di componenti da tenere.
        Se None, tiene il massimo possibile (min(n_samples, n_features)).

    Ritorna
    -------
    PCAModel
    """
    # 1. Calcolo della media (mean face) e centratura dei dati
    mean = X.mean(axis=0)
    X_centered = X - mean

    n_samples, n_features = X_centered.shape

    # 2. SVD compatta
    # X_centered = U Σ V^T
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 3. Se richiesto, taglio al numero di componenti desiderato
    max_components = Vt.shape[0]  # = min(n_samples, n_features)
    if n_components is None or n_components > max_components:
        n_components = max_components

    S = S[:n_components]
    Vt = Vt[:n_components, :]

    # 4. Varianza spiegata da ogni componente
    # λ_i = S_i^2 / (n_samples - 1)
    explained_variance = (S ** 2) / (n_samples - 1)

    # Varianza totale (somma varianze per feature)
    total_var = np.sum(np.var(X_centered, axis=0, ddof=1))
    explained_variance_ratio = explained_variance / total_var

    return PCAModel(
        mean_=mean,
        components_=Vt,               # ogni riga è un componente principale
        singular_values_=S,
        explained_variance_=explained_variance,
        explained_variance_ratio_=explained_variance_ratio,
    )


def transform(X: np.ndarray, model: PCAModel) -> np.ndarray:
    """
    Proietta nuovi dati nello spazio delle componenti principali.

    X: (n_samples, n_features)
    Ritorna: (n_samples, n_components)
    """
    X_centered = X - model.mean_
    return X_centered @ model.components_.T


def inverse_transform(Z: np.ndarray, model: PCAModel) -> np.ndarray:
    """
    Ricostruisce i dati originali a partire dalle coordinate Z nello spazio PCA.

    Z: (n_samples, n_components)
    Ritorna: (n_samples, n_features)
    """
    return Z @ model.components_ + model.mean_
