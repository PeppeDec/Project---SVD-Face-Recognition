"""
Script principale per:
1. Caricare il DataSet_Uno
2. Dividere in train / test
3. Estrarre feature con PCA o SVD
4. Addestrare il classificatore SVM
5. Valutare le prestazioni
"""
import json

from sklearn.model_selection import train_test_split

from config import TEST_SIZE, RANDOM_STATE, FEATURE_METHOD, N_COMPONENTS
from data_loader import load_dataset
from pca_svd import PCABasedProjector, SVDBasedProjector
from classifier import SVMFaceClassifier
from classifier2 import (
    SVMFaceClassifier,
    KNNFaceClassifier,
    RandomForestFaceClassifier,
    LogisticRegressionFaceClassifier,
    NaiveBayesFaceClassifier,
    compare_classifiers
)

def main():
    # 1. Caricamento dati grezzi (immagini flatten)
    X, y, class_names = load_dataset()

    # 2. Train/Test split (stratificato per mantenere la proporzione tra identità)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"[INFO] Train set: {X_train.shape[0]} campioni, Test set: {X_test.shape[0]} campioni.")

    # 3. Scelta del metodo di feature extraction (PCA o SVD manuale)
    if FEATURE_METHOD.lower() == "pca":
        print(f"[INFO] Uso PCA con {N_COMPONENTS} componenti.")
        projector = PCABasedProjector(n_components=N_COMPONENTS, whiten=True)
    elif FEATURE_METHOD.lower() == "svd":
        print(f"[INFO] Uso SVD manuale con {N_COMPONENTS} componenti.")
        projector = SVDBasedProjector(n_components=N_COMPONENTS)
    else:
        raise ValueError(f"FEATURE_METHOD non riconosciuto: {FEATURE_METHOD} (usa 'pca' o 'svd').")

    # 4. Fit sul train e trasformazione
    projector.fit(X_train)
    X_train_proj = projector.transform(X_train)
    X_test_proj = projector.transform(X_test)

    print(f"[INFO] Feature space: {X_train_proj.shape[1]} dimensioni.")

    # 5. Definizione e training del classificatore (per ora un solo modello: SVM)
    #    Più avanti potrai aggiungere KNN, Logistic Regression, RandomForest, ecc.
    clf = SVMFaceClassifier(
        C=10.0,
        kernel="rbf",   # puoi provare anche "linear"
        gamma="scale"
    )
    # clf = [
    #     SVMFaceClassifier(C=10.0, kernel="rbf", gamma="scale"),
    #     SVMFaceClassifier(C=1.0, kernel="linear"),
    #     KNNFaceClassifier(n_neighbors=1, metric="euclidean"),
    #     KNNFaceClassifier(n_neighbors=5, metric="cosine"),
    #     LogisticRegressionFaceClassifier(C=10.0, max_iter=2000)
    # ]

    clf.fit(X_train_proj, y_train)

    # 6. Valutazione
    clf.evaluate(X_test_proj, y_test, class_names)
    # 6. Confronta tutti i modelli
    # results = compare_classifiers(
    #     clf,
    #     X_train_proj, y_train,
    #     X_test_proj, y_test,
    #     class_names
    # )
    #
    # # 7. Identifica il migliore
    # best_name = max(results, key=lambda k: results[k]['accuracy'])
    # best_acc = results[best_name]['accuracy']
    #
    # print(f"\n{'=' * 60}")
    # print(f"🏆 Miglior classificatore: {best_name}")
    # print(f"   Accuracy: {best_acc:.4f}")
    # print('=' * 60)

if __name__ == "__main__":
    main()
