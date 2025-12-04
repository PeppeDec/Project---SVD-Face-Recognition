# main_yale.py
from pathlib import Path

from data_loader import load_yale_faces
from pca_svd import fit_pca_svd, transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread

def predict_1nn(Z_train: np.ndarray,
                y_train: np.ndarray,
                Z_test: np.ndarray) -> np.ndarray:
    """
    1-Nearest Neighbor “a mano”.

    Per ogni campione in Z_test trova il campione di Z_train
    con distanza euclidea minima e ne copia la label.
    """
    y_pred = []

    for z in Z_test:
        # distanza euclidea da tutti i punti del train
        dists = np.linalg.norm(Z_train - z, axis=1)  # (n_train,)
        idx_min = np.argmin(dists)
        y_pred.append(y_train[idx_min])

    return np.array(y_pred, dtype=int)
def predict_single_from_test(idx,
                             X_test,
                             y_test,
                             Z_train,
                             y_train,
                             pca,
                             img_shape,
                             label_map):
    """
    Prende l'immagine idx dal test set, la proietta in PCA,
    fa 1-NN e stampa vero soggetto vs soggetto predetto.
    """
    # 1) vero vettore immagine
    x = X_test[idx].reshape(1, -1)  # (1, n_features)
    true_label = y_test[idx]

    # 2) proiezione in spazio PCA
    z = transform(x, pca)  # (1, n_components)

    # 3) 1-NN (usiamo la stessa logica di predict_1nn)
    dists = np.linalg.norm(Z_train - z[0], axis=1)
    idx_min = np.argmin(dists)
    pred_label = y_train[idx_min]

    # 4) converto label numeriche in nomi (s1, subject01, ecc.)
    inv_label_map = {v: k for k, v in label_map.items()}
    true_name = inv_label_map[true_label]
    pred_name = inv_label_map[pred_label]

    print(f"Indice test: {idx}")
    print(f"Vero soggetto     : {true_label} ({true_name})")
    print(f"Soggetto predetto : {pred_label} ({pred_name})")
    print(f"Distanza 1-NN     : {dists[idx_min]:.4f}")

    # opzionale: mostra l'immagine
    import matplotlib.pyplot as plt
    img = x.reshape(img_shape)
    plt.imshow(img, cmap="gray")
    plt.title(f"True: {true_name}  |  Pred: {pred_name}")
    plt.axis("off")
    plt.show()

# 1. Carico Yale
X, y, img_shape, label_map = load_yale_faces("/dataset_yale")
print("Shape X:", X.shape)
print("Shape y:", y.shape)
print("img_shape:", img_shape)
print("Num persone:", len(label_map))

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("\nTrain size:", X_train.shape[0])
print("Test  size:", X_test.shape[0])

# 3. Stesso esperimento di prima: lista di componenti
component_list = [5, 10, 15, 20, 30, 40, 50]
accuracies = []
cum_variances = []

for n_components in component_list:
    pca = fit_pca_svd(X_train, n_components=n_components)

    Z_train = transform(X_train, pca)
    Z_test = transform(X_test, pca)

    y_pred = predict_1nn(Z_train, y_train, Z_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    cum_var = pca.explained_variance_ratio_.sum()
    cum_variances.append(cum_var)

    print(f"k={n_components:3d}  accuracy={acc:.3f}  var_cum={cum_var:.3f}")

# plot uguali a prima...
plt.figure()
plt.plot(component_list, accuracies, marker="o")
plt.xlabel("Numero componenti PCA")
plt.ylabel("Accuracy 1-NN sul test (Yale)")
plt.title("Yale: Accuracy vs numero componenti PCA")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(component_list, cum_variances, marker="o")
plt.xlabel("Numero componenti PCA")
plt.ylabel("Varianza spiegata cumulativa")
plt.title("Yale: Varianza spiegata vs numero componenti PCA")
plt.grid(True)
plt.show()


first_img = X[49].reshape(img_shape)
plt.imshow(first_img, cmap="gray")
plt.title(f"Label: {y[1]}")
plt.axis("off")
plt.show()

# idx_example = 49  # prova 0, 5, 10, ecc.
# predict_single_from_test(
#     idx_example,
#     X_test,
#     y_test,
#     Z_train,
#     y_train,
#     pca,
#     img_shape,
#     label_map
# )

def load_images_from_folder(folder: str, img_shape):
    """
    Carica tutte le immagini dalla cartella 'folder', indipendentemente
    dall'estensione (subject01.sad, subject07.centerlight, ...).

    Le ridimensiona (se serve) a img_shape e le normalizza in [0,1].

    Ritorna:
        X_new     : (n_new, n_features)
        filenames : lista dei nomi dei file
    """
    folder_path = Path(folder)
    H, W = img_shape

    X_new = []
    filenames = []

    # prende TUTTI i file (niente filtro per estensione)
    for img_path in sorted(folder_path.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.name.startswith("."):
            continue

        try:
            img = imread(img_path)
        except Exception as e:
            print(f"Salto {img_path.name}: impossibile leggere il file ({e})")
            continue

        # se è RGB → converto in grayscale
        if img.ndim == 3:
            img = img.mean(axis=2)

        if img.shape != (H, W):
            raise ValueError(
                f"Immagine {img_path.name} ha shape {img.shape}, ma mi aspetto {img_shape}"
            )

        img_flat = img.astype(np.float32).ravel() / 255.0
        X_new.append(img_flat)
        filenames.append(img_path.name)

    if not X_new:
        raise RuntimeError(
            f"Nessun file immagine valido trovato in {folder_path.resolve()}"
        )

    X_new = np.vstack(X_new)
    return X_new, filenames

def predict_folder(folder: str,
                   Z_train: np.ndarray,
                   y_train: np.ndarray,
                   pca,
                   img_shape,
                   label_map,
                   show_images: bool = True):
    """
    Carica tutte le immagini da 'folder', le proietta in PCA e
    fa 1-NN contro Z_train.

    Stampa, per ogni file:
        - nome file
        - label predetta (indice + nome soggetto)
        - distanza dal vicino più vicino
    """
    import matplotlib.pyplot as plt

    # 1) carico e preparo le immagini della cartella
    X_new, filenames = load_images_from_folder(folder, img_shape)
    Z_new = transform(X_new, pca)  # (n_new, n_components)

    inv_label_map = {v: k for k, v in label_map.items()}

    for i, (z, fname) in enumerate(zip(Z_new, filenames)):
        # distanza verso tutti i punti del train
        dists = np.linalg.norm(Z_train - z, axis=1)
        idx_min = np.argmin(dists)
        min_dist = dists[idx_min]
        pred_label = y_train[idx_min]
        pred_name = inv_label_map[pred_label]

        print(f"[{i}] File: {fname}")
        print(f"    Pred label : {pred_label} ({pred_name})")
        print(f"    Min dist   : {min_dist:.4f}")

        if show_images:
            img = X_new[i].reshape(img_shape)
            plt.figure()
            plt.imshow(img, cmap="gray")
            plt.title(f"{fname}\nPred: {pred_name} (dist={min_dist:.3f})")
            plt.axis("off")
            plt.show()

predict_folder(
        folder="five_images",   # nome della cartella che hai creato
        Z_train=Z_train,
        y_train=y_train,
        pca=pca,
        img_shape=img_shape,
        label_map=label_map,
        show_images=True
    )