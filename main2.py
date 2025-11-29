# main.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from pca_svd import fit_pca_svd, transform
from data_loader import load_att_faces
import numpy as np
import matplotlib.pyplot as plt


# X, y, img_shape, label_map = load_att_faces("dataset_yale")
#
# print("Shape X:", X.shape)          # es. (400, 10304)
# print("Shape y:", y.shape)          # es. (400,)
# print("img_shape:", img_shape)      # es. (112, 92)
# print("Persone:", len(label_map))   # es. 40
# print("Mappa etichette:", label_map)
#
#
# # first_img = X[399].reshape(img_shape)
# # plt.imshow(first_img, cmap="gray")
# # plt.title(f"Label: {y[1]}")
# # plt.axis("off")
# # plt.show()
#
# # 2. Fit PCA con SVD (per ora, ad esempio, teniamo 100 componenti)
# n_components = 100
# pca = fit_pca_svd(X, n_components=n_components)
#
# print("\n=== PCA info ===")
# print("Num componenti tenute:", pca.components_.shape[0])
# print("Dimensionalità originale:", X.shape[1])
# print("Somma varianza spiegata:", pca.explained_variance_ratio_.sum())
#
# # 3. Grafico della varianza spiegata cumulativa
# cum_var = np.cumsum(pca.explained_variance_ratio_)
# plt.figure()
# plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
# plt.xlabel("Numero componenti principali")
# plt.ylabel("Varianza spiegata cumulativa")
# plt.title("PCA su AT&T Faces (via SVD)")
# plt.grid(True)
# plt.show()
#
# num_eigenfaces_to_show = 10
# h, w = img_shape
#
# fig, axes = plt.subplots(2, 5, figsize=(10, 4))
# axes = axes.ravel()
#
# for i in range(num_eigenfaces_to_show):
#     ax = axes[i]
#     eigenface = pca.components_[i].reshape(h, w)
#     ax.imshow(eigenface, cmap="gray")
#     ax.set_title(f"PC {i + 1}")
#     ax.axis("off")
#
# plt.suptitle("Prime eigenfaces")
# plt.tight_layout()
# plt.show()

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


X, y, img_shape, label_map = load_att_faces("dataset")
print("Shape X:", X.shape)      # (400, 10304)
print("Shape y:", y.shape)      # (400,)
print("img_shape:", img_shape)  # (h, w)
print("Num persone:", len(label_map))

# 2. Train/Test split (stratificato sulle persone)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,         # 30% test → ~120 immagini
    stratify=y,            # mantiene la distribuzione delle classi
    random_state=42
)

print("\nTrain size:", X_train.shape[0])
print("Test  size:", X_test.shape[0])

# 3. Fit PCA SOLO sul train (niente leakage)
# 3. Esperimento: varia il numero di componenti PCA
component_list = [5, 10, 15, 20, 30, 40, 50, 75, 100]
accuracies = []
cum_variances = []

for n_components in component_list:
    # Fit PCA solo sul train
    pca = fit_pca_svd(X_train, n_components=n_components)

    # Proietta train e test nello spazio PCA
    Z_train = transform(X_train, pca)
    Z_test = transform(X_test, pca)

    # 1-NN
    y_pred = predict_1nn(Z_train, y_train, Z_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Varianza spiegata cumulativa
    cum_var = pca.explained_variance_ratio_.sum()
    cum_variances.append(cum_var)

    print(f"k={n_components:3d}  accuracy={acc:.3f}  var_cum={cum_var:.3f}")

# 4. Plot: n_components vs accuracy

plt.figure()
plt.plot(component_list, accuracies, marker="o")
plt.xlabel("Numero componenti PCA")
plt.ylabel("Accuracy 1-NN sul test")
plt.title("Accuracy vs numero componenti PCA")
plt.grid(True)
plt.show()

# 5. Plot: n_components vs varianza spiegata cumulativa
plt.figure()
plt.plot(component_list, cum_variances, marker="o")
plt.xlabel("Numero componenti PCA")
plt.ylabel("Varianza spiegata cumulativa")
plt.title("Varianza spiegata vs numero componenti PCA")
plt.grid(True)
plt.show()

# === Baseline: 1-NN direttamente sui pixel grezzi (senza PCA) ===
print("\n=== Baseline: 1-NN su pixel grezzi (nessuna PCA) ===")

y_pred_raw = predict_1nn(X_train, y_train, X_test)
acc_raw = accuracy_score(y_test, y_pred_raw)

print(f"Accuracy 1-NN su pixel grezzi: {acc_raw:.3f}")
