from pathlib import Path

# Directory radice del DataSet_Uno (cambia se necessario)
DATASET_DIR: Path = Path("dataset")

# Tutte le immagini verranno convertite in scala di grigi e ridimensionate a questa dimensione
IMAGE_SIZE = (100, 100)  # (width, height)

# Train/Test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Numero di componenti principali per PCA/SVD
# (puoi giocare con questo parametro: 50, 100, 150...)
N_COMPONENTS = 100

# Per ora useremo PCA "classica"; più avanti confronterai PCA vs SVD manuale
FEATURE_METHOD = "pca"  # "pca" oppure "svd"
