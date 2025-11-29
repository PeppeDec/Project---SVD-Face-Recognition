# data_loader.py
from pathlib import Path
from imageio.v2 import imread   # se non ce l'hai: pip install imageio
import numpy as np


def load_att_faces(dataset_dir: str):
    """
    Carica il dataset_yale AT&T/ORL da una directory con sottocartelle s1, s2, ..., s40.

    Ritorna:
        X          : array (n_samples, n_pixels) con i pixel normalizzati in [0, 1]
        y          : array (n_samples,) con le etichette intere (0, 1, ..., n_persone-1)
        img_shape  : tuple (height, width) per poter rifare il reshape delle immagini
        label_map  : dict {nome_cartella -> label_intera}
    """
    dataset_path = Path(dataset_dir)

    X = []
    y = []
    label_map = {}
    current_label = 0
    img_shape = None

    # Scorre le cartelle degli individui: s1, s2, ...
    for person_dir in sorted(dataset_path.iterdir()):
        if not person_dir.is_dir():
            continue
        if person_dir.name.startswith("."):   # ignora .DS_Store e simili
            continue

        # assegna una label intera a ogni persona
        label_map[person_dir.name] = current_label

        # legge tutte le immagini .pgm dentro quella cartella
        for img_path in sorted(person_dir.glob("*.pgm")):
            img = imread(img_path)  # array 2D (H, W), grayscale

            if img_shape is None:
                img_shape = img.shape  # es. (112, 92) o (92, 112)

            # flatten + normalizzazione in [0, 1]
            img_flat = img.astype(np.float32).ravel() / 255.0
            X.append(img_flat)
            y.append(current_label)

        current_label += 1

    X = np.vstack(X)       # (n_samples, n_pixels)
    y = np.array(y, int)   # (n_samples,)

    return X, y, img_shape, label_map
