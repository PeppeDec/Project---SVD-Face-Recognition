from pathlib import Path
from imageio.v2 import imread
import numpy as np


def load_yale_faces(dataset_dir: str):
    """
    Carica lo Yale Face Database da una directory che contiene file tipo:
    subject01.centerlight, subject01.happy, ...

    Non facciamo filtro per estensione, perché i file non hanno suffisso.
    """
    dataset_path = Path(dataset_dir)

    X = []
    y = []
    label_map = {}
    current_label = 0
    img_shape = None

    # prendiamo TUTTI i file nella cartella (solo livello corrente)
    img_paths = [p for p in sorted(dataset_path.iterdir()) if p.is_file()]

    if not img_paths:
        raise RuntimeError(
            f"Nessun file trovato in {dataset_path.resolve()}.\n"
            "Controlla che i file (subject01.centerlight, ...) siano davvero in questa cartella."
        )

    for img_path in img_paths:
        # es: "subject01.centerlight" -> "subject01"
        stem = img_path.stem          # "subject01.centerlight"
        subject_id = stem.split(".")[0]  # "subject01"

        if subject_id not in label_map:
            label_map[subject_id] = current_label
            current_label += 1
        label = label_map[subject_id]

        try:
            img = imread(img_path)
        except Exception as e:
            print(f"Impossibile leggere {img_path.name}: {e}")
            continue

        # se fosse RGB, facciamo media sui canali → grayscale
        if img.ndim == 3:
            img = img.mean(axis=2)

        if img_shape is None:
            img_shape = img.shape

        img_flat = img.astype(np.float32).ravel() / 255.0
        X.append(img_flat)
        y.append(label)

    if not X:
        raise RuntimeError(
            "Nessuna immagine valida letta. Controlla che i file Yale siano nel formato giusto."
        )

    X = np.vstack(X)
    y = np.array(y, dtype=int)

    return X, y, img_shape, label_map
