from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image

from config import DATASET_DIR, IMAGE_SIZE


def _load_from_subfolders(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Caso 1: struttura del tipo
        DataSet_Uno/
            person1/
                img1.jpg
                ...
            person2/
                ...
    """
    X = []
    y = []

    class_names = sorted(
        [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    )
    if len(class_names) == 0:
        raise RuntimeError(f"Nessuna sottocartella trovata in {dataset_dir}.")

    label_map = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = dataset_dir / class_name
        image_paths = [p for p in class_dir.glob("*.*") if p.is_file()]

        if len(image_paths) == 0:
            print(f"[ATTENZIONE] Nessuna immagine in {class_dir}, salto.")
            continue

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("L")
                img = img.resize(IMAGE_SIZE)

                arr = np.asarray(img, dtype=np.float32) / 255.0
                X.append(arr.flatten())
                y.append(label_map[class_name])
            except Exception as e:
                print(f"[WARNING] Impossibile leggere {img_path}: {e}")

    if len(X) == 0:
        raise RuntimeError("Nessuna immagine valida caricata dalle sottocartelle.")

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)

    print(f"[INFO] (subfolders) Dataset: {X.shape[0]} immagini, {len(class_names)} identità.")
    return X, y, class_names


# 🔴 QUI È LA PARTE CHE TI INTERESSA PER LA CARTELLA PIATTA 🔴

from pathlib import Path

def extract_identity_from_filename(path: Path) -> str:
    """
    Estrae l'identità dal nome del file.

    Nel tuo DataSet_Uno i file sono del tipo:
        "1-01.jpg" -> identità "1"
        "37-14.jpg" -> identità "37"

    Se un domani avrai nomi con "_" useremo anche quello.
    """
    stem = path.stem  # es: "1-01"

    # Caso del tuo DataSet_Uno: "<ID>-<indice>.jpg"
    if "-" in stem:
        return stem.split("-", 1)[0]

    # fallback generico: "<ID>_<qualcosa>.jpg"
    if "_" in stem:
        return stem.split("_", 1)[0]

    # Ultimo fallback: tutto il nome senza estensione
    return stem



def _load_from_flat_folder(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Caso 2: TUTTI i file immagine sono direttamente dentro 'DataSet_Uno/',
    senza sottocartelle.

        DataSet_Uno/
            001_01.jpg
            001_02.jpg
            ...
            100_13.jpg
    """
    image_paths = sorted([p for p in dataset_dir.glob("*.*") if p.is_file()])

    if len(image_paths) == 0:
        raise RuntimeError(f"Nessuna immagine trovata in {dataset_dir}.")

    # Ricavo tutte le identità leggendo i nomi file
    identities = sorted({extract_identity_from_filename(p) for p in image_paths})
    if len(identities) == 0:
        raise RuntimeError("Non sono riuscito a ricavare nessuna identità dai nomi file.")

    label_map = {name: idx for idx, name in enumerate(identities)}

    X = []
    y = []

    for img_path in image_paths:
        try:
            identity = extract_identity_from_filename(img_path)
            if identity not in label_map:
                # non dovrebbe succedere, ma per sicurezza
                print(f"[WARNING] Identità sconosciuta in {img_path}, salto.")
                continue

            label = label_map[identity]

            img = Image.open(img_path).convert("L")
            img = img.resize(IMAGE_SIZE)
            arr = np.asarray(img, dtype=np.float32) / 255.0

            X.append(arr.flatten())
            y.append(label)

        except Exception as e:
            print(f"[WARNING] Impossibile leggere {img_path}: {e}")

    if len(X) == 0:
        raise RuntimeError("Nessuna immagine valida caricata dalla cartella piatta.")

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)

    print(f"[INFO] (flat) Dataset: {X.shape[0]} immagini, {len(identities)} identità.")
    return X, y, identities


def load_dataset(dataset_dir: Path = DATASET_DIR) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Funzione principale che decide automaticamente se:
    - usare la modalità 'subfolders' (person1/person2/...)
    - oppure 'flat' (tutti i file nella stessa cartella)

    In questo modo tu non devi cambiare nulla nel resto del codice.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"La cartella DataSet_Uno non esiste: {dataset_dir.resolve()}")

    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

    if len(subdirs) > 0:
        # Abbiamo sottocartelle: usiamo il caso 1
        print("[INFO] Rilevata struttura a sottocartelle, uso modalità 'subfolders'.")
        return _load_from_subfolders(dataset_dir)
    else:
        # Nessuna sottocartella, tutti i file sono direttamente dentro 'DataSet_Uno/'
        print("[INFO] Rilevata cartella piatta, uso modalità 'flat'.")
        return _load_from_flat_folder(dataset_dir)
