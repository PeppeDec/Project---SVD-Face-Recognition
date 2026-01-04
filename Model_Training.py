import numpy as np
import os
from pathlib import Path
from PIL import Image
from skimage import exposure, transform
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier

from tqdm import tqdm
import gc
import time
import warnings
import re
import csv
import joblib

warnings.filterwarnings('ignore')

# -------------------------
#  REPORTING HELPERS
# -------------------------
def print_model_table(rows, sort_by="cv_mean", descending=True, top_n=None):
    """Pretty-print a fixed-width comparison table.

    rows: list of dicts like:
      {"model": str, "cv_mean": float, "cv_std": float}
    """
    if not rows:
        print("\n⚠️  Nessun risultato da mostrare.")
        return []

    rows_sorted = sorted(rows, key=lambda r: r.get(sort_by, float('-inf')), reverse=descending)
    if top_n is not None:
        rows_sorted = rows_sorted[: int(top_n)]

    print("\n" + "=" * 70)
    print("CONFRONTO MODELLI (CV)")
    print("=" * 70)
    print(f"{'Model':<40} {'CV Mean %':>10} {'CV Std %':>10}")
    print("-" * 70)

    for r in rows_sorted:
        name = str(r.get("model", ""))
        mean = float(r.get("cv_mean", 0.0)) * 100.0
        std = float(r.get("cv_std", 0.0)) * 100.0

        if len(name) > 40:
            name = name[:37] + "..."

        print(f"{name:<40} {mean:>10.2f} {std:>10.2f}")

    print("-" * 70)
    return rows_sorted


def save_model_table_csv(rows, out_path="results/model_comparison.csv"):
    """Save comparison rows to CSV."""
    if not rows:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "cv_mean", "cv_std"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "model": r.get("model", ""),
                "cv_mean": float(r.get("cv_mean", 0.0)),
                "cv_std": float(r.get("cv_std", 0.0)),
            })

    print(f"✅ Salvato CSV confronto modelli: {out_path}")


# -------------------------
#  MODEL BUNDLE (for app.py inference)
# -------------------------
BUNDLE_PATH_DEFAULT = "results/svd_face_model.joblib"


def save_model_bundle(path, *, model_name, model, scaler, svd, normalizer,
                      reverse_label_mapping, img_size, config=None):
    """Save everything needed for inference without retraining."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "svd": svd,
        "normalizer": normalizer,
        "reverse_label_mapping": reverse_label_mapping,
        "img_size": tuple(img_size),
        "config": dict(config) if config is not None else {},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    joblib.dump(bundle, path)
    print(f"✅ Salvato bundle modello: {path}")


def load_model_bundle(path=BUNDLE_PATH_DEFAULT):
    """Load a previously saved model bundle."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bundle non trovato: {path.resolve()}")
    bundle = joblib.load(path)

    required = ["model", "scaler", "svd", "normalizer", "reverse_label_mapping", "img_size", "model_name"]
    for k in required:
        if k not in bundle:
            raise ValueError(f"Bundle invalido: manca '{k}'")

    return bundle


# -------------------------
#  HELPERS
# -------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png")
IMG_EXTS_SET = {e.lower() for e in IMG_EXTS}


def list_image_files(root: Path, recursive: bool = False):
    """Return image files under `root` (non-recursive by default), case-insensitive by extension."""
    it = root.rglob("*") if recursive else root.iterdir()
    return [p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS_SET]


def infer_identity_from_filename(path: Path) -> str:
    """Infer identity label from filename for flat datasets.

    Examples:
      - '1-01.jpg' -> '1'
      - '12_034.png' -> '12'
      - 'subject01-xyz.jpg' -> 'subject01'
    """
    stem = path.stem

    m = re.match(r"^([^_-]+)[_-].+$", stem)
    if m:
        return m.group(1)

    m = re.match(r"^(\d+).*$", stem)
    if m:
        return m.group(1)

    return stem


# =========================
#  CLASS
# =========================
class SVDTopKFaceRecognition:
    """Face recognition pipeline: preprocessing -> split (no leakage) -> SVD -> L2 norm -> classifier.

    Key design points:
      - Split BEFORE augmentation (no leakage)
      - Augmentation ONLY on TRAIN
      - GroupKFold on TRAIN (group = original image id) to avoid augmented clones across folds
      - TruncatedSVD for dimensionality reduction (top-k)
      - L2 Normalizer after SVD (helps linear models / kNN)
      - Light hyperparameter search (C/gamma/alpha/k)
    """

    def __init__(self, dataset_path, img_size=(128, 128)):
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size

        self.scaler = StandardScaler()
        self.svd = None
        self.normalizer = Normalizer(norm="l2")

        self.label_mapping = {}
        self.reverse_label_mapping = {}

    # -------------------------
    # Preprocessing
    # -------------------------
    def robust_preprocessing(self, img_array: np.ndarray) -> np.ndarray:
        """normalize -> CLAHE -> per-image standardization."""
        img_array = img_array.astype(np.float32) / 255.0
        img_array = exposure.equalize_adapthist(img_array, clip_limit=0.03)

        mean = float(np.mean(img_array))
        std = float(np.std(img_array))
        if std > 0:
            img_array = (img_array - mean) / std

        return img_array.astype(np.float32)

    # -------------------------
    # Label mapping
    # -------------------------
    def set_label_mapping(self, selected_identities):
        self.label_mapping.clear()
        self.reverse_label_mapping.clear()
        for idx, identity_info in enumerate(selected_identities):
            self.label_mapping[identity_info["name"]] = idx
            self.reverse_label_mapping[idx] = identity_info["name"]

    # -------------------------
    # Path collection (NO augmentation)
    # -------------------------
    def collect_image_paths(self, selected_identities, max_images_per_identity, base_seed=42):
        """Collect (path, label) for ORIGINAL images only (no augmentation)."""
        if not self.label_mapping:
            self.set_label_mapping(selected_identities)

        file_label_pairs = []

        print(f"\n{'=' * 70}")
        print("RACCOLTA PATH IMMAGINI (SENZA AUGMENTATION)")
        print(f"{'=' * 70}")
        print(f"📊 Identità: {len(selected_identities)}")
        print(f"📊 Max ORIGINALI/identità: {max_images_per_identity}")

        for idx, identity_info in enumerate(tqdm(selected_identities, desc="Scanning")):
            label = self.label_mapping[identity_info["name"]]

            # Flat-mode
            if "files" in identity_info and identity_info["files"] is not None:
                image_files = list(identity_info["files"])
            else:
                # Folder-mode
                folder = identity_info["folder"]
                image_files = list_image_files(folder, recursive=False)
                if not image_files:
                    image_files = list_image_files(folder, recursive=True)

            # Reproducible subsample per identity
            if len(image_files) > max_images_per_identity:
                rng = np.random.default_rng(base_seed + idx)
                chosen = rng.choice(image_files, size=max_images_per_identity, replace=False)
                image_files = list(chosen)

            for p in image_files:
                file_label_pairs.append((Path(p), label))

        print(f"\n✅ Trovate immagini originali: {len(file_label_pairs):,}")
        return file_label_pairs

    # -------------------------
    # Load images + optional augmentation
    # -------------------------
    def load_from_paths(self, file_label_pairs, augment=False, return_groups=False, rng=None):
        """Load images from (path,label).

        If augment=True: for each original generates [original, flip, small rotation].
        If return_groups=True: returns group id repeated for augmented versions.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        images = []
        labels = []
        groups = []

        print(f"\n{'=' * 70}")
        print(f"CARICAMENTO IMMAGINI - augment={'ON' if augment else 'OFF'}")
        print(f"{'=' * 70}")
        if augment:
            print("📊 Augmentation: 3x (originale + flip + rotazione) SOLO su TRAIN")
        else:
            print("📊 Nessuna augmentation (tipico per TEST)")

        group_id = 0

        for (img_path, lab) in tqdm(file_label_pairs, desc="Loading"):
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32)

                img_processed = self.robust_preprocessing(img_array)

                # 1) Original
                images.append(img_processed.flatten())
                labels.append(lab)
                if return_groups:
                    groups.append(group_id)

                if augment:
                    # 2) Flip
                    images.append(np.fliplr(img_processed).flatten())
                    labels.append(lab)
                    if return_groups:
                        groups.append(group_id)

                    # 3) Small rotation
                    angle = float(rng.choice([-7, -5, 5, 7]))
                    img_rotated = transform.rotate(
                        img_processed, angle,
                        mode='edge', preserve_range=True
                    ).astype(np.float32)

                    images.append(img_rotated.flatten())
                    labels.append(lab)
                    if return_groups:
                        groups.append(group_id)

            except Exception:
                pass

            group_id += 1

        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        g = np.array(groups, dtype=np.int32) if return_groups else None

        print(f"\n✅ Caricamento completato!")
        print(f"   • Campioni totali: {len(X):,}")
        print(f"   • Features: {X.shape[1]:,}")
        print(f"   • Classi: {len(np.unique(y))}")

        del images
        gc.collect()

        if return_groups:
            return X, y, g
        return X, y

    # -------------------------
    # SVD + L2 normalization
    # -------------------------
    def apply_optimal_svd(self, X_train, X_test, n_components=400, n_iter=7):
        """Standardize -> TruncatedSVD -> L2 normalize."""
        print(f"\n{'=' * 70}")
        print(f"APPLICAZIONE SVD (n_components={n_components})")
        print(f"{'=' * 70}")

        start = time.time()

        print("🔧 Standardizzazione...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        max_comp = min(n_components, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])
        if max_comp < 2:
            raise ValueError("n_components too small for current training set.")

        print(f"🔧 TruncatedSVD fit (k={max_comp}, n_iter={n_iter})...")
        self.svd = TruncatedSVD(n_components=max_comp, random_state=42, n_iter=n_iter)
        X_train_svd = self.svd.fit_transform(X_train_scaled)
        X_test_svd = self.svd.transform(X_test_scaled)

        var_sum = float(np.sum(self.svd.explained_variance_ratio_))
        print(f"   • Varianza spiegata (somma ratio): {var_sum * 100:.2f}%")

        print("🔧 Normalizzazione L2 post-SVD...")
        X_train_svd = self.normalizer.fit_transform(X_train_svd)
        X_test_svd = self.normalizer.transform(X_test_svd)

        print(f"\n✅ SVD completata in {time.time() - start:.2f}s")
        print(f"   • Riduzione: {X_train.shape[1]:,} → {max_comp}")

        return X_train_svd.astype(np.float32), X_test_svd.astype(np.float32)

    # -------------------------
    # Candidate models
    # -------------------------
    def _candidate_models(self):
        grids = []

        # Linear SVM
        for C in [0.3, 1.0, 3.0, 10.0]:
            grids.append((
                f"SVM-linear(C={C})",
                SVC(kernel='linear', C=C, class_weight='balanced', cache_size=3000, random_state=42)
            ))

        # RBF SVM
        for C in [1.0, 3.0, 10.0]:
            for gamma in ['scale', 0.005, 0.002, 0.001]:
                grids.append((
                    f"SVM-RBF(C={C},gamma={gamma})",
                    SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced', cache_size=3000, random_state=42)
                ))

        # Logistic Regression
        for C in [0.3, 1.0, 3.0]:
            grids.append((
                f"LogReg(C={C})",
                LogisticRegression(
                    max_iter=5000,
                    solver='saga',
                    multi_class='multinomial',
                    C=C,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42
                )
            ))

        # Ridge
        for alpha in [0.3, 1.0, 3.0, 10.0]:
            grids.append((
                f"Ridge(alpha={alpha})",
                RidgeClassifier(alpha=alpha, class_weight='balanced')
            ))

        # kNN
        for k in [1, 3, 5, 7]:
            grids.append((
                f"kNN(k={k})",
                KNeighborsClassifier(n_neighbors=k, weights='distance', metric='minkowski', p=2, n_jobs=-1)
            ))

        return grids

    def tune_and_train(self, X_train, y_train, groups_train=None):
        """Evaluate candidates with CV, pick best, fit on full train.

        Returns: best_name, best_model, best_cv_mean, cv_rows
        """
        print(f"\n{'=' * 70}")
        print("SELEZIONE MODELLO (CV anti-leakage)")
        print(f"{'=' * 70}")

        if groups_train is not None:
            cv = GroupKFold(n_splits=5)
            cv_kwargs = dict(cv=cv, groups=groups_train)
            print("🔄 CV: GroupKFold (anti-leakage) ✅")
        else:
            cv_kwargs = dict(cv=5)
            print("🔄 CV: Stratified 5-fold")

        candidates = self._candidate_models()

        best = None
        best_score = -1.0
        cv_rows = []

        for name, model in candidates:
            scores = cross_val_score(model, X_train, y_train, scoring='accuracy', n_jobs=-1, **cv_kwargs)
            mean = float(scores.mean())
            std = float(scores.std())

            cv_rows.append({
                "model": name,
                "cv_mean": mean,
                "cv_std": std,
            })

            print(f"   • {name:<25s}  CV: {mean*100:6.2f}%  ±{std*100:5.2f}%")

            if mean > best_score:
                best_score = mean
                best = (name, model)

        if best is None:
            raise RuntimeError("No model candidate evaluated.")

        best_name, best_model = best
        print(f"\n🏁 Miglior modello CV: {best_name}  (mean={best_score*100:.2f}%)")

        print("\n🔧 Training finale su tutto il TRAIN...")
        start = time.time()
        best_model.fit(X_train, y_train)
        print(f"   ✅ Completato in {time.time() - start:.2f}s")

        train_acc = float(best_model.score(X_train, y_train))
        print(f"   • Training accuracy: {train_acc * 100:.2f}%")

        return best_name, best_model, best_score, cv_rows

    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate(self, model_name, model, X_train, X_test, y_train, y_test, cv_mean):
        print(f"\n{'=' * 70}")
        print("VALUTAZIONE FINALE")
        print(f"{'=' * 70}")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = float(accuracy_score(y_train, y_train_pred))
        test_acc = float(accuracy_score(y_test, y_test_pred))

        print(f"\n🤖 Modello scelto: {model_name}")
        print(f"\n📊 RISULTATI:")
        print(f"   • Test Accuracy:     {test_acc * 100:.2f}% ⭐")
        print(f"   • CV Accuracy:       {cv_mean * 100:.2f}%")
        print(f"   • Training Accuracy: {train_acc * 100:.2f}%")
        print(f"   • Gap Train-Test:    {(train_acc - test_acc) * 100:.2f}%")

        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='weighted', zero_division=0
        )

        print(f"\n📈 METRICHE (weighted):")
        print(f"   • Precision: {precision * 100:.2f}%")
        print(f"   • Recall:    {recall * 100:.2f}%")
        print(f"   • F1-Score:  {f1 * 100:.2f}%")

        return {
            'model_name': model_name,
            'test_acc': test_acc,
            'train_acc': train_acc,
            'cv_acc': cv_mean,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

    def plot_results(self, results, save_path='results'):
        os.makedirs(save_path, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Test', 'CV', 'Training']
        values = [
            results['test_acc'] * 100,
            results['cv_acc'] * 100,
            results['train_acc'] * 100
        ]

        bars = ax.bar(categories, values, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(
            f"{results['model_name']} | Test: {results['test_acc'] * 100:.2f}%",
            fontsize=13, fontweight='bold', pad=12
        )
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height + 1,
                f'{val:.2f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=11
            )

        plt.tight_layout()
        out = Path(save_path) / 'accuracy.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"\n✅ Salvato: {out}")
        plt.close()

    # -------------------------
    # Evaluate ALL 5 model families (fixed baselines)
    # -------------------------
    def _five_baseline_models(self):
        """Return exactly 5 baseline models (one per family).

        Families:
          1) Linear SVM
          2) RBF SVM
          3) Logistic Regression
          4) Ridge Classifier
          5) kNN

        Note: hyperparameters are chosen to be reasonable defaults given your current grid.
        You can tweak them in CONFIG (see main) if you want.
        """
        cfg = getattr(self, "baseline_cfg", {}) or {}

        svm_lin_C = float(cfg.get("SVM_LINEAR_C", 3.0))
        svm_rbf_C = float(cfg.get("SVM_RBF_C", 3.0))
        svm_rbf_gamma = cfg.get("SVM_RBF_GAMMA", "scale")
        logreg_C = float(cfg.get("LOGREG_C", 3.0))
        ridge_alpha = float(cfg.get("RIDGE_ALPHA", 3.0))
        knn_k = int(cfg.get("KNN_K", 1))

        return [
            (
                f"SVM-linear(C={svm_lin_C})",
                SVC(kernel='linear', C=svm_lin_C, class_weight='balanced', cache_size=3000, random_state=42)
            ),
            (
                f"SVM-RBF(C={svm_rbf_C},gamma={svm_rbf_gamma})",
                SVC(kernel='rbf', C=svm_rbf_C, gamma=svm_rbf_gamma, class_weight='balanced', cache_size=3000, random_state=42)
            ),
            (
                f"LogReg(C={logreg_C})",
                LogisticRegression(
                    max_iter=5000,
                    solver='saga',
                    multi_class='multinomial',
                    C=logreg_C,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42
                )
            ),
            (
                f"Ridge(alpha={ridge_alpha})",
                RidgeClassifier(alpha=ridge_alpha, class_weight='balanced')
            ),
            (
                f"kNN(k={knn_k})",
                KNeighborsClassifier(n_neighbors=knn_k, weights='distance', metric='minkowski', p=2, n_jobs=-1)
            ),
        ]

    def evaluate_all_five_models(self, X_train, y_train, X_test, y_test, out_csv_path="results/all_models_metrics.csv"):
        """Train + evaluate 5 model families on the SAME split and export a CSV.

        Metrics exported:
          - accuracy
          - balanced_accuracy
          - precision/recall/f1 (weighted)
          - precision/recall/f1 (macro)
          - train_time_sec
          - test_infer_time_sec
        """
        out_csv_path = Path(out_csv_path)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        models = self._five_baseline_models()

        print(f"\n{'=' * 70}")
        print("VALUTAZIONE 5 MODELLI (SPLIT FISSO)")
        print(f"{'=' * 70}")

        for name, model in models:
            print(f"\n▶ {name}")
            # Train
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            # Predict (test)
            t1 = time.time()
            y_pred = model.predict(X_test)
            infer_time = time.time() - t1

            acc = float(accuracy_score(y_test, y_pred))
            bacc = float(balanced_accuracy_score(y_test, y_pred))

            p_w, r_w, f1_w, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )
            p_m, r_m, f1_m, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro', zero_division=0
            )

            row = {
                "model": name,
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "precision_weighted": float(p_w),
                "recall_weighted": float(r_w),
                "f1_weighted": float(f1_w),
                "precision_macro": float(p_m),
                "recall_macro": float(r_m),
                "f1_macro": float(f1_m),
                "train_time_sec": float(train_time),
                "test_infer_time_sec": float(infer_time),
            }
            rows.append(row)

            print(f"   • Accuracy:          {acc * 100:.2f}%")
            print(f"   • Balanced Accuracy: {bacc * 100:.2f}%")
            print(f"   • F1 (weighted):     {float(f1_w) * 100:.2f}%")
            print(f"   • F1 (macro):        {float(f1_m) * 100:.2f}%")
            print(f"   • Train time:        {train_time:.2f}s")
            print(f"   • Infer time (test): {infer_time:.2f}s")

        # Sort by accuracy desc
        rows_sorted = sorted(rows, key=lambda d: d["accuracy"], reverse=True)

        # Save CSV
        fieldnames = list(rows_sorted[0].keys()) if rows_sorted else []
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows_sorted:
                w.writerow(r)

        print(f"\n✅ Salvato CSV metriche 5 modelli: {out_csv_path}")
        return rows_sorted


# =========================
#  DATASET ANALYSIS
# =========================
def analyze_dataset(dataset_path):
    """Support:
      - folder-per-identity datasets
      - flat datasets (identity inferred from filename)
    """
    print(f"\n{'=' * 70}")
    print("ANALISI DATASET")
    print(f"{'=' * 70}")

    root = Path(dataset_path)
    if not root.exists():
        print(f"❌ Path non trovato: {root.resolve()}")
        return []

    # 1) Folder-mode
    folders = sorted([f for f in root.iterdir() if f.is_dir()])
    folder_has_images = False
    distribution = []

    if folders:
        for folder in tqdm(folders, desc="Analisi (folders)"):
            files = list_image_files(folder, recursive=False)
            if not files:
                files = list_image_files(folder, recursive=True)
            n_images = len(files)
            if n_images > 0:
                folder_has_images = True
            distribution.append({'folder': folder, 'name': folder.name, 'count': n_images})

        if folder_has_images:
            distribution = sorted(distribution, key=lambda x: x['count'], reverse=True)
            counts = [d['count'] for d in distribution if d['count'] > 0]
            if not counts:
                print("❌ Nessuna immagine trovata nelle cartelle.")
                return []

            print("\n📊 STATISTICHE (folder-mode):")
            print(f"   • Identità totali: {len(distribution)}")
            print(f"   • Immagini totali: {sum(d['count'] for d in distribution):,}")
            print(f"   • Media: {np.mean(counts):.1f}")
            print(f"   • Min: {min(counts)}, Max: {max(counts)}")
            return distribution

    # 2) Flat-mode
    all_images = list_image_files(root, recursive=False)

    if not all_images:
        print(f"❌ Nessuna immagine trovata in: {root.resolve()}")
        return []

    groups = {}
    for p in tqdm(all_images, desc="Analisi (flat)"):
        ident = infer_identity_from_filename(p)
        groups.setdefault(ident, []).append(p)

    distribution = []
    for ident, files in groups.items():
        distribution.append({'folder': None, 'name': ident, 'count': len(files), 'files': files})

    distribution = sorted(distribution, key=lambda x: x['count'], reverse=True)
    counts = [d['count'] for d in distribution]

    print("\n📊 STATISTICHE (flat-mode):")
    print(f"   • Identità totali: {len(distribution)}")
    print(f"   • Immagini totali: {sum(counts):,}")
    print(f"   • Media: {np.mean(counts):.1f}")
    print(f"   • Min: {min(counts)}, Max: {max(counts)}")

    return distribution


def select_identities(distribution, n_identities, min_samples):
    valid = [d for d in distribution if d['count'] >= min_samples]
    selected = valid[:n_identities]

    print("\n💡 SELEZIONE:")
    print(f"   • Identità con ≥{min_samples} img: {len(valid)}")
    print(f"   • Identità selezionate: {len(selected)}")
    print(f"   • Immagini totali (raw): {sum(d['count'] for d in selected):,}")

    return selected


# =========================
#  MAIN
# =========================
def main():
    print("\n" + "=" * 70)
    print("  SVD & FACE RECOGNITION - NO LEAKAGE + TUNING LIGHT")
    print("=" * 70 + "\n")

    # 👇 CAMBIA QUI (deve puntare alla cartella che contiene le cartelle identità)
    DATASET_PATH = "/Users/giuseppe/PycharmProjects/Project---SVD-Face-Recognition/DataSet_Tre/dataset"

    CONFIG = {
        'N_IDENTITIES': 250,
        'MIN_SAMPLES': 12,
        'MAX_IMAGES_PER_ID': 120,
        'IMG_SIZE': (128, 128),
        'N_COMPONENTS': 400,        # prova 300/400/600
        'SVD_N_ITER': 7,            # prova 7/10/15
        'TEST_SIZE': 0.20,

        # Baseline hyperparams for the 5-family comparison CSV
        'BASELINE_SVM_LINEAR_C': 3.0,
        'BASELINE_SVM_RBF_C': 3.0,
        'BASELINE_SVM_RBF_GAMMA': 'scale',
        'BASELINE_LOGREG_C': 3.0,
        'BASELINE_RIDGE_ALPHA': 3.0,
        'BASELINE_KNN_K': 1,

        # Export CSV with metrics for all 5 model families
        'EXPORT_ALL_MODELS_CSV': True,
        'ALL_MODELS_CSV_PATH': 'results/all_models_metrics.csv',

        # Model persistence
        'BUNDLE_PATH': BUNDLE_PATH_DEFAULT,
        'SKIP_TRAIN_IF_BUNDLE_EXISTS': True,
    }

    print("⚙️  CONFIGURAZIONE:")
    print(f"   • Identità target: {CONFIG['N_IDENTITIES']}")
    print(f"   • Min samples/identità: {CONFIG['MIN_SAMPLES']}")
    print(f"   • Max ORIGINALI/identità: {CONFIG['MAX_IMAGES_PER_ID']}")
    print(f"   • Dimensione immagini: {CONFIG['IMG_SIZE']}")
    print(f"   • SVD n_components: {CONFIG['N_COMPONENTS']}")
    print(f"   • SVD n_iter: {CONFIG['SVD_N_ITER']}")
    print(f"   • Augmentation: SOLO TRAIN (3x)")
    print(f"   • Test split: {CONFIG['TEST_SIZE'] * 100:.0f}%")

    # If bundle already exists, skip training (use app.py for inference)
    bundle_path = Path(CONFIG.get('BUNDLE_PATH', BUNDLE_PATH_DEFAULT))
    if CONFIG.get('SKIP_TRAIN_IF_BUNDLE_EXISTS', True) and bundle_path.exists():
        print(f"\n✅ Bundle già presente: {bundle_path}")
        print("   → Salto l'allenamento. Per ri-addestrare, elimina il file .joblib o imposta SKIP_TRAIN_IF_BUNDLE_EXISTS=False")
        return

    # 0) Analyze & select identities
    distribution = analyze_dataset(DATASET_PATH)
    selected = select_identities(distribution, CONFIG['N_IDENTITIES'], CONFIG['MIN_SAMPLES'])

    if not selected:
        print("\n❌ Nessuna identità valida trovata!")
        return

    input("\n⏸️  Premi ENTER per continuare...")

    # 1) Init system + mapping
    system = SVDTopKFaceRecognition(dataset_path=DATASET_PATH, img_size=CONFIG['IMG_SIZE'])
    system.set_label_mapping(selected)

    # 2) Collect ORIGINAL paths (no augmentation)
    all_pairs = system.collect_image_paths(
        selected_identities=selected,
        max_images_per_identity=CONFIG['MAX_IMAGES_PER_ID']
    )

    # 3) Split BEFORE augmentation (no leakage)
    print(f"\n{'=' * 70}")
    print("SPLIT TRAIN/TEST (PRIMA DELL'AUGMENTATION)")
    print(f"{'=' * 70}")

    all_paths = [p for (p, _) in all_pairs]
    all_labels = [lab for (_, lab) in all_pairs]

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=CONFIG['TEST_SIZE'],
        random_state=42,
        stratify=all_labels
    )

    train_pairs = list(zip(train_paths, train_labels))
    test_pairs = list(zip(test_paths, test_labels))

    print(f"✅ Originali TRAIN: {len(train_pairs):,} | Originali TEST: {len(test_pairs):,}")

    del all_pairs, all_paths, all_labels
    gc.collect()

    # 4) Load TRAIN with augmentation and groups
    rng = np.random.default_rng(42)
    X_train, y_train, groups_train = system.load_from_paths(
        train_pairs, augment=True, return_groups=True, rng=rng
    )

    # 5) Load TEST without augmentation
    X_test, y_test = system.load_from_paths(
        test_pairs, augment=False, return_groups=False, rng=rng
    )

    del train_pairs, test_pairs
    gc.collect()

    # 6) SVD + L2 norm
    X_train_svd, X_test_svd = system.apply_optimal_svd(
        X_train, X_test,
        n_components=CONFIG['N_COMPONENTS'],
        n_iter=CONFIG['SVD_N_ITER']
    )

    del X_train, X_test
    gc.collect()

    # 6b) Evaluate and export CSV for all 5 baseline model families (optional)
    system.baseline_cfg = {
        "SVM_LINEAR_C": CONFIG.get('BASELINE_SVM_LINEAR_C', 3.0),
        "SVM_RBF_C": CONFIG.get('BASELINE_SVM_RBF_C', 3.0),
        "SVM_RBF_GAMMA": CONFIG.get('BASELINE_SVM_RBF_GAMMA', 'scale'),
        "LOGREG_C": CONFIG.get('BASELINE_LOGREG_C', 3.0),
        "RIDGE_ALPHA": CONFIG.get('BASELINE_RIDGE_ALPHA', 3.0),
        "KNN_K": CONFIG.get('BASELINE_KNN_K', 1),
    }

    if CONFIG.get('EXPORT_ALL_MODELS_CSV', True):
        system.evaluate_all_five_models(
            X_train_svd, y_train,
            X_test_svd, y_test,
            out_csv_path=CONFIG.get('ALL_MODELS_CSV_PATH', 'results/all_models_metrics.csv')
        )

    # 7) Tune + train classifier
    best_name, best_model, cv_mean, cv_rows = system.tune_and_train(
        X_train_svd, y_train, groups_train=groups_train
    )

    # 7b) Print + save comparison table
    sorted_rows = print_model_table(cv_rows, sort_by="cv_mean", descending=True)
    save_model_table_csv(sorted_rows, out_path="results/model_comparison.csv")

    # 8) Evaluate
    results = system.evaluate(best_name, best_model, X_train_svd, X_test_svd, y_train, y_test, cv_mean)

    # Save pipeline for app.py inference (no retraining needed)
    save_model_bundle(
        CONFIG.get('BUNDLE_PATH', BUNDLE_PATH_DEFAULT),
        model_name=best_name,
        model=best_model,
        scaler=system.scaler,
        svd=system.svd,
        normalizer=system.normalizer,
        reverse_label_mapping=system.reverse_label_mapping,
        img_size=system.img_size,
        config=CONFIG,
    )

    # 9) Plot
    system.plot_results(results)

    # Summary
    print(f"\n{'=' * 70}")
    print("🏆 RIEPILOGO FINALE")
    print(f"{'=' * 70}")
    print(f"\n   🤖 Best model: {results['model_name']}")
    print(f"   📊 Classi: {len(selected)} identità")
    print(f"   🧪 Test set: {len(y_test):,} campioni (NO augmentation)")
    print(f"   🏋️ Train set: {len(y_train):,} campioni (con augmentation)")
    print(f"   🎯 Test Accuracy: {results['test_acc'] * 100:.2f}%")
    print(f"   📈 F1-Score: {results['f1'] * 100:.2f}%")
    print(f"   📉 Gap: {(results['train_acc'] - results['test_acc']) * 100:.2f}%")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()