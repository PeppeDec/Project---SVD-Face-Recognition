import numpy as np
import os
from pathlib import Path
from PIL import Image
from skimage import exposure, transform
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import gc
import time
import warnings

warnings.filterwarnings('ignore')


class OptimalFaceRecognition:
    """Sistema OTTIMIZZATO per performance reali"""

    def __init__(self, dataset_path, img_size=(128, 128)):
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.pca = None
        self.scaler = StandardScaler()
        self.label_mapping = {}
        self.reverse_label_mapping = {}

    def robust_preprocessing(self, img_array):
        """Preprocessing efficace e testato"""
        # Normalizzazione base
        img_array = img_array.astype(np.float32) / 255.0

        # CLAHE per equalizzazione istogramma
        img_array = exposure.equalize_adapthist(img_array, clip_limit=0.03)

        # Standardizzazione
        mean = np.mean(img_array)
        std = np.std(img_array)
        if std > 0:
            img_array = (img_array - mean) / std

        return img_array

    def load_balanced_dataset(self, selected_identities, max_images_per_identity):
        """Carica dataset con augmentation EFFICACE"""
        print(f"\n{'=' * 70}")
        print(f"CARICAMENTO DATASET BILANCIATO")
        print(f"{'=' * 70}")

        images = []
        labels = []

        # Mapping label
        for idx, identity_info in enumerate(selected_identities):
            self.label_mapping[identity_info['name']] = idx
            self.reverse_label_mapping[idx] = identity_info['name']

        print(f"📊 Caricamento di {len(selected_identities)} identità...")
        print(f"📊 Max {max_images_per_identity} immagini base per identità")
        print(f"📊 Augmentation: 3x (originale + flip + rotazione)")

        for idx, identity_info in enumerate(tqdm(selected_identities, desc="Caricamento")):
            folder = identity_info['folder']
            image_files = list(folder.glob("*.jpg")) + \
                          list(folder.glob("*.png")) + \
                          list(folder.glob("*.jpeg"))

            # Limita numero immagini
            if len(image_files) > max_images_per_identity:
                np.random.seed(42 + idx)
                image_files = list(np.random.choice(
                    image_files, max_images_per_identity, replace=False
                ))

            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img, dtype=np.float32)

                    # Preprocessing
                    img_processed = self.robust_preprocessing(img_array)

                    # 1. Originale
                    images.append(img_processed.flatten())
                    labels.append(idx)

                    # 2. Flip orizzontale
                    img_flipped = np.fliplr(img_processed)
                    images.append(img_flipped.flatten())
                    labels.append(idx)

                    # 3. Rotazione leggera
                    angle = np.random.choice([-7, -5, 5, 7])
                    img_rotated = transform.rotate(
                        img_processed, angle, mode='edge', preserve_range=True
                    )
                    images.append(img_rotated.flatten())
                    labels.append(idx)

                except Exception:
                    pass

        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        print(f"\n✅ Dataset caricato!")
        print(f"   • Campioni totali: {len(X):,}")
        print(f"   • Features: {X.shape[1]:,}")
        print(f"   • Identità: {len(np.unique(y))}")
        print(f"   • Campioni/identità: ~{len(X) // len(np.unique(y))}")

        del images
        gc.collect()

        return X, y

    def apply_optimal_pca(self, X_train, X_test, variance_ratio=0.98):
        """PCA ottimale con controllo varianza"""
        print(f"\n{'=' * 70}")
        print(f"APPLICAZIONE PCA (varianza target: {variance_ratio * 100:.0f}%)")
        print(f"{'=' * 70}")

        start = time.time()

        # Standardizzazione
        print(f"🔧 Standardizzazione...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Determina numero componenti ottimale
        max_comp = min(500, X_train.shape[0] - 1, X_train.shape[1])

        print(f"🔧 Calcolo componenti ottimali (max: {max_comp})...")
        pca_test = PCA(n_components=max_comp, random_state=42)
        pca_test.fit(X_train_scaled)

        cumsum_var = np.cumsum(pca_test.explained_variance_ratio_)
        # n_components = np.argmax(cumsum_var >= variance_ratio) + 1
        # n_components = max(200, min(n_components, 400))  # Range 200-400
        n_components = 300

        print(f"   • Componenti selezionate: {n_components}")
        print(f"   • Varianza con {n_components} comp: {cumsum_var[n_components - 1] * 100:.2f}%")

        # PCA finale
        self.pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)

        variance = np.sum(self.pca.explained_variance_ratio_)

        print(f"\n✅ PCA completata in {time.time() - start:.2f}s")
        print(f"   • Riduzione: {X_train.shape[1]:,} → {n_components}")
        print(f"   • Varianza spiegata: {variance * 100:.2f}%")

        return X_train_pca, X_test_pca

    def train_optimal_svm(self, X_train, y_train):
        """SVM con parametri ottimizzati"""
        print(f"\n{'=' * 70}")
        print(f"TRAINING SVM OTTIMIZZATO")
        print(f"{'=' * 70}")

        # SVM RBF con parametri bilanciati
        model = SVC(
            kernel='rbf',
            #C=10.0,
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            cache_size=3000,
            random_state=42
        )

        print(f"\n🤖 Modello: SVM RBF")
        print(f"   • C=10.0 (bilanciato)")
        print(f"   • gamma='scale'")
        print(f"   • class_weight='balanced'")

        # Cross-validation
        print(f"\n🔄 Cross-validation 5-fold...")
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1
        )

        print(f"   • CV Mean: {cv_scores.mean() * 100:.2f}%")
        print(f"   • CV Std: ±{cv_scores.std() * 100:.2f}%")

        # Training
        print(f"\n🔧 Training finale...")
        start = time.time()
        model.fit(X_train, y_train)
        print(f"   ✅ Completato in {time.time() - start:.2f}s")

        train_acc = model.score(X_train, y_train)
        print(f"   • Training accuracy: {train_acc * 100:.2f}%")

        return model, cv_scores

    def evaluate(self, model, X_train, X_test, y_train, y_test, cv_scores):
        """Valutazione completa"""
        print(f"\n{'=' * 70}")
        print(f"VALUTAZIONE FINALE")
        print(f"{'=' * 70}")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        cv_mean = cv_scores.mean()

        print(f"\n📊 RISULTATI:")
        print(f"   • Test Accuracy:     {test_acc * 100:.2f}% ⭐")
        print(f"   • CV Accuracy:       {cv_mean * 100:.2f}%")
        print(f"   • Training Accuracy: {train_acc * 100:.2f}%")
        print(f"   • Gap Train-Test:    {(train_acc - test_acc) * 100:.2f}%")

        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='weighted', zero_division=0
        )

        print(f"\n📈 METRICHE:")
        print(f"   • Precision: {precision * 100:.2f}%")
        print(f"   • Recall:    {recall * 100:.2f}%")
        print(f"   • F1-Score:  {f1 * 100:.2f}%")

        return {
            'test_acc': test_acc,
            'train_acc': train_acc,
            'cv_acc': cv_mean,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_test_pred': y_test_pred
        }

    def plot_results(self, results, y_test, save_path='results'):
        """Grafici risultati"""
        os.makedirs(save_path, exist_ok=True)

        # Accuracy plot
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Test', 'CV', 'Training']
        values = [
            results['test_acc'] * 100,
            results['cv_acc'] * 100,
            results['train_acc'] * 100
        ]
        colors = ['#3498db', '#9b59b6', '#2ecc71']

        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Performance - Test Accuracy: {results["test_acc"] * 100:.2f}%',
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.tight_layout()
        plt.savefig(f'{save_path}/accuracy.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Salvato: {save_path}/accuracy.png")
        plt.close()


def analyze_dataset(dataset_path):
    """Analisi dataset"""
    print(f"\n{'=' * 70}")
    print(f"ANALISI DATASET")
    print(f"{'=' * 70}")

    folders = sorted([f for f in Path(dataset_path).iterdir() if f.is_dir()])

    distribution = []
    for folder in tqdm(folders, desc="Analisi"):
        n_images = len(list(folder.glob("*.jpg")) +
                       list(folder.glob("*.png")) +
                       list(folder.glob("*.jpeg")))
        distribution.append({
            'folder': folder,
            'name': folder.name,
            'count': n_images
        })

    distribution = sorted(distribution, key=lambda x: x['count'], reverse=True)
    counts = [d['count'] for d in distribution]

    print(f"\n📊 STATISTICHE:")
    print(f"   • Identità totali: {len(folders)}")
    print(f"   • Immagini totali: {sum(counts):,}")
    print(f"   • Media: {np.mean(counts):.1f}")
    print(f"   • Min: {min(counts)}, Max: {max(counts)}")

    return distribution


def select_identities(distribution, n_identities, min_samples):
    """Selezione identità"""
    valid = [d for d in distribution if d['count'] >= min_samples]
    selected = valid[:n_identities]

    print(f"\n💡 SELEZIONE:")
    print(f"   • Identità con ≥{min_samples} img: {len(valid)}")
    print(f"   • Identità selezionate: {len(selected)}")
    print(f"   • Immagini totali: {sum(d['count'] for d in selected):,}")

    return selected


def main():
    print("\n" + "=" * 70)
    print("  SVD FACE RECOGNITION - VERSIONE OTTIMIZZATA")
    print("=" * 70 + "\n")

    DATASET_PATH = "DataSet_Tre/dataset"

    # PARAMETRI OTTIMIZZATI
    CONFIG = {
        'N_IDENTITIES': 100,  # Ridotto per focus su qualità
        'MIN_SAMPLES': 60,  # Soglia più alta
        'MAX_IMAGES_PER_ID': 120,  # Più dati per identità
        'IMG_SIZE': (128, 128),  # Risoluzione maggiore
        'PCA_VARIANCE': 0.98,  # Alta varianza
        'TEST_SIZE': 0.20
    }

    print("⚙️  CONFIGURAZIONE:")
    print(f"   • Identità target: {CONFIG['N_IDENTITIES']}")
    print(f"   • Min samples/identità: {CONFIG['MIN_SAMPLES']}")
    print(f"   • Max images/identità: {CONFIG['MAX_IMAGES_PER_ID']}")
    print(f"   • Dimensione immagini: {CONFIG['IMG_SIZE']}")
    print(f"   • PCA varianza: {CONFIG['PCA_VARIANCE'] * 100:.0f}%")
    print(f"   • Augmentation: 3x")
    print(f"   • Test split: {CONFIG['TEST_SIZE'] * 100:.0f}%")

    # Analisi dataset
    distribution = analyze_dataset(DATASET_PATH)
    selected = select_identities(
        distribution,
        CONFIG['N_IDENTITIES'],
        CONFIG['MIN_SAMPLES']
    )

    if not selected:
        print("\n❌ Nessuna identità valida trovata!")
        return

    input("\n⏸️  Premi ENTER per continuare...")

    # Inizializza sistema
    system = OptimalFaceRecognition(
        dataset_path=DATASET_PATH,
        img_size=CONFIG['IMG_SIZE']
    )

    # Carica dataset
    X, y = system.load_balanced_dataset(
        selected_identities=selected,
        max_images_per_identity=CONFIG['MAX_IMAGES_PER_ID']
    )

    # Split
    print(f"\n{'=' * 70}")
    print(f"SPLIT TRAIN/TEST")
    print(f"{'=' * 70}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['TEST_SIZE'], random_state=42, stratify=y
    )
    print(f"✅ Training: {len(X_train):,} | Test: {len(X_test):,}")

    del X, y
    gc.collect()

    # PCA
    X_train_pca, X_test_pca = system.apply_optimal_pca(
        X_train, X_test, variance_ratio=CONFIG['PCA_VARIANCE']
    )

    del X_train, X_test
    gc.collect()

    # Training
    model, cv_scores = system.train_optimal_svm(X_train_pca, y_train)

    # Valutazione
    results = system.evaluate(model, X_train_pca, X_test_pca, y_train, y_test, cv_scores)

    # Grafici
    system.plot_results(results, y_test)

    # Riepilogo
    print(f"\n{'=' * 70}")
    print(f"🏆 RIEPILOGO FINALE")
    print(f"{'=' * 70}")
    print(f"\n   📊 Dataset: {len(selected)} identità, {len(y_train) + len(y_test):,} campioni")
    print(f"   🎯 Test Accuracy: {results['test_acc'] * 100:.2f}%")
    print(f"   📈 F1-Score: {results['f1'] * 100:.2f}%")
    print(f"   📉 Gap: {(results['train_acc'] - results['test_acc']) * 100:.2f}%")

    if results['test_acc'] >= 0.75:
        print(f"\n   🏆 ECCELLENTE! Performance >75%")
    elif results['test_acc'] >= 0.65:
        print(f"\n   ✅ BUONO! Performance solida")
    else:
        print(f"\n   ⚠️  Migliorabile - Prova ad aumentare MAX_IMAGES_PER_ID")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()