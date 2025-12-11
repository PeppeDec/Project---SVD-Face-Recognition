import numpy as np
import os
from pathlib import Path
from PIL import Image
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import gc
import time
import warnings

warnings.filterwarnings('ignore')


class SmartDatasetSelector:
    """Selettore intelligente del dataset"""

    @staticmethod
    def analyze_dataset(dataset_path):
        print(f"\n{'=' * 70}")
        print(f"ANALISI DATASET")
        print(f"{'=' * 70}")

        identity_folders = sorted([f for f in Path(dataset_path).iterdir() if f.is_dir()])

        distribution = []
        for folder in tqdm(identity_folders, desc="Analisi identità"):
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

        print(f"\n📊 STATISTICHE DATASET:")
        print(f"   • Identità totali: {len(identity_folders)}")
        print(f"   • Immagini totali: {sum(counts):,}")
        print(f"   • Media immagini/identità: {np.mean(counts):.1f}")
        print(f"   • Mediana: {np.median(counts):.0f}")
        print(f"   • Min: {min(counts)}, Max: {max(counts)}")

        return distribution

    @staticmethod
    def suggest_configuration(distribution, target_identities=None, min_samples=30):
        valid_identities = [d for d in distribution if d['count'] >= min_samples]

        if target_identities:
            selected = valid_identities[:target_identities]
        else:
            selected = valid_identities

        if len(selected) == 0:
            print(f"\n⚠️  ERRORE: Nessuna identità con ≥{min_samples} immagini!")
            return None

        selected_counts = [d['count'] for d in selected]
        total_images = sum(selected_counts)

        print(f"\n💡 CONFIGURAZIONE:")
        print(f"   • Identità selezionate: {len(selected)}")
        print(f"   • Immagini totali: {total_images:,}")
        print(f"   • Media img/identità: {np.mean(selected_counts):.1f}")

        return selected


class FaceRecognitionAntiOverfit:
    """Sistema con AGGRESSIVE tecniche anti-overfitting"""

    def __init__(self, dataset_path, img_size=(112, 112), n_components=150):
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.n_components = n_components
        self.svd = None
        self.scaler = StandardScaler()
        self.label_mapping = {}
        self.reverse_label_mapping = {}

    def load_dataset_with_augmentation(self, selected_identities, samples_per_identity,
                                       use_preprocessing=True):
        """Carica dataset con DATA AUGMENTATION"""
        print(f"\n{'=' * 70}")
        print(f"CARICAMENTO DATASET CON DATA AUGMENTATION")
        print(f"{'=' * 70}")
        print(f"📊 Identità: {len(selected_identities)}")
        print(f"📊 Campioni/identità: {samples_per_identity}")

        images = []
        labels = []

        for idx, identity_info in enumerate(selected_identities):
            self.label_mapping[identity_info['name']] = idx
            self.reverse_label_mapping[idx] = identity_info['name']

        print(f"\n🔄 Caricamento con preprocessing + augmentation...")

        for idx, identity_info in enumerate(tqdm(selected_identities, desc="Caricamento")):
            folder = identity_info['folder']
            image_files = list(folder.glob("*.jpg")) + \
                          list(folder.glob("*.png")) + \
                          list(folder.glob("*.jpeg"))

            # Limita campioni
            if len(image_files) > samples_per_identity:
                np.random.seed(42)
                image_files = list(np.random.choice(image_files, samples_per_identity, replace=False))

            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img, dtype=np.float32)

                    if use_preprocessing:
                        img_array = img_array / 255.0
                        # CLAHE più conservativo
                        img_array = exposure.equalize_adapthist(
                            img_array, clip_limit=0.02, nbins=256
                        )
                        # Normalizzazione standard
                        mean, std = np.mean(img_array), np.std(img_array)
                        if std > 0:
                            img_array = (img_array - mean) / std

                    # Aggiungi immagine originale
                    images.append(img_array.flatten())
                    labels.append(idx)

                    # DATA AUGMENTATION: Aggiungi rumore gaussiano leggero
                    # Questo FORZA il modello a non memorizzare pixel esatti
                    noisy = img_array + np.random.normal(0, 0.05, img_array.shape)
                    images.append(noisy.flatten())
                    labels.append(idx)

                except Exception as e:
                    pass

        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        print(f"\n✅ Dataset caricato con augmentation!")
        print(f"   • Campioni totali: {len(X):,} (2x per augmentation)")
        print(f"   • Features: {X.shape[1]:,}")
        print(f"   • Classi: {len(np.unique(y))}")

        del images
        gc.collect()

        return X, y

    def apply_pca_aggressive(self, X_train, X_test, variance_target=0.90):
        """PCA invece di SVD - più conservativo"""
        print(f"\n{'=' * 70}")
        print(f"APPLICAZIONE PCA (più conservativo di SVD)")
        print(f"{'=' * 70}")

        start_time = time.time()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Usa PCA invece di TruncatedSVD
        max_components = min(self.n_components, X_train.shape[0] - 1, X_train.shape[1])

        self.svd = PCA(
            n_components=max_components,
            random_state=42,
            whiten=True  # WHITENING riduce overfitting
        )

        X_train_pca = self.svd.fit_transform(X_train_scaled)
        X_test_pca = self.svd.transform(X_test_scaled)

        variance_explained = np.sum(self.svd.explained_variance_ratio_)

        print(f"\n✅ PCA completata in {time.time() - start_time:.2f}s")
        print(f"   • Varianza spiegata: {variance_explained * 100:.2f}%")
        print(f"   • Componenti: {X_train_pca.shape[1]}")
        print(f"   • Whitening: ATTIVO (riduce overfitting)")

        return X_train_pca, X_test_pca

    def train_with_cross_validation(self, X_train, y_train, model_type='logistic'):
        """Training con CROSS-VALIDATION per evitare overfitting"""
        print(f"\n{'=' * 70}")
        print(f"TRAINING CON CROSS-VALIDATION (K-FOLD)")
        print(f"{'=' * 70}")

        # Scegli modello SEMPLICE
        if model_type == 'logistic':
            model = LogisticRegression(
                C=0.01,  # MOLTO restrittivo
                penalty='l2',  # Ridge regularization
                max_iter=1000,
                solver='saga',
                multi_class='multinomial',
                random_state=42,
                n_jobs=-1
            )
            print(f"\n🤖 Modello: Logistic Regression")
            print(f"   • C=0.01 (forte regolarizzazione)")
            print(f"   • Penalty: L2 (Ridge)")

        elif model_type == 'svm_linear':
            model = SVC(
                kernel='linear',
                C=0.01,  # MOLTO conservativo
                class_weight='balanced',
                random_state=42,
                cache_size=2000
            )
            print(f"\n🤖 Modello: SVM Linear")
            print(f"   • C=0.01 (MOLTO restrittivo)")

        else:  # random forest
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,  # Limita profondità
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            print(f"\n🤖 Modello: Random Forest")
            print(f"   • Max depth=10 (limitato)")

        # CROSS-VALIDATION per vedere il vero potere predittivo
        print(f"\n🔄 Cross-validation 5-fold...")
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        print(f"\n📊 CV Scores: {cv_scores}")
        print(f"   • CV Mean: {cv_scores.mean() * 100:.2f}%")
        print(f"   • CV Std: ±{cv_scores.std() * 100:.2f}%")

        # Training finale
        print(f"\n🔧 Training modello finale...")
        start_time = time.time()
        model.fit(X_train, y_train)
        print(f"   ✅ Completato in {time.time() - start_time:.2f}s")

        train_acc = model.score(X_train, y_train)
        print(f"   • Training accuracy: {train_acc * 100:.2f}%")
        print(f"   • CV accuracy: {cv_scores.mean() * 100:.2f}%")

        gap_train_cv = train_acc - cv_scores.mean()
        if gap_train_cv > 0.15:
            print(f"   ⚠️  Gap train-CV: {gap_train_cv * 100:.1f}% - OVERFITTING PRESENTE")
        else:
            print(f"   ✅ Gap train-CV basso: {gap_train_cv * 100:.1f}%")

        return model, cv_scores

    def evaluate_comprehensive(self, model, X_train, X_test, y_train, y_test, cv_scores):
        """Valutazione completa"""
        print(f"\n{'=' * 70}")
        print(f"VALUTAZIONE COMPLETA")
        print(f"{'=' * 70}")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        cv_mean = cv_scores.mean()

        print(f"\n{'─' * 70}")
        print(f"📊 RISULTATI FINALI")
        print(f"{'─' * 70}")
        print(f"   Cross-Val Accuracy: {cv_mean * 100:.2f}% (predizione realistica)")
        print(f"   Training Accuracy:  {train_acc * 100:.2f}%")
        print(f"   Test Accuracy:      {test_acc * 100:.2f}%")
        print(f"\n   Gap Train-Test:     {(train_acc - test_acc) * 100:.2f}%")
        print(f"   Gap Train-CV:       {(train_acc - cv_mean) * 100:.2f}%")

        # Analisi gap
        gap = train_acc - test_acc
        if gap < 0.05:
            print(f"   ✅✅ ECCELLENTE! Gap molto basso")
        elif gap < 0.10:
            print(f"   ✅ OTTIMO! Buona generalizzazione")
        elif gap < 0.15:
            print(f"   ✓ ACCETTABILE")
        else:
            print(f"   ⚠️ Ancora overfitting presente")

        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_test_pred, average=None, zero_division=0
        )

        print(f"\n📈 METRICHE:")
        print(f"   • Precision: {np.mean(precision) * 100:.2f}%")
        print(f"   • Recall: {np.mean(recall) * 100:.2f}%")
        print(f"   • F1-Score: {np.mean(f1) * 100:.2f}%")

        return {
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_accuracy': cv_mean,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def plot_results(self, result, y_test, save_path='results'):
        """Visualizzazioni"""
        os.makedirs(save_path, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"GENERAZIONE GRAFICI")
        print(f"{'=' * 70}")

        # Accuracy comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        categories = ['CV (realistico)', 'Training', 'Test']
        accuracies = [
            result['cv_accuracy'] * 100,
            result['train_accuracy'] * 100,
            result['test_accuracy'] * 100
        ]
        colors = ['#9b59b6', '#2ecc71', '#3498db']

        bars = ax.bar(categories, accuracies, color=colors, edgecolor='black',
                      linewidth=2, alpha=0.8, width=0.6)

        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'Performance - Gap Train-Test: {(result["train_accuracy"] - result["test_accuracy"]) * 100:.1f}%',
                     fontsize=15, fontweight='bold', pad=15)
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{acc:.2f}%', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)

        plt.tight_layout()
        plt.savefig(f'{save_path}/accuracy.png', dpi=300, bbox_inches='tight')
        print(f"✅ Salvato: {save_path}/accuracy.png")
        plt.close()

        # Metriche
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        metrics = [
            (result['precision'], 'Precision', '#3498db'),
            (result['recall'], 'Recall', '#2ecc71'),
            (result['f1'], 'F1-Score', '#e74c3c')
        ]

        for ax, (values, name, color) in zip(axes, metrics):
            ax.hist(values, bins=20, color=color, edgecolor='black', alpha=0.7)
            ax.set_xlabel(name, fontsize=11, fontweight='bold')
            ax.set_ylabel('Numero Classi', fontsize=11, fontweight='bold')
            ax.set_title(f'{name} - Media: {np.mean(values):.3f}',
                         fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.axvline(np.mean(values), color='red', linestyle='--',
                       linewidth=2, label='Media')
            ax.legend()

        plt.tight_layout()
        plt.savefig(f'{save_path}/metrics.png', dpi=300, bbox_inches='tight')
        print(f"✅ Salvato: {save_path}/metrics.png")
        plt.close()


def main():
    """Pipeline ANTI-OVERFITTING AGGRESSIVA"""

    print("\n" + "=" * 70)
    print(" " * 3 + "FACE RECOGNITION: ANTI-OVERFITTING AGGRESSIVO")
    print("=" * 70 + "\n")

    DATASET_PATH = "DataSet_Tre/dataset"

    # CONFIGURAZIONE ULTRA-CONSERVATIVA
    print("🎯 STRATEGIA: ULTRA-CONSERVATIVA ANTI-OVERFITTING")
    print("   • POCHE classi (20-30)")
    print("   • MOLTI dati per classe (150+)")
    print("   • Data augmentation")
    print("   • PCA con whitening")
    print("   • Regolarizzazione forte (C=0.01)")
    print("   • Cross-validation per verifica\n")

    CONFIG = {
        'TARGET_IDENTITIES': 100,  # POCHE classi
        'MIN_SAMPLES': 50,  # MOLTI dati
        'SAMPLES_PER_IDENTITY': 150,
        'IMG_SIZE': (96, 96),  # Ridotto per forzare generalizzazione
        'N_COMPONENTS': 100,  # Meno componenti
        'MODEL_TYPE': 'logistic',  # Prova: 'logistic', 'svm_linear', 'random_forest'
        'TEST_SIZE': 0.30  # 30% test
    }

    print(f"⚙️  PARAMETRI:")
    print(f"   • Identità: {CONFIG['TARGET_IDENTITIES']}")
    print(f"   • Campioni/identità: {CONFIG['SAMPLES_PER_IDENTITY']}")
    print(f"   • IMG_SIZE: {CONFIG['IMG_SIZE']}")
    print(f"   • PCA components: {CONFIG['N_COMPONENTS']}")
    print(f"   • Modello: {CONFIG['MODEL_TYPE']}")

    # Analisi
    selector = SmartDatasetSelector()
    distribution = selector.analyze_dataset(DATASET_PATH)

    selected_identities = selector.suggest_configuration(
        distribution,
        target_identities=CONFIG['TARGET_IDENTITIES'],
        min_samples=CONFIG['MIN_SAMPLES']
    )

    if selected_identities is None:
        print("\n❌ Impossibile continuare.")
        return

    input("\nPremi ENTER per continuare...")

    # Caricamento
    face_system = FaceRecognitionAntiOverfit(
        dataset_path=DATASET_PATH,
        img_size=CONFIG['IMG_SIZE'],
        n_components=CONFIG['N_COMPONENTS']
    )

    X, y = face_system.load_dataset_with_augmentation(
        selected_identities=selected_identities,
        samples_per_identity=CONFIG['SAMPLES_PER_IDENTITY'],
        use_preprocessing=True
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
    X_train_pca, X_test_pca = face_system.apply_pca_aggressive(
        X_train, X_test, variance_target=0.90
    )

    del X_train, X_test
    gc.collect()

    # Training con CV
    model, cv_scores = face_system.train_with_cross_validation(
        X_train_pca, y_train,
        model_type=CONFIG['MODEL_TYPE']
    )

    # Valutazione
    result = face_system.evaluate_comprehensive(
        model, X_train_pca, X_test_pca, y_train, y_test, cv_scores
    )

    # Grafici
    face_system.plot_results(result, y_test)

    # Riepilogo
    print(f"\n{'=' * 70}")
    print(f"🏆 RIEPILOGO FINALE")
    print(f"{'=' * 70}")
    print(f"\n   📊 DATASET:")
    print(f"      • Identità: {len(selected_identities)}")
    print(f"      • Training: {len(y_train):,} | Test: {len(y_test):,}")
    print(f"\n   🎯 PERFORMANCE:")
    print(f"      • CV Accuracy (reale):  {result['cv_accuracy'] * 100:.2f}%")
    print(f"      • Test Accuracy:        {result['test_accuracy'] * 100:.2f}%")
    print(f"      • Training Accuracy:    {result['train_accuracy'] * 100:.2f}%")
    print(f"      • Gap Train-Test:       {(result['train_accuracy'] - result['test_accuracy']) * 100:.2f}%")

    if result['test_accuracy'] >= 0.70 and (result['train_accuracy'] - result['test_accuracy']) < 0.15:
        print(f"\n   ✅✅ BUON RISULTATO! Generalizzazione OK!")
    elif result['test_accuracy'] >= 0.60:
        print(f"\n   ✅ Accettabile - prova altri modelli")
        print(f"      • Cambia MODEL_TYPE: 'random_forest' o 'svm_linear'")
    else:
        print(f"\n   ⚠️  Risultato basso - suggerimenti:")
        print(f"      • Aumenta MIN_SAMPLES a 200")
        print(f"      • Riduci TARGET_IDENTITIES a 15-20")
        print(f"      • Aumenta IMG_SIZE a (96, 96)")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()