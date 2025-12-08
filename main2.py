import numpy as np
import matplotlib.pyplot as plt


    def __init__(self, dataset_path, img_size=(112, 112), n_components=300):
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.n_components = n_components
        self.svd = None
        self.scaler = StandardScaler()
        self.label_mapping = {}
        self.reverse_label_mapping = {}

    def load_dataset_smart(self, selected_identities, samples_per_identity=None,
                           use_preprocessing=True, balance_classes=True):
        """
        Carica dataset con identità pre-selezionate

        balance_classes: se True, equilibra il numero di campioni per classe
        """
        print(f"\n{'=' * 70}")
        print(f"CARICAMENTO DATASET INTELLIGENTE")
        print(f"{'=' * 70}")
        print(f"📊 Identità da caricare: {len(selected_identities)}")

        if balance_classes:
            # Trova il numero di campioni della classe più piccola
            min_count = min(d['count'] for d in selected_identities)
            if samples_per_identity:
                target_per_class = min(samples_per_identity, min_count)
            else:
                target_per_class = min_count

            print(f"⚖️  Bilanciamento classi: {target_per_class} campioni per identità")
        else:
            target_per_class = samples_per_identity

        images = []
        labels = []

        # Mapping
        for idx, identity_info in enumerate(selected_identities):
            self.label_mapping[identity_info['name']] = idx
            self.reverse_label_mapping[idx] = identity_info['name']

        print(f"\n🔄 Caricamento con preprocessing avanzato...")
        if use_preprocessing:
            print("   ✨ CLAHE + Normalizzazione robusta")

        for idx, identity_info in enumerate(tqdm(selected_identities, desc="Caricamento")):
            folder = identity_info['folder']
            image_files = list(folder.glob("*.jpg")) + \
                          list(folder.glob("*.png")) + \
                          list(folder.glob("*.jpeg"))

            # Limita campioni
            if target_per_class:
                # Seleziona casualmente per variabilità
                if len(image_files) > target_per_class:
                    np.random.seed(42)
                    image_files = list(np.random.choice(image_files, target_per_class, replace=False))

            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img, dtype=np.float32)

                    if use_preprocessing:
                        # CLAHE
                        img_array = img_array / 255.0
                        img_array = exposure.equalize_adapthist(
                            img_array, clip_limit=0.01, nbins=256
                        )

                        # Normalizzazione robusta
                        mean, std = np.mean(img_array), np.std(img_array)
                        if std > 0:
                            img_array = (img_array - mean) / (std + 1e-7)
                        img_array = np.clip(img_array, -3, 3)
                    else:
                        img_array = img_array / 255.0

                    images.append(img_array.flatten())
                    labels.append(idx)

                except Exception as e:
                    pass

        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        print(f"\n✅ Dataset caricato!")
        print(f"   • Campioni totali: {len(X):,}")
        print(f"   • Features: {X.shape[1]:,}")
        print(f"   • Classi: {len(np.unique(y))}")
        print(f"   • Memoria: ~{X.nbytes / (1024 ** 2):.1f} MB")

        class_counts = np.bincount(y)
        print(
            f"   • Campioni/classe: min={class_counts.min()}, max={class_counts.max()}, media={class_counts.mean():.1f}")

        if class_counts.max() - class_counts.min() > class_counts.mean() * 0.3:
            print(f"   ⚠️  Dataset sbilanciato! Considera balance_classes=True")
        else:
            print(f"   ✅ Dataset ben bilanciato")

        del images
        gc.collect()

        return X, y

    def apply_svd_with_validation(self, X_train, X_test, target_variance=0.95):
        """SVD con validazione ottimale componenti"""
        print(f"\n{'=' * 70}")
        print(f"APPLICAZIONE SVD CON VALIDAZIONE")
        print(f"{'=' * 70}")

        start_time = time.time()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"🔧 Target varianza: {target_variance * 100:.0f}%")
        print(f"🔧 Componenti richieste: {self.n_components}")

        self.svd = TruncatedSVD(
            n_components=self.n_components,
            algorithm='randomized',
            n_iter=7,
            random_state=42
        )

        X_train_svd = self.svd.fit_transform(X_train_scaled)
        X_test_svd = self.svd.transform(X_test_scaled)

        elapsed = time.time() - start_time
        variance_explained = np.sum(self.svd.explained_variance_ratio_)

        print(f"\n✅ SVD completata in {elapsed:.2f}s")
        print(f"   • Varianza spiegata: {variance_explained * 100:.2f}%")
        print(f"   • Compressione: {X_train.shape[1]:,} → {X_train_svd.shape[1]} features")

        # Validazione
        if variance_explained < target_variance:
            cum_var = np.cumsum(self.svd.explained_variance_ratio_)
            needed = np.argmax(cum_var >= target_variance) + 1
            print(f"\n⚠️  ATTENZIONE: Varianza sotto target!")
            print(f"   Servono {needed} componenti per {target_variance * 100:.0f}% varianza")
            print(f"   Considera di aumentare n_components a {needed}")
        else:
            print(f"   ✅ Target varianza raggiunto!")

        return X_train_svd, X_test_svd

    def train_svm_with_validation(self, X_train, y_train, X_val=None, y_val=None):
        """Training SVM con validazione opzionale"""
        print(f"\n{'=' * 70}")
        print(f"TRAINING SVM OTTIMIZZATO")
        print(f"{'=' * 70}")

        print(f"\n🤖 Configurazione SVM:")
        print(f"   • Kernel: RBF")
        print(f"   • C: 10.0 (regolarizzazione moderata)")
        print(f"   • Gamma: scale (automatico)")
        print(f"   • Class weight: balanced")

        svm = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            cache_size=1000,
            class_weight='balanced',
            random_state=42,
            verbose=False
        )

        start_time = time.time()
        svm.fit(X_train, y_train)
        train_time = time.time() - start_time

        print(f"\n✅ Training completato in {train_time:.2f}s")

        # Validazione su train set
        train_acc = svm.score(X_train, y_train)
        print(f"   • Training accuracy: {train_acc * 100:.2f}%")

        # Validazione opzionale
        if X_val is not None and y_val is not None:
            val_acc = svm.score(X_val, y_val)
            print(f"   • Validation accuracy: {val_acc * 100:.2f}%")

            if train_acc - val_acc > 0.20:
                print(f"   ⚠️  Gap train-val alto ({(train_acc - val_acc) * 100:.1f}%) - possibile overfitting")

        return svm

    def evaluate_comprehensive(self, model, X_train, X_test, y_train, y_test):
        """Valutazione completa"""
        print(f"\n{'=' * 70}")
        print(f"VALUTAZIONE COMPLETA")
        print(f"{'=' * 70}")

        start_time = time.time()

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        pred_time = time.time() - start_time

        print(f"\n{'─' * 70}")
        print(f"📊 RISULTATI FINALI")
        print(f"{'─' * 70}")
        print(f"   Training Accuracy:  {train_acc * 100:.2f}%")
        print(f"   Test Accuracy:      {test_acc * 100:.2f}%")
        print(f"   Gap:                {(train_acc - test_acc) * 100:.2f}%")

        # Analisi gap
        gap = train_acc - test_acc
        if gap < 0.05:
            print(f"   ✅✅ ECCELLENTE! Gap molto basso")
        elif gap < 0.10:
            print(f"   ✅ OTTIMO! Buona generalizzazione")
        elif gap < 0.15:
            print(f"   ✓ ACCETTABILE")
        elif gap < 0.25:
            print(f"   ⚠️  Gap moderato - leggero overfitting")
        else:
            print(f"   ❌ Gap alto - overfitting significativo")

        # Analisi per classe
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_test_pred, average=None, zero_division=0
        )

        print(f"\n📈 STATISTICHE PER CLASSE:")
        print(f"   • Precision media: {np.mean(precision) * 100:.2f}%")
        print(f"   • Recall media: {np.mean(recall) * 100:.2f}%")
        print(f"   • F1-Score media: {np.mean(f1) * 100:.2f}%")

        # Identifica classi problematiche
        worst_classes = np.argsort(f1)[:5]
        if len(worst_classes) > 0 and f1[worst_classes[0]] < 0.5:
            print(f"\n⚠️  Classi con performance basse:")
            for idx in worst_classes[:3]:
                if f1[idx] < 0.7:
                    print(f"      Classe {idx}: F1={f1[idx]:.2f}, Support={support[idx]}")

        return {
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'prediction_time': pred_time,
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

        # 1. Train vs Test Accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Training', 'Test']
        accuracies = [result['train_accuracy'] * 100, result['test_accuracy'] * 100]
        colors = ['#2ecc71' if result['train_accuracy'] < 0.95 else '#e74c3c',
                  '#3498db' if result['test_accuracy'] >= 0.75 else '#e67e22']

        bars = ax.bar(categories, accuracies, color=colors, edgecolor='black',
                      linewidth=2, alpha=0.8, width=0.5)

        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'Performance SVM - Gap: {(result["train_accuracy"] - result["test_accuracy"]) * 100:.1f}%',
                     fontsize=15, fontweight='bold', pad=15)
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{acc:.2f}%', ha='center', va='bottom',
                    fontweight='bold', fontsize=12)

        plt.tight_layout()
        plt.savefig(f'{save_path}/accuracy.png', dpi=300, bbox_inches='tight')
        print(f"✅ Salvato: {save_path}/accuracy.png")
        plt.close()

        # 2. Metriche per classe
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
                       linewidth=2, label=f'Media')
            ax.legend()

        plt.tight_layout()
        plt.savefig(f'{save_path}/metrics_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✅ Salvato: {save_path}/metrics_distribution.png")
        plt.close()

        # 3. SVD Variance
        if self.svd is not None:
            fig, ax = plt.subplots(figsize=(12, 6))
            cum_var = np.cumsum(self.svd.explained_variance_ratio_) * 100

            ax.plot(range(1, len(cum_var) + 1), cum_var,
                    'b-', linewidth=2.5, marker='o', markersize=3, label='Varianza Cumulativa')
            ax.axhline(y=95, color='green', linestyle='--', linewidth=2, label='95% Target')
            ax.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90%')

            ax.set_xlabel('Numero Componenti SVD', fontsize=12, fontweight='bold')
            ax.set_ylabel('Varianza Spiegata (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Analisi SVD - {self.n_components} componenti = {cum_var[-1]:.2f}% varianza',
                         fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{save_path}/svd_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✅ Salvato: {save_path}/svd_analysis.png")
            plt.close()

        print(f"\n✅ Grafici salvati in '{save_path}/'")


# ============================================================================
# FUNZIONE PRINCIPALE CON SELEZIONE INTELLIGENTE
# ============================================================================

def main():
    """Pipeline intelligente per Face Recognition"""

    print("\n" + "=" * 70)
    print(" " * 8 + "FACE RECOGNITION: SELEZIONE INTELLIGENTE DATASET")
    print("=" * 70 + "\n")

    DATASET_PATH = "../Prova4/dataset"  # Percorso del tuo dataset


    # ========================================================================
    # FASE 1: ANALISI DATASET
    # ========================================================================

    selector = SmartDatasetSelector()
    distribution = selector.analyze_dataset(DATASET_PATH)

    # ========================================================================
    # FASE 2: CONFIGURAZIONE INTELLIGENTE
    # ========================================================================

    print(f"\n{'=' * 70}")
    print(f"CONFIGURAZIONE PARAMETRI")
    print(f"{'=' * 70}")

    # OPZIONI DISPONIBILI (scegli una strategia):

    # STRATEGIA 1: MASSIMA ACCURACY (CONSIGLIATA)
    TARGET_IDENTITIES = None  # Usa tutte quelle valide
    MIN_SAMPLES = 35  # Solo identità ben rappresentate
    BALANCE_CLASSES = True  # Bilancia campioni per classe
    SAMPLES_PER_IDENTITY = 60  # Max campioni per identità (per bilanciamento)

    # STRATEGIA 2: DATASET GRANDE
    # TARGET_IDENTITIES = 300
    # MIN_SAMPLES = 30
    # BALANCE_CLASSES = True
    # SAMPLES_PER_IDENTITY = 50

    # STRATEGIA 3: FOCUS QUALITÀ
    # TARGET_IDENTITIES = 100
    # MIN_SAMPLES = 50
    # BALANCE_CLASSES = True
    # SAMPLES_PER_IDENTITY = 80

    # Parametri tecnici
    IMG_SIZE = (112, 112)  # Buon compromesso qualità/memoria
    N_COMPONENTS = 300  # Componenti SVD
    TARGET_VARIANCE = 0.95  # Target varianza spiegata
    TEST_SIZE = 0.25  # 25% per test

    print(f"\n⚙️  PARAMETRI SCELTI:")
    print(f"   • Target identità: {TARGET_IDENTITIES if TARGET_IDENTITIES else 'Tutte valide'}")
    print(f"   • Min campioni/identità: {MIN_SAMPLES}")
    print(f"   • Bilanciamento classi: {BALANCE_CLASSES}")
    print(f"   • Max campioni/identità: {SAMPLES_PER_IDENTITY if BALANCE_CLASSES else 'Tutti'}")
    print(f"   • Dimensione immagini: {IMG_SIZE}")
    print(f"   • Componenti SVD: {N_COMPONENTS}")
    print(f"   • Target varianza: {TARGET_VARIANCE * 100:.0f}%")

    # Selezione identità
    selected_identities = selector.suggest_configuration(
        distribution,
        target_identities=TARGET_IDENTITIES,
        min_samples=MIN_SAMPLES
    )

    if selected_identities is None:
        print("\n❌ Impossibile continuare. Riduci MIN_SAMPLES.")
        return

    # Conferma utente
    print(f"\n{'=' * 70}")
    input("Premi ENTER per continuare con questa configurazione...")

    # ========================================================================
    # FASE 3: CARICAMENTO E TRAINING
    # ========================================================================

    face_system = FaceRecognitionSVD(
        dataset_path=DATASET_PATH,
        img_size=IMG_SIZE,
        n_components=N_COMPONENTS
    )

    # Carica dataset
    X, y = face_system.load_dataset_smart(
        selected_identities=selected_identities,
        samples_per_identity=SAMPLES_PER_IDENTITY if BALANCE_CLASSES else None,
        use_preprocessing=True,
        balance_classes=BALANCE_CLASSES
    )

    # Split
    print(f"\n{'=' * 70}")
    print(f"SPLIT TRAIN/TEST STRATIFICATO")
    print(f"{'=' * 70}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"✅ Training: {len(X_train):,} | Test: {len(X_test):,}")

    del X, y
    gc.collect()

    # SVD
    X_train_svd, X_test_svd = face_system.apply_svd_with_validation(
        X_train, X_test, target_variance=TARGET_VARIANCE
    )

    del X_train, X_test
    gc.collect()

    # Training
    svm_model = face_system.train_svm_with_validation(X_train_svd, y_train)

    # Valutazione
    result = face_system.evaluate_comprehensive(
        svm_model, X_train_svd, X_test_svd, y_train, y_test
    )

    # Visualizzazioni
    face_system.plot_results(result, y_test)

    # ========================================================================
    # RIEPILOGO FINALE
    # ========================================================================

    print(f"\n{'=' * 70}")
    print(f"🏆 RIEPILOGO FINALE")
    print(f"{'=' * 70}")
    print(f"\n   📊 DATASET:")
    print(f"      • Identità utilizzate: {len(selected_identities)}")
    print(f"      • Campioni totali: {len(y_train) + len(y_test):,}")
    print(f"      • Training samples: {len(y_train):,}")
    print(f"      • Test samples: {len(y_test):,}")
    print(f"\n   🎯 PERFORMANCE:")
    print(f"      • Test Accuracy: {result['test_accuracy'] * 100:.2f}%")
    print(f"      • Training Accuracy: {result['train_accuracy'] * 100:.2f}%")
    print(f"      • Gap: {(result['train_accuracy'] - result['test_accuracy']) * 100:.2f}%")

    if result['test_accuracy'] >= 0.80:
        print(f"\n   ✅✅ OBIETTIVO RAGGIUNTO! (≥80%)")
    elif result['test_accuracy'] >= 0.75:
        print(f"\n   ✅ Molto vicino all'obiettivo!")
        print(f"      Suggerimento: Aumenta N_COMPONENTS o MIN_SAMPLES")
    else:
        print(f"\n   📈 SUGGERIMENTI:")
        if result['train_accuracy'] - result['test_accuracy'] > 0.20:
            print(f"      • OVERFITTING: Aumenta MIN_SAMPLES a {MIN_SAMPLES + 10}")
            print(f"      • Riduci N_COMPONENTS a {N_COMPONENTS - 50}")
        else:
            print(f"      • Aumenta N_COMPONENTS a {N_COMPONENTS + 100}")
            print(f"      • Aumenta MIN_SAMPLES a {MIN_SAMPLES + 5}")
            print(f"      • Prova IMG_SIZE = (128, 128)")

    print(f"\n   📁 Risultati salvati in: ./results/")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()