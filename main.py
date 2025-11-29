import numpy as np
from PIL import Image
import os


def load_dataset(dataset_path):
    images = []
    labels = []

    for subject_id in range(1, 41):
        folder = os.path.join(dataset_path, f's{subject_id}')
        for img_id in range(1, 11):
            img_path = os.path.join(folder, f'{img_id}.pgm')
            img = Image.open(img_path)
            images.append(np.array(img).flatten())
            labels.append(subject_id)

    return np.array(images), np.array(labels)

from sklearn.model_selection import train_test_split

def split_dataset(images, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size,
        stratify=labels, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


class PCAFaceRecognizer:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None

    def fit(self, X_train):
        # Calcola la faccia media
        self.mean_face = np.mean(X_train, axis=0)

        # Centra i dati
        X_centered = X_train - self.mean_face

        # Applica SVD
        U, S, Vt = np.linalg.svd(X_centered.T, full_matrices=False)

        # Seleziona le prime n_components eigenfaces
        self.eigenfaces = U[:, :self.n_components]

        return self

    def transform(self, X):
        # Proietta le immagini nello spazio delle eigenfaces
        X_centered = X - self.mean_face
        return X_centered @ self.eigenfaces

from sklearn.neighbors import KNeighborsClassifier

def train_classifier(X_train_pca, y_train):
    classifier = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    classifier.fit(X_train_pca, y_train)
    return classifier


if __name__ == "__main__":
    # Carica dataset_yale
    dataset_path = "dataset"
    images, labels = load_dataset(dataset_path)

    # Split
    X_train, X_test, y_train, y_test = split_dataset(images, labels)

    # PCA
    pca = PCAFaceRecognizer(n_components=50)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train classifier
    classifier = train_classifier(X_train_pca, y_train)

    # Predict
    y_pred = classifier.predict(X_test_pca)

    # Accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
