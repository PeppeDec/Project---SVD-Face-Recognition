import joblib
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from skimage import exposure

BUNDLE_PATH = "results/svd_face_model.joblib"

def robust_preprocessing(img_array: np.ndarray) -> np.ndarray:
    img_array = img_array.astype(np.float32) / 255.0
    img_array = exposure.equalize_adapthist(img_array, clip_limit=0.03)
    mean = np.mean(img_array)
    std = np.std(img_array)
    if std > 0:
        img_array = (img_array - mean) / std
    return img_array

def preprocess_single_image(img_path: Path, img_size):
    img = Image.open(img_path).convert("L")
    img = img.resize(tuple(img_size), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = robust_preprocessing(arr)
    return arr.flatten().astype(np.float32).reshape(1, -1)

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def predict_topk(bundle, img_path: Path, top_k=5):
    X = preprocess_single_image(img_path, bundle["img_size"])

    Xs = bundle["scaler"].transform(X)
    Xsvd = bundle["svd"].transform(Xs)
    Xsvd = bundle["normalizer"].transform(Xsvd)

    model = bundle["model"]

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(Xsvd)[0]
    elif hasattr(model, "decision_function"):
        scores = softmax(np.asarray(model.decision_function(Xsvd)).ravel())
    else:
        pred = int(model.predict(Xsvd)[0])
        name = bundle["reverse_label_mapping"].get(pred, str(pred))
        return [(name, 1.0)]

    idxs = np.argsort(scores)[::-1][:top_k]
    out = []
    for i in idxs:
        name = bundle["reverse_label_mapping"].get(int(i), str(i))
        out.append((name, float(scores[i])))
    return out

def main():
    if not Path(BUNDLE_PATH).exists():
        print(f"❌ Non trovo {BUNDLE_PATH}. Allena prima il modello con main4.py")
        return

    bundle = joblib.load(BUNDLE_PATH)

    root = tk.Tk()
    root.title("SVD Face Recognition - Inference")
    root.geometry("560x560")

    info = tk.Label(root, text=f"Loaded: {bundle.get('model_name','model')} | img_size={bundle.get('img_size')}")
    info.pack(pady=6)

    img_panel = tk.Label(root)
    img_panel.pack(pady=6)

    result_box = tk.Text(root, height=12, width=70)
    result_box.pack(pady=6)

    state = {"img_path": None, "tkimg": None}

    def show(lines):
        result_box.delete("1.0", tk.END)
        result_box.insert(tk.END, "\n".join(lines))

    def choose_image():
        fp = filedialog.askopenfilename(
            title="Scegli un'immagine",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG"), ("All files", "*")]
        )
        if not fp:
            return
        state["img_path"] = Path(fp)

        try:
            pil_img = Image.open(state["img_path"]).convert("RGB")
            pil_img.thumbnail((380, 380))
            tkimg = ImageTk.PhotoImage(pil_img)
            state["tkimg"] = tkimg
            img_panel.configure(image=tkimg)
            show([f"Selezionata: {state['img_path'].name}", "Premi Predict per classificare."])
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire immagine: {e}")

    def predict_now():
        if state["img_path"] is None:
            messagebox.showwarning("Attenzione", "Seleziona prima un'immagine.")
            return
        try:
            top = predict_topk(bundle, state["img_path"], top_k=5)
            lines = [f"Immagine: {state['img_path'].name}", "", "Top-5 predictions:"]
            for r, (name, sc) in enumerate(top, 1):
                lines.append(f"  {r}) {name} | score={sc*100:.2f}%")
            show(lines)
        except Exception as e:
            messagebox.showerror("Errore", f"Prediction fallita: {e}")

    btn = tk.Frame(root)
    btn.pack(pady=8)
    tk.Button(btn, text="Choose Image", command=choose_image, width=18).grid(row=0, column=0, padx=10)
    tk.Button(btn, text="Predict", command=predict_now, width=18).grid(row=0, column=1, padx=10)

    root.mainloop()

if __name__ == "__main__":
    main()