#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def safe_read_image(path: Path) -> Optional[np.ndarray]:
    """Reads image with OpenCV (BGR). Returns None if fails."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def compute_basic_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    """
    Basic quality metrics (no detector needed):
    - sharpness_lap_var: variance of Laplacian on gray
    - luminance_mean: mean gray intensity
    - contrast_std: std of gray intensity
    - pct_dark: % pixels close to black
    - pct_bright: % pixels close to white
    - pct_saturated: % pixels saturated (dark or bright)
    - noise_proxy: std of (gray - gaussian_blur(gray))
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Sharpness / blur proxy
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_lap_var = float(lap.var())

    # Illumination and contrast
    luminance_mean = float(gray.mean())
    contrast_std = float(gray.std())

    # Saturation/extremes
    dark_thr = 10
    bright_thr = 245
    pct_dark = float((gray <= dark_thr).mean() * 100.0)
    pct_bright = float((gray >= bright_thr).mean() * 100.0)
    pct_saturated = pct_dark + pct_bright

    # Noise proxy
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    resid = (gray.astype(np.float32) - blur.astype(np.float32))
    noise_proxy = float(resid.std())

    h, w = gray.shape[:2]
    return {
        "img_h": float(h),
        "img_w": float(w),
        "sharpness_lap_var": sharpness_lap_var,
        "luminance_mean": luminance_mean,
        "contrast_std": contrast_std,
        "pct_dark": pct_dark,
        "pct_bright": pct_bright,
        "pct_saturated": pct_saturated,
        "noise_proxy": noise_proxy,
    }


def get_face_detector(detector_name: str) -> Optional[cv2.CascadeClassifier]:
    """
    Returns Haar cascade detector from OpenCV.
    detector_name options:
      - "haar"  -> haarcascade_frontalface_default.xml
      - "alt"   -> haarcascade_frontalface_alt2.xml
      - "none"  -> None
    """
    if detector_name == "none":
        return None

    if detector_name == "haar":
        fname = "haarcascade_frontalface_default.xml"
    else:
        fname = "haarcascade_frontalface_alt2.xml"

    cascade_path = cv2.data.haarcascades + fname
    det = cv2.CascadeClassifier(cascade_path)
    if det.empty():
        return None
    return det


def detect_primary_face(det: cv2.CascadeClassifier, img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Detect faces and return the biggest (x,y,w,h).
    If none found, returns (-1,-1,-1,-1).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = det.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(40, 40),
    )
    if faces is None or len(faces) == 0:
        return (-1, -1, -1, -1)

    # Choose biggest face (by area)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    return (int(x), int(y), int(w), int(h))


def compute_face_metrics(img_bgr: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
    """
    Face-related quality metrics (requires bbox):
    - face_detected: 0/1
    - face_area_ratio: face_area / image_area
    - face_center_dx_norm, face_center_dy_norm: distance from image center (normalized by width/height)
    """
    h, w = img_bgr.shape[:2]
    x, y, fw, fh = face_bbox

    if fw <= 0 or fh <= 0:
        return {
            "face_detected": 0.0,
            "face_area_ratio": 0.0,
            "face_center_dx_norm": np.nan,
            "face_center_dy_norm": np.nan,
        }

    face_area_ratio = float((fw * fh) / (w * h))

    img_cx, img_cy = w / 2.0, h / 2.0
    face_cx, face_cy = (x + fw / 2.0), (y + fh / 2.0)

    face_center_dx_norm = float((face_cx - img_cx) / w)
    face_center_dy_norm = float((face_cy - img_cy) / h)

    return {
        "face_detected": 1.0,
        "face_area_ratio": face_area_ratio,
        "face_center_dx_norm": face_center_dx_norm,
        "face_center_dy_norm": face_center_dy_norm,
    }


def quality_score(row: pd.Series) -> float:
    """
    Simple 0-100 quality score to use in plots/slides.
    It's a heuristic: sharpness + contrast, penalize saturation and low face ratio, bonus if face detected.
    """
    # Normalize with soft saturations (avoid division by zero)
    sharp = row.get("sharpness_lap_var", 0.0)
    contr = row.get("contrast_std", 0.0)
    sat = row.get("pct_saturated", 0.0)
    noise = row.get("noise_proxy", 0.0)
    face_det = row.get("face_detected", 0.0)
    face_ratio = row.get("face_area_ratio", 0.0)

    # Heuristic scales (tune if needed)
    sharp_n = np.tanh(sharp / 150.0)          # ~0..1
    contr_n = np.tanh(contr / 50.0)           # ~0..1
    noise_pen = np.tanh(noise / 40.0)         # ~0..1 (penalty)
    sat_pen = np.tanh(sat / 20.0)             # ~0..1 (penalty)

    # Face ratio: prefer medium/large faces, but clamp
    face_n = np.clip(face_ratio / 0.12, 0.0, 1.0)  # 0.12 means face ~12% of image

    score = (
        55.0 * sharp_n +
        25.0 * contr_n +
        15.0 * face_n +
        10.0 * float(face_det) -
        15.0 * sat_pen -
        10.0 * noise_pen
    )
    # Clamp to 0..100
    return float(np.clip(score, 0.0, 100.0))


def plot_hist(df: pd.DataFrame, col: str, outpath: Path, title: str):
    # Robust histogram: force numeric 1D array and drop non-finite values.
    series = pd.to_numeric(df[col], errors="coerce")
    arr = series.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    # Use explicit bin edges (bins+1) to avoid rare numpy broadcasting issues.
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmin == vmax:
        # Degenerate case: all values identical
        vmin -= 0.5
        vmax += 0.5
    bin_edges = np.linspace(vmin, vmax, 41)  # 40 bins -> 41 edges

    plt.figure()
    plt.hist(arr, bins=bin_edges)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_box_by_group(df: pd.DataFrame, col: str, group_col: str, outpath: Path, title: str, max_groups: int = 20):
    """
    Boxplot by group, limited to top groups by count (to keep slides readable).
    """
    tmp = df[[group_col, col]].replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return

    counts = tmp[group_col].value_counts()
    groups = counts.head(max_groups).index.tolist()

    data = [tmp.loc[tmp[group_col] == g, col].values for g in groups]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=groups, showfliers=False)
    plt.title(title)
    plt.xlabel(group_col)
    plt.ylabel(col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Compute image quality metrics for a dataset and export CSV + plots.")
    ap.add_argument("--dataset", required=True, help="Path to dataset root folder")
    ap.add_argument("--outdir", default="quality_out", help="Output folder for CSV and plots")
    ap.add_argument("--group_by", default="parent", choices=["parent", "none"],
                    help="Grouping for summaries: 'parent' uses immediate parent folder as identity; 'none' no grouping")
    ap.add_argument("--detector", default="haar", choices=["haar", "alt", "none"],
                    help="Face detector: haar/alt/none")
    ap.add_argument("--max_images", type=int, default=0, help="If >0, process only first N images (debug)")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    det = get_face_detector(args.detector)
    use_det = det is not None

    # Collect image paths
    all_imgs: List[Path] = []
    for p in dataset_path.rglob("*"):
        if p.is_file() and is_image_file(p):
            all_imgs.append(p)

    all_imgs.sort()
    if args.max_images and args.max_images > 0:
        all_imgs = all_imgs[:args.max_images]

    rows = []
    for i, path in enumerate(all_imgs, 1):
        img = safe_read_image(path)
        if img is None:
            continue

        basic = compute_basic_metrics(img)

        if args.group_by == "parent":
            group = path.parent.name
        else:
            group = "all"

        face_bbox = (-1, -1, -1, -1)
        face_metrics = {
            "face_detected": np.nan,
            "face_area_ratio": np.nan,
            "face_center_dx_norm": np.nan,
            "face_center_dy_norm": np.nan,
        }

        if use_det:
            face_bbox = detect_primary_face(det, img)
            face_metrics = compute_face_metrics(img, face_bbox)

        row = {
            "path": str(path),
            "group": group,
            "detector_used": float(1.0 if use_det else 0.0),
        }
        row.update(basic)
        row.update(face_metrics)
        rows.append(row)

        if i % 500 == 0:
            print(f"Processed {i}/{len(all_imgs)}")

    df = pd.DataFrame(rows)

    # Quality score
    df["quality_score_0_100"] = df.apply(quality_score, axis=1)

    # Save per-image CSV
    metrics_csv = outdir / "metrics.csv"
    df.to_csv(metrics_csv, index=False)

    # Summary overall
    def agg_stats(s: pd.Series) -> pd.Series:
        return pd.Series({
            "count": s.count(),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "p10": s.quantile(0.10),
            "p90": s.quantile(0.90),
        })

    cols_to_summarize = [
        "sharpness_lap_var", "luminance_mean", "contrast_std",
        "pct_saturated", "noise_proxy",
        "face_detected", "face_area_ratio",
        "quality_score_0_100",
    ]
    overall = df[cols_to_summarize].apply(agg_stats).T
    overall_csv = outdir / "summary_overall.csv"
    overall.to_csv(overall_csv)

    # Summary by group (identity folder)
    if args.group_by != "none":
        by_group = df.groupby("group")[cols_to_summarize].agg(["count", "mean", "median", "std"])
        by_group.columns = ["_".join(c).strip() for c in by_group.columns.values]
        by_group = by_group.sort_values("quality_score_0_100_median", ascending=False)
        by_group_csv = outdir / "summary_by_folder.csv"
        by_group.to_csv(by_group_csv)

    # Plots for slides
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_hist(df, "sharpness_lap_var", plots_dir / "hist_sharpness.png", "Sharpness (Variance of Laplacian)")
    plot_hist(df, "luminance_mean", plots_dir / "hist_luminance.png", "Luminance (Mean gray level)")
    plot_hist(df, "contrast_std", plots_dir / "hist_contrast.png", "Contrast (Std gray level)")
    plot_hist(df, "pct_saturated", plots_dir / "hist_saturation.png", "Saturation % (dark+bright pixels)")
    plot_hist(df, "noise_proxy", plots_dir / "hist_noise.png", "Noise proxy (std of residual)")
    plot_hist(df, "quality_score_0_100", plots_dir / "hist_quality_score.png", "Quality Score (0-100)")

    if args.group_by != "none":
        # Boxplots (limit to top 20 folders by count)
        plot_box_by_group(df, "quality_score_0_100", "group",
                          plots_dir / "box_quality_by_group.png",
                          "Quality Score by folder (top groups by #images)",
                          max_groups=20)

    print("\nDone!")
    print(f"- Per-image metrics CSV: {metrics_csv}")
    print(f"- Overall summary CSV: {overall_csv}")
    if args.group_by != "none":
        print(f"- Per-folder summary CSV: {outdir / 'summary_by_folder.csv'}")
    print(f"- Plots saved in: {plots_dir}")


if __name__ == "__main__":
    main()