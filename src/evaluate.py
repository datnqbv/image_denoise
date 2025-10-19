import cv2
import os
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Evaluate PSNR/SSIM and export figures")
parser.add_argument("--orig_dir", default="data/original")
parser.add_argument("--den_dir", default="results/denoised")
parser.add_argument("--noisy_dir", default="data/noisy")
parser.add_argument("--out_csv", default="report/report_data.csv")
parser.add_argument("--plot_hist", action="store_true")
parser.add_argument("--plot_error", action="store_true")
parser.add_argument("--plot_edges", action="store_true")
args = parser.parse_args()

orig_dir = args.orig_dir
den_dir = args.den_dir
noisy_dir = args.noisy_dir
out_csv = args.out_csv
os.makedirs(os.path.dirname(out_csv) or "report", exist_ok=True)

rows = []
for orig_fname in os.listdir(orig_dir):
    if not orig_fname.lower().endswith((".png",".jpg",".jpeg")):
        continue
    base = os.path.splitext(orig_fname)[0]
    orig_path = os.path.join(orig_dir, orig_fname)
    # Read color if present; evaluation metrics support multichannel via skimage
    orig = cv2.imread(orig_path, cv2.IMREAD_COLOR)
    if orig is None:
        continue
    # find denoised files matching this base
    for den_fname in os.listdir(den_dir):
        if not den_fname.startswith(base):
            continue
        den_path = os.path.join(den_dir, den_fname)
        den = cv2.imread(den_path, cv2.IMREAD_COLOR)
        if den is None:
            continue
        # Ensure same dimensions (noisy/denoised may be resized earlier)
        if orig.shape[:2] != den.shape[:2]:
            orig = cv2.resize(orig, (den.shape[1], den.shape[0]), interpolation=cv2.INTER_AREA)
        # Convert to RGB for skimage and compute metrics per color image
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        den_rgb = cv2.cvtColor(den, cv2.COLOR_BGR2RGB)
        p = psnr(orig_rgb, den_rgb, data_range=255)
        s = ssim(orig_rgb, den_rgb, data_range=255, channel_axis=2)
        # Additional metrics
        mse = float(np.mean((orig_rgb.astype(np.float32) - den_rgb.astype(np.float32)) ** 2))
        mae = float(np.mean(np.abs(orig_rgb.astype(np.float32) - den_rgb.astype(np.float32))))
        rows.append({
            "image": base,
            "den_file": den_fname,
            "PSNR": p,
            "SSIM": s,
            "MSE": mse,
            "MAE": mae
        })
        # optional: histogram, error map, edges
        den_base = os.path.splitext(den_fname)[0]
        if args.plot_hist:
            try:
                noisy_candidate = os.path.join(noisy_dir, f"{den_base}.png")
                noisy_img = cv2.imread(noisy_candidate, cv2.IMREAD_COLOR)
                if noisy_img is not None:
                    if noisy_img.shape[:2] != den.shape[:2]:
                        noisy_img = cv2.resize(noisy_img, (den.shape[1], den.shape[0]), interpolation=cv2.INTER_AREA)
                    fig, ax = plt.subplots(1,3, figsize=(12,4))
                    ax[0].hist(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).ravel(), bins=256); ax[0].set_title("Orig hist")
                    ax[1].hist(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY).ravel(), bins=256); ax[1].set_title("Noisy hist")
                    ax[2].hist(cv2.cvtColor(den, cv2.COLOR_BGR2GRAY).ravel(), bins=256); ax[2].set_title("Denoised hist")
                    fig.suptitle(f"Histogram: {den_fname}")
                    fig.savefig(os.path.join(os.path.dirname(out_csv) or "report", f"hist_{den_base}.png"))
                    plt.close(fig)
            except Exception:
                pass
        if args.plot_error:
            try:
                # error map in gray for visualization
                err = cv2.absdiff(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), cv2.cvtColor(den, cv2.COLOR_BGR2GRAY))
                fig, ax = plt.subplots(1,1, figsize=(5,4))
                ax.imshow(err, cmap="inferno")
                ax.set_axis_off()
                fig.suptitle(f"Error map: {den_fname}")
                fig.savefig(os.path.join(os.path.dirname(out_csv) or "report", f"error_{den_base}.png"), bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass
        if args.plot_edges:
            try:
                orig_edges = cv2.Canny(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), 100, 200)
                den_edges = cv2.Canny(cv2.cvtColor(den, cv2.COLOR_BGR2GRAY), 100, 200)
                fig, ax = plt.subplots(1,2, figsize=(8,4))
                ax[0].imshow(orig_edges, cmap="gray"); ax[0].set_title("Edges: orig"); ax[0].set_axis_off()
                ax[1].imshow(den_edges, cmap="gray"); ax[1].set_title("Edges: denoised"); ax[1].set_axis_off()
                fig.suptitle(f"Edges: {den_fname}")
                fig.savefig(os.path.join(os.path.dirname(out_csv) or "report", f"edges_{den_base}.png"), bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass

df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
print("Evaluation done. CSV saved to", out_csv)
