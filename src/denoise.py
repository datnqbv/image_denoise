import cv2
import os
import numpy as np
import argparse
import csv
from utils import maybe_resize, Timer

def gaussian_filter(img, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def median_filter(img, ksize=5):
    # cv2.medianBlur works on single channel or 3-channel images; use directly
    return cv2.medianBlur(img, ksize)

def bilateral_filter(img, d=9, sigmaColor=75, sigmaSpace=75):
    # Deprecated from CLI: kept for potential internal use (not exposed)
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)


def nlm_filter_colored(img, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)


def unsharp_mask(img, amount: float = 1.2, sigma: float = 1.0):
    # Amount > 0 increases sharpening strength; sigma controls blur radius
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened


# TV/Wavelet filters removed from CLI to keep only 3 filters in project

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise images with selected filters")
    parser.add_argument("--noisy_dir", default="data/noisy")
    parser.add_argument("--out_dir", default="results/denoised")
    parser.add_argument("--filters", default="gaussian,median,nlm", help="Comma-separated filters")
    parser.add_argument("--ksize", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--nlm_h", type=float, default=10)
    parser.add_argument("--nlm_hColor", type=float, default=10)
    parser.add_argument("--nlm_tws", type=int, default=7)
    parser.add_argument("--nlm_sws", type=int, default=21)
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--gentle", action="store_true", help="Use lighter denoise parameters to avoid over-smoothing")
    parser.add_argument("--unsharp", action="store_true", help="Apply light unsharp mask after denoise")
    parser.add_argument("--unsharp_amount", type=float, default=0.2, help="Strength of unsharp mask (default 0.2)")
    parser.add_argument("--unsharp_sigma", type=float, default=1.0, help="Gaussian sigma for unsharp mask (default 1.0)")
    parser.add_argument("--timings_csv", type=str, default=None, help="Optional CSV to append per-image runtimes")
    args = parser.parse_args()

    noisy_dir = args.noisy_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    chosen = [x.strip() for x in args.filters.split(",") if x.strip()]

    # Apply gentle defaults if requested
    if args.gentle:
        if "gaussian" in chosen:
            args.ksize = min(args.ksize, 3)
            args.sigma = min(args.sigma, 0.8)
        if "median" in chosen:
            args.ksize = min(args.ksize, 3)
        if "nlm" in chosen:
            args.nlm_h = min(args.nlm_h, 5)
            args.nlm_hColor = min(args.nlm_hColor, 5)

    for fname in os.listdir(noisy_dir):
        if not fname.lower().endswith((".png",".jpg",".jpeg")):
            continue
        # Read color (BGR) to preserve colors end-to-end
        img = cv2.imread(os.path.join(noisy_dir, fname), cv2.IMREAD_COLOR)
        img = maybe_resize(img, args.max_size)
        base = os.path.splitext(fname)[0]
        if "gaussian" in chosen:
            with Timer() as t:
                g = gaussian_filter(img, ksize=args.ksize, sigma=args.sigma)
            if args.unsharp:
                g = unsharp_mask(g, amount=args.unsharp_amount, sigma=args.unsharp_sigma)
            cv2.imwrite(os.path.join(out_dir, f"{base}_gaussian.png"), g)
            if args.profile:
                print(f"gaussian,{fname},{t.elapsed:.4f}s")
            if args.timings_csv:
                with open(args.timings_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([fname, "gaussian", f"{t.elapsed:.6f}"])
        if "median" in chosen:
            with Timer() as t:
                m = median_filter(img, ksize=args.ksize)
            if args.unsharp:
                m = unsharp_mask(m, amount=args.unsharp_amount, sigma=args.unsharp_sigma)
            cv2.imwrite(os.path.join(out_dir, f"{base}_median.png"), m)
            if args.profile:
                print(f"median,{fname},{t.elapsed:.4f}s")
            if args.timings_csv:
                with open(args.timings_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([fname, "median", f"{t.elapsed:.6f}"])
        if "nlm" in chosen:
            with Timer() as t:
                n = nlm_filter_colored(img, h=args.nlm_h, hColor=args.nlm_hColor, templateWindowSize=args.nlm_tws, searchWindowSize=args.nlm_sws)
            if args.unsharp:
                n = unsharp_mask(n, amount=args.unsharp_amount, sigma=args.unsharp_sigma)
            cv2.imwrite(os.path.join(out_dir, f"{base}_nlm.png"), n)
            if args.profile:
                print(f"nlm,{fname},{t.elapsed:.4f}s")
            if args.timings_csv:
                with open(args.timings_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([fname, "nlm", f"{t.elapsed:.6f}"])
    print("Denoising finished.")
