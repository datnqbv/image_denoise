import cv2
import numpy as np
import os
import argparse
from utils import maybe_resize

def add_gaussian_noise(img, sigma):
    gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper(img, amount):
    out = img.copy()
    h, w = img.shape[:2]
    num_salt = int(amount * h * w / 2)
    # salt
    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt))
    out[coords] = 255
    # pepper
    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt))
    out[coords] = 0
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate noisy images")
    parser.add_argument("--src_dir", default="data/original")
    parser.add_argument("--out_dir", default="data/noisy")
    parser.add_argument("--sigmas", type=str, default="10,20,30", help="Gaussian sigmas, comma-separated")
    parser.add_argument("--sp_levels", type=str, default="0.01,0.03,0.05", help="Salt&pepper levels, comma-separated")
    parser.add_argument("--max_size", type=int, default=None, help="Resize max side before adding noise")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)

    src_dir = args.src_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    sigmas = [int(s) for s in args.sigmas.split(",") if s]
    sp_levels = [float(p) for p in args.sp_levels.split(",") if p]
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith((".png",".jpg",".jpeg")):
            continue
        # Read color to preserve colors through the pipeline
        img = cv2.imread(os.path.join(src_dir, fname), cv2.IMREAD_COLOR)
        img = maybe_resize(img, args.max_size)
        basename = os.path.splitext(fname)[0]
        # gaussian noises
        for s in sigmas:
            noisy = add_gaussian_noise(img, s)
            cv2.imwrite(os.path.join(out_dir, f"{basename}_gauss_sigma{s}.png"), noisy)
        # salt & pepper
        for p in sp_levels:
            noisy = add_salt_pepper(img, p)
            cv2.imwrite(os.path.join(out_dir, f"{basename}_sp_p{int(p*100)}.png"), noisy)
    print("Done generating noisy images.")
