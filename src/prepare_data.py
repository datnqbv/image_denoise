import os
import cv2
import numpy as np
from skimage import data, img_as_ubyte


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_gray(img, path):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, img)


def save_color(img, path):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(path, img)


if __name__ == "__main__":
    out_dir = os.path.join("data", "original")
    ensure_dir(out_dir)

    # Load a few sample images from skimage
    images = {
        "camera_gray": img_as_ubyte(data.camera()),
        "coins_gray": img_as_ubyte(data.coins()),
        "astronaut_color": cv2.cvtColor(img_as_ubyte(data.astronaut()), cv2.COLOR_RGB2BGR),
        "coffee_color": cv2.cvtColor(img_as_ubyte(data.coffee()), cv2.COLOR_RGB2BGR),
    }

    for name, img in images.items():
        save_path = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(save_path, img)

    print(f"Saved {len(images)} images to {out_dir}")


