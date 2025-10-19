import cv2
import numpy as np
import time


def maybe_resize(img, max_size=None):
    if not max_size:
        return img
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


class Timer:
    def __init__(self):
        self.start_time = None
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start_time


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


