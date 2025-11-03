import cv2
import numpy as np
import time


def maybe_resize(img, max_size=None): # Thay đổi kích thước hình ảnh nếu cạnh lớn nhất vượt quá max_size
    if not max_size: # Nếu không có giới hạn kích thước, trả về hình ảnh gốc
        return img
    h, w = img.shape[:2] # Lấy chiều cao và chiều rộng của hình ảnh
    scale = min(max_size / max(h, w), 1.0) # Tính tỷ lệ thay đổi kích thước
    if scale < 1.0: # Nếu cần thay đổi kích thước
        new_w = int(w * scale) # Tính chiều rộng mới
        new_h = int(h * scale) # Tính chiều cao mới
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA) # Thay đổi kích thước hình ảnh
    return img


class Timer: # Đơn giản để đo thời gian thực thi của một khối mã
    def __init__(self): # Khởi tạo bộ đếm thời gian
        self.start_time = None
    def __enter__(self): # Bắt đầu đo thời gian
        self.start_time = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb): # Kết thúc đo thời gian
        self.elapsed = time.perf_counter() - self.start_time


def to_gray(img): # Chuyển đổi hình ảnh sang thang độ xám
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


