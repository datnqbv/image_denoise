import cv2
import numpy as np
import os
import argparse
from utils import maybe_resize

# Hàm thêm nhiễu  gauusian
def add_gaussian_noise(img, sigma):
    gauss = np.random.normal(0, sigma, img.shape).astype(np.float32) # Tạo nhiễu Gaussian với độ lệch chuẩn sigma
    noisy = img.astype(np.float32) + gauss # Thêm nhiễu vào ảnh gốc
    noisy = np.clip(noisy, 0, 255).astype(np.uint8) # Đảm bảo giá trị pixel hợp lệ
    return noisy 
# hàm thêm nhiễu  salt pepper
def add_salt_pepper(img, amount):
    out = img.copy() # Tạo bản sao của ảnh gốc để thêm nhiễu
    h, w = img.shape[:2] # Lấy kích thước ảnh
    num_salt = int(amount * h * w / 2) # Số điểm nhiễu muối
    # salt
    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)) # Tọa độ ngẫu nhiên cho điểm muối
    out[coords] = 255 # Đặt điểm muối thành trắng
    # pepper
    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)) 
    out[coords] = 0 # Đặt điểm tiêu thành đen
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate noisy images") # Chương trình này thêm nhiễu Gaussian và Salt & Pepper vào các hình ảnh trong thư mục nguồn và lưu kết quả vào thư mục đích.
    parser.add_argument("--src_dir", default="data/original")   
    parser.add_argument("--out_dir", default="data/noisy")
    parser.add_argument("--sigmas", type=str, default="10,20,30", help="Gaussian sigmas, comma-separated") # Các mức độ lệch chuẩn Gaussian, phân tách bằng dấu phẩy.
    parser.add_argument("--sp_levels", type=str, default="0.01,0.03,0.05", help="Salt&pepper levels, comma-separated") # Các mức độ nhiễu Salt & Pepper, phân tách bằng dấu phẩy.
    parser.add_argument("--max_size", type=int, default=None, help="Resize max side before adding noise") # Kích thước tối đa để thay đổi kích thước hình ảnh nguồn trước khi thêm nhiễu.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility") # Hạt giống ngẫu nhiên để tái tạo kết quả.
    args = parser.parse_args() 

    # Thiết lập hạt giống ngẫu nhiên nếu được chỉ định
    if args.seed is not None:
        np.random.seed(args.seed)
    
    src_dir = args.src_dir # Thư mục chứa hình ảnh nguồn.
    out_dir = args.out_dir # Thư mục để lưu hình ảnh bị nhiễu.
    os.makedirs(out_dir, exist_ok=True) # Tạo thư mục đích nếu chưa tồn tại.
    sigmas = [int(s) for s in args.sigmas.split(",") if s] # Phân tích danh sách độ lệch chuẩn Gaussian từ tham số dòng lệnh.
    sp_levels = [float(p) for p in args.sp_levels.split(",") if p] # Phân tích danh sách mức độ nhiễu Salt & Pepper từ tham số dòng lệnh.
    for fname in os.listdir(src_dir): # Lặp qua tất cả các hình ảnh trong thư mục nguồn.
        if not fname.lower().endswith((".png",".jpg",".jpeg")): # Bỏ qua các tệp không phải hình ảnh.
            continue
        # Read color to preserve colors through the pipeline
        img = cv2.imread(os.path.join(src_dir, fname), cv2.IMREAD_COLOR) # Đọc ảnh màu (BGR) để giữ nguyên màu sắc từ đầu đến cuối
        img = maybe_resize(img, args.max_size) # Thay đổi kích thước hình ảnh nếu cần thiết.
        basename = os.path.splitext(fname)[0] # Tên tệp cơ sở không có phần mở rộng.
        # gaussian noises
        for s in sigmas: 
            noisy = add_gaussian_noise(img, s) # Thêm nhiễu Gaussian với độ lệch chuẩn s
            cv2.imwrite(os.path.join(out_dir, f"{basename}_gauss_sigma{s}.png"), noisy) # Lưu hình ảnh bị nhiễu
        # salt & pepper
        for p in sp_levels:
            noisy = add_salt_pepper(img, p)
            cv2.imwrite(os.path.join(out_dir, f"{basename}_sp_p{int(p*100)}.png"), noisy)
    print("Done generating noisy images.")
