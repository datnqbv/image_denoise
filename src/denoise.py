import cv2
import os
import numpy as np
import argparse
import csv
from utils import maybe_resize, Timer

# Denoising filters
def gaussian_filter(img, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)
# Median filter
def median_filter(img, ksize=5): 
    # cv2.medianBlur works on single channel or 3-channel images; use directly
    return cv2.medianBlur(img, ksize)
# Bilateral filter
def bilateral_filter(img, d=9, sigmaColor=75, sigmaSpace=75):
    # Deprecated from CLI: kept for potential internal use (not exposed)
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

# Non-Local Means filter for colored images
def nlm_filter_colored(img, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)

# Unsharp Masking
def unsharp_mask(img, amount: float = 1.2, sigma: float = 1.0):
    # Amount > 0 increases sharpening strength; sigma controls blur radius
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise images with selected filters") # Chương trình này áp dụng các bộ lọc khử nhiễu khác nhau cho các hình ảnh trong thư mục đầu vào và lưu kết quả vào thư mục đầu ra.
    parser.add_argument("--noisy_dir", default="data/noisy") # Thư mục chứa hình ảnh bị nhiễu đầu vào.
    parser.add_argument("--out_dir", default="results/denoised") # Thư mục để lưu hình ảnh đã được khử nhiễu.
    parser.add_argument("--filters", default="gaussian,median,nlm", help="Comma-separated filters") # Danh sách các bộ lọc để áp dụng, phân tách bằng dấu phẩy. Các lựa chọn bao gồm: gaussian, median, nlm.
    parser.add_argument("--ksize", type=int, default=5) # Kích thước kernel cho bộ lọc Gaussian và Median (phải là số lẻ).
    parser.add_argument("--sigma", type=float, default=1.5) # Độ lệch chuẩn cho bộ lọc Gaussian.
    parser.add_argument("--nlm_h", type=float, default=10) # Tham số h cho bộ lọc Non-Local Means (NLM) ảnh màu.
    parser.add_argument("--nlm_hColor", type=float, default=10) # Tham số hColor cho bộ lọc Non-Local Means (NLM) ảnh màu.
    parser.add_argument("--nlm_tws", type=int, default=7) # Kích thước cửa sổ mẫu cho bộ lọc Non-Local Means (NLM).
    parser.add_argument("--nlm_sws", type=int, default=21) # Kích thước cửa sổ tìm kiếm cho bộ lọc Non-Local Means (NLM).
    parser.add_argument("--max_size", type=int, default=None) # Kích thước tối đa để thay đổi kích thước hình ảnh đầu vào (giữ tỷ lệ khung hình). Nếu không đặt, không thay đổi kích thước.
    parser.add_argument("--profile", action="store_true") # Bật chế độ hồ sơ để in thời gian xử lý cho mỗi hình ảnh và bộ lọc.
    parser.add_argument("--gentle", action="store_true", help="Use lighter denoise parameters to avoid over-smoothing") # Sử dụng các tham số khử nhiễu nhẹ hơn để tránh làm mờ quá mức.
    parser.add_argument("--unsharp", action="store_true", help="Apply light unsharp mask after denoise") # Áp dụng mặt nạ unsharp nhẹ sau khi khử nhiễu.
    parser.add_argument("--unsharp_amount", type=float, default=0.2, help="Strength of unsharp mask (default 0.2)") # Độ mạnh của mặt nạ unsharp (mặc định 0.2).
    parser.add_argument("--unsharp_sigma", type=float, default=1.0, help="Gaussian sigma for unsharp mask (default 1.0)") # Độ lệch chuẩn Gaussian cho mặt nạ unsharp (mặc định 1.0).
    # parser.add_argument("--timings_csv", type=str, default=None, help="Optional CSV to append per-image runtimes") # Đã loại bỏ chức năng timings_csv
    args = parser.parse_args()

    noisy_dir = args.noisy_dir # Thư mục chứa hình ảnh bị nhiễu đầu vào.
    out_dir = args.out_dir # Thư mục để lưu hình ảnh đã được khử nhiễu.
    os.makedirs(out_dir, exist_ok=True) # Tạo thư mục đầu ra nếu chưa tồn tại.
    chosen = [x.strip() for x in args.filters.split(",") if x.strip()] # Phân tích danh sách bộ lọc được chọn từ tham số dòng lệnh.


    # Nếu tham số gentle được đặt, điều chỉnh các tham số bộ lọc để tránh làm mờ quá mức.
    if args.gentle:
        if "gaussian" in chosen:
            args.ksize = min(args.ksize, 3) # Giới hạn kích thước kernel tối đa cho bộ lọc Gaussian.
            args.sigma = min(args.sigma, 0.8) # Giới hạn sigma tối đa cho bộ lọc Gaussian.
        if "median" in chosen:
            args.ksize = min(args.ksize, 3) # Giới hạn kích thước kernel tối đa cho bộ lọc Median.
        if "nlm" in chosen:
            args.nlm_h = min(args.nlm_h, 5) # Giới hạn tham số h tối đa cho bộ lọc NLM.
            args.nlm_hColor = min(args.nlm_hColor, 5) # Giới hạn tham số hColor tối đa cho bộ lọc NLM.
    # Lặp qua tất cả các hình ảnh trong thư mục đầu vào và áp dụng các bộ lọc đã chọn.
    for fname in os.listdir(noisy_dir):
        if not fname.lower().endswith((".png",".jpg",".jpeg")): # Bỏ qua các tệp không phải hình ảnh.
            continue

        
        img = cv2.imread(os.path.join(noisy_dir, fname), cv2.IMREAD_COLOR) # Đọc ảnh màu (BGR) để giữ nguyên màu sắc từ đầu đến cuối
        img = maybe_resize(img, args.max_size) # Thay đổi kích thước hình ảnh nếu cần thiết.
        base = os.path.splitext(fname)[0] # Tên tệp cơ sở không có phần mở rộng.
        if "gaussian" in chosen: 
            with Timer() as t:
                g = gaussian_filter(img, ksize=args.ksize, sigma=args.sigma) # Áp dụng bộ lọc Gaussian.
            if args.unsharp: # Áp dụng mặt nạ unsharp nếu được chỉ định.
                g = unsharp_mask(g, amount=args.unsharp_amount, sigma=args.unsharp_sigma) # Áp dụng mặt nạ unsharp nếu được chỉ định.
            cv2.imwrite(os.path.join(out_dir, f"{base}_gaussian.png"), g) # Lưu hình ảnh đã được khử nhiễu bằng bộ lọc Gaussian.
            if args.profile: # In thời gian xử lý nếu chế độ hồ sơ được bật.
                print(f"gaussian,{fname},{t.elapsed:.4f}s") # In thời gian xử lý nếu chế độ hồ sơ được bật.
            # Đã loại bỏ chức năng ghi timings_csv
        if "median" in chosen:
            with Timer() as t: 
                m = median_filter(img, ksize=args.ksize)
            if args.unsharp: 
                m = unsharp_mask(m, amount=args.unsharp_amount, sigma=args.unsharp_sigma)
            cv2.imwrite(os.path.join(out_dir, f"{base}_median.png"), m)
            if args.profile:
                print(f"median,{fname},{t.elapsed:.4f}s")
            # Đã loại bỏ chức năng ghi timings_csv
        if "nlm" in chosen:
            with Timer() as t:
                n = nlm_filter_colored(img, h=args.nlm_h, hColor=args.nlm_hColor, templateWindowSize=args.nlm_tws, searchWindowSize=args.nlm_sws) # Áp dụng bộ lọc Non-Local Means (NLM) cho ảnh màu.
            if args.unsharp: 
                n = unsharp_mask(n, amount=args.unsharp_amount, sigma=args.unsharp_sigma) 
            cv2.imwrite(os.path.join(out_dir, f"{base}_nlm.png"), n)
            if args.profile:
                print(f"nlm,{fname},{t.elapsed:.4f}s") 
            # Đã loại bỏ chức năng ghi timings_csv
    print("Denoising finished.")
