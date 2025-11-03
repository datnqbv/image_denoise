import cv2
import os
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Evaluate PSNR/SSIM and export figures") # Chương trình này đánh giá chất lượng hình ảnh đã được khử nhiễu bằng cách tính toán các chỉ số PSNR và SSIM so với hình ảnh gốc, và xuất kết quả vào tệp CSV cùng với các biểu đồ tùy chọn.
parser.add_argument("--orig_dir", default="data/original")
parser.add_argument("--den_dir", default="results/denoised")
parser.add_argument("--noisy_dir", default="data/noisy") 
parser.add_argument("--out_csv", default="report/report_data.csv") 
parser.add_argument("--plot_hist", action="store_true") # Vẽ biểu đồ histogram
parser.add_argument("--plot_error", action="store_true") # Vẽ bản đồ lỗi
parser.add_argument("--plot_edges", action="store_true") # Vẽ cạnh
args = parser.parse_args() # Phân tích các đối số dòng lệnh.

orig_dir = args.orig_dir # Thư mục chứa hình ảnh gốc.
den_dir = args.den_dir # Thư mục chứa hình ảnh đã được khử nhiễu.
noisy_dir = args.noisy_dir  # Thư mục chứa hình ảnh bị nhiễu.
out_csv = args.out_csv # Tệp CSV để lưu kết quả đánh giá.
os.makedirs(os.path.dirname(out_csv) or "report", exist_ok=True) # Tạo thư mục cho tệp CSV nếu chưa tồn tại.

rows = []
for orig_fname in os.listdir(orig_dir): 
    if not orig_fname.lower().endswith((".png",".jpg",".jpeg")):
        continue
    base = os.path.splitext(orig_fname)[0] 
    orig_path = os.path.join(orig_dir, orig_fname) # Đường dẫn đến hình ảnh gốc.
    # Load original image
    orig = cv2.imread(orig_path, cv2.IMREAD_COLOR) # Đọc ảnh màu (BGR) để giữ nguyên màu sắc từ đầu đến cuối
    if orig is None: 
        continue
    # Find corresponding denoised images
    for den_fname in os.listdir(den_dir):
        if not den_fname.startswith(base): # Bỏ qua các tệp không khớp với hình ảnh gốc.
            continue
        den_path = os.path.join(den_dir, den_fname) # Đường dẫn đến hình ảnh đã được khử nhiễu.
        den = cv2.imread(den_path, cv2.IMREAD_COLOR) # Đọc ảnh màu (BGR) để giữ nguyên màu sắc từ đầu đến cuối
        if den is None: # Bỏ qua nếu không thể đọc hình ảnh đã được khử nhiễu.
            continue
        # Ensure same dimensions (noisy/denoised may be resized earlier)
        if orig.shape[:2] != den.shape[:2]: # Nếu kích thước không khớp, thay đổi kích thước hình ảnh gốc cho phù hợp.
            orig = cv2.resize(orig, (den.shape[1], den.shape[0]), interpolation=cv2.INTER_AREA) # Thay đổi kích thước hình ảnh gốc để khớp với hình ảnh đã được khử nhiễu.
        # Convert to RGB for skimage and compute metrics per color image
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) # Chuyển đổi hình ảnh gốc từ BGR sang RGB
        den_rgb = cv2.cvtColor(den, cv2.COLOR_BGR2RGB) # Chuyển đổi hình ảnh đã được khử nhiễu từ BGR sang RGB
        p = psnr(orig_rgb, den_rgb, data_range=255) # Tính toán PSNR
        s = ssim(orig_rgb, den_rgb, data_range=255, channel_axis=2) # Tính toán SSIM
        # Additional metrics
        mse = float(np.mean((orig_rgb.astype(np.float32) - den_rgb.astype(np.float32)) ** 2)) # Tính toán MSE
        mae = float(np.mean(np.abs(orig_rgb.astype(np.float32) - den_rgb.astype(np.float32)))) # Tính toán MAE
        rows.append({
            "image": base,
            "den_file": den_fname,
            "PSNR": p,
            "SSIM": s,
            "MSE": mse,
            "MAE": mae
        })
       
        den_base = os.path.splitext(den_fname)[0]
        # Loại bỏ hậu tố bộ lọc khỏi tên file denoised để tìm đúng file noisy
        for suffix in ["_gaussian", "_median", "_nlm"]:
            if den_base.endswith(suffix):
                noisy_base = den_base[: -len(suffix)]
                break
        else:
            noisy_base = den_base
        if args.plot_hist: # Vẽ biểu đồ histogram
            try:
                noisy_candidate = os.path.join(noisy_dir, f"{noisy_base}.png") # Tìm hình ảnh bị nhiễu tương ứng
                noisy_img = cv2.imread(noisy_candidate, cv2.IMREAD_COLOR) # Đọc ảnh màu (BGR) để giữ nguyên màu sắc từ đầu đến cuối
                if noisy_img is not None: # Nếu hình ảnh bị nhiễu được đọc thành công
                    if noisy_img.shape[:2] != den.shape[:2]: # Nếu kích thước không khớp, thay đổi kích thước hình ảnh bị nhiễu cho phù hợp.
                        noisy_img = cv2.resize(noisy_img, (den.shape[1], den.shape[0]), interpolation=cv2.INTER_AREA) # Thay đổi kích thước hình ảnh bị nhiễu để khớp với hình ảnh đã được khử nhiễu.
                    fig, ax = plt.subplots(1,3, figsize=(12,4)) # Tạo biểu đồ với 3 cột
                    ax[0].hist(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).ravel(), bins=256); ax[0].set_title("Orig hist") # Biểu đồ histogram của hình ảnh gốc
                    ax[1].hist(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY).ravel(), bins=256); ax[1].set_title("Noisy hist") # Biểu đồ histogram của hình ảnh bị nhiễu
                    ax[2].hist(cv2.cvtColor(den, cv2.COLOR_BGR2GRAY).ravel(), bins=256); ax[2].set_title("Denoised hist") # Biểu đồ histogram của hình ảnh đã được khử nhiễu
                    fig.suptitle(f"Histogram: {den_fname}") # Tiêu đề chính của biểu đồ
                    fig.savefig(os.path.join(os.path.dirname(out_csv) or "report", f"hist_{den_base}.png")) # Lưu biểu đồ vào tệp
                    plt.close(fig) # Đóng biểu đồ để giải phóng bộ nhớ
            except Exception:
                pass
        if args.plot_error: # Vẽ bản đồ lỗi
            try:
                # error map in gray for visualization
                err = cv2.absdiff(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)) # Tính toán bản đồ lỗi tuyệt đối giữa hình ảnh gốc và hình ảnh đã được khử nhiễu
                fig, ax = plt.subplots(1,1, figsize=(5,4)) # Tạo biểu đồ với 1 cột
                ax.imshow(err, cmap="inferno") # Hiển thị bản đồ lỗi với bản đồ màu "inferno"
                ax.set_axis_off() # Tắt trục
                fig.suptitle(f"Error map: {den_fname}") # Tiêu đề chính của biểu đồ
                fig.savefig(os.path.join(os.path.dirname(out_csv) or "report", f"error_{den_base}.png"), bbox_inches="tight") # Lưu biểu đồ vào tệp
                plt.close(fig)
            except Exception:
                pass
        if args.plot_edges: # Vẽ cạnh
            try:
                orig_edges = cv2.Canny(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), 100, 200) # Tính toán cạnh của hình ảnh gốc sử dụng thuật toán Canny
                den_edges = cv2.Canny(cv2.cvtColor(den, cv2.COLOR_BGR2GRAY), 100, 200) # Tính toán cạnh của hình ảnh đã được khử nhiễu sử dụng thuật toán Canny
                fig, ax = plt.subplots(1,2, figsize=(8,4)) # Tạo biểu đồ với 2 cột
                ax[0].imshow(orig_edges, cmap="gray"); ax[0].set_title("Edges: orig"); ax[0].set_axis_off() # Hiển thị cạnh của hình ảnh gốc
                ax[1].imshow(den_edges, cmap="gray"); ax[1].set_title("Edges: denoised"); ax[1].set_axis_off() # Hiển thị cạnh của hình ảnh đã được khử nhiễu
                fig.suptitle(f"Edges: {den_fname}") # Tiêu đề chính của biểu đồ
                fig.savefig(os.path.join(os.path.dirname(out_csv) or "report", f"edges_{den_base}.png"), bbox_inches="tight") # Lưu biểu đồ vào tệp
                plt.close(fig) 
            except Exception:
                pass

df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
print("Evaluation done. CSV saved to", out_csv)
