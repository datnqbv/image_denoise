
# Image Denoising Project

## Tổng quan

Dự án xử lý ảnh khử nhiễu tự động, gồm pipeline tạo ảnh nhiễu, khử nhiễu bằng nhiều bộ lọc (Gaussian, Median, Non-Local Means), đánh giá chất lượng (PSNR, SSIM, MSE, MAE), trực quan hóa histogram, bản đồ lỗi, cạnh, và giao diện Streamlit UI thân thiện.

---

## Công nghệ sử dụng

- Python 3.9+
- OpenCV
- scikit-image
- NumPy, Pandas
- Matplotlib
- Streamlit (giao diện web)
- PIL (Pillow)

---

## Cấu trúc dự án

```
image-denoise-project/
│
├── data/
│   ├── original/      # Ảnh gốc (input)
│   └── noisy/         # Ảnh đã thêm nhiễu
│
├── results/
│   └── denoised/      # Ảnh đã khử nhiễu
│
├── report/
│   ├── report_data.csv    # Bảng kết quả PSNR, SSIM, MSE, MAE
│   ├── timings.csv        # Thời gian xử lý từng ảnh/bộ lọc
│   ├── hist_*.png         # Biểu đồ histogram so sánh
│   ├── error_*.png        # Bản đồ lỗi
│   └── edges_*.png        # Bản đồ cạnh
│
├── src/
│   ├── prepare_data.py    # Tải/lưu ảnh mẫu vào data/original
│   ├── add_noise.py       # Thêm nhiễu Gaussian/Salt-Pepper vào ảnh
│   ├── denoise.py         # Khử nhiễu bằng các bộ lọc
│   ├── evaluate.py        # Đánh giá, xuất chỉ số, vẽ biểu đồ
│   ├── app.py             # Giao diện Streamlit UI
│   ├── utils.py           # Hàm tiện ích (resize, timer, v.v.)
│   └── __pycache__/       # File biên dịch Python
│
├── notebooks/
│   └── analysis.ipynb     # Notebook phân tích, trực quan hóa kết quả
│
├── requirements.txt       # Danh sách thư viện cần cài đặt
└── README.md              # Tài liệu này
```

---

## Vai trò các file chính

- **src/prepare_data.py**: Tải và lưu các ảnh mẫu (từ thư viện skimage) vào thư mục `data/original` để làm dữ liệu gốc.
- **src/add_noise.py**: Thêm nhiễu Gaussian và Salt & Pepper vào ảnh gốc, lưu vào `data/noisy`.
- **src/denoise.py**: Khử nhiễu cho ảnh trong `data/noisy` bằng các bộ lọc (Gaussian, Median, NLM), lưu kết quả vào `results/denoised`.
- **src/evaluate.py**: So sánh ảnh gốc và ảnh đã khử nhiễu, tính toán các chỉ số PSNR, SSIM, MSE, MAE, xuất bảng kết quả và các biểu đồ (histogram, error map, edge map) vào thư mục `report`.
- **src/app.py**: Giao diện web Streamlit cho phép tải ảnh, thêm nhiễu, chọn bộ lọc, xem kết quả, so sánh chỉ số, tải ảnh đã xử lý.
- **src/utils.py**: Các hàm tiện ích như resize ảnh, đo thời gian thực thi, chuyển đổi màu.
- **notebooks/analysis.ipynb**: Notebook Jupyter để phân tích, tổng hợp, trực quan hóa kết quả từ file CSV (so sánh các bộ lọc, vẽ biểu đồ, xem chi tiết từng ảnh).
- **requirements.txt**: Danh sách các thư viện Python cần thiết cho dự án.

---

## Hướng dẫn cài đặt & chạy

### 1. Cài đặt môi trường

```sh
# (Tùy chọn) Tạo môi trường ảo
python -m venv .venv
# Kích hoạt (Windows)
.\.venv\Scripts\Activate.ps1

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

```sh
python src/prepare_data.py
```

### 3. Tạo ảnh nhiễu

```sh
python src/add_noise.py --sigmas 10,20,30 --sp_levels 0.01,0.03,0.05 --seed 42 --max_size 1024
```


### 4. Khử nhiễu

```sh
python src/denoise.py --filters gaussian,median,nlm --ksize 5 --sigma 1.5 --nlm_h 10 --nlm_hColor 10 --nlm_tws 7 --nlm_sws 21 --max_size 1024 --profile
```




### 5. Đánh giá & xuất kết quả

```sh
python src/evaluate.py --plot_hist --plot_error --plot_edges
```


### 6. Chạy giao diện web (tùy chọn)

```sh
streamlit run src/app.py
```
Sau đó mở đường dẫn được cung cấp trên trình duyệt để sử dụng giao diện.

---


## Chức năng chính

- Tạo ảnh nhiễu tự động (Gaussian, Salt & Pepper)
- Khử nhiễu bằng nhiều bộ lọc (Gaussian, Median, Non-Local Means)
- Đánh giá chất lượng bằng PSNR, SSIM, MSE, MAE
- Trực quan hóa: histogram, error map, edge map
- Giao diện web trực quan, dễ sử dụng, hỗ trợ tải lên/xuống ảnh, so sánh kết quả
- Notebook phân tích, tổng hợp, vẽ biểu đồ cho báo cáo

---

**Lưu ý:**
- File `timings.csv` (nếu còn) chỉ để tham khảo tốc độ, không còn được cập nhật tự động.
- Các bước chạy lại dự án: cài thư viện, chuẩn bị dữ liệu, tạo nhiễu, khử nhiễu, đánh giá, (tùy chọn) chạy app hoặc notebook.

---

Nếu cần bổ sung chi tiết hoặc giải thích thêm về bất kỳ file nào, hãy hỏi tiếp!


