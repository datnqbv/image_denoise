import os
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from utils import maybe_resize
from denoise import (
    gaussian_filter,
    median_filter,
    nlm_filter_colored,
    unsharp_mask,
)
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Cấu hình trang với icon và theme
st.set_page_config( # Cấu hình trang Streamlit với tiêu đề, biểu tượng và bố cục tùy chỉnh
    page_title="Image Denoising Studio",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh cho giao diện đẹp hơn
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
    }
    .filter-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header với gradient
st.markdown('<h1 class="main-title">🖼️ Image Denoising Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Khử nhiễu ảnh chuyên nghiệp với Gaussian, Median & Non-Local Means</p>', unsafe_allow_html=True)

# Sidebar với thiết kế đẹp hơn
with st.sidebar:
    st.markdown("### 📤 Upload & Settings")
    
    # Upload section
    with st.expander("📁 Tải ảnh lên", expanded=True):
        uploaded = st.file_uploader(
            "Chọn ảnh (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            help="Tải lên ảnh bạn muốn xử lý"
        )
        max_size = st.number_input(
            "Kích thước tối đa (px)",
            min_value=256,
            max_value=4096,
            value=1024,
            step=64,
            help="Giới hạn kích thước ảnh để xử lý nhanh hơn"
        )
    
    st.divider()
    
    # Noise section
    st.markdown("### 🎲 Mô phỏng Nhiễu")
    with st.expander("⚙️ Cài đặt nhiễu", expanded=False):
        add_noise = st.checkbox("Thêm nhiễu vào ảnh", value=False)
        if add_noise:
            noise_type = st.selectbox(
                "Loại nhiễu",
                ["gaussian", "saltpepper"],
                format_func=lambda x: "🌫️ Gaussian" if x == "gaussian" else "🧂 Salt & Pepper"
            )
            if noise_type == "gaussian":
                sigma = st.slider("Độ mạnh Gaussian (σ)", 1, 50, 20, help="Giá trị càng cao, nhiễu càng mạnh")
            else:
                sp_level = st.slider("Mật độ nhiễu (%)", 0.0, 0.2, 0.03, 0.01, help="Tỷ lệ pixel bị nhiễu")
        else:
            noise_type = None
            sigma = None
            sp_level = None
    
    st.divider()
    
    # Filters section
    st.markdown("### 🔧 Bộ lọc khử nhiễu")
    with st.expander("🎯 Chọn bộ lọc", expanded=True):
        st.markdown("**Chọn một hoặc nhiều bộ lọc:**")
        chosen = st.multiselect(
            "Bộ lọc",
            ["gaussian", "median", "nlm"],
            default=["median", "nlm"],
            format_func=lambda x: {
                "gaussian": "🌀 Gaussian Filter",
                "median": "📊 Median Filter",
                "nlm": "🎨 Non-Local Means"
            }[x],
            help="Chọn bộ lọc để so sánh kết quả"
        )
    
    # Advanced settings
    with st.expander("⚙️ Cài đặt nâng cao", expanded=False):
        # Đề xuất tham số
        st.markdown("**💡 Đề xuất tham số tối ưu:**")
        if st.button("✨ Áp dụng tham số đề xuất"):
            if add_noise and noise_type == "gaussian":
                st.session_state['ksize'] = 3
                st.session_state['g_sigma'] = 1.0
                st.session_state['nlm_h'] = 8
                st.session_state['nlm_hColor'] = 8
                st.success("✅ Đã áp dụng tham số tối ưu cho nhiễu Gaussian!")
            elif add_noise and noise_type == "saltpepper":
                st.session_state['ksize'] = 3
                st.session_state['g_sigma'] = 1.0
                st.session_state['nlm_h'] = 7
                st.session_state['nlm_hColor'] = 7
                st.success("✅ Đã áp dụng tham số tối ưu cho nhiễu Salt & Pepper!")
            else:
                st.info("ℹ️ Vui lòng chọn loại nhiễu trước để nhận đề xuất phù hợp.")
        
        st.divider()
        
        st.markdown("**Gaussian & Median:**") # Tham số cho bộ lọc Gaussian và Median
        ksize = st.slider("Kernel size", 3, 11, 
                         st.session_state.get('ksize', 5), 2, # Bước nhảy 2 để chỉ chọn số lẻ
                         help="Kích thước kernel (số lẻ). Khuyến nghị: 3-5") 
        g_sigma = st.slider("Gaussian σ", 0.1, 5.0, # Độ lệch chuẩn cho bộ lọc Gaussian
                           st.session_state.get('g_sigma', 1.5), 0.1, 
                           help="Độ mượt của Gaussian. Khuyến nghị: 0.8-1.5")
        
        st.markdown("**Non-Local Means:**") # Tham số cho bộ lọc NLM
        nlm_h = st.slider("NLM h", 1, 30, 
                         st.session_state.get('nlm_h', 10), # Độ mạnh khử nhiễu
                         help="Độ mạnh khử nhiễu. Khuyến nghị: 7-12 (quá cao sẽ làm mờ ảnh)")
        nlm_hColor = st.slider("NLM hColor", 1, 30, # Độ mạnh cho màu sắc
                              st.session_state.get('nlm_hColor', 10), 
                              help="Độ mạnh cho màu sắc. Khuyến nghị: 7-12")
        nlm_tws = st.slider("Template Window", 3, 15, 7, 2, # Bước nhảy 2 để chỉ chọn số lẻ
                           help="Kích thước cửa sổ mẫu. Mặc định: 7")
        nlm_sws = st.slider("Search Window", 7, 31, 21, 2,  # Bước nhảy 2 để chỉ chọn số lẻ
                           help="Kích thước vùng tìm kiếm. Mặc định: 21")
        
        st.markdown("**Tăng độ sắc nét:**")
        apply_unsharp = st.checkbox( # Áp dụng bộ lọc làm sắc nét
            "Unsharp mask",
            value=False,
            help="Làm ảnh rõ nét hơn sau khử nhiễu. Nên dùng khi ảnh bị mờ, nhưng tránh đặt quá cao để không bị gắt hoặc xuất hiện viền giả."
        )
        if apply_unsharp: # Nếu áp dụng unsharp mask, hiển thị thêm tùy chọn
            unsharp_amount = st.slider("Độ mạnh", 0.0, 2.0, 0.2, 0.05) # Độ mạnh của unsharp mask
            unsharp_sigma = st.slider("Unsharp σ", 0.1, 3.0, 1.0, 0.1) # Độ lệch chuẩn của unsharp mask
        else:
            unsharp_amount = 0.2 # Giá trị mặc định
            unsharp_sigma = 1.0 # Giá trị mặc định
        
        st.divider()
        
        # Giải thích các chỉ số đánh giá
        st.markdown("**📊 Giải thích các chỉ số đánh giá:**")
        st.info("""
**PSNR (Peak Signal-to-Noise Ratio):**
- Đo tỷ lệ tín hiệu cực đại so với nhiễu
- Càng cao càng tốt (>30 dB là tốt, >40 dB là rất tốt)
- Đơn vị: dB (decibel)

**SSIM (Structural Similarity Index):**
- Đo độ tương đồng cấu trúc giữa 2 ảnh
- Giá trị từ 0 đến 1 (1 là giống hệt nhau)
- >0.9 là tốt, >0.95 là rất tốt

**MSE (Mean Squared Error):**
- Trung bình bình phương sai số giữa 2 ảnh
- Càng nhỏ càng tốt (gần 0 là tốt nhất)

**MAE (Mean Absolute Error):**
- Trung bình sai số tuyệt đối giữa 2 ảnh
- Càng nhỏ càng tốt

**💡 Gợi ý sử dụng:**
- PSNR & SSIM: Quan trọng nhất, dễ hiểu
- MSE & MAE: Bổ trợ, cho cái nhìn chi tiết hơn
- Nên xem cả 4 chỉ số để đánh giá toàn diện
        """)
    
    st.divider()
    st.markdown("### ℹ️ Thông tin")
    st.info("📚 **Dự án:** Khử nhiễu ảnh\n\n🎓 **Phương pháp:** Gaussian, Median, NLM")
    

# Hàm tiện ích
def to_bgr(np_img: np.ndarray) -> np.ndarray: # Chuyển ảnh sang định dạng BGR
    if np_img.ndim == 2:
        return cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR) # Chuyển ảnh xám sang BGR
    return np_img

# Hàm tải ảnh từ file upload
def load_image(file) -> np.ndarray: # Tải ảnh từ file upload và chuyển sang định dạng BGR
    if file is None: 
        return None
    bytes_data = file.read() # Đọc dữ liệu ảnh dưới dạng bytes
    pil = Image.open(io.BytesIO(bytes_data)).convert("RGB") # Mở ảnh với PIL và chuyển sang RGB
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR) # Chuyển sang BGR
    return to_bgr(img) # Trả về ảnh ở định dạng BGR

# Hàm thêm nhiễu
def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray: # Thêm nhiễu Gaussian vào ảnh
    gauss = np.random.normal(0, sigma, img.shape).astype(np.float32) # Tạo nhiễu Gaussian
    noisy = img.astype(np.float32) + gauss # Thêm nhiễu vào ảnh
    return np.clip(noisy, 0, 255).astype(np.uint8) # Trả về ảnh nhiễu đã được cắt gọn

# Hàm thêm nhiễu muối tiêu
def add_sp_noise(img: np.ndarray, amount: float) -> np.ndarray: # Thêm nhiễu Salt & Pepper vào ảnh
    out = img.copy() # Sao chép ảnh gốc
    h, w = img.shape[:2] # Lấy kích thước ảnh
    num_salt = int(amount * h * w / 2) # Số pixel muối
    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)) # Tọa độ ngẫu nhiên
    out[coords] = 255 # Thêm muối (trắng)
    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)) # Tọa độ ngẫu nhiên
    out[coords] = 0 # Thêm tiêu (đen)
    return out # Trả về ảnh nhiễu

# Hàm tính các chỉ số đánh giá
def calculate_metrics(original, processed):
    """Tính toán các chỉ số đánh giá chất lượng ảnh"""
    # Chuyển sang RGB để tính toán
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    proc_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Tính các chỉ số
    psnr_value = psnr(orig_rgb, proc_rgb, data_range=255)
    ssim_value = ssim(orig_rgb, proc_rgb, data_range=255, channel_axis=2)
    mse_value = float(np.mean((orig_rgb.astype(np.float32) - proc_rgb.astype(np.float32)) ** 2))
    mae_value = float(np.mean(np.abs(orig_rgb.astype(np.float32) - proc_rgb.astype(np.float32))))
    
    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'MSE': mse_value,
        'MAE': mae_value
    }


img = load_image(uploaded)

if img is not None: # Nếu ảnh đã được tải lên
    img = maybe_resize(img, max_size)
    
    # Tạo tabs để hiển thị khác nhau
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ So sánh kết quả", "📊 Chỉ số đánh giá", "📈 Thống kê", "💾 Tải xuống"])
    
    with tab1: # Tab so sánh kết quả
        # Xử lý nhiễu nếu có
        work = img.copy()
        if add_noise and noise_type:
            if noise_type == "gaussian":
                work = add_gaussian_noise(work, sigma)
            else:
                work = add_sp_noise(work, sp_level)
        
        # Xử lý khử nhiễu
        results = []
        if "gaussian" in chosen:
            g = gaussian_filter(work, ksize=ksize, sigma=g_sigma) # Áp dụng bộ lọc Gaussian
            results.append(("Gaussian Filter", g))
        if "median" in chosen:
            m = median_filter(work, ksize=ksize) # Áp dụng bộ lọc Median
            results.append(("Median Filter", m))
        if "nlm" in chosen:
            n = nlm_filter_colored(work, h=nlm_h, hColor=nlm_hColor, templateWindowSize=nlm_tws, searchWindowSize=nlm_sws)
            results.append(("Non-Local Means", n))
        
        if apply_unsharp: 
            results = [(name, unsharp_mask(img_, amount=unsharp_amount, sigma=unsharp_sigma)) for name, img_ in results] # Áp dụng unsharp mask nếu được chọn
        
        # Hiển thị kết quả theo grid
        st.markdown("### 🎯 Kết quả so sánh")
        
        # Dòng 1: Ảnh gốc và ảnh nhiễu (nếu có)
        cols = st.columns(2 if add_noise else 1) # Tạo 2 cột nếu có ảnh nhiễu, 1 cột nếu không
        with cols[0]: # Cột ảnh gốc
            st.markdown("#### 📸 Ảnh gốc")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        if add_noise:
            with cols[1]: # Cột ảnh nhiễu
                st.markdown(f"#### 🎲 Ảnh nhiễu ({noise_type})")
                st.image(cv2.cvtColor(work, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        st.divider()
        
        # Dòng 2: Kết quả các bộ lọc
        if results:
            st.markdown("#### 🔧 Kết quả khử nhiễu")
            cols = st.columns(len(results)) # Tạo cột cho mỗi bộ lọc đã chọn
            for idx, (name, out) in enumerate(results): # Hiển thị kết quả từng bộ lọc
                with cols[idx]: # Cột tương ứng với bộ lọc
                    st.markdown(f"**{name}**")
                    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)
        else:
            st.warning("⚠️ Vui lòng chọn ít nhất một bộ lọc!")
    
    with tab2: # Tab chỉ số đánh giá
        st.markdown("### 📊 So sánh chỉ số đánh giá chất lượng")
        
        if results and add_noise: # Chỉ hiển thị nếu có kết quả và ảnh nhiễu
            st.info("💡 **Lưu ý:** Chỉ số được tính so sánh giữa ảnh gốc và ảnh sau khử nhiễu")
            
            # Tính toán chỉ số cho từng bộ lọc
            metrics_data = [] # Danh sách lưu trữ dữ liệu chỉ số
            for name, out in results: # Tính toán chỉ số cho từng kết quả
                metrics = calculate_metrics(img, out) # Tính các chỉ số
                metrics_data.append({ # Thêm dữ liệu vào danh sách
                    'Bộ lọc': name,
                    'PSNR (dB)': f"{metrics['PSNR']:.2f}",
                    'SSIM': f"{metrics['SSIM']:.4f}",
                    'MSE': f"{metrics['MSE']:.2f}",
                    'MAE': f"{metrics['MAE']:.2f}"
                })
            
            # Hiển thị bảng so sánh
            df_metrics = pd.DataFrame(metrics_data) # Tạo DataFrame từ dữ liệu chỉ số
            st.markdown("#### 📋 Bảng so sánh các chỉ số")
            st.dataframe(df_metrics, use_container_width=True)
            
            st.divider()
            
            # Hiển thị biểu đồ so sánh
            st.markdown("#### 📊 Biểu đồ so sánh")
            
            import matplotlib.pyplot as plt
            
            # PSNR & SSIM (chỉ số quan trọng nhất)
            col1, col2 = st.columns(2)
            
            with col1: # Biểu đồ PSNR
                fig, ax = plt.subplots(figsize=(6, 4))
                psnr_values = [calculate_metrics(img, out)['PSNR'] for _, out in results]
                colors = ['#667eea', '#764ba2', '#f093fb']
                ax.bar([name for name, _ in results], psnr_values, color=colors[:len(results)])
                ax.set_ylabel('PSNR (dB)')
                ax.set_title('So sánh PSNR (càng cao càng tốt)')
                ax.grid(axis='y', alpha=0.3)
                # Thêm đường tham chiếu
                ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Tốt (>30)')
                ax.axhline(y=40, color='darkgreen', linestyle='--', alpha=0.5, label='Rất tốt (>40)')
                ax.legend()
                st.pyplot(fig)
                plt.close()
            
            with col2: # Biểu đồ SSIM
                fig, ax = plt.subplots(figsize=(6, 4))
                ssim_values = [calculate_metrics(img, out)['SSIM'] for _, out in results]
                ax.bar([name for name, _ in results], ssim_values, color=colors[:len(results)])
                ax.set_ylabel('SSIM')
                ax.set_ylim([0, 1])
                ax.set_title('So sánh SSIM (càng gần 1 càng tốt)')
                ax.grid(axis='y', alpha=0.3)
                # Thêm đường tham chiếu
                ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Tốt (>0.9)')
                ax.axhline(y=0.95, color='darkgreen', linestyle='--', alpha=0.5, label='Rất tốt (>0.95)')
                ax.legend()
                st.pyplot(fig)
                plt.close()
            
            st.divider()
            
            # MSE & MAE (chỉ số bổ trợ)
            col3, col4 = st.columns(2)
            
            with col3: # Biểu đồ MSE
                fig, ax = plt.subplots(figsize=(6, 4))
                mse_values = [calculate_metrics(img, out)['MSE'] for _, out in results]
                ax.bar([name for name, _ in results], mse_values, color=colors[:len(results)])
                ax.set_ylabel('MSE')
                ax.set_title('So sánh MSE (càng thấp càng tốt)')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col4: # Biểu đồ MAE
                fig, ax = plt.subplots(figsize=(6, 4))
                mae_values = [calculate_metrics(img, out)['MAE'] for _, out in results]
                ax.bar([name for name, _ in results], mae_values, color=colors[:len(results)])
                ax.set_ylabel('MAE')
                ax.set_title('So sánh MAE (càng thấp càng tốt)')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            st.divider()
            
            # Đánh giá và gợi ý
            st.markdown("#### 💡 Nhận xét & Gợi ý")
            
            # Tìm bộ lọc tốt nhất dựa trên PSNR và SSIM
            best_psnr_idx = psnr_values.index(max(psnr_values)) # Chỉ số PSNR cao nhất
            best_ssim_idx = ssim_values.index(max(ssim_values)) # Chỉ số SSIM cao nhất
            best_filter_psnr = results[best_psnr_idx][0] # Tên bộ lọc có PSNR cao nhất
            best_filter_ssim = results[best_ssim_idx][0] # Tên bộ lọc có SSIM cao nhất
            
            col_a, col_b = st.columns(2) # Hiển thị bộ lọc tốt nhất
            with col_a: # Hiển thị bộ lọc tốt nhất
                st.success(f"🏆 **PSNR cao nhất:** {best_filter_psnr}\n\nGiá trị: {psnr_values[best_psnr_idx]:.2f} dB")
            with col_b: # Hiển thị bộ lọc tốt nhất
                st.success(f"🏆 **SSIM cao nhất:** {best_filter_ssim}\n\nGiá trị: {ssim_values[best_ssim_idx]:.4f}")
            
            # Gợi ý dựa trên kết quả
            if best_filter_psnr == best_filter_ssim: # Nếu cùng bộ lọc tốt nhất
                st.info(f"✅ **Kết luận:** Bộ lọc **{best_filter_psnr}** cho kết quả tốt nhất cho ảnh này!")
            else:
                st.info(f"ℹ️ **Kết luận:** Bộ lọc **{best_filter_psnr}** có PSNR cao nhất, nhưng **{best_filter_ssim}** có SSIM cao nhất. Nên xem xét cả hai chỉ số để đánh giá tổng thể.")
                
        elif results and not add_noise: # Nếu có kết quả nhưng không có ảnh nhiễu
            st.warning("⚠️ Cần có ảnh nhiễu (bật 'Thêm nhiễu vào ảnh') để tính toán các chỉ số so sánh với ảnh gốc.")
        else:
            st.info("Chưa có kết quả để hiển thị chỉ số đánh giá.")
    
    with tab3: # Tab thống kê
        st.markdown("### 📊 Thống kê ảnh")
        
        if results:
            col1, col2, col3 = st.columns(3)
            
            # Thông tin ảnh gốc
            with col1:
                st.metric("📐 Kích thước", f"{img.shape[1]} × {img.shape[0]} px")
            with col2:
                st.metric("🎨 Số kênh màu", img.shape[2] if len(img.shape) > 2 else 1)
            with col3:
                st.metric("🔢 Độ sâu bit", f"{img.dtype}")
            
            st.divider()
            
            # So sánh histogram
            st.markdown("#### 📈 Phân bố mức xám")
            fig_cols = st.columns(len(results) + 1)
            
            import matplotlib.pyplot as plt
            
            # Histogram ảnh gốc/nhiễu
            with fig_cols[0]: # Cột đầu tiên cho ảnh gốc hoặc ảnh nhiễu
                fig, ax = plt.subplots(figsize=(4, 3)) # Tạo biểu đồ
                gray_work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) # Chuyển ảnh sang xám
                ax.hist(gray_work.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7) # Vẽ histogram
                ax.set_title('Input' if not add_noise else f'Noisy ({noise_type})')
                ax.set_xlabel('Pixel value')
                ax.set_ylabel('Frequency')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            # Histogram các kết quả
            for idx, (name, out) in enumerate(results): # Vòng lặp qua từng kết quả để hiển thị histogram
                with fig_cols[idx + 1]: # Cột tiếp theo cho từng bộ lọc
                    fig, ax = plt.subplots(figsize=(4, 3)) # Tạo biểu đồ
                    gray_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) # Chuyển ảnh sang xám 
                    ax.hist(gray_out.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
                    ax.set_title(name)
                    ax.set_xlabel('Pixel value')
                    ax.set_ylabel('Frequency')
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
        else:
            st.info("Chưa có kết quả để hiển thị thống kê.")
    
    with tab4: # Tab tải xuống
        st.markdown("### 💾 Tải xuống kết quả")
        
        if results: # Nếu có kết quả để tải xuống
            st.success(f"✅ Đã xử lý thành công {len(results)} ảnh!")
            
            # Tạo nút download cho từng kết quả
            for idx, (name, out) in enumerate(results): # Vòng lặp qua từng kết quả để tạo nút tải xuống
                col1, col2 = st.columns([3, 1]) # Tạo 2 cột: tên bộ lọc và nút tải xuống
                with col1: # Cột tên bộ lọc
                    st.write(f"**{idx+1}. {name}**")
                with col2: # Cột nút tải xuống
                    st.download_button( # Tạo nút tải xuống
                        label="⬇️ Tải xuống",
                        data=cv2.imencode('.png', out)[1].tobytes(), # Mã hóa ảnh sang PNG và chuyển thành bytes
                        file_name=f"denoised_{name.lower().replace(' ', '_')}.png", # Tên file tải xuống
                        mime="image/png", # Định dạng MIME
                        key=f"download_{idx}" # Khóa duy nhất cho mỗi nút
                    )
            
            st.divider()
            
            # Nút tải tất cả (zip)
            st.markdown("#### 📦 Tải tất cả cùng lúc") # Tab tải xuống
            st.info("💡 Mẹo: Tải từng ảnh ở trên hoặc sử dụng các công cụ nén file nếu cần tải nhiều ảnh.")
        else:
            st.warning("⚠️ Chưa có kết quả để tải xuống. Vui lòng chọn bộ lọc!")

else:
    # Màn hình chào mừng
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2>👋 Chào mừng đến với Image Denoising Studio!</h2>
        <p style='font-size: 1.2rem; color: #666;'>
            Hãy tải lên một ảnh từ sidebar bên trái để bắt đầu
        </p>
        <br>
        <p>
            ✨ <b>Hỗ trợ:</b> PNG, JPG, JPEG<br>
            🎯 <b>Phương pháp:</b> Gaussian, Median, Non-Local Means<br>
            📊 <b>Tính năng:</b> So sánh, Thống kê, Tải xuống
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hướng dẫn sử dụng
    with st.expander("📖 Hướng dẫn sử dụng", expanded=True):
        st.markdown("""
        ### Các bước sử dụng:
        
        1. **📤 Tải ảnh lên** từ sidebar bên trái
        2. **🎲 (Tùy chọn)** Thêm nhiễu để mô phỏng
        3. **🔧 Chọn bộ lọc** khử nhiễu (có thể chọn nhiều để so sánh)
        4. **⚙️ Điều chỉnh tham số** nếu cần (phần Advanced Settings)
        5. **👀 Xem kết quả** ở tab "So sánh kết quả"
        6. **📊 Phân tích** histogram ở tab "Thống kê"
        7. **💾 Tải xuống** ảnh đã xử lý ở tab "Tải xuống"
        
        ### Giải thích các bộ lọc:
        
        - **🌀 Gaussian Filter**: Làm mượt ảnh, phù hợp với nhiễu Gaussian
        - **📊 Median Filter**: Loại bỏ nhiễu muối tiêu hiệu quả, giữ cạnh tốt
        - **🎨 Non-Local Means**: Khử nhiễu cao cấp, giữ chi tiết và texture tốt nhất
        
        ### Mẹo sử dụng:
        
        - Với nhiễu Gaussian: dùng Gaussian hoặc NLM
        - Với nhiễu Salt & Pepper: dùng Median
        - Để so sánh: chọn cả 3 bộ lọc cùng lúc
        - Muốn ảnh sắc nét: bật Unsharp mask
        """)



