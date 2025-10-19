Image Denoising Project
=======================

Overview
--------
Pipeline to generate noisy images, denoise using three filters (Gaussian, Median, NLM), and evaluate with PSNR/SSIM/MSE/MAE plus histogram/error/edge visualizations. Includes a Streamlit UI.

Environment
-----------
- Python 3.9+

Install
-------
1) Create venv (optional)
   - Windows PowerShell:
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
2) Install dependencies:
   pip install -r requirements.txt

Quick start
-----------
1) Prepare data (download sample images into `data/original`):
   python src/prepare_data.py
2) Generate noisy images:
   python src/add_noise.py --sigmas 10,20,30 --sp_levels 0.01,0.03,0.05 --seed 42 --max_size 1024
3) Denoise all noisy images:
   python src/denoise.py --filters gaussian,median,nlm --ksize 5 --sigma 1.5 --nlm_h 10 --nlm_hColor 10 --nlm_tws 7 --nlm_sws 21 --max_size 1024 --profile --timings_csv report/timings.csv
4) Evaluate and export metrics/figures:
   python src/evaluate.py --plot_hist --plot_error --plot_edges

UI
--
Run the Streamlit UI:

   streamlit run src/app.py

Then open the provided local URL in your browser to interactively test noise and filters and download results.

Outputs
-------
- data/noisy: noisy images with naming `{base}_gauss_sigmaX.png`, `{base}_sp_pY.png`
- results/denoised: denoised images `{noisy_base}_{filter}.png`
- report/report_data.csv: PSNR/SSIM table
- report/hist_*.png: histogram comparison figures
 - report/error_*.png: error maps; report/edges_*.png: edge maps

CLI usage examples
------------------
- Only Gaussian noise with sigma 20, on smaller images (max 800px):
  python src/add_noise.py --sigmas 20 --sp_levels "" --max_size 800
- Denoise with only Median and NLM (quick run):
  python src/denoise.py --filters median,nlm --ksize 3 --nlm_h 7 --nlm_hColor 7 --profile
- Evaluate and save extra visuals to a custom CSV path:
  python src/evaluate.py --out_csv report/my_run.csv --plot_hist --plot_error


