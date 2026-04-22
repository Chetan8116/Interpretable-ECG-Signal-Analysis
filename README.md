# Interpretable-ECG-Signal-Analysis


This project combines:

- A React + Vite dashboard for interactive 12-lead ECG visualization.
- A Node.js API server for denoising and feature-extraction orchestration.
- Python pipelines for PTB-XL denoising, feature extraction, model training, and SHAP explainability.

## What Is Included

- Web app and visual analytics in `src/`.
- Node API services in `server/`.
- Data processing and model scripts in `scripts/`.
- PTB-XL dataset and generated features in:
  - `archive/`
  - `ptbxl_denoised/`
  - `ptbxl_comprehensive_features/`
  - `ptbxl_features/`

## Requirements

- Node.js 18+
- Python 3.9+
- pip

## Quick Start

### 1) Install JavaScript dependencies

```bash
npm install
```

### 2) Create/activate Python environment and install dependencies

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Start the full development stack

```bash
npm run dev:all
```

This starts:

- Frontend (Vite): `http://localhost:3000`
- Denoising API: `http://localhost:3001`
- SHAP API (Flask): `http://localhost:5101`

You can also run parts separately:

```bash
npm run dev:web     # frontend only
npm run server      # Node denoising/feature API
npm run shap        # Flask SHAP server
```

## PTB-XL Data Setup

Expected layout:

```text
archive/
  ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
    ptbxl_database.csv
    records500/
    records100/
```

Useful check:

```bash
python check_setup.py
```

## Common Workflows

### Denoise PTB-XL records

```bash
python scripts/denoise_ptbxl_data.py
```

See additional denoising details in `scripts/README_DENOISING.md`.

### Extract comprehensive ECG features

Single process:

```bash
python scripts/extract_comprehensive_features_ptbxl.py
```

Parallel (faster on multi-core machines):

```bash
python scripts/extract_comprehensive_features_ptbxl_parallel.py
```

### Prepare frontend data artifacts

```bash
npm run prepare-data
```

### Build production frontend

```bash
npm run build
npm run preview
```

## Training and Modeling Scripts

This repository contains multiple model training scripts in `scripts/` and at the project root, including:

- `scripts/train_mlp.py`
- `scripts/train_resnet1d_ptbxl.py`
- `scripts/train_resnet1d_hybrid.py`
- `scripts/train_lgbm_best.py`
- `train_resnet1d_hybrid_standalone.py`

Note: `train_resnet1d_hybrid_standalone.py` is a template stub and indicates that full implementation must be copied in before use.

## API Endpoints (Development)

Node server (`server/denoise-server.js`):

- `GET /api/health`
- `GET /api/denoise/status`
- `POST /api/denoise/start`
- `GET /api/denoise/output/:processId`
- `POST /api/denoise/stop/:processId`
- `POST /api/features/extract`

SHAP server (`server/shap_server.py`):

- `GET /api/shap/status`
- `GET /api/shap/classes`
- `GET /api/shap/<ecg_id>`
- `POST /api/shap/batch`

## Troubleshooting

- Python not detected by Node server:
  - Ensure Python is installed and available in PATH as `py`, `python`, or `python3`.
- Missing packages for SHAP server:
  - Install extras with:

```bash
pip install flask flask-cors shap
```

- If frontend cannot load expected data files:
  - Run `npm run prepare-data`.
  - Verify required files exist in `public/`.

## Notes

- This project includes large dataset and generated artifact directories. Keep that in mind when copying or versioning.
- For reproducible research runs, pin Python package versions in `requirements.txt` as needed.
