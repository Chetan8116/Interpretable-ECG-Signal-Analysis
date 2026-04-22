# ECG Denoising Pipeline

This script provides a comprehensive denoising pipeline for PTB-XL ECG signals.

## Overview

The `denoise_ptbxl_data.py` script processes raw ECG signals through a multi-stage denoising pipeline and saves cleaned signals as NumPy arrays.

## Denoising Pipeline Stages

1. **DC Removal**: Removes DC offset using mean or median
2. **Baseline Wander Removal**: Cascade of median filters to remove baseline drift
3. **Notch Filter**: Removes power line interference (50Hz or 60Hz)
4. **Bandpass Filter**: Butterworth filter (default 0.5-40 Hz)
5. **Wavelet Denoising**: Wavelet decomposition with soft thresholding

## Installation

Install required Python packages:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy pandas scipy PyWavelets wfdb tqdm
```

## Usage

### Basic Usage

```python
python scripts/denoise_ptbxl_data.py
```

### Configuration

Edit the script's `__main__` section to customize paths and parameters:

```python
if __name__ == '__main__':
    df_csv_path = r"archive/ptb-xl-.../ptbxl_database.csv"
    base_path   = r"archive/ptb-xl-.../"
    out_dir     = r"ptbxl_denoised"

    df_meta = process_ptbxl_all(
        df_csv_path,
        base_path,
        out_dir,
        prefer_fs=None,     # 'hr', 'lr', 500, 100, or None
        target_fs=500,      # resample to 500Hz
        max_records=5,      # limit for testing (None for all)
        verbose=True
    )
```

### Parameters

#### Main Function Parameters

- `df_csv_path`: Path to `ptbxl_database.csv`
- `base_path`: Base directory containing PTB-XL records
- `out_dir`: Output directory for processed signals
- `prefer_fs`: Preferred sampling rate ('hr'=500Hz or 'lr'=100Hz)
- `target_fs`: Target sampling rate for resampling (e.g., 500)
- `max_records`: Limit number of records to process (None for all)
- `verbose`: Print progress messages

#### Denoising Parameters

**Baseline Removal:**
```python
baseline_params = {
    'method': 'cascade',
    'med_spike_ms': 25,        # spike removal window (ms)
    'med_baseline_ms': 200,    # baseline estimation window (ms)
    'smooth_baseline_ms': 100  # baseline smoothing window (ms)
}
```

**Notch Filter:**
```python
notch_params = {
    'freq': 50.0,  # 50Hz (Europe) or 60Hz (US)
    'Q': 30.0      # quality factor
}
```

**Bandpass Filter:**
```python
bandpass_params = {
    'low': 0.5,    # low cutoff frequency (Hz)
    'high': 40.0,  # high cutoff frequency (Hz)
    'order': 4     # filter order
}
```

**Wavelet Denoising:**
```python
wavelet_params = {
    'wavelet': 'db6',          # Daubechies 6
    'level': 4,                # decomposition level
    'threshold_mode': 'soft'   # 'soft' or 'hard'
}
```

## Output Structure

The script creates the following output structure:

```
ptbxl_denoised/
├── records100/00000/00001_lr/
│   ├── lead0_fs500.npy
│   ├── lead1_fs500.npy
│   ├── ...
│   └── lead11_fs500.npy
├── records100/00000/00002_lr/
│   └── ...
├── processed_metadata_partial.csv  # saved every 5 records
└── processed_metadata.csv          # final metadata
```

### Metadata CSV Columns

- `record_index`: Original CSV row index
- `record_name`: Record filename
- `lead`: Lead index (0-11)
- `orig_fs`: Original sampling frequency
- `working_fs`: Final sampling frequency
- `n_samples`: Number of samples in cleaned signal
- `out_path`: Path to saved .npy file

## Loading Processed Signals

```python
import numpy as np
import pandas as pd

# Load metadata
meta = pd.read_csv('ptbxl_denoised/processed_metadata.csv')

# Load specific signal
signal = np.load(meta.loc[0, 'out_path'])

# Load all leads for a record
record_meta = meta[meta['record_name'] == 'records100/00000/00001_lr']
leads = [np.load(row['out_path']) for _, row in record_meta.iterrows()]
```

## Performance

- Processing time: ~1-2 seconds per record (12 leads)
- Output size: ~50KB per lead (500Hz, 10s recording)
- Memory usage: ~100MB for processing

## Troubleshooting

### Common Issues

1. **"CSV not found"**: Check `df_csv_path` points to `ptbxl_database.csv`
2. **"Could not find record files"**: Verify `base_path` contains the PTB-XL data directory
3. **Memory errors**: Process in batches using `max_records` parameter
4. **Import errors**: Install missing packages with `pip install`

### File Format

The script expects PTB-XL data structure:
```
ptb-xl/
├── ptbxl_database.csv
└── records100/ or records500/
    └── 00000/
        ├── 00001_lr.hea
        ├── 00001_lr.dat
        └── ...
```

## Advanced Usage

### Custom Pipeline

```python
from scripts.denoise_ptbxl_data import run_pipeline
import numpy as np

# Load your signal
signal = np.load('your_signal.npy')
fs = 500  # sampling rate

# Run with custom parameters
results = run_pipeline(
    signal, fs,
    dc_remove='median',
    baseline_params={'method': 'cascade', 'med_baseline_ms': 300},
    notch_params={'freq': 60.0, 'Q': 40.0},
    bandpass_params={'low': 1.0, 'high': 30.0},
    wavelet_params={'wavelet': 'db4', 'level': 5}
)

# Access intermediate results
denoised = results['denoised']
after_baseline = results['baseline_removed']
after_notch = results['notch_filtered']
```

### Batch Processing

For large datasets, process in batches:

```python
# Process first 100 records
process_ptbxl_all(..., max_records=100)

# Then next 100, etc.
```

## References

- PTB-XL Database: https://physionet.org/content/ptb-xl/
- WFDB Python Package: https://github.com/MIT-LCP/wfdb-python
- Wavelet Denoising: Donoho & Johnstone (1994)
