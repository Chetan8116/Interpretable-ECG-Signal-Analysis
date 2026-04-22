"""
Load COMPLETE PTB-XL signals from .dat files and process them
This script reads the full 10-second ECG signals (not just beat segments)
"""

import pandas as pd
import wfdb
import numpy as np
import scipy.signal as sp_signal
import pywt
import json
from pathlib import Path
from tqdm import tqdm
from scipy.signal import medfilt

# -------------------------
# Signal Processing Functions
# -------------------------

def _ensure_odd(k):
    k = int(max(1, round(k)))
    return k if (k % 2 == 1) else k + 1

def moving_average(x, kernel_len):
    if kernel_len <= 1:
        return x.copy()
    kernel = np.ones(kernel_len, dtype=float) / float(kernel_len)
    return np.convolve(x, kernel, mode='same')

def notch_filter(sig, fs, freq=50.0, Q=30.0):
    """Zero-phase notch filter"""
    b, a = sp_signal.iirnotch(w0=freq, Q=Q, fs=fs)
    return sp_signal.filtfilt(b, a, sig)

def bandpass_filter(sig, fs, low=0.5, high=40.0, order=4):
    """Zero-phase Butterworth bandpass"""
    nyq = 0.5 * fs
    if low <= 0 or high >= nyq:
        # Adjust limits if needed
        low = max(0.5, low)
        high = min(nyq - 1, high)
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype='band')
    return sp_signal.filtfilt(b, a, sig)

def wavelet_denoise(sig, wavelet='db6', level=None):
    """Soft-thresholding wavelet denoising"""
    maxlev = pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len)
    if level is None:
        level = max(1, min(6, maxlev))

    coeffs = pywt.wavedec(sig, wavelet, level=level, mode='symmetric')
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745 if detail_coeffs.size else 0.0
    uthresh = sigma * np.sqrt(2 * np.log(len(sig))) if sigma > 0 else 0.0

    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    rec = pywt.waverec(coeffs_thresh, wavelet, mode='symmetric')

    if len(rec) > len(sig):
        rec = rec[:len(sig)]
    elif len(rec) < len(sig):
        rec = np.pad(rec, (0, len(sig) - len(rec)), mode='edge')
    return rec

def remove_baseline_wander(sig, fs, method='cascade', med_spike_ms=25, 
                          med_baseline_ms=200, smooth_baseline_ms=100):
    """Remove baseline wander using cascade median filter"""
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    
    k_spike = _ensure_odd(int(round((med_spike_ms / 1000.0) * fs)))
    k_base  = _ensure_odd(int(round((med_baseline_ms / 1000.0) * fs)))
    
    # Step 1: small median (spike suppression)
    if k_spike <= 1:
        x1 = sig.copy()
    else:
        pad1 = k_spike // 2
        padded1 = np.pad(sig, pad1, mode='edge')
        x1 = medfilt(padded1, kernel_size=k_spike)[pad1:pad1 + n]
    
    # Step 2: large median baseline estimate
    pad2 = k_base // 2
    padded2 = np.pad(x1, pad2, mode='edge')
    baseline = medfilt(padded2, kernel_size=k_base)[pad2:pad2 + n]
    
    # Optional smoothing
    if smooth_baseline_ms is not None and smooth_baseline_ms > 0:
        k_smooth = int(round((smooth_baseline_ms / 1000.0) * fs))
        if k_smooth > 1:
            baseline = moving_average(baseline, k_smooth)
    
    return sig - baseline

def process_signal(ecg_raw, fs):
    """Complete signal processing pipeline"""
    # DC removal
    signal = ecg_raw - np.median(ecg_raw)
    
    # Baseline wander removal
    signal = remove_baseline_wander(signal, fs, method='cascade', 
                                   med_spike_ms=25, med_baseline_ms=200, 
                                   smooth_baseline_ms=100)
    
    # Notch filter (50 Hz)
    signal = notch_filter(signal, fs, freq=50.0, Q=30.0)
    
    # Bandpass filter (0.5-40 Hz)
    signal = bandpass_filter(signal, fs, low=0.5, high=40.0, order=4)
    
    # Wavelet denoising
    signal = wavelet_denoise(signal, wavelet='db6', level=4)
    
    return signal

# -------------------------
# Main Processing
# -------------------------

def load_and_process_ptbxl(base_path, output_path, num_records=50, use_high_res=True):
    """
    Load complete ECG signals from PTB-XL .dat files and process them
    
    Args:
        base_path: Path to PTB-XL dataset
        output_path: Path to save processed JSON
        num_records: Number of records to process
        use_high_res: Use 500Hz (hr) instead of 100Hz (lr) signals
    """
    
    # Load database
    df = pd.read_csv(f"{base_path}/ptbxl_database.csv")
    print(f"Found {len(df)} records in PTB-XL database")
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    processed_records = []
    
    for idx in tqdm(range(min(num_records, len(df))), desc="Processing records"):
        record_info = df.iloc[idx]
        
        # Choose high-res or low-res
        filename = record_info['filename_hr'] if use_high_res else record_info['filename_lr']
        full_path = f"{base_path}/{filename}"
        
        try:
            # Load COMPLETE signal (all 10 seconds)
            record = wfdb.rdrecord(full_path)
            fs = int(record.fs)
            
            # Process all 12 leads
            leads_data = {}
            for i, lead_name in enumerate(lead_names):
                if i >= record.p_signal.shape[1]:
                    continue
                
                # Get FULL signal
                raw_signal = record.p_signal[:, i]
                
                # Process through pipeline
                processed_signal = process_signal(raw_signal, fs)
                
                # Store complete signal (convert to list for JSON)
                leads_data[lead_name] = {
                    'raw': raw_signal.tolist(),
                    'processed': processed_signal.tolist()
                }
            
            # Create record entry
            record_data = {
                'ecg_id': int(record_info['ecg_id']),
                'patient_id': int(record_info['patient_id']),
                'age': int(record_info['age']) if pd.notna(record_info['age']) else 0,
                'sex': 'M' if record_info['sex'] == 1 else 'F',
                'sampling_rate': fs,
                'duration': len(raw_signal) / fs,
                'num_samples': len(raw_signal),
                'diagnosis': str(record_info['report']) if pd.notna(record_info['report']) else '',
                'scp_codes': str(record_info['scp_codes']) if pd.notna(record_info['scp_codes']) else '{}',
                'leads': leads_data
            }
            
            processed_records.append(record_data)
            
        except Exception as e:
            print(f"Error processing record {idx} ({full_path}): {e}")
            continue
    
    # Save to JSON
    print(f"\nSaving {len(processed_records)} processed records to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(processed_records, f, indent=2)
    
    # Print statistics
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✓ Saved successfully!")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Samples per lead: {processed_records[0]['num_samples']}")
    print(f"  Sampling rate: {processed_records[0]['sampling_rate']} Hz")
    print(f"  Duration: {processed_records[0]['duration']:.1f} seconds")

if __name__ == "__main__":
    # Configuration
    BASE_PATH = r"C:\Users\Harsha\Pictures\RM\archive\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    OUTPUT_PATH = r"C:\Users\Harsha\Pictures\RM\public\ptbxl_full_signals.json"
    
    # Process 50 records with full high-resolution signals (500 Hz, 10 seconds each)
    load_and_process_ptbxl(
        base_path=BASE_PATH,
        output_path=OUTPUT_PATH,
        num_records=50,
        use_high_res=True  # 500 Hz for better quality
    )
    
    print("\n" + "="*60)
    print("✓ Processing complete!")
    print("="*60)
    print(f"\nThe web application will now load FULL 10-second signals")
    print(f"from: {OUTPUT_PATH}")
    print("\nRestart your dev server to see real ECG waveforms!")
