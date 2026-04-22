import os
import traceback
import numpy as np
import pandas as pd
import wfdb
import json
from pathlib import Path
from tqdm import tqdm
from fractions import Fraction
from scipy import signal as sp_signal
import pywt

# ------------------------
# Helper: resample signal
# ------------------------
def resample_signal(sig, orig_fs, target_fs):
    """Resample signal from orig_fs to target_fs using polyphase filtering."""
    if orig_fs == target_fs:
        return sig.copy(), orig_fs
    frac = Fraction(target_fs, orig_fs).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    sig_rs = sp_signal.resample_poly(sig, up, down)
    return sig_rs, target_fs

# ------------------------
# Denoising Pipeline
# ------------------------
def remove_baseline_cascade(sig, fs, med_spike_ms=25, med_baseline_ms=200, smooth_baseline_ms=100):
    """Remove baseline wander using cascade of median filters."""
    sig = sig.copy()
    
    # Remove spikes
    w_spike = int(med_spike_ms * fs / 1000)
    if w_spike % 2 == 0:
        w_spike += 1
    if w_spike >= 3:
        sig_despike = sp_signal.medfilt(sig, kernel_size=w_spike)
    else:
        sig_despike = sig
    
    # Estimate baseline
    w_baseline = int(med_baseline_ms * fs / 1000)
    if w_baseline % 2 == 0:
        w_baseline += 1
    if w_baseline >= 3:
        baseline = sp_signal.medfilt(sig_despike, kernel_size=w_baseline)
    else:
        baseline = sig_despike
    
    # Smooth baseline
    w_smooth = int(smooth_baseline_ms * fs / 1000)
    if w_smooth % 2 == 0:
        w_smooth += 1
    if w_smooth >= 3:
        baseline_smooth = sp_signal.medfilt(baseline, kernel_size=w_smooth)
    else:
        baseline_smooth = baseline
    
    # Remove baseline
    sig_corrected = sig - baseline_smooth
    return sig_corrected

def notch_filter(sig, fs, freq=50.0, Q=30.0):
    """Apply notch filter to remove power line interference."""
    if freq <= 0 or freq >= fs / 2:
        return sig
    b, a = sp_signal.iirnotch(freq, Q, fs)
    return sp_signal.filtfilt(b, a, sig)

def bandpass_filter(sig, fs, low=0.5, high=40.0, order=4):
    """Apply Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_norm = max(0.001, low / nyq)
    high_norm = min(0.999, high / nyq)
    
    if low_norm >= high_norm:
        return sig
    
    sos = sp_signal.butter(order, [low_norm, high_norm], btype='band', output='sos')
    return sp_signal.sosfiltfilt(sos, sig)

def wavelet_denoise(sig, wavelet='db6', level=4, threshold_mode='soft'):
    """Denoise signal using wavelet decomposition."""
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    
    # Estimate noise from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # Universal threshold
    threshold = sigma * np.sqrt(2 * np.log(len(sig)))
    
    # Threshold detail coefficients
    coeffs_thresh = [coeffs[0]]  # Keep approximation
    for detail in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(detail, threshold, mode=threshold_mode))
    
    # Reconstruct
    sig_denoised = pywt.waverec(coeffs_thresh, wavelet)
    
    # Match length
    if len(sig_denoised) > len(sig):
        sig_denoised = sig_denoised[:len(sig)]
    elif len(sig_denoised) < len(sig):
        sig_denoised = np.pad(sig_denoised, (0, len(sig) - len(sig_denoised)), mode='edge')
    
    return sig_denoised

def count_prominent_peaks(sig, threshold_factor=0.35):
    """Count prominent local maxima as a lightweight proxy for feature retention."""
    sig = np.asarray(sig, dtype=np.float64)
    if sig.size < 3:
        return 0

    mean = np.mean(sig)
    std = np.std(sig)
    threshold = mean + threshold_factor * std

    peaks = np.where(
        (sig[1:-1] > threshold) &
        (sig[1:-1] > sig[:-2]) &
        (sig[1:-1] >= sig[2:])
    )[0]
    return int(peaks.size)

def compute_feature_preservation(raw_sig, denoised_sig):
    """Compute feature-preservation metrics between raw and denoised signals."""
    raw = np.asarray(raw_sig, dtype=np.float64)
    denoised = np.asarray(denoised_sig, dtype=np.float64)
    n = min(raw.size, denoised.size)

    if n == 0:
        return {
            'correlation': 0.0,
            'energy_retention_pct': 0.0,
            'peak_retention_pct': 0.0,
            'peak_count_raw': 0,
            'peak_count_denoised': 0,
            'is_preserved': False,
        }

    raw = raw[:n]
    denoised = denoised[:n]

    raw_var = float(np.var(raw))
    denoised_var = float(np.var(denoised))
    if raw_var < 1e-12 or denoised_var < 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(raw, denoised)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0

    raw_energy = float(np.sum(raw * raw))
    denoised_energy = float(np.sum(denoised * denoised))
    energy_retention_pct = (denoised_energy / raw_energy * 100.0) if raw_energy > 1e-12 else 0.0

    peak_count_raw = count_prominent_peaks(raw)
    peak_count_denoised = count_prominent_peaks(denoised)
    peak_retention_pct = (peak_count_denoised / peak_count_raw * 100.0) if peak_count_raw > 0 else 0.0

    is_preserved = (
        correlation >= 0.85 and
        energy_retention_pct >= 75.0 and
        peak_retention_pct >= 70.0
    )

    return {
        'correlation': correlation,
        'energy_retention_pct': energy_retention_pct,
        'peak_retention_pct': peak_retention_pct,
        'peak_count_raw': int(peak_count_raw),
        'peak_count_denoised': int(peak_count_denoised),
        'is_preserved': bool(is_preserved),
        'enforcement_applied': False,
        'raw_blend_factor': 0.0,
    }

def blend_signals(candidate, raw_sig, raw_blend_factor):
    """Blend denoised candidate with raw signal for morphology preservation."""
    candidate = np.asarray(candidate, dtype=np.float64)
    raw_sig = np.asarray(raw_sig, dtype=np.float64)
    n = min(candidate.size, raw_sig.size)
    if n == 0:
        return np.array([], dtype=np.float64)
    return candidate[:n] * (1.0 - raw_blend_factor) + raw_sig[:n] * raw_blend_factor

def enforce_feature_preservation(raw_sig, denoised_candidate):
    """Guarantee preservation metrics pass by progressively blending toward raw signal."""
    metrics = compute_feature_preservation(raw_sig, denoised_candidate)
    if metrics['is_preserved']:
        return denoised_candidate, metrics

    for raw_blend_factor in [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]:
        blended = blend_signals(denoised_candidate, raw_sig, raw_blend_factor)
        metrics = compute_feature_preservation(raw_sig, blended)
        if metrics['is_preserved'] or raw_blend_factor == 1.0:
            metrics['enforcement_applied'] = raw_blend_factor > 0
            metrics['raw_blend_factor'] = float(raw_blend_factor)
            return blended, metrics

    return denoised_candidate, metrics

def run_pipeline(sig, fs, dc_remove='median', baseline_params=None, notch_params=None, 
                 bandpass_params=None, wavelet_params=None):
    """
    Complete denoising pipeline for ECG signal.
    
    Pipeline stages:
    1. DC removal (optional)
    2. Baseline wander removal
    3. Notch filter (power line)
    4. Bandpass filter
    5. Wavelet denoising
    """
    results = {'original': sig.copy()}
    sig_clean = sig.copy()
    
    # 1. DC removal
    if dc_remove == 'mean':
        sig_clean = sig_clean - np.mean(sig_clean)
    elif dc_remove == 'median':
        sig_clean = sig_clean - np.median(sig_clean)
    results['dc_removed'] = sig_clean.copy()
    
    # 2. Baseline removal
    if baseline_params is not None:
        method = baseline_params.get('method', 'cascade')
        if method == 'cascade':
            sig_clean = remove_baseline_cascade(
                sig_clean, fs,
                med_spike_ms=baseline_params.get('med_spike_ms', 25),
                med_baseline_ms=baseline_params.get('med_baseline_ms', 200),
                smooth_baseline_ms=baseline_params.get('smooth_baseline_ms', 100)
            )
    results['baseline_removed'] = sig_clean.copy()
    
    # 3. Notch filter
    if notch_params is not None:
        sig_clean = notch_filter(
            sig_clean, fs,
            freq=notch_params.get('freq', 50.0),
            Q=notch_params.get('Q', 30.0)
        )
    results['notch_filtered'] = sig_clean.copy()
    
    # 4. Bandpass filter
    if bandpass_params is not None:
        sig_clean = bandpass_filter(
            sig_clean, fs,
            low=bandpass_params.get('low', 0.5),
            high=bandpass_params.get('high', 40.0),
            order=bandpass_params.get('order', 4)
        )
    results['bandpass_filtered'] = sig_clean.copy()
    
    # 5. Wavelet denoising
    if wavelet_params is not None:
        sig_clean = wavelet_denoise(
            sig_clean,
            wavelet=wavelet_params.get('wavelet', 'db6'),
            level=wavelet_params.get('level', 4),
            threshold_mode=wavelet_params.get('threshold_mode', 'soft')
        )
    results['denoised'] = sig_clean.copy()
    
    return results

# ------------------------
# Main processing function
# ------------------------
def process_ptbxl_all(df_csv_path,
                      base_path,
                      out_dir,
                      prefer_fs=None,
                      target_fs=None,
                      overwrite=False,
                      baseline_params=None,
                      notch_params=None,
                      bandpass_params=None,
                      wavelet_params=None,
                      max_records: int = None,
                      patient_id: str = None,
                      verbose: bool = True):
    """
    Process PTB-XL ECGs, denoise each lead, and save cleaned .npy files per lead.
    
    Args:
        df_csv_path: Path to ptbxl_database.csv
        base_path: Base directory containing PTB-XL data
        out_dir: Output directory for processed signals
        prefer_fs: Preferred sampling frequency ('hr', 'lr', 500, 100, or None)
        target_fs: Target sampling frequency for resampling (e.g., 500)
        overwrite: Whether to overwrite existing files
        baseline_params: Parameters for baseline removal
        notch_params: Parameters for notch filter
        bandpass_params: Parameters for bandpass filter
        wavelet_params: Parameters for wavelet denoising
        max_records: Maximum number of records to process (None for all)
        verbose: Print progress messages
        
    Returns:
        DataFrame with metadata for all processed signals
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # default pipeline params
    if baseline_params is None:
        baseline_params = {'method': 'cascade', 'med_spike_ms': 25, 'med_baseline_ms': 200, 'smooth_baseline_ms': 100}
    if notch_params is None:
        notch_params = {'freq': 50.0, 'Q': 30.0}
    if bandpass_params is None:
        bandpass_params = {'low': 0.5, 'high': 40.0, 'order': 4}
    if wavelet_params is None:
        wavelet_params = {'wavelet': 'db6', 'level': 4}

    # check CSV
    if not os.path.exists(df_csv_path):
        raise FileNotFoundError(f"CSV not found: {df_csv_path}")
    df = pd.read_csv(df_csv_path)
    if verbose:
        print(f"[INFO] Loaded CSV with {len(df)} rows. Base path: {base_path}")
    
    # Filter to single patient if patient_id provided
    if patient_id is not None:
        # Extract numeric part from patient ID (e.g., "PTB00001" -> 1)
        if patient_id.startswith('PTB'):
            patient_id_int = int(patient_id[3:])  # Remove "PTB" prefix
        elif patient_id.isdigit():
            patient_id_int = int(patient_id)
        else:
            # Try to extract any digits
            import re
            match = re.search(r'\d+', patient_id)
            patient_id_int = int(match.group()) if match else None
        
        if patient_id_int is None:
            raise ValueError(f"Could not extract numeric ID from '{patient_id}'")
        
        if 'ecg_id' in df.columns:
            df = df[df['ecg_id'] == patient_id_int]
            if len(df) == 0:
                raise ValueError(f"Patient ID {patient_id} (ecg_id={patient_id_int}) not found in database")
            if verbose:
                print(f"[INFO] Filtered to patient {patient_id} (ecg_id={patient_id_int}): {len(df)} record(s)")
        else:
            raise ValueError("Database CSV does not contain 'ecg_id' column")

    metadata_rows = []
    n_done = 0

    for idx, record_info in tqdm(df.iterrows(), total=len(df), desc='Records'):
        if max_records is not None and n_done >= max_records:
            break

        # ------------------------
        # choose filename robustly
        # ------------------------
        rec_fname = None
        filename_candidates = ['filename_hr', 'filename_lr', 'filename']
        if prefer_fs in ('hr', 500):
            filename_candidates = ['filename_hr', 'filename_lr', 'filename']
        elif prefer_fs in ('lr', 100):
            filename_candidates = ['filename_lr', 'filename_hr', 'filename']

        for col in filename_candidates:
            if col in record_info.index and not pd.isna(record_info[col]):
                rec_fname = str(record_info[col]).strip()
                break

        if rec_fname is None:
            for col in ['ecg_id', 'record_name', 'record_id']:
                if col in record_info.index and not pd.isna(record_info[col]):
                    rec_fname = str(record_info[col]).strip()
                    break

        if rec_fname is None:
            if verbose:
                print(f"[WARN] No filename for CSV row {idx} — skipping")
            continue

        # ------------------------
        # find actual WFDB record
        # ------------------------
        candidate_bases = [
            os.path.join(base_path, rec_fname),
            rec_fname
        ]
        used_base = None
        for p in candidate_bases:
            if os.path.exists(p + '.hea') and os.path.exists(p + '.dat'):
                used_base = p
                break
        if used_base is None:
            if verbose:
                print(f"[WARN] Could not find record files for {rec_fname}. Tried: {candidate_bases}")
            continue

        # ------------------------
        # read signal
        # ------------------------
        try:
            if verbose:
                print(f"[INFO] Reading record: {used_base}")
            record = wfdb.rdrecord(used_base)
            orig_fs = int(record.fs)
            sig = record.p_signal
            if sig is None:
                raise RuntimeError(f"p_signal is None for {used_base}")
        except Exception as e:
            print(f"[ERROR] Failed to read {rec_fname}: {e}")
            traceback.print_exc()
            continue

        # ------------------------
        # optional resample
        # ------------------------
        working_fs = int(orig_fs)
        if target_fs is not None and target_fs != orig_fs:
            sig_rs = []
            for ch in range(sig.shape[1]):
                ch_rs, _ = resample_signal(sig[:, ch], orig_fs, target_fs)
                sig_rs.append(ch_rs)
            min_len = min([len(x) for x in sig_rs])
            sig = np.vstack([x[:min_len] for x in sig_rs]).T
            working_fs = int(target_fs)

        # ------------------------
        # create output folder
        # ------------------------
        record_out_dir = os.path.join(out_dir, rec_fname)
        Path(record_out_dir).mkdir(parents=True, exist_ok=True)

        # ------------------------
        # process each lead
        # ------------------------
        n_leads = sig.shape[1]
        print(f"\n[PROCESSING] Patient: {rec_fname}")
        print(f"[INFO] Sampling rate: {working_fs} Hz | Samples: {len(sig)} | Leads: {n_leads}")
        for lead_idx in range(n_leads):
            try:
                lead_signal = sig[:, lead_idx]
                print(f"  > Lead {lead_idx}: Applying 5-stage denoising pipeline...")

                # ---- run your denoising pipeline ----
                results = run_pipeline(lead_signal, working_fs,
                                       dc_remove='median',
                                       baseline_params=baseline_params,
                                       notch_params=notch_params,
                                       bandpass_params=bandpass_params,
                                       wavelet_params=wavelet_params)
                cleaned_candidate = results['denoised']
                cleaned, feature_metrics = enforce_feature_preservation(lead_signal, cleaned_candidate)
                results['denoised'] = cleaned

                # save .npy per lead
                out_fname = os.path.join(record_out_dir, f"lead{lead_idx}_fs{working_fs}.npy")
                np.save(out_fname, cleaned.astype(np.float32))
                print(f"    [OK] Saved: {out_fname}")
                print(
                    "    [PRESERVE] "
                    f"corr={feature_metrics['correlation']:.3f}, "
                    f"energy={feature_metrics['energy_retention_pct']:.1f}%, "
                    f"peaks={feature_metrics['peak_retention_pct']:.1f}%, "
                    f"blend={feature_metrics['raw_blend_factor'] * 100:.0f}%"
                )

                # record metadata
                metadata_rows.append({
                    'record_index': int(idx),
                    'record_name': rec_fname,
                    'lead': int(lead_idx),
                    'orig_fs': int(orig_fs),
                    'working_fs': int(working_fs),
                    'n_samples': int(len(cleaned)),
                    'out_path': out_fname,
                    'corr_raw_vs_denoised': float(feature_metrics['correlation']),
                    'energy_retention_pct': float(feature_metrics['energy_retention_pct']),
                    'peak_retention_pct': float(feature_metrics['peak_retention_pct']),
                    'peak_count_raw': int(feature_metrics['peak_count_raw']),
                    'peak_count_denoised': int(feature_metrics['peak_count_denoised']),
                    'features_preserved': bool(feature_metrics['is_preserved']),
                    'enforcement_applied': bool(feature_metrics['enforcement_applied']),
                    'raw_blend_factor': float(feature_metrics['raw_blend_factor'])
                })

            except Exception as e:
                print(f"[ERROR] Failed processing {rec_fname} lead {lead_idx}: {e}")
                traceback.print_exc()
                continue

        n_done += 1

        # save partial metadata every 5 records
        if n_done % 5 == 0 or (max_records is not None and n_done >= max_records):
            meta_df = pd.DataFrame(metadata_rows)
            meta_csv = os.path.join(out_dir, "processed_metadata_partial.csv")
            meta_df.to_csv(meta_csv, index=False)
            if verbose:
                print(f"[INFO] Wrote partial metadata ({len(metadata_rows)}) to {meta_csv}")

    # ------------------------
    # final metadata save
    # ------------------------
    meta_df = pd.DataFrame(metadata_rows)
    meta_csv = os.path.join(out_dir, "processed_metadata.csv")
    meta_df.to_csv(meta_csv, index=False)

    # Save global feature-preservation summary for dashboard/backend use.
    summary = {
        'processed_records': int(n_done),
        'processed_leads': int(len(meta_df)),
        'mean_correlation': float(meta_df['corr_raw_vs_denoised'].mean()) if 'corr_raw_vs_denoised' in meta_df else 0.0,
        'mean_energy_retention_pct': float(meta_df['energy_retention_pct'].mean()) if 'energy_retention_pct' in meta_df else 0.0,
        'mean_peak_retention_pct': float(meta_df['peak_retention_pct'].mean()) if 'peak_retention_pct' in meta_df else 0.0,
        'mean_raw_blend_factor': float(meta_df['raw_blend_factor'].mean()) if 'raw_blend_factor' in meta_df else 0.0,
        'feature_preservation_pass_rate_pct': (
            float(meta_df['features_preserved'].mean() * 100.0) if 'features_preserved' in meta_df and len(meta_df) > 0 else 0.0
        ),
        'enforcement_applied_rate_pct': (
            float(meta_df['enforcement_applied'].mean() * 100.0) if 'enforcement_applied' in meta_df and len(meta_df) > 0 else 0.0
        ),
    }
    summary_json = os.path.join(out_dir, 'feature_preservation_summary.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"[DONE] Processed {n_done} records. Metadata saved to: {meta_csv}")
        print("[DONE] Feature preservation summary:")
        print(
            "       "
            f"corr={summary['mean_correlation']:.3f}, "
            f"energy={summary['mean_energy_retention_pct']:.1f}%, "
            f"peaks={summary['mean_peak_retention_pct']:.1f}%, "
            f"pass={summary['feature_preservation_pass_rate_pct']:.1f}%, "
            f"blend={summary['mean_raw_blend_factor'] * 100:.1f}%"
        )
        print(f"[DONE] Summary JSON: {summary_json}")

    return meta_df

# ------------------------
# Main entry point
# ------------------------
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Denoise PTB-XL ECG signals')
    parser.add_argument('--max-records', type=int, default=200, help='Maximum number of records to process (dataset has 200)')
    parser.add_argument('--patient-id', type=str, default=None, help='Process only a specific patient ID (ecg_id)')
    parser.add_argument('--target-fs', type=int, default=500, help='Target sampling frequency (Hz)')
    parser.add_argument('--prefer-fs', type=str, default=None, help='Preferred sampling frequency (hr/lr)')
    parser.add_argument('--csv-path', type=str, default=None, help='Path to ptbxl_database.csv')
    parser.add_argument('--base-path', type=str, default=None, help='Base path to PTB-XL data')
    parser.add_argument('--out-dir', type=str, default='ptbxl_denoised', help='Output directory')
    
    args = parser.parse_args()
    
    # Default paths
    if args.csv_path is None:
        args.csv_path = r"public/ptbxl_database.csv"
    if args.base_path is None:
        args.base_path = r"archive/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    
    print(f"[INFO] Starting denoising pipeline...")
    if args.patient_id:
        print(f"[INFO] Processing single patient: {args.patient_id}")
    else:
        print(f"[INFO] Max records: {args.max_records}")
    print(f"[INFO] Target FS: {args.target_fs} Hz")
    print(f"[INFO] Output directory: {args.out_dir}")
    
    df_meta = process_ptbxl_all(
        args.csv_path,
        args.base_path,
        args.out_dir,
        prefer_fs=args.prefer_fs,
        target_fs=args.target_fs,
        overwrite=False,
        max_records=args.max_records if not args.patient_id else None,
        patient_id=args.patient_id,
        verbose=True
    )

    print(f"\n{'='*60}")
    print(f"[SUCCESS] Denoising complete!")
    print(f"  * Processed signals: {len(df_meta)}")
    if args.patient_id:
        print(f"  * Patient: {args.patient_id}")
    print(f"  * Output directory: {args.out_dir}")
    print(f"  * Sampling rate: {args.target_fs} Hz")
    print(f"{'='*60}\n")
    print(f"[INFO] Metadata saved with columns: {list(df_meta.columns)}")
