"""
ECG Feature Enhancement Pipeline
Extracts advanced features including:
- Window features (variance, skewness, kurtosis, entropy)
- QRS detection using logistic regression
- R-peak detection and refinement
- Enhanced signal processing
"""
import os
import sys
import traceback
import numpy as np
import pandas as pd
import scipy.signal as sp_signal
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json

# Default hyperparameters
WIN_MS = 120
HOP_MS = 20
ENERGY_SMOOTH_MS = 30
PEAK_HEIGHT_FACTOR = 2.5
LOGREG_C = 1.0
NL_METHOD = 'power'
NL_POWER = 2.5
ALPHA = 2.0
MIN_RR_MS = 150
REFINE_MS = 20

def windowed_features(sig, fs, win_ms=WIN_MS, hop_ms=HOP_MS):
    """Extract statistical features from sliding windows"""
    win_samples = int(round(win_ms * fs / 1000.0))
    hop_samples = int(round(hop_ms * fs / 1000.0))
    if win_samples % 2 == 0:
        win_samples += 1
    
    n = len(sig)
    starts = np.arange(0, n - win_samples + 1, hop_samples)
    centers = starts + win_samples // 2
    feats = []
    
    for s in starts:
        w = sig[s:s+win_samples]
        v = np.var(w)
        sk = float(skew(w))
        kt = float(kurtosis(w, fisher=False))
        hist, _ = np.histogram(w, bins=32, density=True)
        hist += 1e-12
        ent = -np.sum(hist * np.log(hist))
        feats.append([v, sk, kt, ent])
    
    return np.array(feats, dtype=float), centers, win_samples, hop_samples

def weak_labels_from_energy(sig, fs, centers, win_samples, hop_samples,
                            energy_smooth_ms=ENERGY_SMOOTH_MS, 
                            peak_height_factor=PEAK_HEIGHT_FACTOR):
    """Generate weak labels for QRS detection using energy"""
    d = np.diff(sig, prepend=sig[0])
    energy = d**2
    smooth_samples = max(1, int(round(energy_smooth_ms * fs / 1000.0)))
    energy_env = sp_signal.convolve(energy, np.ones(smooth_samples)/smooth_samples, mode='same')
    th = np.median(energy_env) + peak_height_factor*np.std(energy_env)
    distance = max(1, int(round(0.15*fs)))
    peaks, _ = sp_signal.find_peaks(energy_env, height=th, distance=distance)
    
    labels = []
    half = win_samples//2
    for c in centers:
        s = max(0, int(c-half))
        e = min(len(sig), int(c+half+1))
        labels.append(1 if any((p>=s and p<e) for p in peaks) else 0)
    
    return np.array(labels, dtype=int), energy_env, peaks

def train_logistic(X, y, C=LOGREG_C):
    """Train logistic regression for QRS detection"""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', 
                            solver='liblinear', C=C)
    clf.fit(Xs, y)
    return clf, scaler

def window_probs_to_sample_map(probs, centers, n_samples, win_samples):
    """Map window probabilities to sample-level scores"""
    score = np.zeros(n_samples, dtype=float)
    counts = np.zeros(n_samples, dtype=float)
    half = win_samples//2
    
    for p, c in zip(probs, centers):
        s = max(0, int(c-half))
        e = min(n_samples, int(c+half+1))
        score[s:e] += p
        counts[s:e] += 1.0
    
    counts[counts==0] = 1.0
    return score/counts

def non_linear_sharpen(map_in, method=NL_METHOD, power=NL_POWER, k=8.0):
    """Apply non-linear sharpening to enhance QRS peaks"""
    m = np.clip(map_in, 0.0, 1.0)
    if method == 'power':
        return m**power
    elif method == 'sigmoid':
        return 1.0/(1.0+np.exp(-k*(m-0.5)))
    elif method == 'tanh':
        return 0.5*(1+np.tanh(k*(m-0.5)))
    return m

def enhance_signal(sig, qrs_sharp, alpha=ALPHA):
    """Enhance signal using differentiator, Hilbert, and morphology"""
    # Differentiator
    diff = np.abs(np.diff(sig, prepend=sig[0]))
    
    # Hilbert envelope
    try:
        env = np.abs(sp_signal.hilbert(sig))
    except:
        env = np.abs(sig)
    
    # Morphological filter
    morph = sig - sp_signal.medfilt(sig, kernel_size=5)
    
    # Combine
    enhanced = diff + env*(1 + alpha*qrs_sharp) + morph
    return enhanced

def detect_peaks_on_map(qrs_map, fs, min_rr_ms=MIN_RR_MS, percentile=70):
    """Detect R-peaks on QRS probability map"""
    threshold = max(0.05, np.percentile(qrs_map, percentile))
    distance = max(1, int(round(min_rr_ms*fs/1000.0)))
    peaks, props = sp_signal.find_peaks(qrs_map, height=threshold, distance=distance)
    return peaks, props

def refine_peak_positions(peaks, sig, fs, refine_ms=REFINE_MS):
    """Refine R-peak positions to local maxima"""
    half = max(1, int(round(refine_ms*fs/1000.0)))
    n = len(sig)
    refined = []
    
    for p in peaks:
        s = max(0, p-half)
        e = min(n, p+half+1)
        if e <= s:
            refined.append(p)
            continue
        local = np.abs(sig[s:e])
        refined_idx = s + int(np.argmax(local))
        refined.append(refined_idx)
    
    return np.unique(np.array(refined, dtype=int))

def feature_enhancement_pipeline(sig, fs, params=None):
    """Complete feature enhancement pipeline for one ECG signal"""
    params = params or {}

    verbose = bool(params.get("verbose", True))
    
    if verbose:
        print(f"  [1/7] Extracting window features...")
    X, centers, win_samples, hop_samples = windowed_features(
        sig, fs, 
        win_ms=params.get('win_ms', WIN_MS),
        hop_ms=params.get('hop_ms', HOP_MS)
    )
    
    if verbose:
        print(f"  [2/7] Generating weak QRS labels...")
    labels, _, _ = weak_labels_from_energy(
        sig, fs, centers, win_samples, hop_samples,
        energy_smooth_ms=params.get('energy_smooth_ms', ENERGY_SMOOTH_MS),
        peak_height_factor=params.get('peak_height_factor', PEAK_HEIGHT_FACTOR)
    )
    
    if verbose:
        print(f"  [3/7] Training logistic regression...")
    clf, scaler = train_logistic(X, labels, C=params.get('clf_C', LOGREG_C))
    Xs = scaler.transform(X)
    probs = clf.predict_proba(Xs)[:,1]
    
    if verbose:
        print(f"  [4/7] Mapping to sample-level QRS scores...")
    qrs_score = window_probs_to_sample_map(probs, centers, len(sig), win_samples)
    if np.max(qrs_score) > 0:
        qrs_score = qrs_score/np.max(qrs_score)
    
    if verbose:
        print(f"  [5/7] Applying non-linear sharpening...")
    qrs_sharp = non_linear_sharpen(
        qrs_score, 
        method=params.get('nl_method', NL_METHOD),
        power=params.get('nl_power', NL_POWER)
    )
    
    if verbose:
        print(f"  [6/7] Enhancing signal...")
    enhanced = enhance_signal(sig, qrs_sharp, alpha=params.get('alpha', ALPHA))
    
    if verbose:
        print(f"  [7/7] Detecting and refining R-peaks...")
    peaks_on_map, _ = detect_peaks_on_map(qrs_sharp, fs)
    peaks_refined = refine_peak_positions(peaks_on_map, sig, fs)
    
    return {
        'enhanced': enhanced,
        'qrs_score': qrs_score,
        'qrs_sharp': qrs_sharp,
        'peaks_refined': peaks_refined,
        'num_peaks': len(peaks_refined),
        'heart_rate': int(60 * fs * len(peaks_refined) / len(sig)) if len(peaks_refined) > 0 else 0,
        'fs': fs
    }

def process_patient_features(patient_id, denoised_dir='ptbxl_denoised', 
                             output_dir='ptbxl_features', params=None):
    """Process all leads for a single patient"""
    print(f"\n[PROCESSING] Patient: {patient_id}")
    
    # Find patient's denoised files
    patient_files = []
    for root, dirs, files in os.walk(denoised_dir):
        for fn in files:
            if fn.endswith('.npy'):
                full_path = os.path.join(root, fn)
                patient_files.append(full_path)
    
    if not patient_files:
        raise ValueError(f"No denoised files found for patient {patient_id}")
    
    # Process each lead
    results = {}
    for idx, npy_file in enumerate(patient_files):
        print(f"\n> Lead {idx}: {npy_file}")
        sig = np.load(npy_file)
        fs = params.get('fs', 500)
        
        result = feature_enhancement_pipeline(sig, fs, params)
        
        # Save outputs
        lead_dir = os.path.join(output_dir, patient_id, f"lead{idx}")
        Path(lead_dir).mkdir(parents=True, exist_ok=True)
        
        # Save enhanced signal
        np.save(os.path.join(lead_dir, "enhanced.npy"), result['enhanced'])
        # Save QRS map
        np.save(os.path.join(lead_dir, "qrs_map.npy"), result['qrs_score'])
        # Save sharpened QRS
        np.save(os.path.join(lead_dir, "qrs_sharp.npy"), result['qrs_sharp'])
        # Save R-peaks
        np.save(os.path.join(lead_dir, "r_peaks.npy"), result['peaks_refined'])
        
        results[f'lead{idx}'] = {
            'num_peaks': result['num_peaks'],
            'heart_rate': result['heart_rate'],
            'output_dir': lead_dir
        }
        
        print(f"    [OK] R-peaks: {result['num_peaks']}, HR: {result['heart_rate']} bpm")
    
    # Save summary
    summary_file = os.path.join(output_dir, patient_id, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SUCCESS] Processed {len(results)} leads for {patient_id}")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ECG Feature Enhancement')
    parser.add_argument('--patient-id', type=str, required=True, help='Patient ID (e.g., PTB00001)')
    parser.add_argument('--denoised-dir', type=str, default='ptbxl_denoised', help='Denoised signals directory')
    parser.add_argument('--output-dir', type=str, default='ptbxl_features', help='Output directory')
    parser.add_argument('--fs', type=int, default=500, help='Sampling frequency')
    
    args = parser.parse_args()
    
    params = {
        'win_ms': WIN_MS,
        'hop_ms': HOP_MS,
        'energy_smooth_ms': ENERGY_SMOOTH_MS,
        'peak_height_factor': PEAK_HEIGHT_FACTOR,
        'clf_C': LOGREG_C,
        'nl_method': NL_METHOD,
        'nl_power': NL_POWER,
        'alpha': ALPHA,
        'fs': args.fs
    }
    
    try:
        results = process_patient_features(
            args.patient_id,
            denoised_dir=args.denoised_dir,
            output_dir=args.output_dir,
            params=params
        )
        
        print(f"\n{'='*60}")
        print(f"[COMPLETE] Feature extraction finished")
        print(f"  * Patient: {args.patient_id}")
        print(f"  * Leads processed: {len(results)}")
        print(f"  * Output: {args.output_dir}/{args.patient_id}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n[ERROR] Feature extraction failed: {e}")
        traceback.print_exc()
        sys.exit(1)
