"""
PARALLEL Comprehensive ECG Feature Extraction for PTB-XL Dataset
========================================================
This version uses multiprocessing to utilize all CPU cores for faster extraction.
Expected speedup: 4-8x faster depending on number of cores.
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
import scipy.signal as signal
from scipy import stats
from scipy.stats import skew, kurtosis
import pywt
from pathlib import Path
import json
from tqdm import tqdm
import traceback
from typing import Dict, List, Tuple, Optional, Any
import warnings
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================
PTBXL_ROOT = r"archive/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
PTBXL_DB_CSV = os.path.join(PTBXL_ROOT, "ptbxl_database.csv")
OUTPUT_DIR = "ptbxl_comprehensive_features"
SAMPLING_FREQ = 500  # Hz - use 500Hz records for full detail

# Parallel Processing Options
NUM_WORKERS = mp.cpu_count() - 1  # Leave 1 core free for system
MAX_RECORDS = None  # Set to number for testing, None for all
LEADS_TO_PROCESS = list(range(12))  # All 12 leads
BATCH_SIZE = 100


# =============================================================================
# Import all feature extraction functions from original script
# =============================================================================

def extract_statistical_features(signal_segment: np.ndarray) -> Dict[str, float]:
    """Extract statistical features from signal segment."""
    features = {}
    
    if len(signal_segment) == 0:
        return {k: np.nan for k in ['mean', 'variance', 'std', 'median', 'min', 'max',
                                     'range', 'skewness', 'kurtosis', 'iqr', 'rms',
                                     'cv', 'mad', 'peak_to_peak', 'energy', 'signal_power']}
    
    features['mean'] = float(np.mean(signal_segment))
    features['variance'] = float(np.var(signal_segment))
    features['std'] = float(np.std(signal_segment))
    features['median'] = float(np.median(signal_segment))
    features['min'] = float(np.min(signal_segment))
    features['max'] = float(np.max(signal_segment))
    features['range'] = features['max'] - features['min']
    features['skewness'] = float(skew(signal_segment))
    features['kurtosis'] = float(kurtosis(signal_segment, fisher=False))
    
    q75, q25 = np.percentile(signal_segment, [75, 25])
    features['iqr'] = float(q75 - q25)
    features['rms'] = float(np.sqrt(np.mean(signal_segment**2)))
    
    if features['mean'] != 0:
        features['cv'] = float(features['std'] / abs(features['mean']))
    else:
        features['cv'] = np.nan
    
    features['mad'] = float(np.mean(np.abs(signal_segment - features['mean'])))
    features['peak_to_peak'] = features['max'] - features['min']
    features['energy'] = float(np.sum(signal_segment**2))
    features['signal_power'] = float(np.mean(signal_segment**2))
    
    return features


def detect_r_peaks(ecg_signal: np.ndarray, fs: int) -> np.ndarray:
    """Detect R-peaks using Pan-Tompkins algorithm simplified"""
    try:
        nyq = 0.5 * fs
        low = 5 / nyq
        high = 15 / nyq
        b, a = signal.butter(2, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        diff = np.diff(filtered)
        squared = diff ** 2
        window_size = int(0.12 * fs)
        ma = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        threshold = np.mean(ma) + 0.5 * np.std(ma)
        min_distance = int(0.2 * fs)
        peaks, _ = signal.find_peaks(ma, height=threshold, distance=min_distance)
        
        return peaks
    except:
        return np.array([])


def extract_temporal_features(ecg_signal: np.ndarray, fs: int, r_peaks: np.ndarray = None) -> Dict[str, float]:
    """Extract temporal features including intervals and HRV metrics."""
    features = {}
    
    if r_peaks is None or len(r_peaks) == 0:
        r_peaks = detect_r_peaks(ecg_signal, fs)
    
    if len(r_peaks) < 2:
        return {k: np.nan for k in ['rr_mean', 'rr_std', 'rr_median', 'hr_mean', 
                                     'sdnn', 'rmssd', 'pnn50', 'num_peaks', 
                                     'zero_crossing_rate', 'beat_to_beat_var']}
    
    rr_intervals = np.diff(r_peaks) / fs * 1000.0
    
    features['rr_mean'] = float(np.mean(rr_intervals))
    features['rr_std'] = float(np.std(rr_intervals))
    features['rr_median'] = float(np.median(rr_intervals))
    
    if features['rr_mean'] > 0:
        features['hr_mean'] = float(60000.0 / features['rr_mean'])
    else:
        features['hr_mean'] = np.nan
    
    features['sdnn'] = float(np.std(rr_intervals))
    
    rr_diff = np.diff(rr_intervals)
    features['rmssd'] = float(np.sqrt(np.mean(rr_diff**2))) if len(rr_diff) > 0 else np.nan
    
    if len(rr_diff) > 0:
        features['pnn50'] = float(100.0 * np.sum(np.abs(rr_diff) > 50) / len(rr_diff))
    else:
        features['pnn50'] = 0.0
    
    features['num_peaks'] = int(len(r_peaks))
    
    zero_crossings = np.where(np.diff(np.sign(ecg_signal)))[0]
    features['zero_crossing_rate'] = float(len(zero_crossings) / len(ecg_signal))
    features['beat_to_beat_var'] = float(np.var(rr_intervals))
    
    return features


def find_p_qrs_t_waves(ecg_beat: np.ndarray, r_peak_idx: int, fs: int) -> Dict[str, int]:
    """Detect P, QRS, T wave boundaries in a single beat."""
    indices = {}
    n = len(ecg_beat)
    
    q_search_start = max(0, r_peak_idx - int(0.08 * fs))
    q_search_end = r_peak_idx
    if q_search_end > q_search_start:
        q_segment = ecg_beat[q_search_start:q_search_end]
        q_idx = q_search_start + np.argmin(q_segment)
    else:
        q_idx = r_peak_idx
    
    s_search_start = r_peak_idx
    s_search_end = min(n, r_peak_idx + int(0.08 * fs))
    if s_search_end > s_search_start:
        s_segment = ecg_beat[s_search_start:s_search_end]
        s_idx = s_search_start + np.argmin(s_segment)
    else:
        s_idx = r_peak_idx
    
    p_search_start = max(0, q_idx - int(0.2 * fs))
    p_search_end = max(0, q_idx - int(0.04 * fs))
    if p_search_end > p_search_start:
        p_segment = ecg_beat[p_search_start:p_search_end]
        p_idx = p_search_start + np.argmax(np.abs(p_segment))
    else:
        p_idx = q_idx
    
    t_search_start = min(n-1, s_idx + int(0.05 * fs))
    t_search_end = min(n, s_idx + int(0.3 * fs))
    if t_search_end > t_search_start:
        t_segment = ecg_beat[t_search_start:t_search_end]
        t_idx = t_search_start + np.argmax(np.abs(t_segment))
    else:
        t_idx = min(n-1, s_idx + int(0.1 * fs))
    
    indices['p_idx'] = p_idx
    indices['q_idx'] = q_idx
    indices['r_idx'] = r_peak_idx
    indices['s_idx'] = s_idx
    indices['t_idx'] = t_idx
    
    return indices


def extract_morphological_features(ecg_signal: np.ndarray, fs: int, r_peaks: np.ndarray = None) -> Dict[str, float]:
    """Extract morphological features from ECG signal."""
    features = {}
    
    if r_peaks is None or len(r_peaks) == 0:
        r_peaks = detect_r_peaks(ecg_signal, fs)
    
    if len(r_peaks) < 2:
        return {k: np.nan for k in ['p_amp_mean', 'qrs_amp_mean', 't_amp_mean',
                                     'p_duration_mean', 'qrs_duration_mean', 't_duration_mean',
                                     'pr_interval_mean', 'qt_interval_mean', 'st_segment_mean',
                                     'area_p_mean', 'area_qrs_mean', 'area_t_mean',
                                     'qrs_slope_mean', 't_slope_mean', 'r_peak_amp_mean',
                                     'q_amp_mean', 's_amp_mean', 'p_qrs_ratio']}
    
    p_amps, qrs_amps, t_amps = [], [], []
    p_durs, qrs_durs, t_durs = [], [], []
    pr_ints, qt_ints, st_segs = [], [], []
    areas_p, areas_qrs, areas_t = [], [], []
    qrs_slopes, t_slopes = [], []
    r_amps, q_amps, s_amps = [], [], []
    
    for i in range(1, len(r_peaks) - 1):
        try:
            beat_start = r_peaks[i-1]
            beat_end = r_peaks[i+1]
            beat = ecg_signal[beat_start:beat_end]
            r_in_beat = r_peaks[i] - beat_start
            
            wave_idx = find_p_qrs_t_waves(beat, r_in_beat, fs)
            
            p_amps.append(abs(beat[wave_idx['p_idx']]))
            r_amps.append(abs(beat[wave_idx['r_idx']]))
            q_amps.append(abs(beat[wave_idx['q_idx']]))
            s_amps.append(abs(beat[wave_idx['s_idx']]))
            t_amps.append(abs(beat[wave_idx['t_idx']]))
            
            qrs_amps.append(beat[wave_idx['r_idx']] - min(beat[wave_idx['q_idx']], beat[wave_idx['s_idx']]))
            
            qrs_dur = (wave_idx['s_idx'] - wave_idx['q_idx']) / fs * 1000
            qrs_durs.append(qrs_dur)
            
            p_dur = int(0.08 * fs) / fs * 1000
            p_durs.append(p_dur)
            
            t_dur = int(0.12 * fs) / fs * 1000
            t_durs.append(t_dur)
            
            pr_int = (wave_idx['r_idx'] - wave_idx['p_idx']) / fs * 1000
            pr_ints.append(pr_int)
            
            qt_int = (wave_idx['t_idx'] - wave_idx['q_idx']) / fs * 1000
            qt_ints.append(qt_int)
            
            st_seg = (wave_idx['t_idx'] - wave_idx['s_idx']) / fs * 1000
            st_segs.append(st_seg)
            
            p_start = max(0, wave_idx['p_idx'] - int(0.04*fs))
            p_end = min(len(beat), wave_idx['p_idx'] + int(0.04*fs))
            areas_p.append(np.sum(np.abs(beat[p_start:p_end])))
                        
            areas_qrs.append(np.sum(np.abs(beat[wave_idx['q_idx']:wave_idx['s_idx']])))
            
            t_start = max(0, wave_idx['t_idx'] - int(0.06*fs))
            t_end = min(len(beat), wave_idx['t_idx'] + int(0.06*fs))
            areas_t.append(np.sum(np.abs(beat[t_start:t_end])))
            
            if wave_idx['r_idx'] != wave_idx['q_idx']:
                qrs_slope = (beat[wave_idx['r_idx']] - beat[wave_idx['q_idx']]) / ((wave_idx['r_idx'] - wave_idx['q_idx']) / fs)
                qrs_slopes.append(abs(qrs_slope))
            
            if wave_idx['t_idx'] != wave_idx['s_idx']:
                t_slope = (beat[wave_idx['t_idx']] - beat[wave_idx['s_idx']]) / ((wave_idx['t_idx'] - wave_idx['s_idx']) / fs)
                t_slopes.append(abs(t_slope))
                
        except Exception:
            continue
    
    features['p_amp_mean'] = float(np.mean(p_amps)) if len(p_amps) > 0 else np.nan
    features['qrs_amp_mean'] = float(np.mean(qrs_amps)) if len(qrs_amps) > 0 else np.nan
    features['t_amp_mean'] = float(np.mean(t_amps)) if len(t_amps) > 0 else np.nan
    
    features['p_duration_mean'] = float(np.mean(p_durs)) if len(p_durs) > 0 else np.nan
    features['qrs_duration_mean'] = float(np.mean(qrs_durs)) if len(qrs_durs) > 0 else np.nan
    features['t_duration_mean'] = float(np.mean(t_durs)) if len(t_durs) > 0 else np.nan
    
    features['pr_interval_mean'] = float(np.mean(pr_ints)) if len(pr_ints) > 0 else np.nan
    features['qt_interval_mean'] = float(np.mean(qt_ints)) if len(qt_ints) > 0 else np.nan
    features['st_segment_mean'] = float(np.mean(st_segs)) if len(st_segs) > 0 else np.nan
    
    features['area_p_mean'] = float(np.mean(areas_p)) if len(areas_p) > 0 else np.nan
    features['area_qrs_mean'] = float(np.mean(areas_qrs)) if len(areas_qrs) > 0 else np.nan
    features['area_t_mean'] = float(np.mean(areas_t)) if len(areas_t) > 0 else np.nan
    
    features['qrs_slope_mean'] = float(np.mean(qrs_slopes)) if len(qrs_slopes) > 0 else np.nan
    features['t_slope_mean'] = float(np.mean(t_slopes)) if len(t_slopes) > 0 else np.nan
    
    features['r_peak_amp_mean'] = float(np.mean(r_amps)) if len(r_amps) > 0 else np.nan
    features['q_amp_mean'] = float(np.mean(q_amps)) if len(q_amps) > 0 else np.nan
    features['s_amp_mean'] = float(np.mean(s_amps)) if len(s_amps) > 0 else np.nan
    
    if features['qrs_amp_mean'] != 0 and not np.isnan(features['p_amp_mean']):
        features['p_qrs_ratio'] = float(features['p_amp_mean'] / features['qrs_amp_mean'])
    else:
        features['p_qrs_ratio'] = np.nan
    
    return features


def extract_spectral_features(ecg_signal: np.ndarray, fs: int) -> Dict[str, float]:
    """Extract frequency-domain features using FFT and Welch's method."""
    features = {}
    
    try:
        freqs, psd = signal.welch(ecg_signal, fs=fs, nperseg=min(256, len(ecg_signal)))
        
        dominant_idx = np.argmax(psd)
        features['dominant_frequency'] = float(freqs[dominant_idx])
        features['spectral_energy'] = float(np.sum(psd))
        
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]
        features['spectral_entropy'] = float(-np.sum(psd_norm * np.log(psd_norm)))
        
        features['spectral_centroid'] = float(np.sum(freqs * psd) / np.sum(psd))
        
        centroid = features['spectral_centroid']
        features['spectral_bandwidth'] = float(np.sqrt(np.sum(((freqs - centroid)**2) * psd) / np.sum(psd)))
        
        cumsum_psd = np.cumsum(psd)
        rolloff_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = float(freqs[rolloff_idx[0]])
        else:
            features['spectral_rolloff'] = np.nan
        
        window_size = min(512, len(ecg_signal) // 4)
        hop_size = window_size // 2
        flux_values = []
        
        for i in range(0, len(ecg_signal) - window_size, hop_size):
            window1 = ecg_signal[i:i+window_size]
            window2 = ecg_signal[i+hop_size:i+hop_size+window_size]
            
            if len(window2) == window_size:
                fft1 = np.abs(np.fft.rfft(window1))
                fft2 = np.abs(np.fft.rfft(window2))
                flux = np.sum((fft2 - fft1)**2)
                flux_values.append(flux)
        
        features['spectral_flux'] = float(np.mean(flux_values)) if len(flux_values) > 0 else np.nan
        features['spectral_kurtosis'] = float(kurtosis(psd, fisher=False))
        features['spectral_skewness'] = float(skew(psd))
        
    except Exception:
        return {k: np.nan for k in ['dominant_frequency', 'spectral_energy', 'spectral_entropy',
                                     'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                                     'spectral_flux', 'spectral_kurtosis', 'spectral_skewness']}
    
    return features


def extract_time_frequency_features(ecg_signal: np.ndarray, fs: int) -> Dict[str, float]:
    """Extract time-frequency features using STFT and Wavelet transforms."""
    features = {}
    
    try:
        f, t, Zxx = signal.stft(ecg_signal, fs=fs, nperseg=min(256, len(ecg_signal)//4))
        Zxx_mag = np.abs(Zxx)
        
        features['tf_energy'] = float(np.sum(Zxx_mag**2))
        
        time_grid, freq_grid = np.meshgrid(t, f)
        total_energy = np.sum(Zxx_mag**2)
        if total_energy > 0:
            features['tf_centroid_time'] = float(np.sum(time_grid * Zxx_mag**2) / total_energy)
            features['tf_centroid_freq'] = float(np.sum(freq_grid * Zxx_mag**2) / total_energy)
        else:
            features['tf_centroid_time'] = np.nan
            features['tf_centroid_freq'] = np.nan
        
        analytic_signal = signal.hilbert(ecg_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
        features['inst_freq_mean'] = float(np.mean(instantaneous_frequency))
        features['inst_freq_std'] = float(np.std(instantaneous_frequency))
        
        scales = np.arange(1, min(128, len(ecg_signal)//4))
        try:
            coefficients, frequencies = pywt.cwt(ecg_signal, scales, 'cmor1.5-1.0', sampling_period=1/fs)
            
            features['wavelet_energy'] = float(np.sum(np.abs(coefficients)**2))
            
            wavelet_power = np.abs(coefficients)**2
            wavelet_power_norm = wavelet_power / np.sum(wavelet_power)
            wavelet_power_norm = wavelet_power_norm[wavelet_power_norm > 0]
            features['wavelet_entropy'] = float(-np.sum(wavelet_power_norm * np.log(wavelet_power_norm)))
            
        except Exception:
            features['wavelet_energy'] = np.nan
            features['wavelet_entropy'] = np.nan
        
        features['spectrogram_mean'] = float(np.mean(Zxx_mag))
        features['spectrogram_std'] = float(np.std(Zxx_mag))
        features['spectrogram_entropy'] = float(-np.sum((Zxx_mag / np.sum(Zxx_mag)) * np.log(Zxx_mag / np.sum(Zxx_mag) + 1e-10)))
        
    except Exception:
        return {k: np.nan for k in ['tf_energy', 'tf_centroid_time', 'tf_centroid_freq',
                                     'inst_freq_mean', 'inst_freq_std', 'wavelet_energy',
                                     'wavelet_entropy', 'spectrogram_mean', 'spectrogram_std',
                                     'spectrogram_entropy']}
    
    return features


def approximate_entropy(signal_data: np.ndarray, m: int = 2, r: float = None) -> float:
    """Calculate Approximate Entropy (ApEn)"""
    try:
        N = len(signal_data)
        if r is None:
            r = 0.2 * np.std(signal_data)
        
        def _phi(m):
            patterns = np.array([signal_data[i:i+m] for i in range(N-m+1)])
            C = np.zeros(N-m+1)
            for i in range(N-m+1):
                distances = np.max(np.abs(patterns - patterns[i]), axis=1)
                C[i] = np.sum(distances <= r) / (N - m + 1)
            return np.sum(np.log(C + 1e-10)) / (N - m + 1)
        
        return float(_phi(m) - _phi(m+1))
    except:
        return np.nan


def sample_entropy(signal_data: np.ndarray, m: int = 2, r: float = None) -> float:
    """Calculate Sample Entropy (SampEn)"""
    try:
        N = len(signal_data)
        if r is None:
            r = 0.2 * np.std(signal_data)
        
        def _get_counts(m):
            patterns = np.array([signal_data[i:i+m] for i in range(N-m)])
            A = 0
            for i in range(len(patterns)-1):
                distances = np.max(np.abs(patterns[i+1:] - patterns[i]), axis=1)
                A += np.sum(distances <= r)
            return A
        
        A = _get_counts(m)
        B = _get_counts(m+1)
        
        if A == 0 or B == 0:
            return np.nan
        
        return float(-np.log(B / A))
    except:
        return np.nan


def detrended_fluctuation_analysis(signal_data: np.ndarray) -> float:
    """Calculate DFA (Detrended Fluctuation Analysis) alpha exponent"""
    try:
        N = len(signal_data)
        y = np.cumsum(signal_data - np.mean(signal_data))
        
        scales = np.unique(np.logspace(0.5, min(3, np.log10(N/4)), 20).astype(int))
        fluctuations = []
        
        for scale in scales:
            segments = N // scale
            F = 0
            for v in range(segments):
                segment = y[v*scale:(v+1)*scale]
                fit = np.polyfit(np.arange(scale), segment, 1)
                trend = np.polyval(fit, np.arange(scale))
                F += np.sum((segment - trend)**2)
            F /= segments
            fluctuations.append(np.sqrt(F / scale))
        
        if len(fluctuations) > 1 and len(scales) > 1:
            fit = np.polyfit(np.log(scales), np.log(fluctuations), 1)
            return float(fit[0])
        else:
            return np.nan
    except:
        return np.nan


def poincare_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """Calculate Poincaré plot indices SD1 and SD2"""
    try:
        if len(rr_intervals) < 2:
            return {'sd1': np.nan, 'sd2': np.nan}
        
        rr_x = rr_intervals[:-1]
        rr_y = rr_intervals[1:]
        
        diff = rr_y - rr_x
        sd1 = np.std(diff) / np.sqrt(2)
        
        sum_rr = rr_y + rr_x
        sd2 = np.sqrt(2 * np.std(rr_intervals)**2 - sd1**2)
        
        return {'sd1': float(sd1), 'sd2': float(sd2)}
    except:
        return {'sd1': np.nan, 'sd2': np.nan}


def extract_nonlinear_features(ecg_signal: np.ndarray, fs: int, r_peaks: np.ndarray = None) -> Dict[str, float]:
    """Extract nonlinear and complexity features."""
    features = {}
    
    features['approximate_entropy'] = approximate_entropy(ecg_signal)
    features['sample_entropy'] = sample_entropy(ecg_signal)
    features['dfa_alpha'] = detrended_fluctuation_analysis(ecg_signal)
    
    try:
        k_max = 10
        L = []
        x = ecg_signal
        N = len(x)
        for k in range(1, k_max):
            Lk = 0
            for m in range(k):
                Lmk = 0
                maxI = int(np.floor((N-m)/k))
                for i in range(1, maxI):
                    Lmk += abs(x[m+i*k] - x[m+(i-1)*k])
                Lmk = Lmk * (N - 1) / (maxI * k * k)
                Lk += Lmk
            L.append(Lk / k)
        
        if len(L) > 1:
            fit = np.polyfit(np.log(range(1, k_max)), np.log(L), 1)
            features['fractal_dimension'] = float(-fit[0])
        else:
            features['fractal_dimension'] = np.nan
    except:
        features['fractal_dimension'] = np.nan
    
    if r_peaks is None or len(r_peaks) == 0:
        r_peaks = detect_r_peaks(ecg_signal, fs)
    
    if len(r_peaks) >= 2:
        rr_intervals = np.diff(r_peaks) / fs * 1000.0
        poincare = poincare_features(rr_intervals)
        features.update(poincare)
    else:
        features['sd1'] = np.nan
        features['sd2'] = np.nan
    
    return features


def extract_all_features_from_signal(ecg_signal: np.ndarray, fs: int, lead_name: str = "") -> Dict[str, Any]:
    """Extract ALL features from a single ECG lead signal (silent mode for parallel)."""
    all_features = {'lead_name': lead_name, 'sampling_freq': fs}
    
    r_peaks = detect_r_peaks(ecg_signal, fs)
    
    all_features.update({f"stat_{k}": v for k, v in extract_statistical_features(ecg_signal).items()})
    all_features.update({f"temp_{k}": v for k, v in extract_temporal_features(ecg_signal, fs, r_peaks).items()})
    all_features.update({f"morph_{k}": v for k, v in extract_morphological_features(ecg_signal, fs, r_peaks).items()})
    all_features.update({f"spec_{k}": v for k, v in extract_spectral_features(ecg_signal, fs).items()})
    all_features.update({f"tf_{k}": v for k, v in extract_time_frequency_features(ecg_signal, fs).items()})
    all_features.update({f"nonlin_{k}": v for k, v in extract_nonlinear_features(ecg_signal, fs, r_peaks).items()})
    
    return all_features


def process_record(record_info: Tuple[str, int, int]) -> Dict[str, Any]:
    """
    Process a single PTB-XL record and extract features from all leads.
    This function is called by parallel workers.
    """
    record_path, record_id, fs = record_info
    
    try:
        record = wfdb.rdrecord(record_path)
        lead_names = record.sig_name
        features_by_lead = {}
        
        for lead_idx in LEADS_TO_PROCESS:
            if lead_idx >= record.n_sig:
                continue
            
            lead_name = lead_names[lead_idx]
            ecg_signal = record.p_signal[:, lead_idx]
            lead_features = extract_all_features_from_signal(ecg_signal, fs, lead_name)
            features_by_lead[f"lead_{lead_idx}_{lead_name}"] = lead_features
        
        return {
            'record_id': record_id,
            'record_path': record_path,
            'success': True,
            'features': features_by_lead
        }
    
    except Exception as e:
        return {
            'record_id': record_id,
            'record_path': record_path,
            'success': False,
            'error': str(e)
        }


def main():
    """Main processing function with parallel execution"""
    print("=" * 80)
    print("PTB-XL COMPREHENSIVE FEATURE EXTRACTION (PARALLEL)")
    print("=" * 80)
    print(f"Using {NUM_WORKERS} CPU cores for parallel processing")
    print()
    
    print(f"Loading PTB-XL database from: {PTBXL_DB_CSV}")
    df = pd.read_csv(PTBXL_DB_CSV)
    print(f"Total records in database: {len(df)}")
    
    if SAMPLING_FREQ == 500:
        df = df[df['filename_hr'].notna()].copy()
        df['filename'] = df['filename_hr']
    else:
        df['filename'] = df['filename_lr']
    
    if MAX_RECORDS is not None:
        df = df.head(MAX_RECORDS)
        print(f"Processing first {MAX_RECORDS} records for testing")
    
    print(f"Processing {len(df)} records at {SAMPLING_FREQ} Hz")
    print()
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_results = []
    
    for batch_start in range(0, len(df), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        print(f"\n{'='*80}")
        print(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(len(df)-1)//BATCH_SIZE + 1}")
        print(f"Records {batch_start} to {batch_end-1}")
        print(f"{'='*80}\n")
        
        # Prepare arguments for parallel processing
        record_info_list = []
        for idx, row in batch_df.iterrows():
            record_id = row['ecg_id']
            filename = row['filename']
            record_path = os.path.join(PTBXL_ROOT, filename)
            record_info_list.append((record_path, record_id, SAMPLING_FREQ))
        
        # Parallel processing using multiprocessing Pool
        with mp.Pool(processes=NUM_WORKERS) as pool:
            batch_results = list(tqdm(
                pool.imap(process_record, record_info_list),
                total=len(record_info_list),
                desc="Batch progress"
            ))
        
        all_results.extend(batch_results)
        
        # Save batch results
        batch_file = output_dir / f"batch_{batch_start//BATCH_SIZE + 1:04d}_features.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        print(f"\nBatch saved to: {batch_file}")
    
    # Save all results
    all_results_file = output_dir / "all_features.json"
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to: {all_results_file}")
    
    # Create summary CSV
    print("\nCreating summary CSV...")
    summary_data = []
    
    for result in all_results:
        if result['success']:
            record_row = {'record_id': result['record_id'], 'record_path': result['record_path']}
            
            for lead_key, lead_features in result['features'].items():
                for feature_name, feature_value in lead_features.items():
                    if feature_name not in ['lead_name', 'sampling_freq']:
                        column_name = f"{lead_key}_{feature_name}"
                        record_row[column_name] = feature_value
            
            summary_data.append(record_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / "comprehensive_features_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")
    
    # Print statistics
    success_count = sum(1 for r in all_results if r['success'])
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total records processed: {len(all_results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(all_results) - success_count}")
    print(f"Features per lead: {len(summary_df.columns) // 12 if len(summary_df.columns) > 0 else 0}")
    print(f"Total feature columns: {len(summary_df.columns)}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
