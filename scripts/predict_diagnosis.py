"""
Patient diagnosis prediction using trained MLP v3.0
Fast path: reads precomputed patient_predictions.json
Fallback: extracts signal features on-demand for unknown patients
"""

import ast
import json
import sys
import pickle
import numpy as np
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis

import warnings
warnings.filterwarnings('ignore')


def load_precomputed(patient_id):
    """Check precomputed predictions first (fast path)."""
    pred_path = Path(__file__).parent.parent / 'public' / 'patient_predictions.json'
    if not pred_path.exists():
        return None
    try:
        with open(pred_path) as f:
            data = json.load(f)
        if patient_id in data:
            return data[patient_id]
    except:
        pass
    return None


def extract_lead_features(lead_signal, fs=500):
    """Extract 20 statistical + freq-domain features per lead (matches training)."""
    sig = lead_signal.astype(float)
    n = len(sig)
    if n == 0:
        return [0.0] * 20

    mean_val = float(np.mean(sig))
    std_val = float(np.std(sig))
    rms_val = float(np.sqrt(np.mean(sig**2)))
    max_val = float(np.max(sig))
    min_val = float(np.min(sig))
    p2p = float(max_val - min_val)
    skew_val = float(skew(sig))
    kurt_val = float(kurtosis(sig))
    diff1 = np.diff(sig)
    energy = float(np.sum(diff1**2))
    max_slope = float(np.max(np.abs(diff1))) if len(diff1) > 0 else 0.0
    zcr = float(np.sum(np.diff(np.sign(sig)) != 0)) / n

    freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(256, n))
    total_power = float(np.sum(psd)) + 1e-10
    lf_power = float(np.sum(psd[(freqs >= 0.5) & (freqs < 15)])) / total_power
    hf_power = float(np.sum(psd[(freqs >= 15) & (freqs < 50)])) / total_power
    vlf_power = float(np.sum(psd[freqs < 0.5])) / total_power
    lf_hf_ratio = float(lf_power / (hf_power + 1e-10))

    threshold = mean_val + 0.5 * std_val
    peaks_above = np.where((sig[1:-1] > threshold) & (sig[1:-1] > sig[:-2]) & (sig[1:-1] > sig[2:]))[0]
    n_peaks = len(peaks_above)
    hr_est = float(n_peaks * 60 * fs / n) if n > 0 else 0.0
    st_mean = float(np.mean(sig[int(0.4 * n):])) if n > 0 else 0.0

    return [
        mean_val, std_val, rms_val, max_val, min_val, p2p, skew_val, kurt_val,
        energy, max_slope, zcr, lf_power, hf_power, vlf_power, lf_hf_ratio,
        float(n_peaks), hr_est, st_mean, float(n_peaks / (n / fs + 1e-6)), total_power
    ]


def extract_features_live(ecg_id):
    """Live feature extraction from raw .dat signal files."""
    project_root = Path(__file__).parent.parent
    data_root = project_root / 'archive' / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    folder = f"{(ecg_id - 1) // 1000 * 1000:05d}"
    fname = data_root / 'records500' / folder / f"{ecg_id:05d}_hr"
    try:
        import wfdb
        sig, meta = wfdb.rdsamp(str(fname), channels=list(range(12)))
        fs = meta['fs']
        feature_vec = []
        for lead_idx in range(min(12, sig.shape[1])):
            feature_vec.extend(extract_lead_features(sig[:, lead_idx], fs))
        return np.array(feature_vec).reshape(1, -1)
    except Exception:
        return None


def get_patient_info(patient_id):
    """Get patient metadata and SCP codes from full CSV."""
    import pandas as pd
    csv_path = Path(__file__).parent.parent / 'public' / 'ptbxl_database.csv'
    try:
        df = pd.read_csv(csv_path)
        ecg_id = int(patient_id[3:]) if patient_id.upper().startswith('PTB') else None
        if ecg_id:
            row = df[df['ecg_id'] == ecg_id]
            if not row.empty:
                r = row.iloc[0]
                raw_codes = ast.literal_eval(r['scp_codes']) if isinstance(r['scp_codes'], str) else r['scp_codes']
                return {
                    'ecg_id': int(r['ecg_id']),
                    'age': float(r['age']) if str(r['age']) not in ['nan', ''] else 0,
                    'sex': 'Male' if str(r['sex']) == '0' else 'Female',
                    'actual_scp_codes': raw_codes,
                    'report': str(r.get('report', '')),
                }
    except Exception:
        pass
    return {'ecg_id': None, 'age': 0, 'sex': 'Unknown', 'actual_scp_codes': {}, 'report': ''}


def load_model():
    """Load trained MLP model"""
    model_path = Path(__file__).parent.parent / 'public' / 'mlp_model.pkl'
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['scaler'], model_data['label_encoder']


def predict(patient_id):
    """Make prediction for a patient."""
    CLASS_FULL = {
        'NORM': 'Normal/Sinus Rhythm',
        'MI': 'Myocardial Infarction',
        'HYP': 'Cardiac Hypertrophy',
        'CD': 'Conduction Disturbance',
        'STTC': 'ST-T Wave Changes',
    }

    patient_info = get_patient_info(patient_id)
    ecg_id = patient_info.get('ecg_id')

    # Fast path: precomputed
    precomputed = load_precomputed(patient_id)
    if precomputed:
        return {
            'success': True,
            'patient_id': patient_id,
            'prediction': precomputed['prediction'],
            'prediction_full': CLASS_FULL.get(precomputed['prediction'], precomputed['prediction']),
            'confidence': precomputed['confidence'],
            'probabilities': precomputed['probabilities'],
            'patient_info': patient_info,
            'source': 'precomputed',
        }

    # Live inference fallback
    try:
        mlp, scaler, le = load_model()
    except Exception as e:
        return {'success': False, 'error': f'Failed to load model: {e}'}

    if ecg_id is None:
        return {'success': False, 'error': f'Patient {patient_id} not found.'}

    features = extract_features_live(ecg_id)
    if features is None:
        return {'success': False, 'error': f'Could not load signals for {patient_id}'}

    features_s = scaler.transform(features)
    pred_idx = mlp.predict(features_s)[0]
    probs = mlp.predict_proba(features_s)[0]
    pred_cls = le.classes_[pred_idx]

    return {
        'success': True,
        'patient_id': patient_id,
        'prediction': pred_cls,
        'prediction_full': CLASS_FULL.get(pred_cls, pred_cls),
        'confidence': float(probs.max()),
        'probabilities': {le.classes_[i]: float(probs[i]) for i in range(len(le.classes_))},
        'patient_info': patient_info,
        'source': 'live_inference',
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'Usage: python predict_diagnosis.py PTB00001'}))
        sys.exit(1)

    result = predict(sys.argv[1])
    print(json.dumps(result, indent=2))

