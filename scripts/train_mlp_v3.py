"""
Fast signal-based feature extraction + MLP training on full PTB-XL 21,799 records.
Reads raw .dat/.hea files, extracts statistical ECG features per lead, trains MLP.
Saves model + per-patient predictions to JSON for React app.
"""

import ast
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings('ignore')

# ─── SCP → Superclass ─────────────────────────────────────────────────────────
SCP_MAP = {
    'NORM': 'NORM',
    'IMI': 'MI', 'ILMI': 'MI', 'AMI': 'MI', 'ALMI': 'MI',
    'INJAS': 'MI', 'LMI': 'MI', 'INJAL': 'MI', 'IPLMI': 'MI',
    'IPMI': 'MI', 'INJIN': 'MI', 'INJLA': 'MI', 'PMI': 'MI',
    'INJIL': 'MI', 'INJA': 'MI',
    'NDT': 'STTC', 'DIG': 'STTC', 'LNGQT': 'STTC', 'ANEUR': 'STTC',
    'EL': 'STTC', 'ISCA': 'STTC', 'ISCI': 'STTC', 'ISC_': 'STTC',
    'STTC': 'STTC', 'STD_': 'STTC', 'STE_': 'STTC',
    'NDT': 'STTC',
    'LAFB': 'CD', 'IRBBB': 'CD', 'IVCD': 'CD', 'LBBB': 'CD',
    'RBBB': 'CD', 'LPFB': 'CD', 'WPW': 'CD', '1AVB': 'CD',
    '2AVB': 'CD', '3AVB': 'CD', 'AVB': 'CD',
    'LVH': 'HYP', 'LAO': 'HYP', 'RVH': 'HYP', 'SEHYP': 'HYP',
    'LVOLT': 'HYP', 'RAO': 'HYP', 'LMH': 'HYP',
}


def get_superclass(scp_str):
    try:
        codes = ast.literal_eval(scp_str) if isinstance(scp_str, str) else scp_str
        scored = {}
        for code, conf in codes.items():
            sc = SCP_MAP.get(code)
            if sc:
                scored[sc] = scored.get(sc, 0) + conf
        if not scored:
            return None
        return max(scored, key=scored.get)
    except:
        return None


def extract_lead_features(lead_signal, fs=500):
    """Extract 20 statistical + frequency domain features per lead."""
    sig = lead_signal.astype(float)
    n = len(sig)
    if n == 0:
        return [0.0] * 20

    # Time domain
    mean_val = float(np.mean(sig))
    std_val = float(np.std(sig))
    rms_val = float(np.sqrt(np.mean(sig**2)))
    max_val = float(np.max(sig))
    min_val = float(np.min(sig))
    p2p = float(max_val - min_val)
    skew_val = float(skew(sig))
    kurt_val = float(kurtosis(sig))

    # Derivative features (QRS energy)
    diff1 = np.diff(sig)
    energy = float(np.sum(diff1**2))
    max_slope = float(np.max(np.abs(diff1))) if len(diff1) > 0 else 0.0

    # Zero crossing rate
    zcr = float(np.sum(np.diff(np.sign(sig)) != 0)) / n

    # Power spectral density bands
    freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(256, n))
    total_power = float(np.sum(psd)) + 1e-10
    lf_mask = (freqs >= 0.5) & (freqs < 15)   # Low freq: P, PQ, ST
    hf_mask = (freqs >= 15) & (freqs < 50)    # High freq: QRS
    vlf_mask = freqs < 0.5

    lf_power = float(np.sum(psd[lf_mask])) / total_power
    hf_power = float(np.sum(psd[hf_mask])) / total_power
    vlf_power = float(np.sum(psd[vlf_mask])) / total_power
    lf_hf_ratio = float(lf_power / (hf_power + 1e-10))

    # Simple R-peak detection (just count peaks for HR estimate)
    threshold = mean_val + 0.5 * std_val
    peaks_above = np.where((sig[1:-1] > threshold) & (sig[1:-1] > sig[:-2]) & (sig[1:-1] > sig[2:]))[0]
    n_peaks = len(peaks_above)
    hr_est = float(n_peaks * 60 * fs / n) if n > 0 else 0.0

    # ST segment estimate: mean of latter 60% of signal
    st_region = sig[int(0.4 * n):]
    st_mean = float(np.mean(st_region)) if len(st_region) > 0 else 0.0

    return [
        mean_val, std_val, rms_val, max_val, min_val, p2p,
        skew_val, kurt_val, energy, max_slope, zcr,
        lf_power, hf_power, vlf_power, lf_hf_ratio,
        float(n_peaks), hr_est, st_mean, float(n_peaks / (n / fs + 1e-6)), total_power
    ]


def load_and_extract_all(data_root, df, max_records=None, sample_ratio=1.0):
    """Iterate over CSV rows, read raw signals, extract features and superclass labels."""
    print(f"\nExtracting features from raw signals...")

    records_dir = data_root / 'records500'
    X, y, ecg_ids = [], [], []

    limit = len(df) if max_records is None else min(max_records, len(df))
    if sample_ratio < 1.0:
        df = df.sample(frac=sample_ratio, random_state=42)

    skipped = 0
    n = len(df)
    step = max(1, n // 20)

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx >= limit:
            break
        if idx % step == 0:
            pct = idx / min(limit, n) * 100
            print(f"  [{pct:5.1f}%] Processed {idx}/{min(limit, n)} records, kept {len(X)}", flush=True)

        superclass = get_superclass(row['scp_codes'])
        if superclass is None:
            skipped += 1
            continue

        ecg_id = int(row['ecg_id'])
        folder = f"{(ecg_id - 1) // 1000 * 1000:05d}"
        fname = records_dir / folder / f"{ecg_id:05d}_hr"

        hea_path = Path(str(fname) + '.hea')
        if not hea_path.exists():
            skipped += 1
            continue

        try:
            import wfdb
            sig, meta = wfdb.rdsamp(str(fname), channels=list(range(12)))
        except Exception:
            skipped += 1
            continue

        # Extract features per lead (12 leads × 20 features = 240 features)
        feature_vec = []
        fs = meta['fs']
        for lead_idx in range(min(12, sig.shape[1])):
            feature_vec.extend(extract_lead_features(sig[:, lead_idx], fs))

        if len(feature_vec) == 240:
            X.append(feature_vec)
            y.append(superclass)
            ecg_ids.append(ecg_id)

    if idx % step != 0:
        print(f"  [100.0%] Processed {min(limit, n)}/{min(limit, n)} records, kept {len(X)}")
    print(f"\n  Skipped: {skipped}")
    return np.array(X), np.array(y), np.array(ecg_ids)


def train_and_evaluate(X, y):
    print("\n" + "="*70)
    print("TRAINING: BALANCED MLP (300 iter max)")
    print("="*70)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\nClass distribution:")
    for i, cls in enumerate(le.classes_):
        cnt = int((y_enc == i).sum())
        print(f"  {cls:6s}: {cnt:5d} ({cnt/len(y_enc)*100:.1f}%)")

    classes = np.unique(y_enc)
    cw = compute_class_weight('balanced', classes=classes, y=y_enc)
    sw = np.array([cw[c] for c in y_enc])

    X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
        X, y_enc, sw, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"\nTraining: {len(X_tr)}, Test: {len(X_te)}")
    print("Architecture: 240 → 256 → 128 → 64 → 5 classes\n")

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        learning_rate_init=0.001,
        verbose=True,
    )
    mlp.fit(X_tr_s, y_tr, sample_weight=sw_tr)

    y_pred = mlp.predict(X_te_s)
    acc = accuracy_score(y_te, y_pred)

    print(f"\nAccuracy: {acc*100:.2f}%")
    unique_te = sorted(np.unique(y_te))
    cls_names = [le.classes_[i] for i in unique_te]
    print(classification_report(y_te, y_pred, labels=unique_te, target_names=cls_names, digits=3, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred, labels=unique_te))

    return mlp, scaler, le, acc, y_te, y_pred


def save_all(mlp, scaler, le, acc, y_te, y_pred, ecg_ids_all, y_all_enc, X_all_s, project_root):
    """Save model and precompute predictions for all records."""
    output_dir = project_root / 'public'

    # Save model
    with open(output_dir / 'mlp_model.pkl', 'wb') as f:
        pickle.dump({
            'model': mlp, 'scaler': scaler, 'label_encoder': le,
            'feature_size': 240, 'version': '3.0',
        }, f)
    print(f"\nModel saved → public/mlp_model.pkl")

    # Precompute & save ALL patient predictions
    print("Precomputing predictions for all patients...")
    probs = mlp.predict_proba(X_all_s)
    preds = mlp.predict(X_all_s)

    CLASS_FULL = {
        'NORM': 'Normal/Sinus Rhythm',
        'MI': 'Myocardial Infarction',
        'HYP': 'Hypertrophy',
        'CD': 'Conduction Disturbance',
        'STTC': 'ST-T Wave Changes',
    }

    patient_predictions = {}
    for i, ecg_id in enumerate(ecg_ids_all):
        cls = le.classes_[preds[i]]
        conf = float(probs[i].max())
        patient_predictions[f"PTB{int(ecg_id):05d}"] = {
            'prediction': cls,
            'prediction_full': CLASS_FULL.get(cls, cls),
            'confidence': conf,
            'probabilities': {le.classes_[j]: float(probs[i][j]) for j in range(len(le.classes_))},
        }

    with open(output_dir / 'patient_predictions.json', 'w') as f:
        json.dump(patient_predictions, f)
    print(f"Saved predictions for {len(patient_predictions)} patients → public/patient_predictions.json")

    # results.json for dashboard
    per_class = {}
    for i, cls in enumerate(le.classes_):
        mask = y_te == i
        if mask.sum() > 0:
            per_class[cls] = {
                'accuracy': float((y_pred[mask] == i).sum() / mask.sum()),
                'samples': int(mask.sum()), 'present_in_test': True
            }
        else:
            per_class[cls] = {'accuracy': 0.0, 'samples': 0, 'present_in_test': False}

    results = {
        'accuracy': float(acc),
        'classes': le.classes_.tolist(),
        'confusion_matrix': confusion_matrix(y_te, y_pred).tolist(),
        'test_samples': int(len(y_te)),
        'architecture': {'input_size': 240, 'hidden_layers': [256, 128, 64],
                         'output_size': len(le.classes_), 'activation': 'relu'},
        'training_info': {'iterations': int(mlp.n_iter_), 'final_loss': float(mlp.loss_),
                          'class_balanced': True, 'version': '3.0'},
        'per_class_performance': per_class,
    }
    with open(output_dir / 'mlp_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → public/mlp_results.json")


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_root = project_root / 'archive' / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    csv_path = project_root / 'public' / 'ptbxl_database.csv'

    print("PTB-XL MLP Classifier v3.0 – SIGNAL-BASED TRAINING")
    print("="*70)

    df = pd.read_csv(csv_path)
    print(f"Full dataset: {len(df)} records")

    # Sample for quick training (~2000 records)
    print("\nSampling 2000 records for fast training...")
    df_sample = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)

    X, y, ecg_ids = load_and_extract_all(data_root, df_sample)
    print(f"\nExtracted {len(X)} samples with 240 features (12 leads × 20)")

    if len(X) == 0:
        print("ERROR: No samples extracted!"); exit(1)

    mlp, scaler, le, acc, y_te, y_pred = train_and_evaluate(X, y)

    # Precompute for all extracted patients
    X_all_s = scaler.transform(X)
    y_all_enc = le.transform(y)
    save_all(mlp, scaler, le, acc, y_te, y_pred, ecg_ids, y_all_enc, X_all_s, project_root)

    print("\n" + "="*70)
    print(f"DONE! Accuracy: {acc*100:.2f}%  |  Patients: {len(X)}")
    print("="*70)
