"""
Full PTB-XL Training Pipeline  –  80/20 stratified split
=========================================================
1. Reads all 21,799 records from archive/records500
2. Extracts 170 clinical features (HR, n_beats + 12 leads × 14 features)
   using the SAME column schema as ECG_Diag_pipeline/artifacts/feature_cols.txt
3. Trains a PyTorch MLP (256→128→5) with class-balanced sampling
4. Prints full evaluation: confusion matrix, per-class metrics, predicted vs actual
5. Saves new artifacts to ECG_Diag_pipeline/artifacts/
   – mlp_diagclass.pt  (replaces the old one)
   – scaler.pkl
   – feature_cols.txt  (overwrite)
   – features_all_splits.csv
6. Prints predicted vs actual for a random sample of test records
"""

from __future__ import annotations
import ast, json, pickle, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
PTBXL_DIR  = ROOT / "archive" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
DB_CSV     = PTBXL_DIR / "ptbxl_database.csv"
REC_DIR    = PTBXL_DIR / "records500"

ART        = ROOT / "ECG_Diag_pipeline" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

PUBLIC     = ROOT / "public"
PUBLIC.mkdir(parents=True, exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────────────
FS         = 500          # Hz
LEADS_12   = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
N_EPOCHS   = 60
BATCH      = 256
LR         = 1e-3
PATIENCE   = 10           # early stopping patience

CLASSES    = ["CD", "HYP", "MI", "NORM", "STTC"]

CLASS_LABELS = {
    "CD":   "Conduction Disturbance",
    "HYP":  "Hypertrophy",
    "MI":   "Myocardial Infarction",
    "NORM": "Normal Sinus Rhythm",
    "STTC": "ST-T Change",
}

# ─── SCP → Superclass map (same as streamlit_dashboard.py) ──────────────────
SCP_MAP = {
    "NORM": "NORM",
    "IMI":"MI","ILMI":"MI","AMI":"MI","ALMI":"MI","INJAS":"MI","LMI":"MI",
    "INJAL":"MI","IPLMI":"MI","IPMI":"MI","INJIN":"MI","INJLA":"MI","PMI":"MI","INJIL":"MI","INJA":"MI",
    "NDT":"STTC","DIG":"STTC","LNGQT":"STTC","ANEUR":"STTC","EL":"STTC",
    "ISCA":"STTC","ISCI":"STTC","ISC_":"STTC","STTC":"STTC","STD_":"STTC","STE_":"STTC",
    "LAFB":"CD","IRBBB":"CD","IVCD":"CD","LBBB":"CD","RBBB":"CD","LPFB":"CD","WPW":"CD",
    "1AVB":"CD","2AVB":"CD","3AVB":"CD","AVB":"CD",
    "LVH":"HYP","LAO":"HYP","RVH":"HYP","SEHYP":"HYP","LVOLT":"HYP","RAO":"HYP","LMH":"HYP",
}

def get_superclass(scp_str: str) -> str | None:
    try:
        codes = ast.literal_eval(scp_str) if isinstance(scp_str, str) else scp_str
        scored: dict[str, float] = {}
        for code, conf in codes.items():
            sc = SCP_MAP.get(code)
            if sc:
                scored[sc] = scored.get(sc, 0) + conf
        return max(scored, key=scored.get) if scored else None
    except Exception:
        return None

# ─── ECG Preprocessing ───────────────────────────────────────────────────────
def bandpass(x, lo=0.5, hi=40.0, fs=FS, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x)

def notch(x, f0=50.0, Q=30.0, fs=FS):
    b, a = iirnotch(f0/(fs/2), Q)
    return filtfilt(b, a, x)

def preprocess(sig: np.ndarray) -> np.ndarray:
    """Bandpass + notch filter, NaN → 0."""
    sig = np.nan_to_num(sig, nan=0.0)
    sig = bandpass(sig)
    sig = notch(sig)
    return sig

# ─── Feature extraction (170 cols matching feature_cols.txt) ─────────────────
def detect_r_peaks(lead_sig: np.ndarray, fs: int = FS) -> np.ndarray:
    """Pan-Tompkins inspired R-peak detector."""
    # derivative + squaring
    diff = np.diff(lead_sig, prepend=lead_sig[0])
    sq   = diff ** 2
    # integrate with moving window
    win = int(0.150 * fs)
    integrated = np.convolve(sq, np.ones(win)/win, mode='same')
    threshold = 0.3 * np.max(integrated)
    min_dist  = int(0.25 * fs)
    peaks, _  = find_peaks(integrated, height=threshold, distance=min_dist)
    return peaks

def extract_per_lead_features(lead_sig: np.ndarray, r_peaks: np.ndarray, fs: int = FS) -> dict:
    """
    Returns dict with 14 features per lead (mean + std of 7 clinical measures).
    r_peaks are approximate beat centres from lead II; we refine them per-lead.
    """
    n = len(lead_sig)
    empty = {k: 0.0 for k in [
        "R_amp_mean","R_amp_std",
        "QRS_w_ms_mean","QRS_w_ms_std",
        "P_dur_ms_mean","P_dur_ms_std",
        "P_amp_mean","P_amp_std",
        "PR_int_ms_mean","PR_int_ms_std",
        "QT_int_ms_mean","QT_int_ms_std",
        "ST_dev_mean","ST_dev_std",
    ]}
    if len(r_peaks) < 2:
        return empty

    R_amps, QRS_ws, P_durs, P_amps, PR_ints, QT_ints, ST_devs = [], [], [], [], [], [], []

    # Refine window: ±100 ms around each approximate beat position
    refine_win = int(0.10 * fs)

    for rp_approx in r_peaks:
        # ── Refine R peak: find the max-absolute-value sample in ±refine_win ──
        win_s = max(0, rp_approx - refine_win)
        win_e = min(n, rp_approx + refine_win)
        local_seg = lead_sig[win_s:win_e]
        if len(local_seg) == 0:
            continue
        # Choose max or min, whichever has larger absolute value  (handles inverted leads)
        local_max_idx = int(np.argmax(local_seg))
        local_min_idx = int(np.argmin(local_seg))
        if abs(local_seg[local_max_idx]) >= abs(local_seg[local_min_idx]):
            rp = win_s + local_max_idx
        else:
            rp = win_s + local_min_idx

        # ── R amplitude ──────────────────────────────────────────────────
        R_amps.append(float(lead_sig[rp]))

        # ── QRS width: ±80 ms search for Q and S troughs ─────────────────
        q_window = int(0.08 * fs)
        s_window = int(0.08 * fs)
        q_start = max(0, rp - q_window)
        s_end   = min(n, rp + s_window)

        q_seg  = lead_sig[q_start:rp]
        s_seg  = lead_sig[rp:s_end]

        q_idx = q_start + int(np.argmin(q_seg)) if len(q_seg) else rp
        s_idx = rp + int(np.argmin(s_seg)) if len(s_seg) else rp

        qrs_w_ms = (s_idx - q_idx) / fs * 1000.0
        QRS_ws.append(max(40.0, min(300.0, qrs_w_ms)))   # clip sanity

        # ── P wave: search 250→80 ms before R peak ───────────────────────
        p_search_end   = max(0, rp - int(0.08 * fs))
        p_search_start = max(0, rp - int(0.25 * fs))
        if p_search_end > p_search_start:
            p_seg = lead_sig[p_search_start:p_search_end]
            p_local_peak = int(np.argmax(p_seg))
            p_amp = float(p_seg[p_local_peak])
            P_amps.append(p_amp)
            # P duration heuristic: ±30 ms around peak
            p_peak_global = p_search_start + p_local_peak
            pd_start = max(0, p_peak_global - int(0.06 * fs))
            pd_end   = min(n, p_peak_global + int(0.06 * fs))
            P_durs.append((pd_end - pd_start) / fs * 1000.0)

            # PR interval: p_start → QRS onset (q_idx)
            pr_ms = (q_idx - pd_start) / fs * 1000.0
            PR_ints.append(max(50.0, min(400.0, pr_ms)))
        else:
            P_amps.append(0.0)
            P_durs.append(0.0)
            PR_ints.append(0.0)

        # ── QT interval: QRS onset → T wave end (heuristic: +500 ms) ────
        t_search_start = rp + int(0.05 * fs)
        t_search_end   = min(n, rp + int(0.50 * fs))
        if t_search_end > t_search_start:
            t_seg  = lead_sig[t_search_start:t_search_end]
            t_peak = t_search_start + int(np.argmax(np.abs(t_seg)))
            # T end ≈ 50 ms after T peak
            t_end  = min(n, t_peak + int(0.05 * fs))
            qt_ms  = (t_end - q_idx) / fs * 1000.0
            QT_ints.append(max(200.0, min(700.0, qt_ms)))
        else:
            QT_ints.append(0.0)

        # ── ST deviation: mean of 60–120 ms after R peak ─────────────────
        st_start = min(n, rp + int(0.06 * fs))
        st_end   = min(n, rp + int(0.12 * fs))
        if st_end > st_start:
            ST_devs.append(float(np.mean(lead_sig[st_start:st_end])))
        else:
            ST_devs.append(0.0)

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0
    def safe_std(lst):
        return float(np.std(lst))  if lst else 0.0

    return {
        "R_amp_mean":    safe_mean(R_amps),  "R_amp_std":    safe_std(R_amps),
        "QRS_w_ms_mean": safe_mean(QRS_ws),  "QRS_w_ms_std": safe_std(QRS_ws),
        "P_dur_ms_mean": safe_mean(P_durs),  "P_dur_ms_std": safe_std(P_durs),
        "P_amp_mean":    safe_mean(P_amps),  "P_amp_std":    safe_std(P_amps),
        "PR_int_ms_mean":safe_mean(PR_ints), "PR_int_ms_std":safe_std(PR_ints),
        "QT_int_ms_mean":safe_mean(QT_ints), "QT_int_ms_std":safe_std(QT_ints),
        "ST_dev_mean":   safe_mean(ST_devs), "ST_dev_std":   safe_std(ST_devs),
    }


# Column order that must match feature_cols.txt exactly
FEAT_COLS = ["HR_bpm", "n_beats"]
for _lead in LEADS_12:
    for _suf in ["R_amp_mean","R_amp_std",
                 "QRS_w_ms_mean","QRS_w_ms_std",
                 "P_dur_ms_mean","P_dur_ms_std",
                 "P_amp_mean","P_amp_std",
                 "PR_int_ms_mean","PR_int_ms_std",
                 "QT_int_ms_mean","QT_int_ms_std",
                 "ST_dev_mean","ST_dev_std"]:
        FEAT_COLS.append(f"{_lead}_{_suf}")
# 2 + 12×14 = 170


def extract_record(sig: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    sig: (samples, 12) float32 array in LEADS_12 order.
    Returns a 170-dim feature vector.
    """
    # Use lead II (idx 1) for global HR / n_beats
    lead_II = preprocess(sig[:, 1])
    r_peaks = detect_r_peaks(lead_II, fs)
    n_beats = float(len(r_peaks))
    duration_s = sig.shape[0] / fs
    hr_bpm = float(n_beats / duration_s * 60.0) if duration_s > 0 else 0.0

    row = {"HR_bpm": hr_bpm, "n_beats": n_beats}

    for li, lead_name in enumerate(LEADS_12):
        lead_proc = preprocess(sig[:, li])
        # Use the same R peaks (lead II) for all leads — consistent segmentation
        feats = extract_per_lead_features(lead_proc, r_peaks, fs)
        for suf, val in feats.items():
            row[f"{lead_name}_{suf}"] = val

    return np.array([row[c] for c in FEAT_COLS], dtype=np.float32)


# ─── MLP Model ───────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),    nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Feature extraction loop ─────────────────────────────────────────────────
def extract_all_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import wfdb  # lazy import
    X, y, ids = [], [], []
    n = len(df)
    t0 = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        ecg_id     = int(row["ecg_id"])
        superclass = get_superclass(row["scp_codes"])
        if superclass is None or superclass not in CLASSES:
            continue

        folder = f"{(ecg_id - 1) // 1000 * 1000:05d}"
        rec    = REC_DIR / folder / f"{ecg_id:05d}_hr"

        if not (rec.with_suffix(".hea")).exists():
            continue

        try:
            sig, meta = wfdb.rdsamp(str(rec), channels=list(range(12)))
        except Exception:
            continue

        # Reorder leads to canonical LEADS_12 order
        sig_names = [s.upper().replace("AVR","AVR").replace("AVF","AVF").replace("AVL","AVL")
                     for s in meta["sig_name"]]
        # Map column indices
        col_map = {name: idx for idx, name in enumerate(sig_names)}
        try:
            ordered = np.stack([sig[:, col_map[l.upper()]] for l in LEADS_12], axis=1).astype(np.float32)
        except KeyError:
            ordered = sig[:, :12].astype(np.float32)

        fvec = extract_record(ordered, fs=meta["fs"])
        if not np.all(np.isfinite(fvec)):
            fvec = np.nan_to_num(fvec, nan=0.0)

        X.append(fvec)
        y.append(superclass)
        ids.append(ecg_id)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1:>5}/{n}]  kept={len(X):>5}  elapsed={elapsed:.0f}s", flush=True)

    print(f"  Done: {len(X)}/{n} records extracted in {time.time()-t0:.0f}s")
    return np.array(X, dtype=np.float32), np.array(y), np.array(ids, dtype=np.int64)


# ─── Training ────────────────────────────────────────────────────────────────
def train_model(X_tr: np.ndarray, y_tr: np.ndarray,
                X_va: np.ndarray, y_va: np.ndarray,
                class_to_idx: dict) -> tuple[MLP, list]:
    n_classes = len(class_to_idx)
    in_dim    = X_tr.shape[1]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}  |  in_dim={in_dim}  n_classes={n_classes}")

    # class-balanced sampler
    class_counts = np.bincount([class_to_idx[c] for c in y_tr])
    weights_cls  = 1.0 / (class_counts + 1e-6)
    sample_wts   = np.array([weights_cls[class_to_idx[c]] for c in y_tr])
    sampler      = WeightedRandomSampler(torch.tensor(sample_wts, dtype=torch.float32),
                                         num_samples=len(y_tr), replacement=True)

    # tensors
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor([class_to_idx[c] for c in y_tr], dtype=torch.long)
    Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
    yv = torch.tensor([class_to_idx[c] for c in y_va], dtype=torch.long).to(device)

    ds     = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=BATCH, sampler=sampler)

    model     = MLP(in_dim, n_classes).to(device)
    cw        = compute_class_weight("balanced", classes=np.arange(n_classes),
                                     y=[class_to_idx[c] for c in y_tr])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    best_acc = 0.0
    best_state = None
    no_improve = 0
    history = []

    print(f"\n  Training: {len(X_tr)} | Val: {len(X_va)} | Epochs: {N_EPOCHS}")
    print(f"  {'Epoch':>6} | {'Train Loss':>11} | {'Val Acc':>8}")
    print("  " + "-"*35)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = torch.argmax(model(Xv), dim=1)
            val_acc   = (val_preds == yv).float().mean().item()

        avg_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc})

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6} | {avg_loss:>11.4f} | {val_acc*100:>7.2f}%")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}  (best val_acc={best_acc*100:.2f}%)")
                break

    model.load_state_dict(best_state)
    print(f"\n  Best val accuracy: {best_acc*100:.2f}%")
    return model, history


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(model: MLP, X_te: np.ndarray, y_te: np.ndarray,
             class_to_idx: dict, idx_to_class: dict, ecg_ids_te: np.ndarray,
             X_raw_te: np.ndarray) -> dict:
    device = next(model.parameters()).device
    model.eval()
    Xt = torch.tensor(X_te, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(Xt)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = torch.argmax(logits, dim=1).cpu().numpy()

    y_true_idx = np.array([class_to_idx[c] for c in y_te])
    accuracy   = float(accuracy_score(y_true_idx, preds))

    ordered_cls = [c for c in CLASSES if c in class_to_idx]
    ordered_idx = [class_to_idx[c] for c in ordered_cls]

    print("\n" + "="*70)
    print(f"TEST SET RESULTS  (n={len(y_te)}, accuracy={accuracy*100:.2f}%)")
    print("="*70)
    print(classification_report(
        y_true_idx, preds,
        labels=ordered_idx,
        target_names=ordered_cls,
        digits=3, zero_division=0
    ))

    cm = confusion_matrix(y_true_idx, preds, labels=ordered_idx)
    print("Confusion Matrix (rows=Actual, cols=Predicted):")
    header = "         " + "  ".join(f"{c:>6}" for c in ordered_cls)
    print(header)
    for ci, cls in enumerate(ordered_cls):
        row_str = "  ".join(f"{cm[ci, j]:>6}" for j in range(len(ordered_cls)))
        print(f"  {cls:>5}:  {row_str}")

    # ── Predicted vs Actual sample ─────────────────────────────────────────
    print("\n" + "="*70)
    print("PREDICTED vs ACTUAL – random sample of 20 test records")
    print("="*70)
    sample_n = min(20, len(y_te))
    sample_indices = np.random.choice(len(y_te), sample_n, replace=False)
    sample_indices = sorted(sample_indices)
    print(f"  {'ecg_id':>7}  {'Actual':>6}  {'Predicted':>10}  {'Confidence':>11}  {'Correct':>7}")
    print("  " + "-"*50)
    for si in sample_indices:
        ecg_id    = ecg_ids_te[si]
        actual    = y_te[si]
        predicted = idx_to_class[int(preds[si])]
        conf      = float(probs[si, int(preds[si])])
        correct   = "✓" if actual == predicted else "✗"
        print(f"  {ecg_id:>7}  {actual:>6}  {predicted:>10}  {conf*100:>10.1f}%  {correct:>7}")

    per_class = {}
    for ci, cls in enumerate(ordered_cls):
        mask = y_true_idx == class_to_idx[cls]
        if mask.sum() > 0:
            per_class[cls] = {
                "precision": float(cm[ci, ci] / (cm[:, ci].sum() + 1e-9)),
                "recall":    float(cm[ci, ci] / (cm[ci, :].sum() + 1e-9)),
                "support":   int(mask.sum()),
                "accuracy":  float((preds[mask] == class_to_idx[cls]).sum() / mask.sum()),
            }
        else:
            per_class[cls] = {"precision":0.0,"recall":0.0,"support":0,"accuracy":0.0}

    return {
        "accuracy":      accuracy,
        "classes":       CLASSES,
        "ordered_cls":   ordered_cls,
        "confusion_matrix": cm.tolist(),
        "test_samples":  int(len(y_te)),
        "per_class":     per_class,
        "probs":         probs,
        "preds_idx":     preds,
        "y_true_idx":    y_true_idx,
    }


# ─── Save artifacts ───────────────────────────────────────────────────────────
def save_artifacts(model: MLP, scaler: StandardScaler,
                   class_to_idx: dict, eval_info: dict,
                   X_all: np.ndarray, y_all: np.ndarray, ids_all: np.ndarray,
                   history: list):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    device       = next(model.parameters()).device

    # ── mlp_diagclass.pt ──────────────────────────────────────────────────
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim":  X_all.shape[1],
        "classes":    [idx_to_class[i] for i in range(len(class_to_idx))],
        "accuracy":   eval_info["accuracy"],
    }, ART / "mlp_diagclass.pt")
    print(f"\n✓ Model saved   → {ART}/mlp_diagclass.pt")

    # ── scaler.pkl ────────────────────────────────────────────────────────
    with open(ART / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved  → {ART}/scaler.pkl")

    # ── feature_cols.txt ─────────────────────────────────────────────────
    (ART / "feature_cols.txt").write_text("\n".join(FEAT_COLS))
    print(f"✓ Feat cols     → {ART}/feature_cols.txt  ({len(FEAT_COLS)} features)")

    # ── features_all_splits.csv ───────────────────────────────────────────
    df_feat = pd.DataFrame(X_all, columns=FEAT_COLS)
    df_feat.insert(0, "ecg_id", ids_all)
    df_feat["y"]  = y_all
    df_feat["split"] = "train"          # placeholder; precompute script doesn't use it
    df_feat.to_csv(ART / "features_all_splits.csv", index=False)
    print(f"✓ Features CSV  → {ART}/features_all_splits.csv  ({len(df_feat)} rows)")

    # ── public/mlp_results.json ───────────────────────────────────────────
    results = {
        "accuracy":      eval_info["accuracy"],
        "classes":       eval_info["ordered_cls"],
        "confusion_matrix": eval_info["confusion_matrix"],
        "test_samples":  eval_info["test_samples"],
        "architecture": {
            "input_size":    X_all.shape[1],
            "hidden_layers": [256, 128, 64],
            "output_size":   len(class_to_idx),
            "activation":    "relu",
        },
        "training_info": {
            "total_records":  int(len(ids_all)),
            "train_fraction": 0.8,
            "epochs_run":     len(history),
            "best_val_acc":   max(h["val_acc"] for h in history),
            "version":        "4.0-full",
        },
        "per_class_performance": {
            cls: {
                "accuracy": info["accuracy"],
                "precision": info["precision"],
                "recall":    info["recall"],
                "samples":   info["support"],
                "present_in_test": info["support"] > 0,
            }
            for cls, info in eval_info["per_class"].items()
        },
    }
    with open(PUBLIC / "mlp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results       → {PUBLIC}/mlp_results.json")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("PTB-XL FULL TRAINING  v4.0  (80/20 stratified split)")
    print("="*70)
    print(f"  PTB-XL dir:  {PTBXL_DIR}")
    print(f"  Records dir: {REC_DIR}")

    # ── Load metadata ──────────────────────────────────────────────────────
    df_meta = pd.read_csv(DB_CSV)
    print(f"\n  Metadata: {len(df_meta)} rows")

    # ── Feature extraction ────────────────────────────────────────────────
    print("\n[STEP 1/4] Feature extraction from 500 Hz signals…")
    X, y, ids = extract_all_features(df_meta)
    print(f"\n  Extracted {len(X)} usable records  |  features={X.shape[1]}")

    # Class distribution
    print("\n  Class distribution (full dataset):")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {u:>6}: {c:>5}  ({c/len(y)*100:.1f}%)")

    # ── 80/20 stratified split ────────────────────────────────────────────
    print("\n[STEP 2/4] Train/Test split (80/20, stratified)…")
    X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
        X, y, ids, test_size=0.20, random_state=42, stratify=y
    )
    # Further split train → 90% train, 10% val (for early stopping)
    X_tr2, X_va, y_tr2, y_va = train_test_split(
        X_tr, y_tr, test_size=0.10, random_state=42, stratify=y_tr
    )

    print(f"  Train: {len(X_tr2)}  |  Val: {len(X_va)}  |  Test: {len(X_te)}")

    # ── Scaler (fit on train only) ─────────────────────────────────────────
    scaler = StandardScaler()
    X_tr2_s = scaler.fit_transform(X_tr2)
    X_va_s  = scaler.transform(X_va)
    X_te_s  = scaler.transform(X_te)
    X_all_s = scaler.transform(X)

    class_to_idx = {c: i for i, c in enumerate(CLASSES) if c in np.unique(y)}
    # Reindex to be consecutive  
    present = sorted(np.unique(y))
    class_to_idx = {c: i for i, c in enumerate(present)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # ── Training ──────────────────────────────────────────────────────────
    print("\n[STEP 3/4] Training PyTorch MLP…")
    model, history = train_model(X_tr2_s, y_tr2, X_va_s, y_va, class_to_idx)

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n[STEP 4/4] Evaluating on held-out 20% test set…")
    eval_info = evaluate(model, X_te_s, y_te, class_to_idx, idx_to_class, ids_te, X_te)

    # ── Save ─────────────────────────────────────────────────────────────
    print("\n--- Saving artifacts ---")
    save_artifacts(model, scaler, class_to_idx, eval_info, X, y, ids, history)

    print("\n" + "="*70)
    print(f"DONE  |  Test accuracy: {eval_info['accuracy']*100:.2f}%  |  Records: {len(X)}")
    print("="*70)
    print("\nNext step: run  scripts/precompute_mlp_predictions.py  to refresh the dashboard JSON")
