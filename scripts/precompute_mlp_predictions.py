"""
Pre-compute MLP predictions (from ECG_Diag_pipeline) for all patients.
Outputs:  public/mlp_predictions.json
          - keyed by ecg_id (int)
          - includes predicted class, probabilities, lead influence, abnormal features
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
PIPELINE   = ROOT / "ECG_Diag_pipeline" / "artifacts"
MODEL_PATH = PIPELINE / "mlp_diagclass.pt"
SCHEMA     = PIPELINE / "feature_cols.txt"
FEAT_ALL   = PIPELINE / "features_all_splits.csv"
FEAT_TRAIN = PIPELINE / "features_train.csv"
SCALER_PKL = PIPELINE / "scaler.pkl"
OUT        = ROOT / "public" / "mlp_predictions.json"

LEADS_12 = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# ─── MLP model (mirrors train_full_ptbxl.py architecture) ───────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),    nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─── Normal ranges (copied from streamlit_dashboard.py) ─────────────────────
NORMAL_RANGES = {
    "P_dur_ms_mean":  (0.0,   120.0,  "ms"),
    "PR_int_ms_mean": (120.0, 200.0,  "ms"),
    "QRS_w_ms_mean":  (0.0,   120.0,  "ms"),
    "QT_int_ms_mean": (300.0, 460.0,  "ms"),
    "ST_dev_mean":    (-0.10, 0.10,   "mV"),
    "R_amp_mean":     (0.0,   2.5,    "mV"),
    "P_amp_mean":     (0.0,   0.3,    "mV"),
}

CONDITION_LABELS = {
    "CD":   "Conduction Disturbance",
    "HYP":  "Hypertrophy",
    "MI":   "Myocardial Infarction",
    "NORM": "Normal Sinus Rhythm",
    "STTC": "ST-T Change",
}


def suffix_match(col: str, suffix: str) -> bool:
    return col.endswith("_" + suffix)


def check_abnormal(col: str, value: float):
    """Return (normalRange_str, status, unit) or None if no known range."""
    for key, (lo, hi, unit) in NORMAL_RANGES.items():
        if suffix_match(col, key):
            if value < lo:
                return f"{lo}–{hi}", "Low",  unit
            if value > hi:
                return f"{lo}–{hi}", "High", unit
            return f"{lo}–{hi}", "Normal", unit
    return None


def lead_from_col(col: str) -> str:
    head = col.split("_", 1)[0]
    return head if head in LEADS_12 else "GLOBAL"


def feature_label(col: str) -> str:
    """Human-readable label for a feature column."""
    mapping = {
        "QRS_w_ms_mean":  "QRS Width",
        "P_dur_ms_mean":  "P Wave Duration",
        "PR_int_ms_mean": "PR Interval",
        "QT_int_ms_mean": "QT Interval",
        "ST_dev_mean":    "ST Deviation",
        "R_amp_mean":     "R Amplitude",
        "P_amp_mean":     "P Amplitude",
    }
    for key, label in mapping.items():
        if suffix_match(col, key):
            return label
    return col


# ─── Load artifacts ──────────────────────────────────────────────────────────
print("Loading model checkpoint …")
ckpt      = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
classes   = ckpt["classes"]          # list like ['CD','HYP','MI','NORM','STTC']
in_dim    = int(ckpt["input_dim"])
model     = MLP(in_dim, len(classes))
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"  classes={classes}, in_dim={in_dim}")

feature_cols = SCHEMA.read_text().splitlines()
feature_cols = [c for c in feature_cols if c.strip()]
print(f"  feature_cols  count={len(feature_cols)}")

# Load features
feat = pd.read_csv(FEAT_ALL)
if "ecg_id" in feat.columns:
    feat = feat.set_index("ecg_id")

# Build / load scaler
if SCALER_PKL.exists():
    import pickle
    scaler = pickle.loads(SCALER_PKL.read_bytes())
    print("  scaler loaded from pkl")
else:
    from sklearn.preprocessing import StandardScaler
    print("  scaler.pkl missing → fitting on training features …")
    train_feat = pd.read_csv(FEAT_TRAIN)
    if "ecg_id" in train_feat.columns:
        train_feat = train_feat.set_index("ecg_id")
    X_train = train_feat.reindex(columns=feature_cols).to_numpy(dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0)
    scaler = StandardScaler().fit(X_train)
    print("  scaler fitted")


# ─── Compute ─────────────────────────────────────────────────────────────────
results: dict = {}

ecg_ids = feat.index.astype(int).tolist()
print(f"\nProcessing {len(ecg_ids)} patients …")

for i, ecg_id in enumerate(ecg_ids):
    if i % 200 == 0:
        print(f"  {i}/{len(ecg_ids)}")

    row = feat.loc[ecg_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    X_raw = row.reindex(feature_cols).to_numpy(dtype=np.float32)[None, :]
    X_raw = np.nan_to_num(X_raw, nan=0.0)
    X_s   = scaler.transform(X_raw)

    # ── Prediction ──
    xt = torch.tensor(X_s, dtype=torch.float32, requires_grad=False)
    with torch.no_grad():
        logits = model(xt)
        probs  = torch.softmax(logits, dim=1).numpy()[0]

    pred_i     = int(np.argmax(probs))
    pred_class = classes[pred_i]
    confidence = float(probs[pred_i])

    class_probs = {c: float(p) for c, p in zip(classes, probs)}

    # ── Gradient-based feature importance (gradient × input) ──
    xt2 = torch.tensor(X_s, dtype=torch.float32, requires_grad=True)
    logits2 = model(xt2)
    prob2    = torch.softmax(logits2, dim=1)
    score    = prob2[0, pred_i]
    score.backward()
    grads    = xt2.grad.detach().numpy()[0]          # (in_dim,)
    attr     = (grads * X_s[0])                      # gradient × scaled input

    # ── Per-lead importance ──
    lead_attr: dict[str, float] = {l: 0.0 for l in LEADS_12}
    lead_top_features: dict[str, list] = {l: [] for l in LEADS_12}

    for col_i, col in enumerate(feature_cols):
        lead = lead_from_col(col)
        if lead in lead_attr:
            lead_attr[lead] += abs(float(attr[col_i]))

    # Identify top 3 contributing features per lead
    for lead in LEADS_12:
        lead_feat_scores = []
        for col_i, col in enumerate(feature_cols):
            if lead_from_col(col) == lead:
                lead_feat_scores.append((abs(float(attr[col_i])), col_i, col))
        lead_feat_scores.sort(reverse=True)
        lead_top_features[lead] = [
            {"col": c, "label": feature_label(c), "attr": round(v, 6)}
            for v, _, c in lead_feat_scores[:5]
        ]

    sorted_leads = sorted(LEADS_12, key=lambda l: lead_attr[l], reverse=True)

    # ── Abnormal features ──
    raw_values = X_raw[0]   # unscaled original values
    abnormal   = []
    for col_i, col in enumerate(feature_cols):
        val    = float(raw_values[col_i])
        result = check_abnormal(col, val)
        if result is None:
            continue
        norm_str, status, unit = result
        if status == "Normal":
            continue
        lead = lead_from_col(col)
        abnormal.append({
            "feature":     col,
            "lead":        lead,
            "label":       feature_label(col),
            "value":       round(val, 4),
            "normalRange": norm_str,
            "status":      status,
            "unit":        unit,
            "attr":        round(abs(float(attr[col_i])), 6),
        })
    # Sort by attribution (most influential first)
    abnormal.sort(key=lambda x: x["attr"], reverse=True)

    # ── Top leads with their abnormal features ──
    top_leads_out = []
    for lead in sorted_leads[:6]:
        lead_abnormal = [a for a in abnormal if a["lead"] == lead]
        top_leads_out.append({
            "lead":             lead,
            "influence":        round(lead_attr[lead], 6),
            "topFeatures":      lead_top_features[lead],
            "abnormalFeatures": lead_abnormal[:5],
        })

    results[str(ecg_id)] = {
        "ecg_id":           ecg_id,
        "predictedClass":   pred_class,
        "predictedLabel":   CONDITION_LABELS.get(pred_class, pred_class),
        "confidence":       round(confidence, 4),
        "classProbabilities": class_probs,
        "topLeads":         top_leads_out,
        "allAbnormal":      abnormal[:20],
    }

# ─── Save ─────────────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=None))   # compact JSON
print(f"\n✓ Saved {len(results)} records → {OUT}")
