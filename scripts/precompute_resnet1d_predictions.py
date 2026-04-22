"""
Pre-compute ResNet1D predictions for all PTB-XL patients.
• Prediction + lead attribution : from ResNet1D trained on raw ECG signals
• Abnormal feature details (Q3/Q4/Q5): from clinical features CSV
Outputs: public/mlp_predictions.json  (same schema, dashboard-compatible)

Run AFTER train_resnet1d_ptbxl.py finishes:
    python scripts/precompute_resnet1d_predictions.py
"""

from __future__ import annotations

import ast, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
ARCHIVE   = ROOT / "archive" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
REC_ROOT  = ARCHIVE / "records500"
ARTIFACTS = ROOT / "ECG_Diag_pipeline" / "artifacts"
PUBLIC    = ROOT / "public"
MODEL_PT  = ARTIFACTS / "resnet1d_ptbxl.pt"
FEAT_CSV  = ARTIFACTS / "features_all_splits.csv"
SCHEMA    = ARTIFACTS / "feature_cols.txt"
OUT       = PUBLIC    / "mlp_predictions.json"

FS_TARGET = 100
SEQ_LEN   = 1000
N_LEADS   = 12
CLASSES   = ["CD", "HYP", "MI", "NORM", "STTC"]
LEADS_12  = ["I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]

CONDITION_LABELS = {
    "CD":   "Conduction Disturbance",
    "HYP":  "Hypertrophy",
    "MI":   "Myocardial Infarction",
    "NORM": "Normal Sinus Rhythm",
    "STTC": "ST-T Change",
}

NORMAL_RANGES = {
    "P_dur_ms_mean":  (0.0,   120.0, "ms"),
    "PR_int_ms_mean": (120.0, 200.0, "ms"),
    "QRS_w_ms_mean":  (0.0,   120.0, "ms"),
    "QT_int_ms_mean": (300.0, 460.0, "ms"),
    "ST_dev_mean":    (-0.10, 0.10,  "mV"),
    "R_amp_mean":     (0.0,   2.5,   "mV"),
    "P_amp_mean":     (0.0,   0.3,   "mV"),
}


def suffix_match(col: str, suffix: str) -> bool:
    return col.endswith("_" + suffix)

def check_abnormal(col: str, value: float):
    for key, (lo, hi, unit) in NORMAL_RANGES.items():
        if suffix_match(col, key):
            if value < lo:  return f"{lo}–{hi}", "Low",  unit
            if value > hi:  return f"{lo}–{hi}", "High", unit
            return f"{lo}–{hi}", "Normal", unit
    return None

def feature_label(col: str) -> str:
    mapping = {
        "QRS_w_ms_mean": "QRS Width",    "P_dur_ms_mean": "P Wave Duration",
        "PR_int_ms_mean":"PR Interval",  "QT_int_ms_mean":"QT Interval",
        "ST_dev_mean":   "ST Deviation", "R_amp_mean":    "R Amplitude",
        "P_amp_mean":    "P Amplitude",
    }
    for key, label in mapping.items():
        if suffix_match(col, key): return label
    return col

def lead_from_col(col: str) -> str:
    head = col.split("_", 1)[0]
    return head if head in LEADS_12 else "GLOBAL"


# ═══════════════════════════════════════════════════════════
# Model (must mirror train_resnet1d_ptbxl.py exactly)
# ═══════════════════════════════════════════════════════════
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1    = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1,      padding=pad, bias=False)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.drop     = nn.Dropout(dropout)
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch))
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))

class ResNet1D(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(12, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(ResBlock1D(32,  64,7,1,0.2), ResBlock1D(64, 64,7,1,0.2))
        self.layer2 = nn.Sequential(ResBlock1D(64, 128,7,2,0.2), ResBlock1D(128,128,7,1,0.2))
        self.layer3 = nn.Sequential(ResBlock1D(128,256,7,2,0.3), ResBlock1D(256,256,7,1,0.3))
        self.layer4 = nn.Sequential(ResBlock1D(256,256,7,2,0.3), ResBlock1D(256,256,7,1,0.3))
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Linear(256,128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, n_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.head(self.gap(x))


# ═══════════════════════════════════════════════════════════
# Load artifacts
# ═══════════════════════════════════════════════════════════
print("Loading ResNet1D checkpoint …")
ckpt  = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
model = ResNet1D(n_classes=len(CLASSES))
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"  val_accuracy ={ckpt.get('val_accuracy','?')}")
print(f"  test_accuracy={ckpt.get('test_accuracy','?')}")

feat_df      = None
feature_cols = []
if FEAT_CSV.exists() and SCHEMA.exists():
    print("Loading clinical features CSV …")
    feat_df = pd.read_csv(FEAT_CSV)
    if "ecg_id" in feat_df.columns:
        feat_df = feat_df.set_index("ecg_id")
    feature_cols = [c for c in SCHEMA.read_text().splitlines() if c.strip()]
    print(f"  {len(feat_df)} rows, {len(feature_cols)} feature cols")

# PTB-XL metadata
meta = pd.read_csv(ARCHIVE / "ptbxl_database.csv")
SCP_TO_SUPER = {
    "NORM":"NORM",
    "IMI":"MI","ILMI":"MI","AMI":"MI","ALMI":"MI","INJAS":"MI","LMI":"MI",
    "INJAL":"MI","IPLMI":"MI","IPMI":"MI","INJIN":"MI","INJLA":"MI","PMI":"MI","INJIL":"MI",
    "STD_":"STTC","ISCA":"STTC","ISCI":"STTC","ISC_":"STTC","INVT":"STTC",
    "LNGQT":"STTC","TAB_":"STTC","ANEUR":"STTC","EL":"STTC",
    "LAFB":"CD","IRBBB":"CD","CLBBB":"CD","CRBBB":"CD","LPFB":"CD",
    "WPW":"CD","IVCD":"CD","ILBBB":"CD","AVB":"CD","1AVB":"CD",
    "2AVB":"CD","3AVB":"CD","LBBB":"CD","RBBB":"CD",
    "LVH":"HYP","RVH":"HYP","SEHYP":"HYP","LAO/LAE":"HYP","RAO/RAE":"HYP","VCLVH":"HYP",
}
def parse_super(s):
    try: d = ast.literal_eval(s)
    except: return None
    best, bc = None, -1
    for code, conf in d.items():
        sp = SCP_TO_SUPER.get(code)
        if sp and conf > bc: best, bc = sp, conf
    return best
meta["superclass"] = meta["scp_codes"].apply(parse_super)
meta = meta.dropna(subset=["superclass"])
print(f"\nProcessing {len(meta)} patients …\n")


# ═══════════════════════════════════════════════════════════
# Signal helpers
# ═══════════════════════════════════════════════════════════
def normalize(sig):
    mu  = sig.mean(axis=1, keepdims=True)
    std = sig.std(axis=1, keepdims=True) + 1e-6
    return (sig - mu) / std

def load_signal(row):
    fname = str(row["filename_hr"]).replace("records500/", "")
    try:
        import wfdb
        sig, _ = wfdb.rdsamp(str(REC_ROOT / fname), channels=list(range(N_LEADS)))
        sig = np.nan_to_num(np.array(sig, dtype=np.float32).T, nan=0.0, posinf=0.0, neginf=0.0)
        return resample(sig, SEQ_LEN, axis=1).astype(np.float32)
    except: return None

def get_lead_influence(sig_norm):
    x_t    = torch.tensor(sig_norm[None], dtype=torch.float32, requires_grad=True)
    logits = model(x_t)
    pred_i = logits.argmax(dim=1).item()
    torch.softmax(logits, dim=1)[0, pred_i].backward()
    grads  = x_t.grad.detach().numpy()[0]
    return np.abs(grads).mean(axis=1), pred_i   # (12,)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
results: dict = {}
t0 = time.time()

for i, (_, row) in enumerate(meta.iterrows()):
    if i % 500 == 0:
        elapsed = time.time() - t0
        rate    = i / elapsed if elapsed > 0 else 0
        eta     = (len(meta) - i) / rate if rate > 0 else 0
        print(f"  {i:>6}/{len(meta)}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    ecg_id     = int(row["ecg_id"])
    true_class = str(row["superclass"])

    sig = load_signal(row)
    if sig is None: continue
    sig_norm = normalize(sig)

    # ── ResNet1D prediction ───────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(torch.tensor(sig_norm[None], dtype=torch.float32))
        probs  = torch.softmax(logits, dim=1).numpy()[0]
    pred_i     = int(np.argmax(probs))
    pred_class = CLASSES[pred_i]
    confidence = float(probs[pred_i])
    class_probs = {c: round(float(p), 4) for c, p in zip(CLASSES, probs)}

    # ── Gradient-based lead attribution ──────────────────────────────────────
    try:
        lead_inf, _ = get_lead_influence(sig_norm)
    except Exception:
        lead_inf = np.ones(N_LEADS) / N_LEADS
    lead_inf_norm = lead_inf / (lead_inf.sum() + 1e-9)

    # ── Clinical feature annotation ───────────────────────────────────────────
    clinical_abnormal: list      = []
    clinical_lead_top: dict      = {l: [] for l in LEADS_12}

    if feat_df is not None and ecg_id in feat_df.index:
        feat_row = feat_df.loc[ecg_id]
        if isinstance(feat_row, pd.DataFrame): feat_row = feat_row.iloc[0]
        X_raw = np.nan_to_num(feat_row.reindex(feature_cols).to_numpy(dtype=np.float32), nan=0.0)

        for col_i, col in enumerate(feature_cols):
            val    = float(X_raw[col_i])
            result = check_abnormal(col, val)
            if result is None: continue
            norm_str, status, unit = result
            lead    = lead_from_col(col)
            li      = LEADS_12.index(lead) if lead in LEADS_12 else -1
            attr_w  = round(float(lead_inf_norm[li]) if li >= 0 else 0.0, 6)
            entry = {
                "feature": col, "lead": lead, "label": feature_label(col),
                "value": round(val, 4), "normalRange": norm_str,
                "status": status, "unit": unit, "attr": attr_w,
            }
            clinical_lead_top[lead].append({"col":col,"label":feature_label(col),"attr":attr_w})
            if status != "Normal":
                clinical_abnormal.append(entry)

        clinical_abnormal.sort(key=lambda x: x["attr"], reverse=True)

    # ── Build top leads ───────────────────────────────────────────────────────
    top_leads_out = []
    for li, lead in enumerate(LEADS_12):
        influence     = round(float(lead_inf_norm[li]), 6)
        lead_abnormal = [a for a in clinical_abnormal if a["lead"] == lead]
        top_feats     = sorted(clinical_lead_top.get(lead, []),
                               key=lambda x: x["attr"], reverse=True)[:5]
        top_leads_out.append({
            "lead": lead, "influence": influence,
            "topFeatures": top_feats, "abnormalFeatures": lead_abnormal[:5],
        })
    top_leads_out.sort(key=lambda x: x["influence"], reverse=True)

    results[str(ecg_id)] = {
        "ecg_id": ecg_id, "trueClass": true_class,
        "predictedClass": pred_class,
        "predictedLabel": CONDITION_LABELS.get(pred_class, pred_class),
        "confidence": round(confidence, 4),
        "classProbabilities": class_probs,
        "topLeads": top_leads_out[:6],
        "allAbnormal": clinical_abnormal[:20],
    }

# ── Save ──────────────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=None))
total_time = time.time() - t0
correct = sum(1 for v in results.values() if v.get("trueClass") == v["predictedClass"])
print(f"\n✓ Saved {len(results)} records → {OUT}")
print(f"  Total time : {total_time:.0f}s  ({total_time/60:.1f} min)")
print(f"  Accuracy   : {correct/len(results):.4f}  ({correct}/{len(results)})")
