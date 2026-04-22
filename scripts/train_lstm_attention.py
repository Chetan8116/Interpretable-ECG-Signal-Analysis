"""
PTB-XL BiLSTM + Multi-head Attention + Extracted Features — Target: 85%+

Architecture rationale vs InceptionTime:
  - CNN stem (stride-8) reduces 1250 → 157 timesteps before LSTM
  - BiLSTM captures long-range ECG patterns (P→QRS→T wave relationships) in
    both temporal directions — something purely local convolutions miss
  - Multi-head self-attention (Transformer-style) learns which ECG segments
    are most diagnostic per class
  - Additive attention pooling focuses the time-dimension into a single vector
  - Feature MLP processes extracted features independently, fused at classifier

Key fixes vs previous runs:
  - Early stopping on macro-F1 (not accuracy) — avoids NORM-inflation bias
  - ReduceLROnPlateau on macro-F1 with factor=0.4, patience=10
  - Gradient clipping=0.5 (tighter → better long-range LSTM stability)
  - Weight decay=5e-4 + dropout=0.45 (stronger regularization)
  - Label smoothing=0.10 (reduces overconfident wrong predictions)
  - MixUp alpha=0.15 (light, preserves decision boundaries)

Usage:
    python scripts/train_lstm_attention.py
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
def find_project_root():
    script_path = Path(__file__).resolve()
    candidates = [
        script_path.parent.parent,
        script_path.parent,
        Path.cwd(),
        Path.home() / "Music",
        Path.home() / "Pictures" / "RM",
    ]
    for c in candidates:
        if (c / "ptbxl_comprehensive_features").exists():
            return c
        if (c / "archive").exists():
            if any("ptb-xl" in str(d).lower() for d in (c / "archive").iterdir()):
                return c
    return script_path.parent.parent if script_path.parent.name == "scripts" else script_path.parent

ROOT = find_project_root()
print(f"[paths] Project root: {ROOT}")

ARCHIVE = None
archive_dir = ROOT / "archive"
if archive_dir.exists():
    for sub in archive_dir.iterdir():
        if sub.is_dir() and "ptb-xl" in sub.name.lower():
            ARCHIVE = sub; break

if ARCHIVE is None:
    for p in [
        ROOT / "archive" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        ROOT / "archive",
    ]:
        if p.exists() and (p / "ptbxl_database.csv").exists():
            ARCHIVE = p; break

if ARCHIVE is None or not ARCHIVE.exists():
    print(f"\n⚠️  ERROR: PTB-XL dataset not found in {ROOT / 'archive'}"); import sys; sys.exit(1)

CSV_PATH  = ARCHIVE / "ptbxl_database.csv"
REC_ROOT  = ARCHIVE / "records500"
FEAT_DIR  = ROOT / "ptbxl_comprehensive_features"
ARTIFACTS = ROOT / "ECG_Diag_pipeline" / "artifacts"
PUBLIC    = ROOT / "public"

for p, name in [(CSV_PATH, "CSV"), (REC_ROOT, "Signals"), (FEAT_DIR, "Features")]:
    if not p.exists():
        print(f"\n⚠️  ERROR: {name} not found: {p}"); import sys; sys.exit(1)

ARTIFACTS.mkdir(parents=True, exist_ok=True)
PUBLIC.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
FS_TARGET = 125
SEQ_LEN   = 1250
N_LEADS   = 12
N_CLASSES = 5
CLASSES   = ["CD", "HYP", "MI", "NORM", "STTC"]

N_SELECTED_FEATS = 300

# BiLSTM + Attention architecture
CNN_CHANNELS  = [32, 64, 128]   # CNN stem output channels; last = LSTM input size
CNN_STRIDES   = [2,  2,  2]     # 1250 → 625 → 313 → 157 timesteps
LSTM_HIDDEN   = 128             # per direction; total = 256 bidirectional
LSTM_LAYERS   = 2
ATTN_HEADS    = 8               # MultiheadAttention heads
ATTN_DIM      = 256             # = LSTM_HIDDEN * 2
DROPOUT       = 0.45

# Training
BATCH_SIZE   = 64
MAX_EPOCHS   = 150
PATIENCE     = 40
LR_INIT      = 1e-3
WEIGHT_DECAY = 5e-4
FOCAL_GAMMA  = 2.0
LABEL_SMOOTH = 0.10
MIXUP_ALPHA  = 0.15
GRAD_CLIP    = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  FS={FS_TARGET}Hz  SEQ={SEQ_LEN}  N_FEATS={N_SELECTED_FEATS}")
print(f"[paths] CSV:      {CSV_PATH}")
print(f"[paths] Signals:  {REC_ROOT}")
print(f"[paths] Features: {FEAT_DIR}")
print(f"[paths] Output:   {ARTIFACTS}")


# ── SCP → superclass ──────────────────────────────────────────────────────────
SCP_TO_SUPER = {
    "NORM": "NORM",
    "IMI":"MI","ILMI":"MI","AMI":"MI","ALMI":"MI","INJAS":"MI","LMI":"MI",
    "INJAL":"MI","IPLMI":"MI","IPMI":"MI","INJIN":"MI","INJLA":"MI",
    "PMI":"MI","INJIL":"MI","INJA":"MI",
    "STD_":"STTC","ISCA":"STTC","ISCI":"STTC","ISC_":"STTC","INVT":"STTC",
    "NDT":"STTC","DIG":"STTC","LNGQT":"STTC","TAB_":"STTC","ANEUR":"STTC",
    "EL":"STTC","STTC":"STTC","STE_":"STTC",
    "LAFB":"CD","IRBBB":"CD","CLBBB":"CD","CRBBB":"CD","LPFB":"CD",
    "WPW":"CD","IVCD":"CD","ILBBB":"CD","AVB":"CD","1AVB":"CD",
    "2AVB":"CD","3AVB":"CD","LBBB":"CD","RBBB":"CD",
    "LVH":"HYP","RVH":"HYP","SEHYP":"HYP","LAO/LAE":"HYP","LAO":"HYP",
    "RAO/RAE":"HYP","RAO":"HYP","VCLVH":"HYP","LVOLT":"HYP","LMH":"HYP",
}
SKIP_KEYS = {"lead_name", "sampling_freq"}

def parse_superclass(scp_str: str):
    try: d = ast.literal_eval(scp_str)
    except: return None
    best, bc = None, -1
    for code, conf in d.items():
        sup = SCP_TO_SUPER.get(code)
        if sup and conf > bc: best, bc = sup, conf
    return best


# ── Feature loading ───────────────────────────────────────────────────────────
def flatten_record(feat_obj: dict) -> dict:
    flat: dict[str, float] = {}
    for lead_key, lead_feats in feat_obj.items():
        if not isinstance(lead_feats, dict): continue
        for fname, val in lead_feats.items():
            if fname in SKIP_KEYS: continue
            try: flat[f"{lead_key}_{fname}"] = float(val) if val is not None else np.nan
            except (TypeError, ValueError): flat[f"{lead_key}_{fname}"] = np.nan
    return flat

def load_features() -> pd.DataFrame:
    batch_files = sorted(FEAT_DIR.glob("batch_*_features.json"))
    print(f"  Found {len(batch_files)} batch files")
    records, t0, n_ok = [], time.time(), 0
    for bi, bf in enumerate(batch_files):
        try: batch = json.loads(bf.read_text(encoding="utf-8"))
        except Exception as e: print(f"  [!] {bf.name}: {e}"); continue
        if isinstance(batch, dict): batch = [batch]
        for rec in batch:
            if not rec.get("success", False): continue
            rid = rec.get("record_id"); feat = rec.get("features")
            if rid is None or feat is None: continue
            fl = flatten_record(feat if isinstance(feat, dict) else {})
            fl["record_id"] = int(rid)
            records.append(fl); n_ok += 1
        if (bi + 1) % 50 == 0:
            print(f"  [{bi+1:>3}/{len(batch_files)}] loaded={n_ok}  {time.time()-t0:.0f}s", flush=True)
    print(f"  Loaded {n_ok} feature records in {time.time()-t0:.0f}s")
    return pd.DataFrame(records)


# ── Signal loading ────────────────────────────────────────────────────────────
def load_signal(row) -> np.ndarray | None:
    fname = str(row["filename_hr"]).replace("records500/", "")
    try:
        import wfdb
        sig, _ = wfdb.rdsamp(str(REC_ROOT / fname), channels=list(range(N_LEADS)))
        sig = np.nan_to_num(np.array(sig, dtype=np.float32).T, nan=0.0, posinf=0.0, neginf=0.0)
        return resample(sig, SEQ_LEN, axis=1).astype(np.float32)
    except: return None


# ── Dataset ───────────────────────────────────────────────────────────────────
class HybridECGDataset(Dataset):
    def __init__(self, sigs, feats, labels, augment=False):
        self.sigs    = sigs
        self.feats   = feats
        self.labels  = np.array(labels, dtype=np.int64)
        self.augment = augment

    def __len__(self): return len(self.sigs)

    @staticmethod
    def _normalize(sig: np.ndarray) -> np.ndarray:
        mu  = sig.mean(axis=1, keepdims=True)
        std = sig.std(axis=1, keepdims=True) + 1e-6
        return (sig - mu) / std

    def _augment(self, sig: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.7:
            sig = sig + (np.random.randn(*sig.shape) * 0.02).astype(np.float32)
        if np.random.rand() < 0.6:
            sig = sig * float(np.random.uniform(0.85, 1.15))
        if np.random.rand() < 0.5:
            t  = np.linspace(0, 1, SEQ_LEN, dtype=np.float32)
            bw = (np.sin(2 * np.pi * np.random.uniform(0.05, 0.5) * t)
                  * np.random.uniform(0.01, 0.06)).astype(np.float32)
            sig = sig + bw
        if np.random.rand() < 0.5:
            sig = np.roll(sig, np.random.randint(-25, 26), axis=1)
        if np.random.rand() < 0.3:
            sig = sig.copy()
            sig[np.random.choice(N_LEADS, np.random.randint(1, 3), replace=False)] = 0.0
        if np.random.rand() < 0.4:
            win   = np.random.randint(50, 175)
            start = np.random.randint(0, max(1, SEQ_LEN - win))
            sig   = sig.copy()
            sig[:, start:start + win] = 0.0
        return sig

    def __getitem__(self, idx):
        sig = self.sigs[idx].copy()
        if self.augment:
            sig = self._augment(sig)
        sig = self._normalize(sig)
        return (torch.tensor(sig,             dtype=torch.float32),
                torch.tensor(self.feats[idx], dtype=torch.float32),
                int(self.labels[idx]))


# ── Architecture ──────────────────────────────────────────────────────────────
class CNNStem(nn.Module):
    """
    3-layer strided CNN to downsample temporal dimension before LSTM.
    1250 timesteps → 157 timesteps (stride-8 total), keeping spatial coherence.
    """
    def __init__(self, in_ch=12, channels=(32, 64, 128), strides=(2, 2, 2), dropout=0.1):
        super().__init__()
        layers, ch = [], in_ch
        kernels = [15, 7, 5]
        for out_ch, s, k in zip(channels, strides, kernels):
            layers += [
                nn.Conv1d(ch, out_ch, k, stride=s, padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            ch = out_ch
        self.net     = nn.Sequential(*layers)
        self.out_ch  = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, out_ch, T')


class AdditiveAttentionPool(nn.Module):
    """
    Learns a scalar "importance" score per timestep, then computes a
    weighted sum over time — equivalent to a single-query attention.
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        w = torch.softmax(self.score(x), dim=1)   # (B, T, 1)
        return (x * w).sum(dim=1)                  # (B, H)


class BiLSTMAttentionModel(nn.Module):
    """
    CNN Stem → BiLSTM → Multi-head Self-Attention → Additive Attention Pool
             → concat with Feature MLP → Classifier

    Signal:  (B, 12, 1250)  →  256-dim embedding
    Feature: (B, n_feat)    →  128-dim embedding
    Fusion:  384-dim        →  n_classes
    """
    def __init__(self, n_classes=5, n_features=300, dropout=0.45):
        super().__init__()

        # ── Signal branch ─────────────────────────────────────────────────
        self.cnn_stem = CNNStem(
            in_ch=N_LEADS,
            channels=tuple(CNN_CHANNELS),
            strides=tuple(CNN_STRIDES),
            dropout=0.1,
        )
        lstm_in = CNN_CHANNELS[-1]   # 128

        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if LSTM_LAYERS > 1 else 0.0,
        )

        # Multi-head self-attention over LSTM outputs
        self.self_attn  = nn.MultiheadAttention(
            embed_dim=ATTN_DIM, num_heads=ATTN_HEADS,
            dropout=0.1, batch_first=True,
        )
        self.attn_norm  = nn.LayerNorm(ATTN_DIM)

        # Additive attention for temporal pooling
        self.attn_pool  = AdditiveAttentionPool(ATTN_DIM)

        self.sig_head = nn.Sequential(
            nn.Linear(ATTN_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Feature branch ────────────────────────────────────────────────
        self.feat_head = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Fusion classifier ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, sig: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # CNN stem: (B, 12, 1250) → (B, 128, 157)
        x = self.cnn_stem(sig)
        x = x.permute(0, 2, 1)                 # (B, 157, 128)

        # BiLSTM: (B, 157, 128) → (B, 157, 256)
        lstm_out, _ = self.lstm(x)              # (B, T, 256)

        # Multi-head self-attention with residual + LayerNorm
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        attn_out    = self.attn_norm(attn_out + lstm_out)   # residual

        # Additive attention pooling: (B, T, 256) → (B, 256)
        pooled  = self.attn_pool(attn_out)      # (B, 256)
        sig_emb = self.sig_head(pooled)         # (B, 256)

        # Feature branch
        feat_emb = self.feat_head(feat)         # (B, 128)

        # Fusion
        return self.classifier(torch.cat([sig_emb, feat_emb], dim=1))


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, smooth=0.05, class_weights=None):
        super().__init__()
        self.gamma = gamma; self.smooth = smooth; self.weights = class_weights

    def forward(self, logits, targets):
        n     = logits.size(1)
        log_p = F.log_softmax(logits, dim=1)
        p     = log_p.exp()
        with torch.no_grad():
            st = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
            st = st * (1 - self.smooth) + self.smooth / n
        p_t   = (p * st).sum(1)
        loss  = ((1 - p_t) ** self.gamma) * (-(st * log_p).sum(1))
        if self.weights is not None:
            loss = loss * self.weights[targets]
        return loss.mean()


# ── MixUp ─────────────────────────────────────────────────────────────────────
def mixup_batch(sig, feat, y, alpha=0.15):
    if alpha <= 0:
        return sig, feat, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(sig.size(0), device=sig.device)
    return lam * sig + (1 - lam) * sig[idx], lam * feat + (1 - lam) * feat[idx], y, y[idx], lam

def mixup_criterion(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)


# ── Evaluation (returns macro-F1 and accuracy) ────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, n = 0.0, 0
    preds, trues = [], []
    for sig, feat, y in loader:
        sig, feat, y = sig.to(DEVICE), feat.to(DEVICE), y.to(DEVICE)
        logits  = model(sig, feat)
        total  += criterion(logits, y).item() * len(y)
        n      += len(y)
        preds.extend(logits.argmax(1).cpu().tolist())
        trues.extend(y.cpu().tolist())

    acc = sum(p == t for p, t in zip(preds, trues)) / len(preds)

    # Macro-F1
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(trues, preds):
        cm[t][p] += 1
    f1s = []
    for i in range(N_CLASSES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if (tp + fp) else 0
        re = tp / (tp + fn) if (tp + fn) else 0
        f1s.append(2 * pr * re / (pr + re) if (pr + re) else 0)
    macro_f1 = float(np.mean(f1s))

    return total / n, acc, macro_f1, preds, trues


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  PTB-XL BiLSTM + Multi-head Attention + Extracted Features")
    print("=" * 80)

    # ── [1] Metadata ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading metadata ...")
    df = pd.read_csv(CSV_PATH)
    df["superclass"] = df["scp_codes"].apply(parse_superclass)
    df = df.dropna(subset=["superclass"])
    df["label_idx"]  = df["superclass"].map({c: i for i, c in enumerate(CLASSES)})
    print(f"  Total records: {len(df)}")
    for c in CLASSES:
        nn_ = (df["superclass"] == c).sum()
        print(f"    {c}: {nn_} ({100*nn_/len(df):.1f}%)")

    df_train = df[df["strat_fold"].isin(range(1, 9))].reset_index(drop=True)
    df_val   = df[df["strat_fold"] == 9].reset_index(drop=True)
    df_test  = df[df["strat_fold"] == 10].reset_index(drop=True)
    print(f"\n  Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # ── [2] Features ──────────────────────────────────────────────────────────
    print("\n[2/5] Loading extracted features ...")
    feat_df = load_features()
    merged  = feat_df.merge(
        df[["ecg_id", "superclass", "strat_fold", "filename_hr", "label_idx"]],
        left_on="record_id", right_on="ecg_id", how="inner",
    )
    print(f"  Merged: {len(merged)} records with features")

    meta_cols = {"record_id", "ecg_id", "superclass", "strat_fold", "filename_hr", "label_idx"}
    feat_cols = [c for c in merged.columns if c not in meta_cols]
    X_raw = merged[feat_cols].values.astype(np.float32)
    print(f"  Raw features: {X_raw.shape[1]} dimensions")

    train_mask = merged["strat_fold"].isin(range(1, 9))
    val_mask   = merged["strat_fold"] == 9
    test_mask  = merged["strat_fold"] == 10

    # ── [3] Feature preprocessing ─────────────────────────────────────────────
    print("\n[3/5] Preprocessing features ...")
    X_tr_raw = X_raw[train_mask]; X_va_raw = X_raw[val_mask]; X_te_raw = X_raw[test_mask]

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr_raw)
    X_va = imputer.transform(X_va_raw)
    X_te = imputer.transform(X_te_raw)

    var_sel = VarianceThreshold(threshold=1e-6)
    X_tr = var_sel.fit_transform(X_tr)
    X_va = var_sel.transform(X_va)
    X_te = var_sel.transform(X_te)
    print(f"  After variance filter: {X_tr.shape[1]} features")

    p01 = np.percentile(X_tr, 1, axis=0); p99 = np.percentile(X_tr, 99, axis=0)
    X_tr = np.clip(X_tr, p01, p99); X_va = np.clip(X_va, p01, p99); X_te = np.clip(X_te, p01, p99)

    y_tr_enc = LabelEncoder().fit_transform(merged.loc[train_mask, "superclass"].values)
    print(f"  Selecting top {N_SELECTED_FEATS} features via mutual info ...")
    mi_sel = SelectKBest(mutual_info_classif, k=min(N_SELECTED_FEATS, X_tr.shape[1]))
    X_tr   = mi_sel.fit_transform(X_tr, y_tr_enc)
    X_va   = mi_sel.transform(X_va)
    X_te   = mi_sel.transform(X_te)
    print(f"  Final feature count: {X_tr.shape[1]}")

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_va   = scaler.transform(X_va).astype(np.float32)
    X_te   = scaler.transform(X_te).astype(np.float32)

    # ── [4] Signals ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Loading signals @ {FS_TARGET}Hz → {SEQ_LEN} samples ...")

    def load_split(split_df, name):
        sigs, labels, ids = [], [], []
        t0 = time.time()
        for i, (_, row) in enumerate(split_df.iterrows()):
            if i % 1000 == 0: print(f"  {name}: {i}/{len(split_df)}", flush=True)
            sig = load_signal(row)
            if sig is None: continue
            sigs.append(sig); labels.append(int(row["label_idx"])); ids.append(int(row["ecg_id"]))
        print(f"  {name}: {len(sigs)}/{len(split_df)} in {time.time()-t0:.0f}s")
        return sigs, labels, ids

    df_tr_m = merged[train_mask].reset_index(drop=True)
    df_va_m = merged[val_mask].reset_index(drop=True)
    df_te_m = merged[test_mask].reset_index(drop=True)

    tr_sigs, tr_labels, tr_ids = load_split(df_tr_m, "train")
    va_sigs, va_labels, va_ids = load_split(df_va_m, "val  ")
    te_sigs, te_labels, te_ids = load_split(df_te_m, "test ")

    tr_id2i = {row["ecg_id"]: i for i, row in df_tr_m.iterrows()}
    va_id2i = {row["ecg_id"]: i for i, row in df_va_m.iterrows()}
    te_id2i = {row["ecg_id"]: i for i, row in df_te_m.iterrows()}

    X_tr_f = X_tr[[tr_id2i[e] for e in tr_ids]]
    X_va_f = X_va[[va_id2i[e] for e in va_ids]]
    X_te_f = X_te[[te_id2i[e] for e in te_ids]]

    print(f"\n  Final train: {len(tr_sigs)} signals, {X_tr_f.shape[0]} features")
    print(f"  Final val  : {len(va_sigs)} signals, {X_va_f.shape[0]} features")
    print(f"  Final test : {len(te_sigs)} signals, {X_te_f.shape[0]} features")

    # ── [5] Datasets + loaders ────────────────────────────────────────────────
    print("\n[5/5] Creating datasets ...")
    ds_train = HybridECGDataset(tr_sigs, X_tr_f, tr_labels, augment=True)
    ds_val   = HybridECGDataset(va_sigs, X_va_f, va_labels, augment=False)
    ds_test  = HybridECGDataset(te_sigs, X_te_f, te_labels, augment=False)

    counts   = np.bincount(tr_labels, minlength=N_CLASSES).astype(float)
    w_class  = 1.0 / np.sqrt(np.maximum(counts, 1))
    w_sample = [float(w_class[l]) for l in tr_labels]
    sampler  = WeightedRandomSampler(w_sample, num_samples=len(w_sample), replacement=True)

    train_loader = DataLoader(ds_train, BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(ds_val,   BATCH_SIZE, shuffle=False,   num_workers=0, pin_memory=True)
    test_loader  = DataLoader(ds_test,  BATCH_SIZE, shuffle=False,   num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nInitializing BiLSTM+Attention model ...")
    model = BiLSTMAttentionModel(
        n_classes=N_CLASSES, n_features=X_tr_f.shape[1], dropout=DROPOUT
    ).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: BiLSTMAttentionModel  {n_p/1e6:.2f}M params  |  device={DEVICE}")

    # CNN stem output size info
    with torch.no_grad():
        dummy = torch.zeros(1, N_LEADS, SEQ_LEN, device=DEVICE)
        t_len = model.cnn_stem(dummy).shape[-1]
    print(f"  CNN stem: {SEQ_LEN} → {t_len} timesteps for LSTM")

    # ── Loss + optimizer ──────────────────────────────────────────────────────
    cw        = torch.tensor(w_class / w_class.sum() * N_CLASSES, dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, smooth=LABEL_SMOOTH, class_weights=cw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    # Monitor macro-F1 rather than accuracy (avoids NORM-class inflation)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.4, patience=10, min_lr=1e-6,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print("\nStarting training ...")
    print("  (Early stopping and LR scheduling on macro-F1, not raw accuracy)")
    best_f1    = 0.0
    best_path  = ARTIFACTS / "lstm_attention_best.pt"
    no_improve = 0

    print(f"\n{'Ep':>4}  {'LR':>9}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'VaF1':>7}  {'BestF1':>7}")
    print("─" * 80)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss, tr_ok, tr_n = 0.0, 0, 0
        t0 = time.time()

        for sig, feat, y in train_loader:
            sig, feat, y = sig.to(DEVICE), feat.to(DEVICE), y.to(DEVICE)

            m_sig, m_feat, ya, yb, lam = mixup_batch(sig, feat, y, MIXUP_ALPHA)
            optimizer.zero_grad()
            logits = model(m_sig, m_feat)
            loss   = mixup_criterion(criterion, logits, ya, yb, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            tr_loss += loss.item() * len(y)
            tr_ok   += (logits.argmax(1) == ya).sum().item()
            tr_n    += len(y)

        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(va_f1)              # LR reduction driven by macro-F1
        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        marker = ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "state_dict":   model.state_dict(),
                "classes":      CLASSES,
                "architecture": "BiLSTMAttentionModel",
                "seq_len":      SEQ_LEN,
                "fs":           FS_TARGET,
                "n_features":   X_tr_f.shape[1],
                "preprocessors": {
                    "imputer": imputer,
                    "var_sel": var_sel,
                    "mi_sel":  mi_sel,
                    "scaler":  scaler,
                    "clip":    (p01, p99),
                },
            }, best_path)
            no_improve = 0
            marker = " ✓"
        else:
            no_improve += 1

        print(f"{epoch:4d}  {lr_now:9.2e}  {tr_loss/tr_n:8.4f}  {tr_ok/tr_n:6.2%}  "
              f"{va_loss:8.4f}  {va_acc:6.2%}  {va_f1:6.3f}  {best_f1:6.3f}{marker}  [{elapsed:.0f}s]")

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} — {PATIENCE} epochs without improvement")
            break

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Loading best checkpoint for test evaluation ...")
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    _, te_acc, te_f1, preds, trues = evaluate(model, test_loader, criterion)
    print(f"\nTEST ACCURACY: {te_acc:.4f}  ({te_acc*100:.2f}%)")
    print(f"TEST MACRO-F1: {te_f1:.4f}  ({te_f1*100:.2f}%)")
    print("=" * 80)

    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(trues, preds): cm[t][p] += 1

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print("      " + "   ".join(f"{c:>5}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:4s}: " + "  ".join(f"{cm[i,j]:>6}" for j in range(N_CLASSES)))

    print("\nPer-class metrics:")
    print(f"{'Class':<6}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'Supp':>8}")
    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if (tp + fp) else 0
        re = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * pr * re / (pr + re) if (pr + re) else 0
        print(f"{c:<6}  {pr:8.3f}  {re:8.3f}  {f1:8.3f}  {cm[i,:].sum():8d}")
        per_class[c] = {"precision": round(pr,4), "recall": round(re,4),
                        "f1": round(f1,4), "support": int(cm[i,:].sum())}

    final_path = ARTIFACTS / "lstm_attention.pt"
    torch.save(ckpt, final_path)
    print(f"\n✓ Model saved → {final_path}")

    results = {
        "model":          "BiLSTMAttentionHybrid (signals + features)",
        "architecture":   "CNNStem + BiLSTM-256d + MultiheadAttn + FeatureMLP-128d",
        "classes":        CLASSES,
        "n_features":     int(X_tr_f.shape[1]),
        "val_macro_f1":   round(float(best_f1), 4),
        "test_accuracy":  round(float(te_acc), 4),
        "test_macro_f1":  round(float(te_f1), 4),
        "test_samples":   len(preds),
        "confusion_matrix": cm.tolist(),
        "per_class":      per_class,
    }
    (PUBLIC / "lstm_attention_results.json").write_text(json.dumps(results, indent=2))
    print(f"✓ Results → {PUBLIC / 'lstm_attention_results.json'}")
    print("\nBiLSTM+Attention training complete!")


if __name__ == "__main__":
    main()
