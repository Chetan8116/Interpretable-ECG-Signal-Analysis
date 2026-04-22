"""
PTB-XL InceptionTime + Extracted Features — Target: 85%+ accuracy

InceptionTime (Fawaz et al., 2020) is SOTA for time series classification.
Multi-scale inception modules capture patterns at multiple temporal scales
simultaneously, which is ideal for ECG morphology (P/QRS/T waves).

Key improvements over ResNet1D hybrid:
  - Parallel convolutions at 3 scales: 39, 19, 9 timesteps (0.31s / 0.15s / 0.07s)
  - Residual shortcuts every 3 inception modules
  - OneCycleLR: warm-up + cosine annealing (no premature LR collapse)
  - MixUp augmentation: better minority-class generalization
  - Patience=50 (more time to converge)

Architecture:
  Signals (12, 1250) → 2× InceptionBlock[3 modules] → GAP → 256-dim
  Features (300)     → Feature MLP                        → 128-dim
  Fusion  [256+128]  → Classifier                         → 5 classes

Usage:
    python scripts/train_inceptiontime.py
"""

from __future__ import annotations

import ast, json, os, time, warnings
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
    print(f"\n⚠️  ERROR: PTB-XL dataset not found in {ROOT / 'archive'}")
    import sys; sys.exit(1)

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
FS_ORIG   = 500
FS_TARGET = 125
SEQ_LEN   = 1250
N_LEADS   = 12
N_CLASSES = 5
CLASSES   = ["CD", "HYP", "MI", "NORM", "STTC"]

N_SELECTED_FEATS = 300

# InceptionTime architecture
NB_FILTERS   = 128          # Filters per kernel (out_ch = 128*4 = 512)
KERNEL_SIZES = [39, 19, 9]  # Multi-scale: 0.31s, 0.15s, 0.07s at 125 Hz
BOTTLENECK   = 32           # Bottleneck before large convolutions
DEPTH        = 9            # Total inception modules (3 blocks × 3)

# Training
BATCH_SIZE   = 64
MAX_EPOCHS   = 200
PATIENCE     = 65           # Enough for ≥2 full cosine cycles
LR_INIT      = 5e-4
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA  = 2.0
LABEL_SMOOTH = 0.05
MIXUP_ALPHA  = 0.1          # Reduced: less aggressive interpolation

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
        # Gaussian noise
        if np.random.rand() < 0.7:
            sig = sig + (np.random.randn(*sig.shape) * 0.025).astype(np.float32)
        # Amplitude scaling
        if np.random.rand() < 0.6:
            sig = sig * float(np.random.uniform(0.8, 1.2))
        # Baseline wander
        if np.random.rand() < 0.5:
            t  = np.linspace(0, 1, SEQ_LEN, dtype=np.float32)
            bw = (np.sin(2 * np.pi * np.random.uniform(0.05, 0.5) * t)
                  * np.random.uniform(0.01, 0.08)).astype(np.float32)
            sig = sig + bw
        # Time shift
        if np.random.rand() < 0.5:
            sig = np.roll(sig, np.random.randint(-25, 26), axis=1)
        # Lead dropout
        if np.random.rand() < 0.3:
            k = np.random.randint(1, 3)
            sig = sig.copy()
            sig[np.random.choice(N_LEADS, k, replace=False)] = 0.0
        # Cutout window
        if np.random.rand() < 0.5:
            win = np.random.randint(50, 200)
            start = np.random.randint(0, max(1, SEQ_LEN - win))
            sig = sig.copy()
            sig[:, start:start + win] = 0.0
        return sig

    def __getitem__(self, idx):
        sig = self.sigs[idx].copy()
        if self.augment:
            sig = self._augment(sig)
        sig = self._normalize(sig)
        return (torch.tensor(sig,              dtype=torch.float32),
                torch.tensor(self.feats[idx],  dtype=torch.float32),
                int(self.labels[idx]))


# ── InceptionTime Architecture ────────────────────────────────────────────────
class SEBlock1D(nn.Module):
    """Squeeze-and-Excite channel attention for 1D signals."""
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, max(ch // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(ch // reduction, 4), ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1)


class InceptionModule(nn.Module):
    """
    Single inception module with SE channel attention.
    Output channels = nb_filters * (len(kernel_sizes) + 1)
    e.g. 128 * (3 + 1) = 512 with default settings.
    """
    def __init__(self, in_ch: int, nb_filters: int = 128, bottleneck: int = 32,
                 kernel_sizes: list[int] = None, dropout: float = 0.0):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [39, 19, 9]

        # Bottleneck (1×1) reduces channels before expensive large convolutions
        bn_ch = min(bottleneck, in_ch) if in_ch <= bottleneck else bottleneck
        self.bottleneck = nn.Conv1d(in_ch, bn_ch, 1, bias=False)

        # Parallel multi-scale convolutions on bottleneck output
        self.convs = nn.ModuleList([
            nn.Conv1d(bn_ch, nb_filters, k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])

        # MaxPool path preserves local-extrema information
        self.mp      = nn.MaxPool1d(3, stride=1, padding=1)
        self.mp_conv = nn.Conv1d(in_ch, nb_filters, 1, bias=False)

        out_ch = nb_filters * (len(kernel_sizes) + 1)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU()
        self.se   = SEBlock1D(out_ch)          # Channel attention
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bn = self.bottleneck(x)
        outs = [conv(x_bn) for conv in self.convs]
        outs.append(self.mp_conv(self.mp(x)))
        out = self.se(self.act(self.bn(torch.cat(outs, dim=1))))
        return self.drop(out)


class InceptionBlock(nn.Module):
    """
    3 stacked inception modules with a residual shortcut (original InceptionTime design).
    """
    def __init__(self, in_ch: int, nb_filters: int = 64, bottleneck: int = 32,
                 kernel_sizes: list[int] = None, dropout: float = 0.1):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [39, 19, 9]
        out_ch = nb_filters * (len(kernel_sizes) + 1)

        self.inc1 = InceptionModule(in_ch,  nb_filters, bottleneck, kernel_sizes, dropout)
        self.inc2 = InceptionModule(out_ch, nb_filters, bottleneck, kernel_sizes, dropout)
        self.inc3 = InceptionModule(out_ch, nb_filters, bottleneck, kernel_sizes, dropout)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.inc1(x)
        out = self.inc2(out)
        out = self.inc3(out)
        return self.act(out + res)


class InceptionTimeHybrid(nn.Module):
    """
    InceptionTime+SE signal backbone fused with extracted features.

    Signal path : (B, 12, 1250) → N InceptionBlocks(SE) → GAP → Linear → 256-dim
    Feature path: (B, n_feat)   → MLP                               → 128-dim
    Fusion      : concat [256, 128] → classifier → n_classes
    """
    def __init__(self, n_classes: int = 5, n_features: int = 300,
                 nb_filters: int = 64, depth: int = 6,
                 bottleneck: int = 32, kernel_sizes: list[int] = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [39, 19, 9]

        out_ch   = nb_filters * (len(kernel_sizes) + 1)  # 256 with defaults
        n_blocks = depth // 3                             # depth=6 → 2 blocks

        blocks, in_ch = [], N_LEADS
        for i in range(n_blocks):
            drop = 0.05 if i == 0 else 0.1
            blocks.append(InceptionBlock(in_ch, nb_filters, bottleneck, kernel_sizes, drop))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.gap      = nn.AdaptiveAvgPool1d(1)

        # Signal embedding head
        self.sig_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_ch, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # Feature MLP
        self.feat_head = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, sig: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        sig_emb  = self.sig_head(self.gap(self.backbone(sig)))
        feat_emb = self.feat_head(feat)
        return self.classifier(torch.cat([sig_emb, feat_emb], dim=1))


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, smooth: float = 0.05,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma; self.smooth = smooth; self.weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.size(1)
        log_p = F.log_softmax(logits, dim=1)
        p     = log_p.exp()
        with torch.no_grad():
            smooth_t = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
            smooth_t = smooth_t * (1 - self.smooth) + self.smooth / n
        p_t   = (p * smooth_t).sum(1)
        focal = (1 - p_t) ** self.gamma
        ce    = -(smooth_t * log_p).sum(1)
        loss  = focal * ce
        if self.weights is not None:
            loss = loss * self.weights[targets]
        return loss.mean()


# ── MixUp ─────────────────────────────────────────────────────────────────────
def mixup_batch(sig, feat, y, alpha: float = 0.3):
    """Linear interpolation of two random samples in a batch."""
    if alpha <= 0:
        return sig, feat, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(sig.size(0), device=sig.device)
    return (lam * sig  + (1 - lam) * sig[idx],
            lam * feat + (1 - lam) * feat[idx],
            y, y[idx], lam)

def mixup_criterion(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)


# ── Evaluation ────────────────────────────────────────────────────────────────
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
    return total / n, acc, preds, trues


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  PTB-XL InceptionTime + Extracted Features")
    print("=" * 80)

    # ── [1] Metadata ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading metadata ...")
    df = pd.read_csv(CSV_PATH)
    df["superclass"] = df["scp_codes"].apply(parse_superclass)
    df = df.dropna(subset=["superclass"])
    df["label_idx"]  = df["superclass"].map({c: i for i, c in enumerate(CLASSES)})
    print(f"  Total records: {len(df)}")
    for c in CLASSES:
        n = (df["superclass"] == c).sum()
        print(f"    {c}: {n} ({100*n/len(df):.1f}%)")

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

    X_tr_raw = X_raw[train_mask]
    X_va_raw = X_raw[val_mask]
    X_te_raw = X_raw[test_mask]

    # ── [3] Feature preprocessing ─────────────────────────────────────────────
    print("\n[3/5] Preprocessing features ...")

    imputer = SimpleImputer(strategy="median")
    X_tr    = imputer.fit_transform(X_tr_raw)
    X_va    = imputer.transform(X_va_raw)
    X_te    = imputer.transform(X_te_raw)

    var_sel = VarianceThreshold(threshold=1e-6)
    X_tr    = var_sel.fit_transform(X_tr)
    X_va    = var_sel.transform(X_va)
    X_te    = var_sel.transform(X_te)
    print(f"  After variance filter: {X_tr.shape[1]} features")

    p01  = np.percentile(X_tr, 1,  axis=0)
    p99  = np.percentile(X_tr, 99, axis=0)
    X_tr = np.clip(X_tr, p01, p99)
    X_va = np.clip(X_va, p01, p99)
    X_te = np.clip(X_te, p01, p99)

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
            if i % 1000 == 0:
                print(f"  {name}: {i}/{len(split_df)}", flush=True)
            sig = load_signal(row)
            if sig is None: continue
            sigs.append(sig)
            labels.append(int(row["label_idx"]))
            ids.append(int(row["ecg_id"]))
        print(f"  {name}: {len(sigs)}/{len(split_df)} in {time.time()-t0:.0f}s")
        return sigs, labels, ids

    df_tr_m = merged[train_mask].reset_index(drop=True)
    df_va_m = merged[val_mask].reset_index(drop=True)
    df_te_m = merged[test_mask].reset_index(drop=True)

    tr_sigs, tr_labels, tr_ids = load_split(df_tr_m, "train")
    va_sigs, va_labels, va_ids = load_split(df_va_m, "val  ")
    te_sigs, te_labels, te_ids = load_split(df_te_m, "test ")

    # Align features with successfully loaded signals
    tr_id2i = {row["ecg_id"]: i for i, row in df_tr_m.iterrows()}
    va_id2i = {row["ecg_id"]: i for i, row in df_va_m.iterrows()}
    te_id2i = {row["ecg_id"]: i for i, row in df_te_m.iterrows()}

    X_tr_f = X_tr[[tr_id2i[eid] for eid in tr_ids]]
    X_va_f = X_va[[va_id2i[eid] for eid in va_ids]]
    X_te_f = X_te[[te_id2i[eid] for eid in te_ids]]

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
    print("\nInitializing InceptionTime model ...")
    model = InceptionTimeHybrid(
        n_classes=N_CLASSES, n_features=X_tr_f.shape[1],
        nb_filters=NB_FILTERS, depth=DEPTH,
        bottleneck=BOTTLENECK, kernel_sizes=KERNEL_SIZES,
    ).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: InceptionTimeHybrid  {n_p/1e6:.2f}M params  |  device={DEVICE}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    cw        = torch.tensor(w_class / w_class.sum() * N_CLASSES, dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, smooth=LABEL_SMOOTH, class_weights=cw)

    # ── Optimizer + CosineAnnealingWarmRestarts ──────────────────────────────
    # T_0=30: first cycle 30 epochs, T_mult=2: each restart doubles length
    # → cycles at 30, 60, 120 epochs → model gets multiple fine-tuning phases
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print("\nStarting training ...")
    best_val   = 0.0
    best_path  = ARTIFACTS / "inceptiontime_best.pt"
    no_improve = 0

    print(f"\n{'Ep':>4}  {'LR':>9}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'Best':>7}")
    print("─" * 72)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * len(y)
            tr_ok   += (logits.argmax(1) == ya).sum().item()
            tr_n    += len(y)

        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(epoch - 1)          # CosineAnnealingWarmRestarts: step per epoch
        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        marker = ""
        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "state_dict":   model.state_dict(),
                "classes":      CLASSES,
                "architecture": "InceptionTimeHybrid",
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
              f"{va_loss:8.4f}  {va_acc:6.2%}  {best_val:6.2%}{marker}  [{elapsed:.0f}s]")

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} — {PATIENCE} epochs without improvement")
            break

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Loading best checkpoint for test evaluation ...")
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    _, te_acc, preds, trues = evaluate(model, test_loader, criterion)
    print(f"\nTEST ACCURACY: {te_acc:.4f}  ({te_acc*100:.2f}%)")
    print("=" * 80)

    # Confusion matrix
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(trues, preds):
        cm[t][p] += 1

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print("      " + "   ".join(f"{c:>5}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:4s}: " + "  ".join(f"{cm[i,j]:>6}" for j in range(N_CLASSES)))

    print("\nPer-class metrics:")
    print(f"{'Class':<6}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'Supp':>8}")
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0
        re = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * re / (pr + re) if pr + re else 0
        print(f"{c:<6}  {pr:8.3f}  {re:8.3f}  {f1:8.3f}  {cm[i,:].sum():8d}")

    # ── Save results ──────────────────────────────────────────────────────────
    final_path = ARTIFACTS / "inceptiontime_hybrid.pt"
    torch.save(ckpt, final_path)
    print(f"\n✓ Model saved → {final_path}")

    results = {
        "model":          "InceptionTimeHybrid v2 (signals + features)",
        "architecture":   "InceptionTime-SE-512d + FeatureMLP-128d + Classifier",
        "classes":        CLASSES,
        "n_features":     int(X_tr_f.shape[1]),
        "val_accuracy":   round(float(best_val), 4),
        "test_accuracy":  round(float(te_acc), 4),
        "test_samples":   len(preds),
        "confusion_matrix": cm.tolist(),
        "per_class":      {},
    }
    for i, c in enumerate(CLASSES):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp); fn = int(cm[i, :].sum() - tp)
        pr = tp / (tp + fp) if tp + fp else 0
        re = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * re / (pr + re) if pr + re else 0
        results["per_class"][c] = {
            "precision": round(pr, 4),
            "recall":    round(re, 4),
            "f1":        round(f1, 4),
            "support":   int(cm[i, :].sum()),
        }

    (PUBLIC / "inceptiontime_results.json").write_text(json.dumps(results, indent=2))
    print(f"✓ Results → {PUBLIC / 'inceptiontime_results.json'}")
    print("\nInceptionTime training complete!")


if __name__ == "__main__":
    main()
