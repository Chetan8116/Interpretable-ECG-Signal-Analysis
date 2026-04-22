"""
PTB-XL Hybrid ResNet1D + Extracted Features — Target: 88%+ accuracy

Combines the best of both worlds:
  1. ResNet1D on raw ECG signals  → learns temporal patterns (P/QRS/T morphology)
  2. Extracted features (720 dims) → statistical/morphological insights
  3. Fusion layer → final classification

Architecture:
  - Raw signals (12, 1250) → ResNet1D → 256-dim embedding
  - Extracted features (720) → Feature MLP → 128-dim embedding
  - Concatenate [256 + 128] → Fusion MLP → 5 classes

Expected improvement: +3-7% over pure ResNet1D (85% → 88-92%)

Usage:
    python scripts/train_resnet1d_hybrid.py
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
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

# ── Paths (flexible for different systems) ───────────────────────────────────
def find_project_root():
    """Find project root by looking for key directories."""
    script_path = Path(__file__).resolve()
    
    # Try different possible locations
    candidates = [
        script_path.parent.parent,  # Windows: scripts/ in project
        script_path.parent,         # Linux: script copied to project root
        Path.cwd(),                 # Current working directory
        Path.home() / "Music",      # Linux common location
        Path.home() / "Pictures" / "RM",  # Windows location
    ]
    
    for candidate in candidates:
        # Check if this looks like the project root
        if (candidate / "ptbxl_comprehensive_features").exists():
            return candidate
        # Check if archive folder exists
        if (candidate / "archive").exists():
            archive_contents = list((candidate / "archive").iterdir())
            if any("ptb-xl" in str(d).lower() for d in archive_contents):
                return candidate
    
    # Fallback to parent of script
    return script_path.parent.parent if script_path.parent.name == "scripts" else script_path.parent

ROOT = find_project_root()
print(f"[paths] Project root: {ROOT}")

# Find archive directory (handle different naming)
ARCHIVE = None
archive_dir = ROOT / "archive"
if archive_dir.exists():
    # Look for ptb-xl dataset folder
    for subdir in archive_dir.iterdir():
        if subdir.is_dir() and "ptb-xl" in subdir.name.lower():
            ARCHIVE = subdir
            break

if ARCHIVE is None:
    # Try direct archive path
    possible_archives = [
        ROOT / "archive" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        ROOT / "archive",
    ]
    for path in possible_archives:
        if path.exists() and (path / "ptbxl_database.csv").exists():
            ARCHIVE = path
            break

if ARCHIVE is None or not ARCHIVE.exists():
    print(f"\n⚠️  ERROR: Could not find PTB-XL dataset in {ROOT / 'archive'}")
    print("Expected structure:")
    print("  archive/")
    print("    ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/")
    print("      ptbxl_database.csv")
    print("      records500/")
    print("\nPlease ensure the dataset is extracted in the correct location.")
    import sys
    sys.exit(1)

CSV_PATH = ARCHIVE / "ptbxl_database.csv"
REC_ROOT = ARCHIVE / "records500"
FEAT_DIR = ROOT / "ptbxl_comprehensive_features"
ARTIFACTS = ROOT / "ECG_Diag_pipeline" / "artifacts"
PUBLIC = ROOT / "public"

# Verify critical paths exist
if not CSV_PATH.exists():
    print(f"\n⚠️  ERROR: CSV file not found: {CSV_PATH}")
    import sys
    sys.exit(1)

if not REC_ROOT.exists():
    print(f"\n⚠️  ERROR: Signal directory not found: {REC_ROOT}")
    import sys
    sys.exit(1)

if not FEAT_DIR.exists():
    print(f"\n⚠️  ERROR: Feature directory not found: {FEAT_DIR}")
    print("Please run feature extraction first (see FEATURE_EXTRACTION_GUIDE.md)")
    import sys
    sys.exit(1)

ARTIFACTS.mkdir(parents=True, exist_ok=True)
PUBLIC.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
FS_ORIG     = 500
FS_TARGET   = 125           # 125 Hz → 1250 samples / 10 s
SEQ_LEN     = 1250
N_LEADS     = 12
N_CLASSES   = 5
CLASSES     = ["CD", "HYP", "MI", "NORM", "STTC"]

# Feature selection
N_SELECTED_FEATS = 300      # Reduced to reduce overfitting

BATCH_SIZE   = 64           # Increased from 32 to 64
MAX_EPOCHS   = 150          # More epochs
PATIENCE     = 30           # More patience
LR_INIT      = 4e-4         # Back to 4e-4
LR_MIN       = 1e-6         # Back to original
WEIGHT_DECAY = 3e-4         # More regularization
FOCAL_GAMMA  = 2.5          # Back to original
LABEL_SMOOTH = 0.08         # Back to original

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  FS={FS_TARGET}Hz  SEQ={SEQ_LEN}  N_FEATS={N_SELECTED_FEATS}")
print(f"[paths] CSV:      {CSV_PATH}")
print(f"[paths] Signals:  {REC_ROOT}")
print(f"[paths] Features: {FEAT_DIR}")
print(f"[paths] Output:   {ARTIFACTS}")


# ═══════════════════════════════════════════════════════════════════════════════
# SCP → superclass mapping
# ═══════════════════════════════════════════════════════════════════════════════
SCP_TO_SUPER = {
    "NORM":"NORM",
    "IMI":"MI","ILMI":"MI","AMI":"MI","ALMI":"MI","INJAS":"MI","LMI":"MI",
    "INJAL":"MI","IPLMI":"MI","IPMI":"MI","INJIN":"MI","INJLA":"MI","PMI":"MI","INJIL":"MI","INJA":"MI",
    "STD_":"STTC","ISCA":"STTC","ISCI":"STTC","ISC_":"STTC","INVT":"STTC","NDT":"STTC","DIG":"STTC",
    "LNGQT":"STTC","TAB_":"STTC","ANEUR":"STTC","EL":"STTC","STTC":"STTC","STE_":"STTC",
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


# ═══════════════════════════════════════════════════════════════════════════════
# Feature loading
# ═══════════════════════════════════════════════════════════════════════════════
def flatten_record(features_obj: dict) -> dict:
    flat: dict[str, float] = {}
    for lead_key, lead_feats in features_obj.items():
        if not isinstance(lead_feats, dict):
            continue
        for fname, val in lead_feats.items():
            if fname in SKIP_KEYS:
                continue
            col = f"{lead_key}_{fname}"
            try:
                flat[col] = float(val) if val is not None else np.nan
            except (TypeError, ValueError):
                flat[col] = np.nan
    return flat


def load_features() -> pd.DataFrame:
    batch_files = sorted(FEAT_DIR.glob("batch_*_features.json"))
    print(f"  Found {len(batch_files)} batch files")
    records, t0, n_ok = [], time.time(), 0
    for bi, bf in enumerate(batch_files):
        try:
            batch = json.loads(bf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [!] {bf.name}: {e}"); continue
        if isinstance(batch, dict):
            batch = [batch]
        for rec in batch:
            if not rec.get("success", False): continue
            rid  = rec.get("record_id")
            feat = rec.get("features")
            if rid is None or feat is None: continue
            fl = flatten_record(feat if isinstance(feat, dict) else {})
            fl["record_id"] = int(rid)
            records.append(fl)
            n_ok += 1
        if (bi + 1) % 50 == 0:
            print(f"  [{bi+1:>3}/{len(batch_files)}] loaded={n_ok}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"  Loaded {n_ok} feature records in {time.time()-t0:.0f}s")
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Signal loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_signal(row) -> np.ndarray | None:
    fname = str(row["filename_hr"]).replace("records500/", "")
    try:
        import wfdb
        sig, _ = wfdb.rdsamp(str(REC_ROOT / fname), channels=list(range(N_LEADS)))
        sig = np.array(sig, dtype=np.float32).T                              # (12, 5000)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        sig = resample(sig, SEQ_LEN, axis=1).astype(np.float32)              # (12, 1250)
        return sig
    except: return None


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid Dataset: signals + features
# ═══════════════════════════════════════════════════════════════════════════════
class HybridECGDataset(Dataset):
    def __init__(self, sigs, feats, labels, augment=False):
        """
        sigs: list of (12, 1250) arrays
        feats: (N, D) numpy array of extracted features
        labels: (N,) array
        """
        self.sigs    = sigs
        self.feats   = feats
        self.labels  = np.array(labels, dtype=np.int64)
        self.augment = augment

    def __len__(self): return len(self.sigs)

    @staticmethod
    def _normalize_sig(sig):
        mu  = sig.mean(axis=1, keepdims=True)
        std = sig.std(axis=1, keepdims=True) + 1e-6
        return (sig - mu) / std

    def _augment_sig(self, sig):
        # Enhanced augmentation with cutout for regularization
        if np.random.rand() < 0.7:
            sig = sig + (np.random.randn(*sig.shape) * 0.025).astype(np.float32)
        if np.random.rand() < 0.6:
            sig = sig * float(np.random.uniform(0.8, 1.2))
        if np.random.rand() < 0.5:
            t   = np.linspace(0, 1, SEQ_LEN, dtype=np.float32)
            bw  = (np.sin(2 * np.pi * np.random.uniform(0.1, 0.5) * t)
                   * np.random.uniform(0.02, 0.10)).astype(np.float32)
            sig = sig + bw
        if np.random.rand() < 0.5:
            shift = np.random.randint(-12, 13)
            sig   = np.roll(sig, shift, axis=1)
        if np.random.rand() < 0.2:
            li  = np.random.randint(N_LEADS)
            sig = sig.copy()
            sig[li] = -sig[li]
        if np.random.rand() < 0.3:
            k     = np.random.randint(1, 3)
            leads = np.random.choice(N_LEADS, k, replace=False)
            sig   = sig.copy()
            sig[leads] = 0.0
        # Cutout: zero out a random time window
        if np.random.rand() < 0.4:
            win_size = np.random.randint(50, 200)
            start = np.random.randint(0, max(1, SEQ_LEN - win_size))
            sig = sig.copy()
            sig[:, start:start+win_size] = 0.0
        return sig

    def __getitem__(self, idx):
        sig = self.sigs[idx].copy()
        if self.augment:
            sig = self._augment_sig(sig)
        sig = self._normalize_sig(sig)
        
        feat = self.feats[idx]  # Already preprocessed
        
        return (torch.tensor(sig, dtype=torch.float32),
                torch.tensor(feat, dtype=torch.float32),
                int(self.labels[idx]))


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid Model: ResNet1D + Feature MLP
# ═══════════════════════════════════════════════════════════════════════════════
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1    = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.drop     = nn.Dropout(dropout)
        self.shortcut = (nn.Sequential(
                            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                            nn.BatchNorm1d(out_ch))
                         if (in_ch != out_ch or stride != 1) else nn.Identity())

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class HybridResNet1D(nn.Module):
    """
    Simplified Hybrid model with better regularization:
      - Signal path: ResNet1D → 256-dim embedding
      - Feature path: MLP → 128-dim embedding  
      - Fusion: Simple concat + classifier
    """
    def __init__(self, n_classes=5, n_features=300):
        super().__init__()
        
        # ── Signal processing branch ──────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResBlock1D(32,  64, 7, 1, 0.15), ResBlock1D(64,  64, 7, 1, 0.15))
        self.layer2 = nn.Sequential(ResBlock1D(64, 128, 7, 2, 0.15), ResBlock1D(128,128, 7, 1, 0.15))
        self.layer3 = nn.Sequential(ResBlock1D(128,256, 7, 2, 0.25), ResBlock1D(256,256, 7, 1, 0.25))
        self.layer4 = nn.Sequential(ResBlock1D(256,256, 7, 2, 0.25), ResBlock1D(256,256, 7, 1, 0.25))
        self.gap    = nn.AdaptiveAvgPool1d(1)
        
        # Signal embedding
        self.signal_embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # ── Feature processing branch ────────────────────────────────────
        self.feature_embed = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # ── Classifier ───────────────────────────────────────────────────
        # 256 (signal) + 128 (features) = 384
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, sig, feat):
        # Signal path
        x = self.stem(sig)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        sig_emb = self.signal_embed(x)
        
        # Feature path
        feat_emb = self.feature_embed(feat)
        
        # Concatenate embeddings
        combined = torch.cat([sig_emb, feat_emb], dim=1)
        
        # Classification
        return self.classifier(combined)


# ═══════════════════════════════════════════════════════════════════════════════
# Focal Loss
# ═══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """Multi-class focal loss with optional label smoothing."""
    def __init__(self, gamma: float = 2.0, smooth: float = 0.05,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.gamma   = gamma
        self.smooth  = smooth
        self.weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.size(1)
        log_p  = F.log_softmax(logits, dim=1)
        p      = log_p.exp()

        # Label smoothing
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1.0)
            smooth_targets = smooth_targets * (1 - self.smooth) + self.smooth / n

        # Focal weight: (1 - p_t)^gamma
        p_t    = (p * smooth_targets).sum(dim=1)
        focal  = (1 - p_t) ** self.gamma

        # CE with smooth targets
        ce     = -(smooth_targets * log_p).sum(dim=1)
        loss   = focal * ce

        if self.weights is not None:
            w = self.weights[targets]
            loss = loss * w

        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []

    for sig, feat, y in loader:
        sig  = sig.to(DEVICE)
        feat = feat.to(DEVICE)
        y    = y.to(DEVICE)
        
        logits = model(sig, feat)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        n          += len(y)

        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_preds)
    return total_loss / n, acc, all_preds, all_labels


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("  PTB-XL Hybrid ResNet1D + Extracted Features")
    print("=" * 80)
    
    # ── Load metadata ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading metadata ...")
    df = pd.read_csv(CSV_PATH)
    df["superclass"] = df["scp_codes"].apply(parse_superclass)
    df = df.dropna(subset=["superclass"])
    df["label_idx" ] = df["superclass"].map({c: i for i, c in enumerate(CLASSES)})
    print(f"  Total records: {len(df)}")
    for c in CLASSES:
        n = (df["superclass"] == c).sum()
        print(f"    {c}: {n} ({100*n/len(df):.1f}%)")

    # Split by strat_fold
    df_train = df[df["strat_fold"].isin(range(1, 9))].reset_index(drop=True)
    df_val   = df[df["strat_fold"] == 9].reset_index(drop=True)
    df_test  = df[df["strat_fold"] == 10].reset_index(drop=True)
    print(f"\n  Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # ── Load extracted features ───────────────────────────────────────────────
    print("\n[2/5] Loading extracted features ...")
    feat_df = load_features()
    
    # Merge with labels
    merged = feat_df.merge(df[["ecg_id", "superclass", "strat_fold", "filename_hr", "label_idx"]], 
                           left_on="record_id", right_on="ecg_id", how="inner")
    print(f"  Merged: {len(merged)} records with features")
    
    # Prepare feature matrix
    meta_cols = {"record_id", "ecg_id", "superclass", "strat_fold", "filename_hr", "label_idx"}
    feat_cols = [c for c in merged.columns if c not in meta_cols]
    X_raw = merged[feat_cols].values.astype(np.float32)
    print(f"  Raw features: {X_raw.shape[1]} dimensions")
    
    # Split by strat_fold
    train_mask = merged["strat_fold"].isin(range(1, 9))
    val_mask   = merged["strat_fold"] == 9
    test_mask  = merged["strat_fold"] == 10
    
    X_tr_raw = X_raw[train_mask]
    X_va_raw = X_raw[val_mask]
    X_te_raw = X_raw[test_mask]
    
    # Preprocessing pipeline
    print("\n[3/5] Preprocessing features ...")
    
    # Impute
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr_raw)
    X_va = imputer.transform(X_va_raw)
    X_te = imputer.transform(X_te_raw)
    
    # Variance threshold
    var_sel = VarianceThreshold(threshold=1e-6)
    X_tr = var_sel.fit_transform(X_tr)
    X_va = var_sel.transform(X_va)
    X_te = var_sel.transform(X_te)
    print(f"  After variance filter: {X_tr.shape[1]} features")
    
    # Clip outliers
    p01 = np.percentile(X_tr, 1, axis=0)
    p99 = np.percentile(X_tr, 99, axis=0)
    X_tr = np.clip(X_tr, p01, p99)
    X_va = np.clip(X_va, p01, p99)
    X_te = np.clip(X_te, p01, p99)
    
    # Feature selection via mutual information
    y_tr_temp = merged.loc[train_mask, "superclass"].values
    le_temp = LabelEncoder()
    y_tr_enc_temp = le_temp.fit_transform(y_tr_temp)
    
    # Feature selection via mutual information (simpler)
    print(f"  Selecting top {N_SELECTED_FEATS} features via mutual info ...")
    mi_sel = SelectKBest(mutual_info_classif, k=min(N_SELECTED_FEATS, X_tr.shape[1]))
    X_tr = mi_sel.fit_transform(X_tr, y_tr_enc_temp)
    X_va = mi_sel.transform(X_va)
    X_te = mi_sel.transform(X_te)
    print(f"  Final feature count: {X_tr.shape[1]}")
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    
    # ── Load signals ──────────────────────────────────────────────────────────
    print(f"\n[4/5] Loading signals @ {FS_TARGET}Hz → {SEQ_LEN} samples ...")
    
    def load_split_signals(split_df, name):
        sigs, labels, ids = [], [], []
        t0 = time.time()
        for i, (_, row) in enumerate(split_df.iterrows()):
            if i % 1000 == 0: 
                print(f"  {name}: {i}/{len(split_df)}", flush=True)
            sig = load_signal(row)
            if sig is None: 
                continue
            sigs.append(sig)
            labels.append(int(row["label_idx"]))
            ids.append(int(row["ecg_id"]))
        print(f"  {name}: {len(sigs)}/{len(split_df)} in {time.time()-t0:.0f}s")
        return sigs, labels, ids
    
    # Get dataframes with features
    df_train_merged = merged[train_mask].reset_index(drop=True)
    df_val_merged   = merged[val_mask].reset_index(drop=True)
    df_test_merged  = merged[test_mask].reset_index(drop=True)
    
    tr_sigs, tr_labels, tr_ids = load_split_signals(df_train_merged, "train")
    val_sigs, val_labels, val_ids = load_split_signals(df_val_merged, "val  ")
    te_sigs, te_labels, te_ids = load_split_signals(df_test_merged, "test ")
    
    # Align features with loaded signals (some may have failed to load)
    # Build mapping
    train_id_to_idx = {row["ecg_id"]: i for i, row in df_train_merged.iterrows()}
    val_id_to_idx   = {row["ecg_id"]: i for i, row in df_val_merged.iterrows()}
    test_id_to_idx  = {row["ecg_id"]: i for i, row in df_test_merged.iterrows()}
    
    tr_feat_indices = [train_id_to_idx[eid] for eid in tr_ids]
    va_feat_indices = [val_id_to_idx[eid] for eid in val_ids]
    te_feat_indices = [test_id_to_idx[eid] for eid in te_ids]
    
    X_tr_final = X_tr[tr_feat_indices]
    X_va_final = X_va[va_feat_indices]
    X_te_final = X_te[te_feat_indices]
    
    print(f"\n  Final train: {len(tr_sigs)} signals, {X_tr_final.shape[0]} features")
    print(f"  Final val  : {len(val_sigs)} signals, {X_va_final.shape[0]} features")
    print(f"  Final test : {len(te_sigs)} signals, {X_te_final.shape[0]} features")
    
    # ── Datasets + loaders ────────────────────────────────────────────────────
    print("\n[5/5] Creating datasets ...")
    ds_train = HybridECGDataset(tr_sigs, X_tr_final, tr_labels, augment=True)
    ds_val   = HybridECGDataset(val_sigs, X_va_final, val_labels, augment=False)
    ds_test  = HybridECGDataset(te_sigs, X_te_final, te_labels, augment=False)
    
    # Weighted sampler
    counts   = np.bincount(tr_labels, minlength=N_CLASSES).astype(float)
    w_class  = 1.0 / np.sqrt(np.maximum(counts, 1))
    w_sample = [float(w_class[l]) for l in tr_labels]
    sampler  = WeightedRandomSampler(w_sample, num_samples=len(w_sample), replacement=True)
    
    train_loader = DataLoader(ds_train, BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(ds_val,   BATCH_SIZE, shuffle=False,    num_workers=0, pin_memory=True)
    test_loader  = DataLoader(ds_test,  BATCH_SIZE, shuffle=False,    num_workers=0, pin_memory=True)
    
    # ── Model + loss ──────────────────────────────────────────────────────────
    print("\nInitializing hybrid model ...")
    model = HybridResNet1D(n_classes=N_CLASSES, n_features=X_tr_final.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: HybridResNet1D  {n_params/1e6:.2f}M params  |  device={DEVICE}")
    
    # Class weights for focal loss
    cw_t = torch.tensor(w_class / w_class.sum() * N_CLASSES, dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, smooth=LABEL_SMOOTH, class_weights=cw_t)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=LR_MIN)
    
    # ── Training ──────────────────────────────────────────────────────────────
    print("\nStarting training ...")
    best_val_acc  = 0.0
    best_path     = ARTIFACTS / "resnet1d_hybrid_best.pt"
    no_improve    = 0
    
    print(f"\n{'Ep':>4}  {'LR':>9}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'Best':>7}")
    print("─" * 68)
    
    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        t0 = time.time()
        
        for sig, feat, y in train_loader:
            sig  = sig.to(DEVICE)
            feat = feat.to(DEVICE)
            y    = y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(sig, feat)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            tr_loss    += loss.item() * len(y)
            tr_correct += (logits.argmax(dim=1) == y).sum().item()
            tr_n       += len(y)
        
        tr_acc = tr_correct / tr_n
        
        # ── val ──
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(va_acc)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        
        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": CLASSES,
                "architecture": "HybridResNet1D",
                "seq_len": SEQ_LEN,
                "fs": FS_TARGET,
                "n_features": X_tr_final.shape[1],
                "preprocessors": {
                    "imputer": imputer,
                    "var_sel": var_sel,
                    "mi_sel": mi_sel,
                    "scaler": scaler,
                    "clip": (p01, p99)
                }
            }, best_path)
            no_improve = 0
            marker = " ✓"
        else:
            no_improve += 1
        
        print(f"{epoch:4d}  {lr_now:9.2e}  {tr_loss/tr_n:8.4f}  {tr_acc:6.2%}  "
              f"{va_loss:8.4f}  {va_acc:6.2%}  {best_val_acc:6.2%}{marker}  [{elapsed:.0f}s]")
        
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping — {PATIENCE} epochs without improvement")
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
        tp = cm[i, i]; fp = cm[:, i].sum()-tp; fn = cm[i, :].sum()-tp
        pr = tp/(tp+fp) if tp+fp else 0
        re = tp/(tp+fn) if tp+fn else 0
        f1 = 2*pr*re/(pr+re) if pr+re else 0
        print(f"{c:<6}  {pr:8.3f}  {re:8.3f}  {f1:8.3f}  {cm[i,:].sum():8d}")
    
    # ── Save results ──────────────────────────────────────────────────────────
    final_path = ARTIFACTS / "resnet1d_hybrid.pt"
    torch.save(ckpt, final_path)
    print(f"\n✓ Model saved → {final_path}")
    
    results = {
        "model": "HybridResNet1D (signals + features)",
        "architecture": "ResNet1D-256d + FeatureMLP-128d + Classifier",
        "classes": CLASSES,
        "n_features": int(X_tr_final.shape[1]),
        "val_accuracy": round(float(best_val_acc), 4),
        "test_accuracy": round(float(te_acc), 4),
        "test_samples": len(preds),
        "confusion_matrix": cm.tolist(),
        "per_class": {},
    }
    for i, c in enumerate(CLASSES):
        tp=int(cm[i,i]); fp=int(cm[:,i].sum()-tp); fn=int(cm[i,:].sum()-tp)
        pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0
        f1=2*pr*re/(pr+re) if pr+re else 0
        results["per_class"][c] = {
            "precision": round(pr,4),
            "recall": round(re,4),
            "f1": round(f1,4),
            "support": int(cm[i,:].sum())
        }
    
    (PUBLIC / "hybrid_results.json").write_text(json.dumps(results, indent=2))
    print(f"✓ Results → {PUBLIC/'hybrid_results.json'}")
    print("\nHybrid training complete!")


if __name__ == "__main__":
    main()
