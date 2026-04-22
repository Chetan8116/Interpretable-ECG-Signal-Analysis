"""
PTB-XL Hybrid ResNet1D-SE + Extracted Features — Target: 85%+

Key improvements over all previous runs (ResNet1D-v2, InceptionTime, BiLSTM):

  1. ResNet1D-SE: 8 residual blocks (4 stages × 2 blocks) with Squeeze-and-
     Excite channel attention — deeper and wider than train_resnet1d_v2.py
     (no SE, 4 blocks total), proven best single-lead ECG architecture.

  2. CosineAnnealingWarmRestarts(T_0=50, T_mult=2): LR resets at epoch 50
     and 150, preventing the LR collapse that killed BiLSTM (6.4e-5 by ep 56)
     and ResNet1D-v2 (ReduceLROnPlateau patience=5).

  3. Class weights = 1/count^0.7 — more aggressive minority boost than
     1/sqrt = 1/count^0.5 used in all previous scripts.

  4. Early stopping on macro-F1 (not raw accuracy) — avoids NORM 46% bias.

  5. Focal gamma=2.5 — harder focus on minority class errors than gamma=2.0.

  6. Label smooth=0.05 (half of BiLSTM's 0.10) — sharper decision boundaries.

  7. MixUp alpha=0.05 (very light vs 0.10-0.15) — minimal label noise, keeps
     minority class gradients clear.

  8. Stochastic Weight Averaging (SWA) over last 50 epochs — finds flatter,
     more-generalizable minima; typically yields +0.5-1.5% on plateau models.

  9. TTA ×5 at test time — averages 5 augmented predictions for each sample.

 10. Hybrid: signals + 300 MI-selected extracted features (same as BiLSTM).

Usage:
    python scripts/train_resnet1d_ptbxl.py
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
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

# ── Paths (flexible for Linux / Windows) ─────────────────────────────────────
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
    print(f"\n⚠  ERROR: PTB-XL dataset not found in {ROOT / 'archive'}")
    import sys; sys.exit(1)

CSV_PATH  = ARCHIVE / "ptbxl_database.csv"
REC_ROOT  = ARCHIVE / "records500"
FEAT_DIR  = ROOT / "ptbxl_comprehensive_features"
ARTIFACTS = ROOT / "ECG_Diag_pipeline" / "artifacts"
PUBLIC    = ROOT / "public"

for p, name in [(CSV_PATH, "CSV"), (REC_ROOT, "Signals"), (FEAT_DIR, "Features")]:
    if not p.exists():
        print(f"\n⚠  ERROR: {name} not found: {p}"); import sys; sys.exit(1)

ARTIFACTS.mkdir(parents=True, exist_ok=True)
PUBLIC.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
FS_TARGET = 125
SEQ_LEN   = 1250
N_LEADS   = 12
N_CLASSES = 5
CLASSES   = ["CD", "HYP", "MI", "NORM", "STTC"]

N_SELECTED_FEATS = 300

# Architecture
BASE_CHANNELS = 64     # stem output channels
DROPOUT       = 0.35   # lighter than BiLSTM's 0.45 — ResNet needs less

# Training
BATCH_SIZE   = 64
MAX_EPOCHS   = 200
PATIENCE     = 60      # on macro-F1
SWA_START    = 130     # begin SWA accumulation from this epoch (if training goes long)
LR_INIT      = 3e-4
WEIGHT_DECAY = 2e-4
FOCAL_GAMMA  = 2.5     # harder focus on minority classes
LABEL_SMOOTH = 0.05    # sharper than BiLSTM's 0.10
MIXUP_ALPHA  = 0.05    # very light — preserve minority class signal
GRAD_CLIP    = 1.0
CW_POW       = 0.7     # class weight = 1/count^CW_POW (between 0.5=sqrt, 1.0=linear)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  FS={FS_TARGET}Hz  SEQ={SEQ_LEN}  N_FEATS={N_SELECTED_FEATS}")
print(f"[paths] CSV:      {CSV_PATH}")
print(f"[paths] Signals:  {REC_ROOT}")
print(f"[paths] Features: {FEAT_DIR}")
print(f"[paths] Output:   {ARTIFACTS}")


# ── SCP → superclass mapping (identical to other scripts) ────────────────────
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
        # ① Gaussian noise
        if np.random.rand() < 0.7:
            sig = sig + (np.random.randn(*sig.shape) * 0.02).astype(np.float32)
        # ② Amplitude scaling ±15%
        if np.random.rand() < 0.6:
            sig = sig * float(np.random.uniform(0.85, 1.15))
        # ③ Baseline wander
        if np.random.rand() < 0.5:
            t  = np.linspace(0, 1, SEQ_LEN, dtype=np.float32)
            bw = (np.sin(2 * np.pi * np.random.uniform(0.05, 0.5) * t)
                  * np.random.uniform(0.01, 0.06)).astype(np.float32)
            sig = sig + bw
        # ④ Time shift ±20 samples (160ms @ 125Hz)
        if np.random.rand() < 0.5:
            sig = np.roll(sig, np.random.randint(-20, 21), axis=1)
        # ⑤ Lead dropout (1-2 leads zeroed)
        if np.random.rand() < 0.3:
            sig = sig.copy()
            sig[np.random.choice(N_LEADS, np.random.randint(1, 3), replace=False)] = 0.0
        # ⑥ Cutout window (ECG segment masked)
        if np.random.rand() < 0.4:
            win   = np.random.randint(50, 200)
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
class SEBlock1D(nn.Module):
    """Squeeze-and-Excite: global avg → bottleneck FC → channel gate."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1)   # (B, C, T) * (B, C, 1)


class ResBlockSE(nn.Module):
    """
    Pre-activation residual block with SE attention.
    BN→ReLU→Conv → BN→ReLU→Conv → SE → add shortcut
    (pre-activation = better gradient flow for deep networks)
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7,
                 stride: int = 1, dropout: float = 0.2):
        super().__init__()
        pad = kernel // 2
        self.bn1   = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, bias=False)
        self.se    = SEBlock1D(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.drop(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.se(out)
        return out + self.shortcut(x)


class ECGResNetSE(nn.Module):
    """
    Hybrid ResNet1D-SE + Feature MLP.

    Signal path:  (B, 12, 1250)
      → Stem (stride-4)          → (B, 64, 313)
      → Layer 1 (2 × SE blocks)  → (B,  64, 313)
      → Layer 2 (stride-2 + SE)  → (B, 128, 157)
      → Layer 3 (stride-2 + SE)  → (B, 256,  79)
      → Layer 4 (stride-2 + SE)  → (B, 384,  40)
      → GlobalAvgPool            → (B, 384)
      → sig_head                 → (B, 256)

    Feature path: (B, n_feat) → feat_head → (B, 128)

    Fusion: cat([256, 128]) → classifier → (B, n_classes)
    """

    def __init__(self, n_classes: int = 5, n_features: int = 300,
                 dropout: float = 0.35):
        super().__init__()

        # ── Signal branch ─────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(N_LEADS, BASE_CHANNELS, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(BASE_CHANNELS),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),   # 625 → 313
        )

        def make_layer(in_ch, out_ch, kernel, stride, drop, n_blocks=2):
            layers = [ResBlockSE(in_ch, out_ch, kernel, stride, drop)]
            for _ in range(n_blocks - 1):
                layers.append(ResBlockSE(out_ch, out_ch, kernel, 1, drop))
            return nn.Sequential(*layers)

        # Stage 1: 313 → 313 (same spatial, widen channels)
        self.layer1 = make_layer(64,  64,  7, 1, 0.20, 2)
        # Stage 2: 313 → 157
        self.layer2 = make_layer(64,  128, 7, 2, 0.20, 2)
        # Stage 3: 157 → 79
        self.layer3 = make_layer(128, 256, 7, 2, 0.25, 2)
        # Stage 4: 79 → 40
        self.layer4 = make_layer(256, 384, 5, 2, 0.30, 2)

        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.sig_bn  = nn.BatchNorm1d(384)

        self.sig_head = nn.Sequential(
            nn.Linear(384, 256, bias=False),
            nn.BatchNorm1d(256),
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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, sig: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        x = self.stem(sig)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(self.sig_bn(self.gap(x).squeeze(-1)))
        sig_emb  = self.sig_head(x)
        feat_emb = self.feat_head(feat)
        return self.classifier(torch.cat([sig_emb, feat_emb], dim=1))


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.5, smooth: float = 0.05,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma; self.smooth = smooth; self.weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
def mixup_batch(sig, feat, y, alpha=0.05):
    if alpha <= 0:
        return sig, feat, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)   # always take the larger share → cleaner gradients
    idx = torch.randperm(sig.size(0), device=sig.device)
    return (lam * sig + (1 - lam) * sig[idx],
            lam * feat + (1 - lam) * feat[idx],
            y, y[idx], lam)

def mixup_criterion(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)


# ── Evaluation (macro-F1 + accuracy) ─────────────────────────────────────────
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


# ── TTA evaluation (test-time augmentation) ───────────────────────────────────
@torch.no_grad()
def evaluate_tta(model, loader, criterion, n_aug: int = 5):
    """
    For each sample, collect softmax probs from 1 original + n_aug augmented
    copies and average them before taking argmax.
    """
    model.eval()
    total, n = 0.0, 0
    all_probs: list[np.ndarray] = []
    all_trues: list[int]        = []

    # Gather per-sample original predictions first (for loss)
    for sig, feat, y in loader:
        sig, feat, y = sig.to(DEVICE), feat.to(DEVICE), y.to(DEVICE)
        logits = model(sig, feat)
        total += criterion(logits, y).item() * len(y)
        n     += len(y)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_trues.extend(y.cpu().tolist())

    # Accumulate augmented passes
    for _ in range(n_aug):
        aug_probs: list[np.ndarray] = []
        for sig, feat, _ in loader:
            sig_np = sig.numpy()
            # Apply lightweight TTA augmentation
            scale = np.random.uniform(0.90, 1.10, (sig_np.shape[0], 1, 1)).astype(np.float32)
            shift = np.random.randint(-15, 16)
            aug   = np.roll(sig_np * scale, shift, axis=2)
            aug   = aug + (np.random.randn(*aug.shape) * 0.015).astype(np.float32)
            # Re-normalize per lead
            mu  = aug.mean(axis=2, keepdims=True)
            std = aug.std(axis=2, keepdims=True) + 1e-6
            aug = (aug - mu) / std
            aug_t = torch.tensor(aug, dtype=torch.float32).to(DEVICE)
            feat_t = feat.to(DEVICE)
            logits = model(aug_t, feat_t)
            aug_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

        # Accumulate probabilities
        all_probs = [a + b for a, b in zip(all_probs, aug_probs)]

    # Final predictions from averaged probs
    preds = [int(np.argmax(p)) for p in all_probs]
    acc   = sum(p == t for p, t in zip(preds, all_trues)) / len(preds)

    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(all_trues, preds):
        cm[t][p] += 1
    f1s = []
    for i in range(N_CLASSES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if (tp + fp) else 0
        re = tp / (tp + fn) if (tp + fn) else 0
        f1s.append(2 * pr * re / (pr + re) if (pr + re) else 0)

    return total / n, acc, float(np.mean(f1s)), preds, all_trues


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  PTB-XL Hybrid ResNet1D-SE + Extracted Features")
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

    # Aggressive class weighting: 1/count^0.7 (between sqrt=0.5 and linear=1.0)
    counts   = np.bincount(tr_labels, minlength=N_CLASSES).astype(float)
    w_class  = 1.0 / (np.maximum(counts, 1) ** CW_POW)
    w_sample = [float(w_class[l]) for l in tr_labels]
    sampler  = WeightedRandomSampler(w_sample, num_samples=len(w_sample), replacement=True)

    train_loader = DataLoader(ds_train, BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(ds_val,   BATCH_SIZE, shuffle=False,   num_workers=0, pin_memory=True)
    test_loader  = DataLoader(ds_test,  BATCH_SIZE, shuffle=False,   num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nInitializing ResNet1D-SE model ...")
    model = ECGResNetSE(
        n_classes=N_CLASSES, n_features=X_tr_f.shape[1], dropout=DROPOUT
    ).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: ECGResNetSE  {n_p/1e6:.2f}M params  |  device={DEVICE}")

    # ── Loss + optimizer ──────────────────────────────────────────────────────
    cw_t      = torch.tensor(w_class / w_class.sum() * N_CLASSES, dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, smooth=LABEL_SMOOTH, class_weights=cw_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)

    # CosineAnnealingWarmRestarts: cycle 1 = epochs 1-50, cycle 2 = epochs 51-150
    # LR resets to LR_INIT at the start of each cycle → no permanent LR collapse
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6,
    )

    # SWA (Stochastic Weight Averaging) setup
    swa_model  = AveragedModel(model)
    swa_active = False

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nScheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2) — no LR collapse")
    print(f"Class weights (1/count^{CW_POW:.1f}): {dict(zip(CLASSES, np.round(w_class/w_class.sum()*N_CLASSES, 3)))}")
    print(f"SWA starts at epoch {SWA_START}")
    print("\nStarting training ...")
    print("  (Early stopping + SWA on macro-F1, not raw accuracy)")

    best_f1    = 0.0
    best_path  = ARTIFACTS / "resnet1d_ptbxl_best.pt"
    no_improve = 0

    print(f"\n{'Ep':>4}  {'LR':>9}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'VaF1':>7}  {'BestF1':>7}  {'SWA':>4}")
    print("─" * 88)

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
            # Training accuracy measured vs majority label (ya); MixUp lam≥0.5 always
            tr_ok   += (logits.argmax(1) == ya).sum().item()
            tr_n    += len(y)

        scheduler.step(epoch - 1)   # CAWR uses step(epoch) for epoch-level scheduling

        # SWA: start accumulating weights after SWA_START
        if epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_active = True

        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, criterion)
        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        marker = ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "state_dict":   model.state_dict(),
                "classes":      CLASSES,
                "architecture": "ECGResNetSE",
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

        swa_tag = "ON " if swa_active else "   "
        print(f"{epoch:4d}  {lr_now:9.2e}  {tr_loss/tr_n:8.4f}  {tr_ok/tr_n:6.2%}  "
              f"{va_loss:8.4f}  {va_acc:6.2%}  {va_f1:6.3f}  {best_f1:6.3f}{marker}  {swa_tag}  [{elapsed:.0f}s]")

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} — {PATIENCE} epochs without improvement")
            break

    # ── SWA: update BatchNorm statistics ──────────────────────────────────────
    if swa_active:
        print("\nUpdating SWA BatchNorm statistics ...")
        update_bn(train_loader, swa_model, device=DEVICE)
        print("  Done. Evaluating SWA model:")
        _, swa_acc, swa_f1, _, _ = evaluate(swa_model, val_loader, criterion)
        print(f"  SWA val  → acc={swa_acc:.4f}  macro-F1={swa_f1:.4f}")
        _, best_acc, best_f1_ck, _, _ = evaluate(model, val_loader, criterion)
        # Load best checkpoint to compare
        ckpt_best = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt_best["state_dict"])
        _, ck_acc, ck_f1, _, _ = evaluate(model, val_loader, criterion)
        print(f"  Best ckpt → acc={ck_acc:.4f}  macro-F1={ck_f1:.4f}")

        # Use SWA model if it beats best checkpoint on val
        if swa_f1 >= ck_f1:
            print("  → Using SWA model for test evaluation")
            eval_model = swa_model
        else:
            print("  → Using best checkpoint for test evaluation")
            eval_model = model
    else:
        print("\nSWA not activated (training stopped before SWA_START epoch)")
        print("Loading best checkpoint for test evaluation ...")
        ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        eval_model = model

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Running test evaluation (standard + TTA×5) ...")

    _, te_acc,     te_f1,     preds,     trues = evaluate(eval_model, test_loader, criterion)
    _, te_acc_tta, te_f1_tta, preds_tta, _     = evaluate_tta(eval_model, test_loader, criterion, n_aug=5)

    print(f"\n{'='*80}")
    print(f"TEST ACCURACY  (standard): {te_acc:.4f}  ({te_acc*100:.2f}%)")
    print(f"TEST MACRO-F1  (standard): {te_f1:.4f}  ({te_f1*100:.2f}%)")
    print(f"TEST ACCURACY  (TTA ×5  ): {te_acc_tta:.4f}  ({te_acc_tta*100:.2f}%)")
    print(f"TEST MACRO-F1  (TTA ×5  ): {te_f1_tta:.4f}  ({te_f1_tta*100:.2f}%)")
    print("=" * 80)

    # Use TTA predictions for the confusion matrix
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(trues, preds_tta): cm[t][p] += 1

    print("\nConfusion matrix (rows=actual, cols=predicted) — TTA:")
    print("      " + "   ".join(f"{c:>5}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:4s}: " + "  ".join(f"{cm[i,j]:>6}" for j in range(N_CLASSES)))

    print("\nPer-class metrics — TTA:")
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

    # ── Save ──────────────────────────────────────────────────────────────────
    # Save the weights of eval_model (handles both SWA AveragedModel and regular)
    final_path = ARTIFACTS / "resnet1d_ptbxl.pt"
    if isinstance(eval_model, AveragedModel):
        save_dict = {
            "state_dict":   eval_model.module.state_dict(),
            "classes":      CLASSES,
            "architecture": "ECGResNetSE (SWA)",
            "seq_len":      SEQ_LEN,
            "fs":           FS_TARGET,
            "n_features":   int(X_tr_f.shape[1]),
            "val_macro_f1": round(float(best_f1), 4),
            "test_accuracy_tta": round(float(te_acc_tta), 4),
            "test_macro_f1_tta": round(float(te_f1_tta), 4),
        }
    else:
        ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
        save_dict = ckpt
        save_dict["test_accuracy_tta"]  = round(float(te_acc_tta), 4)
        save_dict["test_macro_f1_tta"]  = round(float(te_f1_tta), 4)

    torch.save(save_dict, final_path)
    print(f"\n✓ Model saved     → {final_path}")

    results = {
        "model":               "ECGResNetSE Hybrid (signals + features)",
        "architecture":        "ResNet1D-SE (4 stages, 8 SE blocks) + FeatureMLP-128d",
        "classes":             CLASSES,
        "n_features":          int(X_tr_f.shape[1]),
        "val_best_macro_f1":   round(float(best_f1), 4),
        "test_accuracy":       round(float(te_acc), 4),
        "test_macro_f1":       round(float(te_f1), 4),
        "test_accuracy_tta":   round(float(te_acc_tta), 4),
        "test_macro_f1_tta":   round(float(te_f1_tta), 4),
        "test_samples":        len(preds),
        "confusion_matrix_tta": cm.tolist(),
        "per_class_tta":       per_class,
    }
    (PUBLIC / "resnet1d_ptbxl_results.json").write_text(json.dumps(results, indent=2))
    print(f"✓ Results saved   → {PUBLIC / 'resnet1d_ptbxl_results.json'}")
    print("\nResNet1D-SE training complete!")


if __name__ == "__main__":
    main()
