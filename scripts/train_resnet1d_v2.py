"""
PTB-XL 5-class ResNet1D — IMPROVED v2  (target: 85%+ accuracy)

Key improvements over v1:
  1. 125 Hz sampling  (1250 samples/lead  vs 1000)  — preserves QRS morphology
  2. Focal Loss (gamma=2) instead of CE — crushes class imbalance penalty
  3. ReduceLROnPlateau instead of OneCycleLR — safe LR decay without premature spikes
  4. Patience 20 (was 15) — more time to escape plateaus
  5. Stronger augmentation: baseline wander, random polarity flip, bandpass noise
  6. Heavier WeightedRandomSampler (square-root inverse freq)
  7. Test-time augmentation (TTA) at evaluation

Architecture unchanged: 4-block ResNet1D, 4.1M params
Input: (batch, 12, 1250)

Run:
    python scripts/train_resnet1d_v2.py
"""

from __future__ import annotations

import ast, json, os, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample, butter, filtfilt
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
ARCHIVE    = ROOT / "archive" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
CSV_PATH   = ARCHIVE / "ptbxl_database.csv"
REC_ROOT   = ARCHIVE / "records500"
ARTIFACTS  = ROOT / "ECG_Diag_pipeline" / "artifacts"
PUBLIC     = ROOT / "public"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
PUBLIC.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
FS_ORIG     = 500
FS_TARGET   = 125           # 125 Hz → 1250 samples / 10 s
SEQ_LEN     = 1250
N_LEADS     = 12
N_CLASSES   = 5
CLASSES     = ["CD", "HYP", "MI", "NORM", "STTC"]

BATCH_SIZE   = 64
MAX_EPOCHS   = 150
PATIENCE     = 20
LR_INIT      = 5e-4
LR_MIN       = 5e-6
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA  = 2.0          # focal loss concentration on hard examples
LABEL_SMOOTH = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  FS={FS_TARGET}Hz  SEQ={SEQ_LEN}")


# ═══════════════════════════════════════════════════════════════════════════════
# SCP → superclass mapping
# ═══════════════════════════════════════════════════════════════════════════════
SCP_TO_SUPER = {
    "NORM":"NORM",
    "IMI":"MI","ILMI":"MI","AMI":"MI","ALMI":"MI","INJAS":"MI","LMI":"MI",
    "INJAL":"MI","IPLMI":"MI","IPMI":"MI","INJIN":"MI","INJLA":"MI","PMI":"MI","INJIL":"MI",
    "STD_":"STTC","ISCA":"STTC","ISCI":"STTC","ISC_":"STTC","INVT":"STTC",
    "LNGQT":"STTC","TAB_":"STTC","ANEUR":"STTC","EL":"STTC",
    "LAFB":"CD","IRBBB":"CD","CLBBB":"CD","CRBBB":"CD","LPFB":"CD",
    "WPW":"CD","IVCD":"CD","ILBBB":"CD","AVB":"CD","1AVB":"CD",
    "2AVB":"CD","3AVB":"CD","LBBB":"CD","RBBB":"CD",
    "LVH":"HYP","RVH":"HYP","SEHYP":"HYP","LAO/LAE":"HYP",
    "RAO/RAE":"HYP","VCLVH":"HYP",
}

def parse_superclass(scp_str: str):
    try: d = ast.literal_eval(scp_str)
    except: return None
    best, bc = None, -1
    for code, conf in d.items():
        sup = SCP_TO_SUPER.get(code)
        if sup and conf > bc: best, bc = sup, conf
    return best


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
# Dataset with stronger augmentation
# ═══════════════════════════════════════════════════════════════════════════════
class ECGDataset(Dataset):
    def __init__(self, sigs, labels, augment=False):
        self.sigs    = sigs
        self.labels  = np.array(labels, dtype=np.int64)
        self.augment = augment

    def __len__(self): return len(self.sigs)

    @staticmethod
    def _normalize(sig):
        mu  = sig.mean(axis=1, keepdims=True)
        std = sig.std(axis=1, keepdims=True) + 1e-6
        return (sig - mu) / std

    def _augment(self, sig):
        # ① Gaussian noise
        if np.random.rand() < 0.7:
            sig = sig + (np.random.randn(*sig.shape) * 0.025).astype(np.float32)
        # ② Amplitude scaling ±20%
        if np.random.rand() < 0.6:
            sig = sig * float(np.random.uniform(0.8, 1.2))
        # ③ Baseline wander: low-freq sinusoid
        if np.random.rand() < 0.5:
            t   = np.linspace(0, 1, SEQ_LEN, dtype=np.float32)
            bw  = (np.sin(2 * np.pi * np.random.uniform(0.1, 0.5) * t)
                   * np.random.uniform(0.02, 0.10)).astype(np.float32)
            sig = sig + bw          # broadcast over leads
        # ④ Random time shift ±100ms = ±12 samples @ 125Hz
        if np.random.rand() < 0.5:
            shift = np.random.randint(-12, 13)
            sig   = np.roll(sig, shift, axis=1)
        # ⑤ Random polarity flip on 1 lead (valid for AVR/limb leads)
        if np.random.rand() < 0.2:
            li  = np.random.randint(N_LEADS)
            sig = sig.copy()
            sig[li] = -sig[li]
        # ⑥ Random lead dropout (zero up to 2 leads)
        if np.random.rand() < 0.3:
            k     = np.random.randint(1, 3)
            leads = np.random.choice(N_LEADS, k, replace=False)
            sig   = sig.copy()
            sig[leads] = 0.0
        return sig

    def __getitem__(self, idx):
        sig = self.sigs[idx].copy()
        if self.augment:
            sig = self._augment(sig)
        sig = self._normalize(sig)
        return torch.tensor(sig, dtype=torch.float32), int(self.labels[idx])


# ═══════════════════════════════════════════════════════════════════════════════
# Model  (identical architecture, adjusted for SEQ_LEN=1250)
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


class ResNet1D(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        # Stem  1250 → 312
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, 15, stride=2, padding=7, bias=False),   # 1250→625
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),                      # 625→313
        )
        self.layer1 = nn.Sequential(ResBlock1D(32,  64, 7, 1, 0.2), ResBlock1D(64,  64, 7, 1, 0.2))
        self.layer2 = nn.Sequential(ResBlock1D(64, 128, 7, 2, 0.2), ResBlock1D(128,128, 7, 1, 0.2))
        self.layer3 = nn.Sequential(ResBlock1D(128,256, 7, 2, 0.3), ResBlock1D(256,256, 7, 1, 0.3))
        self.layer4 = nn.Sequential(ResBlock1D(256,256, 7, 2, 0.3), ResBlock1D(256,256, 7, 1, 0.3))
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.head(self.gap(x))


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
        self.weights = class_weights   # (n_classes,) or None

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
        p_t    = (p * smooth_targets).sum(dim=1)          # (B,)
        focal  = (1 - p_t) ** self.gamma

        # CE with smooth targets
        ce     = -(smooth_targets * log_p).sum(dim=1)     # (B,)
        loss   = focal * ce

        if self.weights is not None:
            w = self.weights[targets]
            loss = loss * w

        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# TTA: average predictions over augmented copies
# ═══════════════════════════════════════════════════════════════════════════════
def tta_predict(model, x: torch.Tensor, n_aug: int = 4) -> np.ndarray:
    """x: (1, 12, SEQ_LEN) — return softmax probs averaged over n_aug+1 passes."""
    model.eval()
    sig = x.numpy()[0]   # (12, SEQ_LEN)
    preds = []
    with torch.no_grad():
        # Original
        preds.append(torch.softmax(model(x), dim=1).numpy()[0])
        # Augmented
        for _ in range(n_aug):
            s = sig.copy()
            # amplitude jitter
            s = s * float(np.random.uniform(0.85, 1.15))
            # small time shift
            s = np.roll(s, int(np.random.randint(-8, 9)), axis=1)
            # tiny noise
            s = s + (np.random.randn(*s.shape) * 0.015).astype(np.float32)
            mu  = s.mean(axis=1, keepdims=True)
            std = s.std(axis=1,  keepdims=True) + 1e-6
            s   = (s - mu) / std
            xt  = torch.tensor(s[None], dtype=torch.float32)
            preds.append(torch.softmax(model(xt), dim=1).numpy()[0])
    return np.mean(preds, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Eval with optional TTA
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, criterion, use_tta=False):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(x)
        n          += len(x)

        if use_tta:
            for xi, yi in zip(x.cpu(), y.cpu()):
                probs = tta_predict(model, xi.unsqueeze(0))
                all_preds.append(int(np.argmax(probs)))
                all_labels.append(int(yi))
        else:
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_preds)
    return total_loss / n, acc, all_preds, all_labels


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Metadata ──────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    df["superclass"] = df["scp_codes"].apply(parse_superclass)
    df = df.dropna(subset=["superclass"])
    df["label_idx" ] = df["superclass"].map({c: i for i, c in enumerate(CLASSES)})
    print(f"Records: {len(df)}")
    for c in CLASSES:
        n = (df["superclass"] == c).sum()
        print(f"  {c}: {n} ({100*n/len(df):.1f}%)")

    df_train = df[df["strat_fold"].isin(range(1, 9))].reset_index(drop=True)
    df_val   = df[df["strat_fold"] == 9].reset_index(drop=True)
    df_test  = df[df["strat_fold"] == 10].reset_index(drop=True)
    print(f"\nSplit: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # ── Load signals ──────────────────────────────────────────────────────────
    print(f"\nLoading signals @ {FS_TARGET}Hz → {SEQ_LEN} samples …")
    def load_split(split_df, name):
        sigs, labels, ids = [], [], []
        t0 = time.time()
        for i, (_, row) in enumerate(split_df.iterrows()):
            if i % 1000 == 0: print(f"  {name}: {i}/{len(split_df)}")
            sig = load_signal(row)
            if sig is None: continue
            sigs.append(sig); labels.append(int(row["label_idx"]))
            ids.append(int(row["ecg_id"]))
        print(f"  {name}: {len(sigs)}/{len(split_df)} in {time.time()-t0:.0f}s")
        return sigs, labels, ids

    tr_sigs,  tr_labels,  tr_ids  = load_split(df_train, "train")
    val_sigs, val_labels, val_ids = load_split(df_val,   "val  ")
    te_sigs,  te_labels,  te_ids  = load_split(df_test,  "test ")

    # ── Datasets + loaders ────────────────────────────────────────────────────
    ds_train = ECGDataset(tr_sigs,  tr_labels, augment=True)
    ds_val   = ECGDataset(val_sigs, val_labels, augment=False)
    ds_test  = ECGDataset(te_sigs,  te_labels,  augment=False)

    # Square-root inverse frequency sampler (gentler than pure inverse freq)
    counts   = np.bincount(tr_labels, minlength=N_CLASSES).astype(float)
    w_class  = 1.0 / np.sqrt(np.maximum(counts, 1))
    w_sample = [float(w_class[l]) for l in tr_labels]
    sampler  = WeightedRandomSampler(w_sample, num_samples=len(w_sample), replacement=True)

    train_loader = DataLoader(ds_train, BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(ds_val,   BATCH_SIZE, shuffle=False,    num_workers=0)
    test_loader  = DataLoader(ds_test,  BATCH_SIZE, shuffle=False,    num_workers=0)

    # ── Model + loss ──────────────────────────────────────────────────────────
    model = ResNet1D(n_classes=N_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: ResNet1D  {n_params/1e6:.2f}M params  |  device={DEVICE}")

    # Class weights for focal loss (inverse-sqrt frequency)
    cw_t  = torch.tensor(w_class / w_class.sum() * N_CLASSES, dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, smooth=LABEL_SMOOTH, class_weights=cw_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=LR_MIN)

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    best_path     = ARTIFACTS / "resnet1d_best.pt"
    no_improve    = 0

    print(f"\n{'Ep':>4}  {'LR':>9}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'Best':>7}")
    print("─" * 68)

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss    += loss.item() * len(x)
            tr_correct += (logits.argmax(dim=1) == y).sum().item()
            tr_n       += len(x)
        tr_acc = tr_correct / tr_n

        # ── val ──
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, use_tta=False)
        scheduler.step(va_acc)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                        "architecture": "ResNet1D_v2", "seq_len": SEQ_LEN,
                        "fs": FS_TARGET}, best_path)
            no_improve = 0
            marker = " ✓"
        else:
            no_improve += 1

        print(f"{epoch:4d}  {lr_now:9.2e}  {tr_loss/tr_n:8.4f}  {tr_acc:6.2%}  "
              f"{va_loss:8.4f}  {va_acc:6.2%}  {best_val_acc:6.2%}{marker}  [{elapsed:.0f}s]")

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping — {PATIENCE} epochs without improvement")
            break

    # ── Test evaluation ────────────────────────────────────────────────────────
    print("\nLoading best checkpoint …")
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    print("Evaluating with TTA …")
    _, te_acc_tta, preds_tta, trues = evaluate(model, test_loader, criterion, use_tta=True)
    _, te_acc,     preds,     _     = evaluate(model, test_loader, criterion, use_tta=False)
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY  (standard ): {te_acc:.4f}  ({te_acc*100:.2f}%)")
    print(f"TEST ACCURACY  (TTA x5   ): {te_acc_tta:.4f}  ({te_acc_tta*100:.2f}%)")
    print("=" * 60)

    # Confusion matrix (TTA)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(trues, preds_tta): cm[t][p] += 1
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

    # ── Save ──────────────────────────────────────────────────────────────────
    final_path = ARTIFACTS / "resnet1d_ptbxl.pt"
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                "architecture": "ResNet1D_v2", "seq_len": SEQ_LEN, "fs": FS_TARGET,
                "val_accuracy": float(best_val_acc),
                "test_accuracy": float(te_acc_tta)}, final_path)
    print(f"\n✓ Model saved → {final_path}")

    results = {
        "model": "ResNet1D-v2 (focal loss + TTA, 125Hz)",
        "architecture": "ResNet1D-4block-256ch-GAP",
        "classes": CLASSES,
        "val_accuracy":  round(float(best_val_acc), 4),
        "test_accuracy": round(float(te_acc_tta), 4),
        "test_accuracy_no_tta": round(float(te_acc), 4),
        "test_samples": len(preds_tta),
        "confusion_matrix": cm.tolist(),
        "per_class": {},
    }
    for i, c in enumerate(CLASSES):
        tp=int(cm[i,i]); fp=int(cm[:,i].sum()-tp); fn=int(cm[i,:].sum()-tp)
        pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0
        f1=2*pr*re/(pr+re) if pr+re else 0
        results["per_class"][c] = {"precision":round(pr,4),"recall":round(re,4),
                                   "f1":round(f1,4),"support":int(cm[i,:].sum())}
    (PUBLIC / "mlp_results.json").write_text(json.dumps(results, indent=2))
    print(f"✓ Results → {PUBLIC/'mlp_results.json'}")

    # ecg_id list for precompute step
    all_ids    = tr_ids + val_ids + te_ids
    all_labels = tr_labels + val_labels + te_labels
    all_splits = ["train"]*len(tr_ids) + ["val"]*len(val_ids) + ["test"]*len(te_ids)
    pd.DataFrame({"ecg_id": all_ids, "label_idx": all_labels,
                  "superclass": [CLASSES[l] for l in all_labels],
                  "split": all_splits}).to_csv(
        ARTIFACTS / "resnet1d_ids.csv", index=False)
    print(f"✓ IDs CSV saved")
    print(f"\nNext step: python scripts/precompute_resnet1d_predictions.py")


if __name__ == "__main__":
    main()
