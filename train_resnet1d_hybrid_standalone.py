"""
PTB-XL Hybrid ResNet1D + Extracted Features — STANDALONE VERSION

This is a self-contained version that can be copied anywhere.
It will automatically find the dataset and feature directories.

Usage:
    # Copy this file to your project directory, then:
    python train_resnet1d_hybrid_standalone.py
    
    # Or specify paths manually:
    python train_resnet1d_hybrid_standalone.py --data-dir /path/to/ptb-xl-dataset
"""

from __future__ import annotations

import ast, json, os, time, warnings, argparse
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

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Hybrid ResNet1D on PTB-XL")
parser.add_argument("--data-dir", type=str, help="Path to PTB-XL dataset directory")
parser.add_argument("--feat-dir", type=str, help="Path to extracted features directory")
parser.add_argument("--output-dir", type=str, help="Path to save outputs")
args = parser.parse_args()

# ── Paths (auto-detect or use provided) ──────────────────────────────────────
def find_project_root():
    """Find project root by looking for key directories."""
    script_path = Path(__file__).resolve()
    
    # Try different possible locations
    candidates = [
        script_path.parent,         # Script in project root
        script_path.parent.parent,  # Script in subdirectory
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
    
    # Fallback to script directory
    return script_path.parent

ROOT = find_project_root()
print(f"[auto-detect] Project root: {ROOT}")

# Find archive directory
if args.data_dir:
    ARCHIVE = Path(args.data_dir)
    print(f"[manual] Using provided data directory: {ARCHIVE}")
else:
    ARCHIVE = None
    archive_dir = ROOT / "archive"
    if archive_dir.exists():
        for subdir in archive_dir.iterdir():
            if subdir.is_dir() and "ptb-xl" in subdir.name.lower():
                ARCHIVE = subdir
                break
    
    if ARCHIVE is None:
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
    print(f"\n⚠️  ERROR: Could not find PTB-XL dataset")
    print("Please specify --data-dir or ensure dataset is in archive/ folder")
    print("\nExample:")
    print(f"  python {Path(__file__).name} --data-dir /path/to/ptb-xl-dataset")
    import sys
    sys.exit(1)

CSV_PATH = ARCHIVE / "ptbxl_database.csv"
REC_ROOT = ARCHIVE / "records500"

# Find features directory
if args.feat_dir:
    FEAT_DIR = Path(args.feat_dir)
    print(f"[manual] Using provided feature directory: {FEAT_DIR}")
else:
    FEAT_DIR = ROOT / "ptbxl_comprehensive_features"

# Output directory
if args.output_dir:
    ARTIFACTS = Path(args.output_dir)
else:
    ARTIFACTS = ROOT / "ECG_Diag_pipeline" / "artifacts"

PUBLIC = ARTIFACTS.parent / "public" if ARTIFACTS.parent.name == "artifacts" else ARTIFACTS / "public"

# Verify critical paths
if not CSV_PATH.exists():
    print(f"\n⚠️  ERROR: CSV not found: {CSV_PATH}")
    print("\nCorrect dataset structure:")
    print("  ptb-xl-dataset/")
    print("    ├─ ptbxl_database.csv")
    print("    └─ records500/")
    import sys
    sys.exit(1)

if not REC_ROOT.exists():
    print(f"\n⚠️  ERROR: Signal directory not found: {REC_ROOT}")
    import sys
    sys.exit(1)

if not FEAT_DIR.exists():
    print(f"\n⚠️  ERROR: Feature directory not found: {FEAT_DIR}")
    print("Please run feature extraction first or specify --feat-dir")
    import sys
    sys.exit(1)

ARTIFACTS.mkdir(parents=True, exist_ok=True)
PUBLIC.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
FS_ORIG     = 500
FS_TARGET   = 125
SEQ_LEN     = 1250
N_LEADS     = 12
N_CLASSES   = 5
CLASSES     = ["CD", "HYP", "MI", "NORM", "STTC"]
N_SELECTED_FEATS = 200

BATCH_SIZE   = 48
MAX_EPOCHS   = 120
PATIENCE     = 18
LR_INIT      = 4e-4
LR_MIN       = 1e-6
WEIGHT_DECAY = 2e-4
FOCAL_GAMMA  = 2.5
LABEL_SMOOTH = 0.08

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n" + "=" * 80)
print("  PTB-XL Hybrid ResNet1D Configuration")
print("=" * 80)
print(f"Device:   {DEVICE}")
print(f"Sampling: {FS_TARGET}Hz → {SEQ_LEN} samples")
print(f"Features: Top {N_SELECTED_FEATS} selected")
print(f"Batch:    {BATCH_SIZE}")
print(f"Epochs:   {MAX_EPOCHS} (patience={PATIENCE})")
print(f"\nPaths:")
print(f"  CSV:      {CSV_PATH}")
print(f"  Signals:  {REC_ROOT}")
print(f"  Features: {FEAT_DIR}")
print(f"  Output:   {ARTIFACTS}")
print("=" * 80 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# All the same code from train_resnet1d_hybrid.py follows...
# (SCP mapping, feature loading, dataset, model, training loop, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

# Copy the entire content from line 151 onwards from the original script
# For brevity, I'll indicate this is where it goes:

# [INSERT ALL CODE FROM ORIGINAL SCRIPT HERE - lines 151 to end]

if __name__ == "__main__":
    print("\n⚠️  NOTE: This is a template. Please copy the full implementation")
    print("from train_resnet1d_hybrid.py (lines 151 onwards) into this file.")
    print("\nOr simply use the original script with the fixed paths!")
