#!/usr/bin/env python3
"""
Quick diagnostic script to check PTB-XL dataset setup.

Usage:
    python check_setup.py
"""

from pathlib import Path
import sys

print("=" * 80)
print("  PTB-XL Dataset Setup Checker")
print("=" * 80)
print()

# Try to find project root
script_path = Path(__file__).resolve()
candidates = [
    script_path.parent,
    script_path.parent.parent,
    Path.cwd(),
    Path.home() / "Music",
    Path.home() / "Pictures" / "RM",
]

print("🔍 Searching for project root...")
print()

found_root = None
for i, candidate in enumerate(candidates, 1):
    print(f"  [{i}] Checking: {candidate}")
    
    checks = []
    
    # Check for features
    if (candidate / "ptbxl_comprehensive_features").exists():
        feat_count = len(list((candidate / "ptbxl_comprehensive_features").glob("*.json")))
        checks.append(f"✓ Features ({feat_count} files)")
    else:
        checks.append("✗ Features missing")
    
    # Check for archive
    if (candidate / "archive").exists():
        archive_subdirs = [d for d in (candidate / "archive").iterdir() if d.is_dir()]
        ptbxl_dirs = [d for d in archive_subdirs if "ptb-xl" in d.name.lower()]
        if ptbxl_dirs:
            checks.append(f"✓ Archive ({ptbxl_dirs[0].name[:30]}...)")
        else:
            checks.append("✗ Archive (no ptb-xl folder)")
    else:
        checks.append("✗ Archive missing")
    
    print(f"      {' | '.join(checks)}")
    
    if all("✓" in c for c in checks):
        found_root = candidate
        print(f"      → FOUND! This looks like the project root")
        break

print()

if found_root is None:
    print("⚠️  WARNING: Could not auto-detect project root")
    print()
    print("Please specify paths manually or ensure folder structure is:")
    print("  project/")
    print("    ├── archive/")
    print("    │   └── ptb-xl-.../")
    print("    │       ├── ptbxl_database.csv")
    print("    │       └── records500/")
    print("    └── ptbxl_comprehensive_features/")
    print()
    sys.exit(1)

print("=" * 80)
print("  Detailed Path Check")
print("=" * 80)
print()

ROOT = found_root

# Find archive
print("📁 Archive Search:")
archive_dir = ROOT / "archive"
if archive_dir.exists():
    for subdir in archive_dir.iterdir():
        if subdir.is_dir() and "ptb-xl" in subdir.name.lower():
            ARCHIVE = subdir
            print(f"  ✓ Found: {ARCHIVE.name}")
            break
else:
    print(f"  ✗ Archive directory not found at {archive_dir}")
    ARCHIVE = None

if ARCHIVE:
    # Check CSV
    CSV_PATH = ARCHIVE / "ptbxl_database.csv"
    if CSV_PATH.exists():
        size_mb = CSV_PATH.stat().st_size / 1024 / 1024
        print(f"  ✓ CSV: {CSV_PATH.name} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ CSV not found: {CSV_PATH}")
    
    # Check signals
    REC_ROOT = ARCHIVE / "records500"
    if REC_ROOT.exists():
        sample_dirs = list(REC_ROOT.glob("*"))[:5]
        print(f"  ✓ Signals: records500/ ({len(sample_dirs)} subdirs checked)")
    else:
        print(f"  ✗ Signals not found: {REC_ROOT}")

print()
print("📊 Features Check:")
FEAT_DIR = ROOT / "ptbxl_comprehensive_features"
if FEAT_DIR.exists():
    json_files = list(FEAT_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in json_files) / 1024 / 1024
    print(f"  ✓ Directory: {FEAT_DIR}")
    print(f"  ✓ Files: {len(json_files)} JSON files")
    print(f"  ✓ Size: {total_size:.1f} MB")
    
    if len(json_files) < 220:
        print(f"  ⚠️  WARNING: Expected 220 files, found {len(json_files)}")
        print(f"      Feature extraction may be incomplete")
else:
    print(f"  ✗ Features directory not found: {FEAT_DIR}")

print()
print("🐍 Python Environment:")
try:
    import torch
    print(f"  ✓ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"    ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"    ⚠️  CUDA not available (will use CPU)")
except ImportError:
    print("  ✗ PyTorch not installed")

try:
    import wfdb
    print(f"  ✓ wfdb-python: installed")
except ImportError:
    print("  ✗ wfdb-python not installed (pip install wfdb)")

try:
    import sklearn
    print(f"  ✓ scikit-learn: {sklearn.__version__}")
except ImportError:
    print("  ✗ scikit-learn not installed")

try:
    import pandas as pd
    print(f"  ✓ pandas: {pd.__version__}")
except ImportError:
    print("  ✗ pandas not installed")

print()
print("=" * 80)
print("  Summary")
print("=" * 80)
print()

# Final verdict
all_good = True
issues = []

if ARCHIVE is None or not ARCHIVE.exists():
    all_good = False
    issues.append("Archive directory not found")

if ARCHIVE and not (ARCHIVE / "ptbxl_database.csv").exists():
    all_good = False
    issues.append("ptbxl_database.csv not found")

if ARCHIVE and not (ARCHIVE / "records500").exists():
    all_good = False
    issues.append("records500/ directory not found")

if not FEAT_DIR.exists():
    all_good = False
    issues.append("Feature directory not found")
elif len(list(FEAT_DIR.glob("*.json"))) < 200:
    all_good = False
    issues.append("Insufficient feature files (need 220)")

if all_good:
    print("✅ READY TO TRAIN!")
    print()
    print("All required files and directories found.")
    print("You can now run:")
    print()
    print(f"  cd {ROOT}")
    print("  python trainres.py")
    print()
else:
    print("❌ SETUP INCOMPLETE")
    print()
    print("Issues found:")
    for issue in issues:
        print(f"  • {issue}")
    print()
    print("Please fix the issues above before training.")
    print()

print("=" * 80)
