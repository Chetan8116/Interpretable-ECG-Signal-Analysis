"""ptbxl_tabular_data.py

Utilities for training tabular ML models from precomputed PTB-XL feature JSON.

This project already contains scripts that flatten the JSON feature batches.
This module centralizes that logic and adds robust label extraction.

Expected feature format (from ptbxl_comprehensive_features/*.json):
[
  {
    "record_id": 1,
    "success": true,
    "features": {
      "lead_0_I": {"stat_mean": ..., ...},
      ...
    }
  },
  ...
]

Labels are loaded from PTB-XL's ptbxl_database.csv (scp_codes).
We support multi-label targets for super-classes: [CD, HYP, MI, NORM, STTC].
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CLASSES: List[str] = ["CD", "HYP", "MI", "NORM", "STTC"]

# Map SCP codes to diagnostic super-classes (same mapping used across scripts).
SCP_MAP: Dict[str, str] = {
    "NORM": "NORM",

    # MI
    "IMI": "MI",
    "ILMI": "MI",
    "AMI": "MI",
    "ALMI": "MI",
    "INJAS": "MI",
    "LMI": "MI",
    "INJAL": "MI",
    "IPLMI": "MI",
    "IPMI": "MI",
    "INJIN": "MI",
    "INJLA": "MI",
    "PMI": "MI",
    "INJIL": "MI",
    "INJA": "MI",

    # ST/T changes
    "NDT": "STTC",
    "DIG": "STTC",
    "LNGQT": "STTC",
    "ANEUR": "STTC",
    "EL": "STTC",
    "ISCA": "STTC",
    "ISCI": "STTC",
    "ISC_": "STTC",
    "STTC": "STTC",
    "STD_": "STTC",
    "STE_": "STTC",

    # Conduction disturbances
    "LAFB": "CD",
    "IRBBB": "CD",
    "IVCD": "CD",
    "LBBB": "CD",
    "RBBB": "CD",
    "LPFB": "CD",
    "WPW": "CD",
    "1AVB": "CD",
    "2AVB": "CD",
    "3AVB": "CD",
    "AVB": "CD",

    # Hypertrophy
    "LVH": "HYP",
    "LAO": "HYP",
    "RVH": "HYP",
    "SEHYP": "HYP",
    "LVOLT": "HYP",
    "RAO": "HYP",
    "LMH": "HYP",
}

SKIP_KEYS = {"lead_name", "sampling_freq"}


def find_project_root() -> Path:
    """Find project root on Windows/Linux without hardcoding paths."""
    script_path = Path(__file__).resolve()
    candidates = [
        script_path.parent.parent,
        script_path.parent,
        Path.cwd(),
        Path.home() / "Pictures" / "RM",
        Path.home() / "RM",
    ]
    for c in candidates:
        if (c / "ptbxl_comprehensive_features").exists():
            return c
        archive = c / "archive"
        if archive.exists():
            # Only accept if it looks like PTB-XL
            if (archive / "ptbxl_database.csv").exists():
                return c
            try:
                if any(p.is_dir() and "ptb-xl" in p.name.lower() for p in archive.iterdir()):
                    return c
            except Exception:
                pass
        if (c / "public" / "ptbxl_database.csv").exists():
            return c
    return script_path.parent.parent


def find_ptbxl_database_csv(root: Path) -> Path:
    archive = root / "archive"
    if archive.exists():
        hits = list(archive.rglob("ptbxl_database.csv"))
        if hits:
            return hits[0]
    # Some scripts use public/ptbxl_database.csv
    for p in [root / "public" / "ptbxl_database.csv", root / "ptbxl_database.csv"]:
        if p.exists():
            return p
    return archive / "ptbxl_database.csv"


def find_features_dir(root: Path) -> Path:
    """Locate the ptbxl_comprehensive_features directory.

    This avoids hardcoding absolute paths and supports moving the feature folder
    around (common when copying to Ubuntu machines).
    """
    direct = root / "ptbxl_comprehensive_features"
    if direct.exists() and direct.is_dir():
        return direct

    # Search a bit (bounded) for a directory named ptbxl_comprehensive_features
    # that actually contains batch_*_features.json or all_features.json.
    candidates = list(root.rglob("ptbxl_comprehensive_features"))
    for c in candidates:
        if not c.is_dir():
            continue
        if list(c.glob("batch_*_features.json")) or (c / "all_features.json").exists():
            return c

    # As a fallback, search from CWD too (helpful when running outside repo root)
    cwd = Path.cwd()
    if cwd != root:
        direct2 = cwd / "ptbxl_comprehensive_features"
        if direct2.exists() and direct2.is_dir():
            return direct2
        candidates2 = list(cwd.rglob("ptbxl_comprehensive_features"))
        for c in candidates2:
            if c.is_dir() and (list(c.glob("batch_*_features.json")) or (c / "all_features.json").exists()):
                return c

    return direct


def _parse_scp_codes(scp_codes: object) -> Dict[str, float]:
    """Return dict(code -> confidence)."""
    if isinstance(scp_codes, dict):
        return {str(k): float(v) for k, v in scp_codes.items()}
    if not isinstance(scp_codes, str):
        return {}
    try:
        parsed = ast.literal_eval(scp_codes)
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
    except Exception:
        return {}
    return {}


def scp_to_multilabel(
    scp_codes: object,
    classes: Sequence[str] = CLASSES,
    min_conf: float = 0.0,
) -> Tuple[np.ndarray, Optional[str]]:
    """Map raw scp_codes to a multi-label indicator vector.

    Returns (y_vec, primary_class) where primary_class is the highest-confidence
    superclass (for stratification convenience).
    """
    codes = _parse_scp_codes(scp_codes)
    scored: Dict[str, float] = {}
    for code, conf in codes.items():
        try:
            conf_f = float(conf)
        except Exception:
            continue
        if conf_f <= float(min_conf):
            continue
        superclass = SCP_MAP.get(code)
        if superclass is None:
            continue
        scored[superclass] = scored.get(superclass, 0.0) + float(conf_f)

    y = np.zeros(len(classes), dtype=int)
    for i, c in enumerate(classes):
        if scored.get(c, 0.0) > 0.0:
            y[i] = 1

    primary = max(scored, key=scored.get) if scored else None
    return y, primary


def flatten_record(features_obj: dict) -> Dict[str, float]:
    """Flatten nested lead feature dict into a single row."""
    flat: Dict[str, float] = {}
    for lead_key, lead_feats in (features_obj or {}).items():
        if not isinstance(lead_feats, dict):
            continue
        for fname, val in lead_feats.items():
            if fname in SKIP_KEYS:
                continue
            col = f"{lead_key}_{fname}"
            try:
                flat[col] = float(val) if val is not None else np.nan
            except Exception:
                flat[col] = np.nan
    return flat


@dataclass(frozen=True)
class LoadedTabular:
    X: pd.DataFrame
    y: np.ndarray
    record_ids: np.ndarray
    primary: np.ndarray
    strat_fold: Optional[np.ndarray]
    classes: Tuple[str, ...]


def iter_feature_batches(features_dir: Path) -> Iterable[Path]:
    # Prefer batch files for memory (all_features.json can be huge)
    batch_files = sorted(features_dir.glob("batch_*_features.json"))
    if batch_files:
        yield from batch_files
    else:
        # fallback
        allf = features_dir / "all_features.json"
        if allf.exists():
            yield allf


def load_features_dataframe(features_dir: Path, max_records: Optional[int] = None) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    n_ok = 0
    t0 = time.time()

    batch_list = list(iter_feature_batches(features_dir))
    if not batch_list:
        # Return an empty frame with the expected key to avoid confusing merge errors.
        print(f"[load] No feature JSON files found under: {features_dir}")
        return pd.DataFrame({"record_id": pd.Series(dtype=int)})

    for bi, bf in enumerate(batch_list):
        try:
            payload = json.loads(bf.read_text(encoding="utf-8"))
        except Exception:
            continue

        if isinstance(payload, dict):
            payload = [payload]

        for rec in payload:
            if not rec.get("success", False):
                continue
            rid = rec.get("record_id")
            feat = rec.get("features")
            if rid is None or not isinstance(feat, dict):
                continue

            row = flatten_record(feat)
            row["record_id"] = int(rid)
            records.append(row)
            n_ok += 1

            if max_records is not None and n_ok >= max_records:
                break

        if (bi + 1) % 50 == 0:
            print(f"[load] {bi+1:>3} files  loaded={n_ok}  {time.time()-t0:.0f}s", flush=True)

        if max_records is not None and n_ok >= max_records:
            break

    df = pd.DataFrame(records)
    if df.empty:
        print(f"[load] rows=0 cols=0  {time.time()-t0:.0f}s")
        return pd.DataFrame({"record_id": pd.Series(dtype=int)})

    if "record_id" not in df.columns:
        # Extremely defensive (shouldn't happen with our writer)
        raise ValueError(
            "Loaded features but no record_id column was produced. "
            "Verify JSON entries contain a top-level record_id."
        )

    df = df.drop_duplicates(subset=["record_id"], keep="first")
    print(f"[load] rows={len(df)} cols={df.shape[1]}  {time.time()-t0:.0f}s")
    return df


def load_labels_multilabel(db_csv: Path, classes: Sequence[str] = CLASSES) -> pd.DataFrame:
    db = pd.read_csv(db_csv)
    # PTB-XL uses ecg_id as unique record id
    if "ecg_id" not in db.columns:
        raise ValueError(f"ptbxl_database.csv missing ecg_id column: {db_csv}")

    y_rows = []
    for _, r in db.iterrows():
        y_vec, primary = scp_to_multilabel(r.get("scp_codes"), classes=classes)
        # Keep rows that map to at least one superclass
        if int(y_vec.sum()) == 0:
            continue
        y_rows.append(
            {
                "record_id": int(r["ecg_id"]),
                "primary": primary if primary is not None else "",
                "strat_fold": int(r["strat_fold"]) if "strat_fold" in db.columns and pd.notna(r.get("strat_fold")) else np.nan,
                **{c: int(y_vec[i]) for i, c in enumerate(classes)},
            }
        )

    out = pd.DataFrame(y_rows)
    if out.empty:
        raise ValueError("No labels produced from scp_codes; check SCP_MAP.")
    return out


def load_tabular_dataset(
    features_dir: Optional[Path] = None,
    db_csv: Optional[Path] = None,
    classes: Sequence[str] = CLASSES,
    max_records: Optional[int] = None,
) -> LoadedTabular:
    root = find_project_root()
    features_dir = features_dir or find_features_dir(root)
    db_csv = db_csv or find_ptbxl_database_csv(root)

    # If user passed a path that doesn't exist, try to auto-resolve.
    if features_dir is not None and not features_dir.exists():
        features_dir = find_features_dir(root)

    X = load_features_dataframe(features_dir, max_records=max_records)
    if "record_id" not in X.columns or len(X) == 0:
        raise ValueError(
            "No features were loaded. Ensure the feature folder exists and contains "
            "batch_*_features.json (or all_features.json). "
            f"Resolved features_dir={features_dir}"
        )

    ydf = load_labels_multilabel(db_csv, classes=classes)

    merged = X.merge(ydf, on="record_id", how="inner")

    if merged.empty:
        raise ValueError(
            "Features and labels loaded, but merge produced 0 rows. "
            "This usually means record_id values in feature JSON don't match ecg_id in ptbxl_database.csv. "
            f"features_dir={features_dir} db_csv={db_csv}"
        )

    y = merged[list(classes)].astype(int).to_numpy()
    record_ids = merged["record_id"].astype(int).to_numpy()
    primary = merged["primary"].astype(str).to_numpy()

    strat_fold = None
    if "strat_fold" in merged.columns:
        # Keep NaNs as None
        sf = merged["strat_fold"].to_numpy()
        strat_fold = sf

    X = merged.drop(columns=["record_id", "primary", "strat_fold", *list(classes)], errors="ignore")

    return LoadedTabular(
        X=X,
        y=y,
        record_ids=record_ids,
        primary=primary,
        strat_fold=strat_fold,
        classes=tuple(classes),
    )
