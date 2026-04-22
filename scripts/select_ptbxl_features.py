"""select_ptbxl_features.py

Feature selection for PTB-XL tabular (JSON) features.

This mirrors the notebook-style workflow:
- Load flattened per-lead features
- Impute missing values
- Remove (near-)constant features
- Score with mutual information
- Select top-k features, with an optional minimum per lead

Outputs:
- artifacts/tabular/selected_features.json
- artifacts/tabular/feature_scores.csv

Usage:
  python scripts/select_ptbxl_features.py --k 70
"""

from __future__ import annotations

# Allow running as: python scripts/select_ptbxl_features.py
# (adds project root to sys.path so `import scripts.*` works)
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer

from scripts.ptbxl_tabular_data import CLASSES, find_project_root, load_tabular_dataset


_LEAD_RE = re.compile(r"^(lead[_-]?(?P<lead>\d+)[^_]*)_", flags=re.IGNORECASE)


def _lead_group(col: str) -> str:
    m = _LEAD_RE.match(col)
    if m:
        return f"lead{m.group('lead')}"
    return "__global__"


def _mi_scores_multilabel(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Compute MI per class (binary) and average."""
    scores = []
    for i in range(y.shape[1]):
        yi = y[:, i].astype(int)
        # If a class is all zeros/ones, MI is undefined; treat as zeros.
        if yi.min() == yi.max():
            scores.append(np.zeros(X.shape[1], dtype=float))
            continue
        s = mutual_info_classif(X, yi, discrete_features=False, random_state=random_state)
        scores.append(np.asarray(s, dtype=float))
    return np.mean(np.vstack(scores), axis=0)


def select_features(
    df_X: pd.DataFrame,
    y: np.ndarray,
    k: int,
    min_per_lead: int = 1,
    variance_threshold: float = 0.0,
    random_state: int = 42,
) -> Tuple[List[str], pd.DataFrame]:
    """Return (selected_columns, score_table)."""

    # Guard: remove columns that are completely missing (or inf)
    # SimpleImputer(median) cannot impute such columns and may drop them,
    # which then desyncs feature names vs score arrays.
    df_work = df_X.replace([np.inf, -np.inf], np.nan)
    non_empty_mask = df_work.notna().any(axis=0).to_numpy()
    cols_all = df_work.columns.tolist()
    cols = [c for c, keep in zip(cols_all, non_empty_mask) if keep]
    df_work = df_work[cols]

    if df_work.shape[1] == 0:
        raise ValueError("All feature columns are empty (all-NaN). Check feature extraction outputs.")

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(df_work.values)

    # Variance filtering (keeps same semantics as notebook pipelines)
    if variance_threshold is not None and variance_threshold > 0:
        vt = VarianceThreshold(threshold=float(variance_threshold))
        X_v = vt.fit_transform(X_imp)
        keep_mask = vt.get_support()
        kept_cols = [c for c, keep in zip(cols, keep_mask) if keep]
    else:
        X_v = X_imp
        kept_cols = cols

    mi = _mi_scores_multilabel(X_v, y, random_state=random_state)

    score_df = pd.DataFrame({
        "feature": kept_cols,
        "mi": mi,
        "lead": [_lead_group(c) for c in kept_cols],
    }).sort_values("mi", ascending=False)

    # Lead-aware selection: guarantee at least `min_per_lead` from each lead group
    selected: List[str] = []
    if min_per_lead and min_per_lead > 0:
        for lead, g in score_df.groupby("lead", sort=False):
            if lead == "__global__":
                continue
            top = g.head(min_per_lead)["feature"].tolist()
            selected.extend(top)

    # Fill remaining slots with best MI overall
    for f in score_df["feature"].tolist():
        if f in selected:
            continue
        selected.append(f)
        if len(selected) >= k:
            break

    # If k > available, just return all.
    if k is None or k <= 0:
        k = len(kept_cols)
    selected = selected[: min(int(k), len(selected))]
    score_df["selected"] = score_df["feature"].isin(selected)
    return selected, score_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features-format",
        type=str,
        default="comprehensive",
        choices=["comprehensive", "agg500"],
        help="Feature source format: comprehensive JSON batches or 500Hz agg_features.json hierarchy",
    )
    ap.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Path to features root (format-dependent). Use 'auto' to search (incl. ~/Music on Ubuntu).",
    )
    ap.add_argument("--db-csv", type=str, default=None, help="Path to ptbxl_database.csv")
    ap.add_argument("--k", type=int, default=70, help="Number of features to select")
    ap.add_argument("--min-per-lead", type=int, default=1, help="Minimum features per lead group")
    ap.add_argument("--var-thresh", type=float, default=0.0, help="Variance threshold")
    ap.add_argument("--max-records", type=int, default=None, help="Limit records for quick tests")
    ap.add_argument("--out-dir", type=str, default=None, help="Output dir (default: ECG_Diag_pipeline/artifacts/tabular)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing selected_features.json")

    args = ap.parse_args()

    root = find_project_root()

    fd_arg = (str(args.features_dir).strip() if args.features_dir is not None else "")
    if fd_arg.lower() in ("", "auto"):
        # For agg500, the loader will auto-discover a real folder if this default is empty.
        features_dir = (
            (root / "ptbxl_feature_enhanced2_features_extracted")
            if args.features_format == "agg500"
            else (root / "ptbxl_comprehensive_features")
        )
    else:
        features_dir = Path(fd_arg)
    db_csv = Path(args.db_csv) if args.db_csv else None

    if args.features_format == "agg500":
        from scripts.ptbxl_agg_tabular_data import load_agg_tabular_dataset

        data = load_agg_tabular_dataset(features_root=features_dir, db_csv=db_csv, classes=CLASSES, max_records=args.max_records)
    else:
        data = load_tabular_dataset(features_dir=features_dir, db_csv=db_csv, classes=CLASSES, max_records=args.max_records)

    selected, score_df = select_features(
        data.X,
        data.y,
        k=int(args.k),
        min_per_lead=int(args.min_per_lead),
        variance_threshold=float(args.var_thresh),
    )

    out_dir = Path(args.out_dir) if args.out_dir else (root / "ECG_Diag_pipeline" / "artifacts" / "tabular")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "selected_features.json"
    if out_path.exists() and not args.force:
        print(f"[info] {out_path} exists; skipping (use --force to overwrite)")
        return 0

    (out_dir / "selected_features.json").write_text(
        json.dumps(
            {
                "k": int(args.k),
                "min_per_lead": int(args.min_per_lead),
                "var_thresh": float(args.var_thresh),
                "classes": list(CLASSES),
                "features_format": str(args.features_format),
                "selected": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    score_df.to_csv(out_dir / "feature_scores.csv", index=False)

    print(f"[OK] Selected {len(selected)} features -> {out_dir / 'selected_features.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
