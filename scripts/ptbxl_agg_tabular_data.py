"""ptbxl_agg_tabular_data.py

Load tabular datasets from a 500Hz feature-enhancement pipeline output.

Expected folder structure (features_root):
  <features_root>/<record_key>/lead<k>/*agg_features*.json

Where <record_key> is typically the PTB-XL filename (without extension), e.g.:
  records500/00000/00001_hr

This loader:
- Builds a wide DataFrame by concatenating per-lead aggregated features.
- Maps record_key back to PTB-XL labels using ptbxl_database.csv.

It returns the same LoadedTabular dataclass used by scripts/ptbxl_tabular_data.py.

Notes
- This is meant for your "original" (records500 / 500Hz) pipeline.
- If some records cannot be matched to ptbxl_database.csv, they are skipped.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.ptbxl_tabular_data import LoadedTabular, find_project_root, find_ptbxl_database_csv, scp_to_multilabel


def _as_posix_rel(path: Path) -> str:
    return path.as_posix().lstrip("./")


def iter_agg_jsons(features_root: Path) -> Iterable[Path]:
    # Accept multiple naming conventions
    yield from features_root.rglob("*agg_features*.json")


def _looks_like_agg_root(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    if (p / "agg_features_metadata.csv").exists():
        return True
    try:
        return next(p.rglob("*agg_features*.json"), None) is not None
    except Exception:
        return False


def _discover_agg_feature_roots(search_root: Path, max_hits: int = 5, max_depth: int = 8) -> List[Path]:
    """Find candidate feature roots by locating agg_features_metadata.csv (preferred).

    This is intentionally bounded: it only walks a limited depth and stops after
    a few hits to avoid scanning an entire home directory.
    """
    hits: List[Path] = []
    if not search_root.exists() or not search_root.is_dir():
        return hits

    try:
        for dirpath, dirnames, filenames in os.walk(str(search_root)):
            cur = Path(dirpath)
            try:
                depth = len(cur.relative_to(search_root).parts)
            except Exception:
                depth = 0
            if depth > max_depth:
                dirnames[:] = []
                continue

            # prune common large/unrelated dirs
            for bad in [".git", "node_modules", "__pycache__", ".venv", "venv"]:
                if bad in dirnames:
                    dirnames.remove(bad)

            if "agg_features_metadata.csv" in filenames:
                hits.append(cur)
                if len(hits) >= int(max_hits):
                    break
    except Exception:
        return hits
    return hits


def find_agg_features_dir(root: Path, preferred: Optional[Path] = None) -> Path:
    """Locate the agg500 features root without hardcoding absolute paths.

    If the caller-provided path is missing/empty, this searches under a few
    likely roots, including ~/Music (Ubuntu use case).
    """
    if preferred is not None:
        preferred = Path(preferred)
        # Treat the common placeholder as "unset"
        if str(preferred).strip() and "path/to" not in str(preferred).replace("\\", "/"):
            if _looks_like_agg_root(preferred):
                return preferred

    direct = root / "ptbxl_feature_enhanced2_features_extracted"
    if _looks_like_agg_root(direct):
        return direct

    search_roots: List[Path] = []
    try:
        search_roots.append(Path.cwd())
    except Exception:
        pass
    search_roots.append(root)
    try:
        home = Path.home()
        # Ubuntu request: search inside Music
        search_roots.append(home / "Music")
        search_roots.append(home)
    except Exception:
        pass

    candidates: List[Path] = []
    for sr in search_roots:
        # Quick direct name hit
        named = sr / "ptbxl_feature_enhanced2_features_extracted"
        if _looks_like_agg_root(named):
            candidates.append(named)
        # Prefer metadata file roots
        candidates.extend(_discover_agg_feature_roots(sr))

    # Deduplicate while keeping order
    seen = set()
    uniq: List[Path] = []
    for c in candidates:
        cc = c.resolve() if c.exists() else c
        if str(cc) in seen:
            continue
        seen.add(str(cc))
        uniq.append(c)

    if not uniq:
        # last resort: return conventional path even if missing
        return direct if preferred is None else preferred

    # Choose the newest (by metadata mtime) when available
    def _score(p: Path) -> float:
        meta = p / "agg_features_metadata.csv"
        try:
            return float(meta.stat().st_mtime) if meta.exists() else float(p.stat().st_mtime)
        except Exception:
            return 0.0

    uniq.sort(key=_score, reverse=True)
    return uniq[0]


def _parse_lead_from_path(p: Path) -> str:
    # lead folder usually named lead0, lead11, etc.
    for part in p.parts[::-1]:
        if part.lower().startswith("lead"):
            return part.lower()
    return "lead_unknown"


def load_agg_features_dataframe(features_root: Path, max_records: Optional[int] = None) -> pd.DataFrame:
    t0 = time.time()

    rows: Dict[str, Dict[str, float]] = {}

    json_paths = sorted(iter_agg_jsons(features_root))
    if not json_paths:
        print(f"[load] No agg feature JSON files found under: {features_root}")
        return pd.DataFrame({"record_key": pd.Series(dtype=str)})

    # Group by record_key (relative path up to the lead folder)
    for p in json_paths:
        try:
            rel = p.relative_to(features_root)
        except Exception:
            rel = p

        # record_key = rel path up to (but excluding) the lead folder
        parts = list(rel.parts)
        if not parts:
            continue

        # find index of lead folder
        lead_idx = None
        for i, part in enumerate(parts):
            if part.lower().startswith("lead"):
                lead_idx = i
                break
        if lead_idx is None or lead_idx == 0:
            # can't infer record
            continue

        record_key = _as_posix_rel(Path(*parts[:lead_idx]))
        lead_name = parts[lead_idx].lower()

        if max_records is not None and record_key not in rows and len(rows) >= int(max_records):
            continue

        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict) or not obj:
            continue

        row = rows.setdefault(record_key, {})
        for k, v in obj.items():
            if isinstance(v, bool):
                continue
            if v is None:
                continue
            # Keep numeric only; JSON may have strings for debug
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isnan(fv) or np.isinf(fv):
                continue
            col = f"{lead_name}__{k}"
            row[col] = fv

    if not rows:
        return pd.DataFrame({"record_key": pd.Series(dtype=str)})

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.insert(0, "record_key", df.index.astype(str))

    elapsed = time.time() - t0
    print(f"[load] agg features: records={df.shape[0]} cols={df.shape[1]-1} in {elapsed:.1f}s")
    return df


def _build_db_lookup(db_csv: Path) -> Dict[str, dict]:
    df = pd.read_csv(db_csv, low_memory=False)

    out: Dict[str, dict] = {}
    for _, r in df.iterrows():
        # filename_hr / filename_lr are stored without extension
        for col in ("filename_hr", "filename_lr", "filename"):
            if col not in df.columns:
                continue
            key = r.get(col)
            if not isinstance(key, str) or not key.strip():
                continue
            key = key.strip().replace("\\", "/")
            out[key] = {
                "ecg_id": r.get("ecg_id"),
                "scp_codes": r.get("scp_codes"),
                "strat_fold": r.get("strat_fold"),
            }
    return out


def load_agg_tabular_dataset(
    features_root: Path,
    db_csv: Optional[Path] = None,
    classes: Sequence[str] = ("CD", "HYP", "MI", "NORM", "STTC"),
    max_records: Optional[int] = None,
) -> LoadedTabular:
    root = find_project_root()
    if db_csv is None:
        db_csv = find_ptbxl_database_csv(root)

    requested_root = Path(features_root)
    features_root = find_agg_features_dir(root=root, preferred=requested_root)
    if features_root != requested_root:
        print(f"[discover] agg500 features-dir: {features_root} (requested: {requested_root})")

    df = load_agg_features_dataframe(features_root=features_root, max_records=max_records)
    if df.shape[0] == 0:
        raise RuntimeError(
            f"No agg features loaded from: {features_root}. "
            f"Tip: run scripts/extract_ptbxl_agg_features_500hz.py first; it should create agg_features_metadata.csv in the output folder."
        )

    lookup = _build_db_lookup(Path(db_csv))

    # Map record_key -> labels
    y_list: List[np.ndarray] = []
    primary_list: List[str] = []
    record_id_list: List[int] = []
    strat_fold_list: List[float] = []

    keep_rows: List[int] = []
    for i, key in enumerate(df["record_key"].astype(str).tolist()):
        key_norm = key.replace("\\", "/")
        meta = lookup.get(key_norm)
        if meta is None:
            # Sometimes enhancement output may only include the basename (e.g. 00001_hr)
            # Try a suffix match (unique only)
            hits = [k for k in lookup.keys() if k.endswith("/" + key_norm) or k.endswith(key_norm)]
            if len(hits) == 1:
                meta = lookup[hits[0]]
        if meta is None:
            continue

        y, primary = scp_to_multilabel(meta.get("scp_codes"), classes=classes)

        ecg_id = meta.get("ecg_id")
        try:
            ecg_id_int = int(ecg_id)
        except Exception:
            continue

        sf = meta.get("strat_fold")
        try:
            sf_val = float(sf) if sf is not None and str(sf) != "" else float("nan")
        except Exception:
            sf_val = float("nan")

        keep_rows.append(i)
        y_list.append(y)
        primary_list.append(primary or "")
        record_id_list.append(ecg_id_int)
        strat_fold_list.append(sf_val)

    if not keep_rows:
        raise RuntimeError(
            "Loaded agg features but could not match any records to ptbxl_database.csv. "
            "Check that your record folders preserve the PTB-XL filename_hr path (e.g., records500/00000/00001_hr)."
        )

    df_keep = df.iloc[keep_rows].reset_index(drop=True)
    X = df_keep.drop(columns=["record_key"]).copy()
    record_ids = np.asarray(record_id_list, dtype=int)
    primary = np.asarray(primary_list, dtype=object)
    strat_fold = np.asarray(strat_fold_list, dtype=float)
    y_arr = np.vstack(y_list).astype(int)

    return LoadedTabular(
        X=X,
        y=y_arr,
        record_ids=record_ids,
        primary=primary,
        strat_fold=strat_fold,
        classes=tuple(classes),
    )
