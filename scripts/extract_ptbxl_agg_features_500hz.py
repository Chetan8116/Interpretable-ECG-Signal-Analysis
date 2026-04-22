"""extract_ptbxl_agg_features_500hz.py

Generate per-lead aggregated JSON features from denoised 500Hz PTB-XL signals.

Input:
- A folder like ptbxl_processed/ or ptbxl_denoised/ produced by scripts/denoise_ptbxl_data.py
- processed_metadata.csv inside that folder (preferred)

Output:
- Hierarchical folder with per-lead agg JSON:
    <out-dir>/<record_name>/lead<k>/agg_features.json

These outputs are compatible with scripts/ptbxl_agg_tabular_data.py and can be
used by scripts/train_ptbxl_tabular_rf_lgbm.py with:
  --features-format agg500 --features-dir <out-dir>

This script intentionally keeps the feature set compact + robust (fast to compute):
- RR/HR stats from refined peaks
- QRS-map / QRS-sharp distribution stats
- Basic signal stats (enhanced + raw)

If you want a bigger feature set later, we can extend this safely.
"""

from __future__ import annotations

# Allow running as: python /abs/path/scripts/extract_ptbxl_agg_features_500hz.py
# (adds project root to sys.path so `import scripts.*` works)
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.feature_enhancement import feature_enhancement_pipeline


def _ms_to_samples(ms: float, fs: float) -> int:
    return max(1, int(round(float(ms) * float(fs) / 1000.0)))


def _rr_stats_ms(peaks: np.ndarray, fs: float) -> Dict[str, float]:
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 2:
        return {
            "RR_mean_ms": float("nan"),
            "RR_std_ms": float("nan"),
            "RR_median_ms": float("nan"),
            "RR_rmssd_ms": float("nan"),
            "pNN50": float("nan"),
            "HR_mean_bpm": float("nan"),
        }
    rr_ms = (np.diff(peaks).astype(float) / float(fs)) * 1000.0
    rr_diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(rr_diff**2))) if rr_diff.size > 0 else float("nan")
    pnn50 = float(np.mean(np.abs(rr_diff) > 50.0) * 100.0) if rr_diff.size > 0 else float("nan")
    rr_mean = float(np.mean(rr_ms))
    hr = 60000.0 / rr_mean if rr_mean > 0 else float("nan")
    return {
        "RR_mean_ms": rr_mean,
        "RR_std_ms": float(np.std(rr_ms)),
        "RR_median_ms": float(np.median(rr_ms)),
        "RR_rmssd_ms": rmssd,
        "pNN50": pnn50,
        "HR_mean_bpm": float(hr),
    }


def _stats(prefix: str, arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_std": float("nan"), f"{prefix}_p95": float("nan")}
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_std": float(np.std(a)),
        f"{prefix}_p95": float(np.percentile(a, 95)),
    }


def discover_cleaned_files(proc_out_dir: Path, proc_meta_csv: Optional[Path]) -> List[dict]:
    rows: List[dict] = []
    if proc_meta_csv and proc_meta_csv.exists():
        df = pd.read_csv(proc_meta_csv)
        for _, r in df.iterrows():
            out_path = str(r.get("out_path", ""))
            if out_path and os.path.exists(out_path):
                rows.append(
                    {
                        "record_name": str(r.get("record_name", "")),
                        "lead": int(r.get("lead", -1)),
                        "working_fs": int(r.get("working_fs", 500)),
                        "out_path": out_path,
                    }
                )
        if rows:
            return rows

    # Fallback recursive scan
    for p in proc_out_dir.rglob("*.npy"):
        name = p.name.lower()
        if "lead" not in name:
            continue
        rows.append({"record_name": str(p.parent.relative_to(proc_out_dir).as_posix()), "lead": -1, "working_fs": 500, "out_path": str(p)})
    return rows


def process_one(row: dict, out_dir: Path, params: dict, overwrite: bool) -> dict:
    rec = str(row.get("record_name") or "")
    lead = int(row.get("lead", -1))
    fs = int(row.get("working_fs", 500))
    p = str(row.get("out_path") or "")

    if not rec or not p or not os.path.exists(p):
        return {"record_name": rec, "lead": lead, "status": "missing"}

    lead_dir = out_dir / Path(rec) / f"lead{lead if lead >= 0 else 0}"
    lead_dir.mkdir(parents=True, exist_ok=True)
    out_json = lead_dir / "agg_features.json"
    if out_json.exists() and not overwrite:
        return {"record_name": rec, "lead": lead, "status": "skip_exists"}

    sig = np.load(p)
    sig = np.asarray(sig, dtype=float).reshape(-1)

    res = feature_enhancement_pipeline(sig, fs, params=params)

    peaks = np.asarray(res.get("peaks_refined", []), dtype=int)
    rr = _rr_stats_ms(peaks, fs=float(fs))

    feat: Dict[str, float] = {}
    feat.update(rr)
    feat["n_samples"] = float(sig.size)
    feat["n_peaks"] = float(peaks.size)

    qrs_score = np.asarray(res.get("qrs_score", []), dtype=float)
    qrs_sharp = np.asarray(res.get("qrs_sharp", []), dtype=float)
    enhanced = np.asarray(res.get("enhanced", []), dtype=float)

    feat.update(_stats("qrs_score", qrs_score))
    feat.update(_stats("qrs_sharp", qrs_sharp))
    feat.update(_stats("enhanced", enhanced))
    feat.update(_stats("raw", sig))

    # Simple peak amplitude stats around peaks (10ms window)
    half = _ms_to_samples(10.0, fs)
    if peaks.size > 0:
        amps = []
        for pk in peaks:
            s = max(0, int(pk) - half)
            e = min(sig.size, int(pk) + half + 1)
            amps.append(float(np.max(np.abs(sig[s:e]))))
        feat.update(_stats("r_amp_abs", np.asarray(amps, dtype=float)))

    out_json.write_text(json.dumps(feat, indent=2), encoding="utf-8")
    return {"record_name": rec, "lead": lead, "status": "ok"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-out-dir", type=str, default=None, help="Folder containing denoised .npy and processed_metadata.csv")
    ap.add_argument("--proc-meta-csv", type=str, default=None, help="Path to processed_metadata.csv (optional)")
    ap.add_argument("--out-dir", type=str, default=None, help="Output folder for agg_features.json hierarchy")
    ap.add_argument("--max-files", type=int, default=None, help="Limit number of lead files for quick runs")
    ap.add_argument("--overwrite", action="store_true")

    # Enhancement hyperparams
    ap.add_argument("--fs", type=int, default=500)
    ap.add_argument("--win-ms", type=int, default=120)
    ap.add_argument("--hop-ms", type=int, default=20)
    ap.add_argument("--nl-power", type=float, default=2.5)
    ap.add_argument("--alpha", type=float, default=2.0)

    args = ap.parse_args()

    root = Path.cwd()
    proc_out_dir = Path(args.proc_out_dir) if args.proc_out_dir else (root / "ptbxl_processed")
    proc_meta_csv = Path(args.proc_meta_csv) if args.proc_meta_csv else (proc_out_dir / "processed_metadata.csv")
    out_dir = Path(args.out_dir) if args.out_dir else (root / "ptbxl_feature_enhanced2_features_extracted")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_cleaned_files(proc_out_dir, proc_meta_csv)
    if args.max_files is not None:
        files = files[: int(args.max_files)]

    if not files:
        raise SystemExit(f"No cleaned .npy files found under {proc_out_dir}")

    params = {
        "fs": int(args.fs),
        "win_ms": int(args.win_ms),
        "hop_ms": int(args.hop_ms),
        "nl_power": float(args.nl_power),
        "alpha": float(args.alpha),
        "verbose": False,
    }

    meta_rows = []
    for row in files:
        meta_rows.append(process_one(row, out_dir=out_dir, params=params, overwrite=bool(args.overwrite)))

    pd.DataFrame(meta_rows).to_csv(out_dir / "agg_features_metadata.csv", index=False)
    print(f"[OK] Wrote agg features to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
