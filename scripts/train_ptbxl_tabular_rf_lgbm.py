"""train_ptbxl_tabular_rf_lgbm.py

Train strong tabular baseline models on PTB-XL comprehensive features:
- RandomForest (multi-label One-vs-Rest via per-class training)
- LightGBM (optional, per-class binary models)
- Optional probability averaging ensemble (rf + lgbm)

Metrics reported (as requested):
- AUC (Macro), AUC (Micro)
- Micro F1, Macro F1
- Hamming Loss

Artifacts saved for integration into project prediction flows:
- artifacts/tabular/selected_features.json (if not already)
- artifacts/tabular/imputer.joblib
- artifacts/tabular/models_*.joblib
- artifacts/tabular/metrics.json

Usage:
  python scripts/train_ptbxl_tabular_rf_lgbm.py --model rf
  python scripts/train_ptbxl_tabular_rf_lgbm.py --model lgbm
  python scripts/train_ptbxl_tabular_rf_lgbm.py --model ensemble

Notes:
- This script auto-discovers the project root and feature folder.
- LightGBM is optional: pip install lightgbm
"""

from __future__ import annotations

# Allow running as: python scripts/train_ptbxl_tabular_rf_lgbm.py
# (adds project root to sys.path so `import scripts.*` works)
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from scripts.ptbxl_tabular_data import CLASSES, LoadedTabular, find_project_root, load_tabular_dataset

try:
    import joblib
except Exception as e:  # pragma: no cover
    raise SystemExit("Install joblib (usually included with scikit-learn): pip install joblib")


def _load_selected_features(artifacts_dir: Path) -> Tuple[Optional[List[str]], Dict[str, object]]:
    p = artifacts_dir / "selected_features.json"
    if not p.exists():
        return None, {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        sel = obj.get("selected")
        if isinstance(sel, list) and sel:
            return [str(x) for x in sel], (obj if isinstance(obj, dict) else {})
    except Exception:
        return None, {}
    return None, {}


def _ensure_selected_features(
    root: Path,
    artifacts_dir: Path,
    features_dir: Path,
    db_csv: Optional[Path],
    k: int,
    min_per_lead: int,
    var_thresh: float,
    max_records: Optional[int],
    force: bool,
    features_format: str,
) -> List[str]:
    existing, meta = _load_selected_features(artifacts_dir)
    if existing is not None and not force:
        try:
            meta_k = int(meta.get("k")) if isinstance(meta, dict) and meta.get("k") is not None else None
        except Exception:
            meta_k = None
        meta_fmt = str(meta.get("features_format")) if isinstance(meta, dict) and meta.get("features_format") else None
        if meta_k == int(k) and (meta_fmt is None or meta_fmt == str(features_format)):
            return existing
        print(
            f"[info] selected_features.json exists but doesn't match requested settings "
            f"(file k={meta_k}, requested k={k}, file fmt={meta_fmt}, requested fmt={features_format}); "
            f"recomputing (use --force-select to always recompute)."
        )

    # Lazy import to avoid heavy deps in inference contexts
    from scripts.select_ptbxl_features import select_features

    if features_format == "agg500":
        from scripts.ptbxl_agg_tabular_data import load_agg_tabular_dataset

        data = load_agg_tabular_dataset(features_root=features_dir, db_csv=db_csv, classes=CLASSES, max_records=max_records)
    else:
        data = load_tabular_dataset(features_dir=features_dir, db_csv=db_csv, classes=CLASSES, max_records=max_records)
    selected, score_df = select_features(
        data.X,
        data.y,
        k=k,
        min_per_lead=min_per_lead,
        variance_threshold=var_thresh,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "selected_features.json").write_text(
        json.dumps(
            {
                "k": int(k),
                "min_per_lead": int(min_per_lead),
                "var_thresh": float(var_thresh),
                "classes": list(CLASSES),
                "features_format": str(features_format),
                "selected": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    score_df.to_csv(artifacts_dir / "feature_scores.csv", index=False)
    return selected


def _split_data(data: LoadedTabular, test_size: float, random_state: int = 42):
    """Create train/val/test split.

    Prefer PTB-XL strat_fold if present; otherwise random split.
    """
    if data.strat_fold is not None:
        sf = data.strat_fold
        # Typical PTB-XL split: folds 1-8 train, 9 val, 10 test
        if np.isfinite(sf).all() and set(np.unique(sf.astype(int))).issuperset({1, 8, 9, 10}):
            train_mask = sf.astype(int) <= 8
            val_mask = sf.astype(int) == 9
            test_mask = sf.astype(int) == 10
            return (
                train_mask,
                val_mask,
                test_mask,
            )

    # Fallback random split using primary label proxy when available
    idx = np.arange(len(data.record_ids))
    strat = None
    if data.primary is not None and len(data.primary) == len(idx):
        # If empty strings, don't stratify
        if np.any(pd.Series(data.primary).astype(str).str.len() > 0):
            strat = data.primary

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=strat if strat is not None else None,
    )
    # carve out val from train
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=min(0.15, max(0.05, test_size / 2.0)),
        random_state=random_state,
        shuffle=True,
        stratify=(strat[train_idx] if strat is not None else None),
    )

    train_mask = np.zeros(len(idx), dtype=bool)
    val_mask = np.zeros(len(idx), dtype=bool)
    test_mask = np.zeros(len(idx), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _safe_auc_multilabel(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    """Compute macro/micro AUC robustly even when some labels are degenerate in y_true."""
    # Macro: average per-class AUC where defined
    aucs: List[float] = []
    for ci in range(y_true.shape[1]):
        yt = y_true[:, ci].astype(int)
        if yt.min() == yt.max():
            continue
        try:
            aucs.append(float(roc_auc_score(yt, prob[:, ci])))
        except Exception:
            continue
    auc_macro = float(np.mean(aucs)) if aucs else float("nan")

    # Micro: flatten labels
    try:
        auc_micro = float(roc_auc_score(y_true.ravel().astype(int), prob.ravel().astype(float)))
    except Exception:
        try:
            auc_micro = float(roc_auc_score(y_true, prob, average="micro"))
        except Exception:
            auc_micro = float("nan")
    return auc_macro, auc_micro


def _apply_thresholds(prob: np.ndarray, thresholds: object) -> np.ndarray:
    if isinstance(thresholds, (float, int)):
        return (prob >= float(thresholds)).astype(int)
    thr = np.asarray(thresholds, dtype=float).reshape(1, -1)
    if thr.shape[1] != prob.shape[1]:
        raise ValueError(f"thresholds shape mismatch: got {thr.shape} for prob {prob.shape}")
    return (prob >= thr).astype(int)


def _evaluate_multilabel(y_true: np.ndarray, prob: np.ndarray, thresholds: object = 0.5) -> Dict[str, float]:
    y_pred = _apply_thresholds(prob, thresholds)
    auc_macro, auc_micro = _safe_auc_multilabel(y_true, prob)
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    hl = float(hamming_loss(y_true, y_pred))
    return {
        "auc_macro": float(auc_macro),
        "auc_micro": float(auc_micro),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "hamming_loss": hl,
    }


def _optimize_thresholds_per_class(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    grid_size: int = 41,
    min_thr: float = 0.05,
    max_thr: float = 0.95,
) -> np.ndarray:
    """Pick per-class thresholds that maximize per-class F1 on the validation set."""
    thr_grid = np.linspace(float(min_thr), float(max_thr), int(grid_size))
    best = np.full(prob_val.shape[1], 0.5, dtype=float)
    for ci in range(prob_val.shape[1]):
        yt = y_val[:, ci].astype(int)
        if yt.min() == yt.max():
            best[ci] = 0.5
            continue
        scores = []
        for t in thr_grid:
            yp = (prob_val[:, ci] >= t).astype(int)
            scores.append(f1_score(yt, yp, zero_division=0))
        best[ci] = float(thr_grid[int(np.argmax(scores))])
    return best


def _fit_rf_per_class(X_tr: np.ndarray, y_tr: np.ndarray, random_state: int = 42):
    from sklearn.ensemble import RandomForestClassifier

    models = []
    for ci in range(y_tr.shape[1]):
        # Strong but still reasonably fast
        rf = RandomForestClassifier(
            n_estimators=1200,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state + ci,
        )
        rf.fit(X_tr, y_tr[:, ci].astype(int))
        models.append(rf)
    return models


def _fit_et_per_class(X_tr: np.ndarray, y_tr: np.ndarray, random_state: int = 42):
    """ExtraTrees is often stronger than RF for tabular ECG features."""
    from sklearn.ensemble import ExtraTreesClassifier

    models = []
    for ci in range(y_tr.shape[1]):
        et = ExtraTreesClassifier(
            n_estimators=2000,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state + 1000 + ci,
        )
        et.fit(X_tr, y_tr[:, ci].astype(int))
        models.append(et)
    return models


def _predict_proba_rf(models, X: np.ndarray) -> np.ndarray:
    prob = np.zeros((X.shape[0], len(models)), dtype=float)
    for ci, m in enumerate(models):
        p = m.predict_proba(X)
        # binary: [:,1]
        prob[:, ci] = p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel()
    return prob


def _predict_proba_sklearn(models, X: np.ndarray) -> np.ndarray:
    """Predict probabilities for a list of sklearn-like binary classifiers."""
    prob = np.zeros((X.shape[0], len(models)), dtype=float)
    for ci, m in enumerate(models):
        p = m.predict_proba(X)
        prob[:, ci] = p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel()
    return prob


def _fit_lgbm_per_class(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, random_state: int = 42):
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        raise SystemExit("LightGBM not installed. Install with: pip install lightgbm")

    try:
        from lightgbm import early_stopping, log_evaluation
    except Exception:  # pragma: no cover
        early_stopping = None
        log_evaluation = None

    models = []
    for ci in range(y_tr.shape[1]):
        yt = y_tr[:, ci].astype(int)
        yv = y_val[:, ci].astype(int)
        pos = int(yt.sum())
        neg = int(len(yt) - pos)
        spw = float(neg / max(1, pos))

        clf = LGBMClassifier(
            n_estimators=20000,
            learning_rate=0.01,
            num_leaves=127,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            min_child_samples=30,
            random_state=random_state + ci,
            n_jobs=-1,
            objective="binary",
            scale_pos_weight=spw,
        )

        callbacks = []
        if early_stopping is not None:
            callbacks.append(early_stopping(stopping_rounds=200, first_metric_only=True))
        if log_evaluation is not None:
            callbacks.append(log_evaluation(period=0))

        clf.fit(X_tr, yt, eval_set=[(X_val, yv)], eval_metric="auc", callbacks=callbacks or None)
        models.append(clf)

    return models


def _predict_proba_lgbm(models, X: np.ndarray) -> np.ndarray:
    prob = np.zeros((X.shape[0], len(models)), dtype=float)
    for ci, m in enumerate(models):
        p = m.predict_proba(X)
        prob[:, ci] = p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel()
    return prob


def _fit_catboost_per_class(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
):
    try:
        from catboost import CatBoostClassifier
    except Exception:
        raise SystemExit("CatBoost not installed. Install with: pip install catboost")

    models = []
    for ci in range(y_tr.shape[1]):
        yt = y_tr[:, ci].astype(int)
        yv = y_val[:, ci].astype(int)
        pos = int(yt.sum())
        neg = int(len(yt) - pos)
        spw = float(neg / max(1, pos))

        clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=20000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3.0,
            random_seed=random_state + ci,
            scale_pos_weight=spw,
            od_type="Iter",
            od_wait=200,
            verbose=False,
            allow_writing_files=False,
        )
        clf.fit(X_tr, yt, eval_set=(X_val, yv), use_best_model=True)
        models.append(clf)
    return models


def _predict_proba_catboost(models, X: np.ndarray) -> np.ndarray:
    prob = np.zeros((X.shape[0], len(models)), dtype=float)
    for ci, m in enumerate(models):
        p = m.predict_proba(X)
        prob[:, ci] = p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel()
    return prob


def _fit_mlp_per_class(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    random_state: int = 42,
):
    from sklearn.neural_network import MLPClassifier

    models = []
    for ci in range(y_tr.shape[1]):
        yt = y_tr[:, ci].astype(int)
        pos = int(yt.sum())
        neg = int(len(yt) - pos)

        # sklearn's early_stopping does an internal stratified split which can
        # fail if either class has < 2 samples.
        val_frac = 0.1
        use_early = bool(min(pos, neg) >= 2 and len(yt) >= 30)
        # If early stopping is enabled, sklearn internally does train_test_split
        # with test_size=val_frac. That uses ceil() for n_test.
        n = int(X_tr.shape[0])
        n_val = int(math.ceil(float(val_frac) * float(n))) if use_early else 0
        internal_train_n = int(max(1, n - n_val))
        max_bs = internal_train_n if use_early else n
        batch_size = int(min(512, max(8, max_bs)))

        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=batch_size,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=250,
            early_stopping=use_early,
            n_iter_no_change=20,
            validation_fraction=val_frac,
            random_state=random_state + 2000 + ci,
        )
        clf.fit(X_tr, yt)
        models.append(clf)
    return models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Path to features root. Use 'auto' to search (incl. ~/Music on Ubuntu).",
    )
    ap.add_argument("--db-csv", type=str, default=None, help="Path to ptbxl_database.csv")
    ap.add_argument(
        "--features-format",
        type=str,
        default="comprehensive",
        choices=["comprehensive", "agg500"],
        help="Feature source format: comprehensive JSON batches or 500Hz agg_features.json hierarchy",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["rf", "et", "lgbm", "catboost", "mlp", "ensemble"],
        help="Back-compat single choice. Prefer --models for multiple.",
    )
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated models to train from {rf,et,lgbm,catboost,mlp}. Example: --models et,lgbm",
    )
    ap.add_argument("--k", type=int, default=70, help="Number of selected features")
    ap.add_argument("--min-per-lead", type=int, default=1)
    ap.add_argument("--var-thresh", type=float, default=0.0)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--thresh", type=float, default=0.5, help="Global threshold (used if --optimize-thresholds is off)")
    ap.add_argument("--optimize-thresholds", action="store_true", help="Tune per-class thresholds on val to maximize F1")
    ap.add_argument("--threshold-grid", type=int, default=41, help="Grid size for threshold search")
    ap.add_argument("--force-select", action="store_true", help="Recompute feature selection even if selected_features.json exists")
    ap.add_argument("--out-dir", type=str, default=None)

    args = ap.parse_args()

    root = find_project_root()

    fd_arg = (str(args.features_dir).strip() if args.features_dir is not None else "")
    if fd_arg.lower() in ("", "auto"):
        features_dir = (
            (root / "ptbxl_feature_enhanced2_features_extracted")
            if str(args.features_format) == "agg500"
            else (root / "ptbxl_comprehensive_features")
        )
    else:
        features_dir = Path(fd_arg)
    db_csv = Path(args.db_csv) if args.db_csv else None

    artifacts_dir = Path(args.out_dir) if args.out_dir else (root / "ECG_Diag_pipeline" / "artifacts" / "tabular")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Resolve models to train
    if args.models:
        models_to_train = [m.strip() for m in str(args.models).split(",") if m.strip()]
    else:
        # Back-compat mapping
        if args.model == "ensemble":
            models_to_train = ["et", "lgbm"]
        else:
            models_to_train = [str(args.model)]
    allowed = {"rf", "et", "lgbm", "catboost", "mlp"}
    bad = [m for m in models_to_train if m not in allowed]
    if bad:
        raise SystemExit(f"Unknown model(s) in --models: {bad}. Allowed: {sorted(allowed)}")

    selected = _ensure_selected_features(
        root=root,
        artifacts_dir=artifacts_dir,
        features_dir=features_dir,
        db_csv=db_csv,
        k=int(args.k),
        min_per_lead=int(args.min_per_lead),
        var_thresh=float(args.var_thresh),
        max_records=args.max_records,
        force=bool(args.force_select),
        features_format=str(args.features_format),
    )

    print(f"[data] loading dataset...")
    if args.features_format == "agg500":
        from scripts.ptbxl_agg_tabular_data import load_agg_tabular_dataset

        data = load_agg_tabular_dataset(features_root=features_dir, db_csv=db_csv, classes=CLASSES, max_records=args.max_records)
    else:
        data = load_tabular_dataset(features_dir=features_dir, db_csv=db_csv, classes=CLASSES, max_records=args.max_records)

    # Keep only selected columns that actually exist (robust to different feature sets)
    selected_existing = [c for c in selected if c in data.X.columns]
    if len(selected_existing) < max(10, int(args.k * 0.5)):
        raise SystemExit(
            f"Too few selected features exist in dataframe ({len(selected_existing)}/{len(selected)}). "
            f"Check that the feature JSON matches the expected naming (lead_#_*_feature)."
        )

    X_df = data.X[selected_existing].copy()
    # Remove inf and columns that are entirely missing; otherwise SimpleImputer may skip them.
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    non_empty = X_df.notna().any(axis=0)
    if non_empty.sum() < len(non_empty):
        dropped = X_df.columns[~non_empty].tolist()
        print(f"[warn] Dropping {len(dropped)} all-missing selected features")
        X_df = X_df.loc[:, non_empty]

    train_mask, val_mask, test_mask = _split_data(data, test_size=float(args.test_size))

    X_train = X_df[train_mask].values
    y_train = data.y[train_mask]
    X_val = X_df[val_mask].values
    y_val = data.y[val_mask]
    X_test = X_df[test_mask].values
    y_test = data.y[test_mask]

    print(f"[split] train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # Impute (do NOT scale by default; tree/boosting models don't need it)
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_val_i = imputer.transform(X_val)
    X_test_i = imputer.transform(X_test)

    joblib.dump(imputer, artifacts_dir / "imputer.joblib")
    (artifacts_dir / "selected_features_used.json").write_text(json.dumps(selected_existing, indent=2), encoding="utf-8")

    results: Dict[str, Dict[str, float]] = {}
    probs_test: Dict[str, np.ndarray] = {}
    probs_val: Dict[str, np.ndarray] = {}
    thresholds: Dict[str, object] = {}

    t0 = time.time()

    if "rf" in models_to_train:
        print("[train] RandomForest per class...")
        rf_models = _fit_rf_per_class(X_train_i, y_train)
        joblib.dump(rf_models, artifacts_dir / "models_rf.joblib")
        probs_val["rf"] = _predict_proba_rf(rf_models, X_val_i)
        probs_test["rf"] = _predict_proba_rf(rf_models, X_test_i)

    if "et" in models_to_train:
        print("[train] ExtraTrees per class...")
        et_models = _fit_et_per_class(X_train_i, y_train)
        joblib.dump(et_models, artifacts_dir / "models_et.joblib")
        probs_val["et"] = _predict_proba_rf(et_models, X_val_i)
        probs_test["et"] = _predict_proba_rf(et_models, X_test_i)

    if "lgbm" in models_to_train:
        print("[train] LightGBM per class (early stopping on val)...")
        lgbm_models = _fit_lgbm_per_class(X_train_i, y_train, X_val_i, y_val)
        joblib.dump(lgbm_models, artifacts_dir / "models_lgbm.joblib")
        probs_val["lgbm"] = _predict_proba_lgbm(lgbm_models, X_val_i)
        probs_test["lgbm"] = _predict_proba_lgbm(lgbm_models, X_test_i)

    if "catboost" in models_to_train:
        print("[train] CatBoost per class (early stopping on val)...")
        cb_models = _fit_catboost_per_class(X_train_i, y_train, X_val_i, y_val)
        joblib.dump(cb_models, artifacts_dir / "models_catboost.joblib")
        probs_val["catboost"] = _predict_proba_catboost(cb_models, X_val_i)
        probs_test["catboost"] = _predict_proba_catboost(cb_models, X_test_i)

    if "mlp" in models_to_train:
        print("[train] MLP per class (scaled features)...")
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_i)
        X_val_s = scaler.transform(X_val_i)
        X_test_s = scaler.transform(X_test_i)
        joblib.dump(scaler, artifacts_dir / "scaler_mlp.joblib")
        mlp_models = _fit_mlp_per_class(X_train_s, y_train)
        joblib.dump(mlp_models, artifacts_dir / "models_mlp.joblib")
        probs_val["mlp"] = _predict_proba_sklearn(mlp_models, X_val_s)
        probs_test["mlp"] = _predict_proba_sklearn(mlp_models, X_test_s)

    # Evaluate each model (optionally optimizing thresholds)
    for name, p_test in probs_test.items():
        if args.optimize_thresholds:
            thr = _optimize_thresholds_per_class(y_val, probs_val[name], grid_size=int(args.threshold_grid))
            thresholds[name] = thr.tolist()
            results[name] = _evaluate_multilabel(y_test, p_test, thresholds=thr)
        else:
            thresholds[name] = float(args.thresh)
            results[name] = _evaluate_multilabel(y_test, p_test, thresholds=float(args.thresh))

    # Ensemble: average member probabilities
    if len(probs_test) >= 2:
        members = sorted(probs_test.keys())
        p_val_ens = np.mean([probs_val[m] for m in members], axis=0)
        p_test_ens = np.mean([probs_test[m] for m in members], axis=0)
        if args.optimize_thresholds:
            thr = _optimize_thresholds_per_class(y_val, p_val_ens, grid_size=int(args.threshold_grid))
            thresholds["ensemble"] = thr.tolist()
            results["ensemble"] = _evaluate_multilabel(y_test, p_test_ens, thresholds=thr)
        else:
            thresholds["ensemble"] = float(args.thresh)
            results["ensemble"] = _evaluate_multilabel(y_test, p_test_ens, thresholds=float(args.thresh))

    elapsed = time.time() - t0

    print("\n=== RESULTS (test) ===")
    for name, m in results.items():
        print(f"\n[{name}]")
        print(f"AUC (Macro): {m['auc_macro']:.4f} | AUC (Micro): {m['auc_micro']:.4f}")
        print(f"Micro F1: {m['micro_f1']:.4f} | Macro F1: {m['macro_f1']:.4f} | Hamming Loss: {m['hamming_loss']:.4f}")

    metrics_out = {
        "classes": list(CLASSES),
        "features_format": str(args.features_format),
        "models": models_to_train,
        "n_selected": len(selected_existing),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "threshold": float(args.thresh),
        "thresholds": thresholds,
        "optimize_thresholds": bool(args.optimize_thresholds),
        "elapsed_sec": float(elapsed),
        "results": results,
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    print(f"\n[OK] Saved artifacts to: {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
