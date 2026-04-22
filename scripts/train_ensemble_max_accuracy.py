# -*- coding: utf-8 -*-
"""
PTB-XL ENSEMBLE for Maximum Accuracy -- Target 75-80%+
======================================================
Combines multiple approaches to maximize accuracy:
1. Focal loss model (per-class alpha)
2. Balanced LightGBM with boosted minority classes
3. XGBoost with scale_pos_weight
4. Soft voting ensemble

Usage:
    python train_ensemble_max_accuracy.py          # full ~19k records
    python train_ensemble_max_accuracy.py --fast   # 5k subset
"""

from __future__ import annotations
import ast, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, balanced_accuracy_score)

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("Install LightGBM: pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] XGBoost not installed. Ensemble will use LightGBM models only.")
    print("          Install with: pip install xgboost")

warnings.filterwarnings("ignore")

FAST_MODE = "--fast" in sys.argv
N_FAST    = 5000
N_FEATS   = 200  # Use more features for ensemble diversity

# ==============================================================================
# Paths
# ==============================================================================
def _find_root() -> Path:
    for c in [Path(__file__).resolve().parent,
              Path(__file__).resolve().parent.parent]:
        if (c / "ptbxl_comprehensive_features").exists():
            return c
    return Path(__file__).resolve().parent.parent

def _find_db_csv(root: Path) -> Path:
    archive = root / "archive"
    if archive.exists():
        hits = list(archive.rglob("ptbxl_database.csv"))
        if hits:
            return hits[0]
    return archive / "ptbxl_database.csv"

ROOT     = _find_root()
FEAT_DIR = ROOT / "ptbxl_comprehensive_features"
DB_CSV   = _find_db_csv(ROOT)
ART_DIR  = ROOT / "ECG_Diag_pipeline" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

print(f"[paths] ROOT     = {ROOT}")
print(f"[paths] FEAT_DIR = {FEAT_DIR}")
print(f"[paths] DB_CSV   = {DB_CSV}")
print(f"[mode ] {'FAST (5k subset)' if FAST_MODE else 'FULL dataset'}")

# ==============================================================================
CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]

SCP_MAP = {
    "NORM":"NORM",
    "IMI":"MI","ILMI":"MI","AMI":"MI","ALMI":"MI","INJAS":"MI","LMI":"MI",
    "INJAL":"MI","IPLMI":"MI","IPMI":"MI","INJIN":"MI","INJLA":"MI",
    "PMI":"MI","INJIL":"MI","INJA":"MI",
    "NDT":"STTC","DIG":"STTC","LNGQT":"STTC","ANEUR":"STTC","EL":"STTC",
    "ISCA":"STTC","ISCI":"STTC","ISC_":"STTC","STTC":"STTC",
    "STD_":"STTC","STE_":"STTC",
    "LAFB":"CD","IRBBB":"CD","IVCD":"CD","LBBB":"CD","RBBB":"CD",
    "LPFB":"CD","WPW":"CD","1AVB":"CD","2AVB":"CD","3AVB":"CD","AVB":"CD",
    "LVH":"HYP","LAO":"HYP","RVH":"HYP","SEHYP":"HYP","LVOLT":"HYP",
    "RAO":"HYP","LMH":"HYP",
}
SKIP_KEYS = {"lead_name", "sampling_freq"}

# Per-class focal loss weights: [CD, HYP, MI, NORM, STTC]
CLASS_ALPHAS = np.array([1.5, 2.5, 1.2, 0.25, 1.8])  # Even more aggressive


# ==============================================================================
# Focal Loss
# ==============================================================================
def focal_loss_lgb(y_true, y_pred, gamma=4.0, num_class=5, class_alphas=CLASS_ALPHAS):
    """Focal loss with per-class alpha (gamma=4.0 for extreme focus)."""
    y_pred = y_pred.reshape(-1, num_class, order='F')
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    
    y_true = y_true.astype(int)
    y_onehot = np.zeros((len(y_true), num_class))
    y_onehot[np.arange(len(y_true)), y_true] = 1
    
    alpha_t = class_alphas[y_true].reshape(-1, 1)
    pt = np.sum(probs * y_onehot, axis=1, keepdims=True)
    focal_weight = alpha_t * (1 - pt) ** gamma
    
    grad = focal_weight * (probs - y_onehot)
    hess = focal_weight * probs * (1 - probs) * 2.5
    
    return grad.flatten('F'), hess.flatten('F')


def focal_loss_eval(y_true, y_pred, gamma=4.0, num_class=5, class_alphas=CLASS_ALPHAS):
    """Focal loss metric."""
    y_pred = y_pred.reshape(-1, num_class, order='F')
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    
    y_true = y_true.astype(int)
    alpha_t = class_alphas[y_true]
    pt = probs[np.arange(len(y_true)), y_true]
    focal = -alpha_t * (1 - pt) ** gamma * np.log(pt)
    
    return 'focal_loss', np.mean(focal), False


# ==============================================================================
# Data helpers
# ==============================================================================
def get_superclass(scp_str):
    try:
        codes = ast.literal_eval(scp_str) if isinstance(scp_str, str) else scp_str
        scored: dict[str, float] = {}
        for code, conf in codes.items():
            sc = SCP_MAP.get(code)
            if sc:
                scored[sc] = scored.get(sc, 0) + conf
        return max(scored, key=scored.get) if scored else None
    except Exception:
        return None


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
    print(f"  Loaded {n_ok} records in {time.time()-t0:.0f}s")
    return pd.DataFrame(records)


def load_labels() -> pd.DataFrame:
    db = pd.read_csv(DB_CSV)
    db["superclass"] = db["scp_codes"].apply(get_superclass)
    db = db.dropna(subset=["superclass"])
    db = db[db["superclass"].isin(CLASSES)]
    return db[["ecg_id", "superclass", "strat_fold"]].rename(
        columns={"ecg_id": "record_id"})


def print_cm(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    w  = 8
    print(" " * 12 + "".join(f"{l:>{w}}" for l in labels))
    for i, lbl in enumerate(labels):
        print(f"  {lbl:<10}" + "".join(f"{cm[i,j]:>{w}}" for j in range(len(labels))))


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 70)
    print("  PTB-XL ENSEMBLE for Maximum Accuracy")
    print("  Focal Loss + Balanced GBDT + XGBoost")
    print("=" * 70)

    # -- Load ------------------------------------------------------------------
    print("\n[1/5] Loading features ...")
    feat_df  = load_features()
    print("\n[2/5] Loading labels ...")
    label_df = load_labels()

    merged = feat_df.merge(label_df, on="record_id", how="inner")
    print(f"  Total merged records: {len(merged)}")
    print(merged["superclass"].value_counts().to_string(header=False))

    if FAST_MODE:
        from sklearn.model_selection import train_test_split as tts
        _, merged = tts(merged, test_size=N_FAST,
                        stratify=merged["superclass"], random_state=42)
        print(f"\n  [FAST MODE] Using {len(merged)} records")
        print(merged["superclass"].value_counts().to_string(header=False))

    # -- Feature matrix --------------------------------------------------------
    meta_cols  = {"record_id", "superclass", "strat_fold", "record_path"}
    feat_cols  = [c for c in merged.columns if c not in meta_cols]
    X_raw      = merged[feat_cols].values.astype(np.float32)
    y          = merged["superclass"].values

    strat_fold = merged["strat_fold"].values
    val_mask   = strat_fold >= 9
    X_tr_r, X_va_r = X_raw[~val_mask], X_raw[val_mask]
    y_tr,   y_va   = y[~val_mask],     y[val_mask]
    print(f"\n  Train: {len(X_tr_r)}  |  Val: {len(X_va_r)}")

    # -- Preprocess ------------------------------------------------------------
    print("\n[3/5] Preprocessing ...")
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr_r)
    X_va = imputer.transform(X_va_r)

    var_sel = VarianceThreshold(threshold=1e-6)
    X_tr = var_sel.fit_transform(X_tr)
    X_va = var_sel.transform(X_va)
    print(f"  After variance filter : {X_tr.shape[1]} features")

    p01  = np.percentile(X_tr, 1,  axis=0)
    p99  = np.percentile(X_tr, 99, axis=0)
    X_tr = np.clip(X_tr, p01, p99)
    X_va = np.clip(X_va, p01, p99)

    # Label encoding
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_va_enc = le.transform(y_va)
    num_class = len(le.classes_)
    print(f"  Classes: {le.classes_}")

    # Feature selection
    print(f"  Selecting top {N_FEATS} features ...")
    mi_sel = SelectKBest(mutual_info_classif, k=min(N_FEATS, X_tr.shape[1]))
    X_tr = mi_sel.fit_transform(X_tr, y_tr_enc)
    X_va = mi_sel.transform(X_va)
    print(f"  Features selected: {X_tr.shape[1]}")

    # Compute class weights
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw_dict = dict(zip(np.unique(y_tr), cw))
    sw = np.array([cw_dict[c] for c in y_tr])
    sw_amplified = sw ** 1.5 * 2.0  # Aggressive amplification

    # -- Train Ensemble --------------------------------------------------------
    print("\n[4/5] Training ensemble models ...")
    t0 = time.time()

    models = []
    model_names = []
    
    # ---- Model 1: Focal Loss LightGBM ----------------------------------------
    print("\n  [1/3] Training Focal Loss LightGBM ...")
    
    def focal_obj(y_true, y_pred):
        return focal_loss_lgb(y_true, y_pred, gamma=4.0, num_class=num_class, class_alphas=CLASS_ALPHAS)
    
    def focal_metric(y_true, y_pred):
        return focal_loss_eval(y_true, y_pred, gamma=4.0, num_class=num_class, class_alphas=CLASS_ALPHAS)

    m1 = lgb.LGBMClassifier(
        boosting_type     = "gbdt",
        objective         = focal_obj,
        num_class         = num_class,
        learning_rate     = 0.012,
        n_estimators      = 1800,
        num_leaves        = 127,
        min_child_samples = 8,
        subsample         = 0.65,
        subsample_freq    = 1,
        colsample_bytree  = 0.55,
        reg_alpha         = 0.25,
        reg_lambda        = 2.5,
        max_depth         = 14,
        force_col_wise    = True,
        n_jobs            = -1,
        random_state      = 42,
        verbosity         = -1,
    )
    m1.fit(
        X_tr, y_tr_enc,
        sample_weight = sw_amplified,
        eval_set      = [(X_va, y_va_enc)],
        eval_metric   = focal_metric,
        callbacks     = [lgb.early_stopping(180, verbose=False),
                         lgb.log_evaluation(50)],
    )
    models.append(m1)
    model_names.append("Focal Loss LGBM")
    print(f"  → Best iteration: {m1.best_iteration_}")

    # ---- Model 2: Balanced LightGBM with Class Weights ----------------------
    print("\n  [2/3] Training Balanced LightGBM ...")
    
    m2 = lgb.LGBMClassifier(
        boosting_type     = "gbdt",
        objective         = "multiclass",
        num_class         = num_class,
        learning_rate     = 0.015,
        n_estimators      = 1500,
        num_leaves        = 95,
        min_child_samples = 12,
        subsample         = 0.75,
        subsample_freq    = 1,
        colsample_bytree  = 0.65,
        reg_alpha         = 0.15,
        reg_lambda        = 1.5,
        max_depth         = 12,
        force_col_wise    = True,
        n_jobs            = -1,
        random_state      = 123,
        verbosity         = -1,
    )
    m2.fit(
        X_tr, y_tr_enc,
        sample_weight = sw_amplified,
        eval_set      = [(X_va, y_va_enc)],
        callbacks     = [lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(50)],
    )
    models.append(m2)
    model_names.append("Balanced LGBM")
    print(f"  → Best iteration: {m2.best_iteration_}")

    # ---- Model 3: XGBoost (if available) -------------------------------------
    if HAS_XGB:
        print("\n  [3/3] Training XGBoost ...")
        
        # Map sample weights to xgb format
        m3 = xgb.XGBClassifier(
            objective         = "multi:softprob",
            num_class         = num_class,
            learning_rate     = 0.012,
            n_estimators      = 1500,
            max_depth         = 10,
            min_child_weight  = 3,
            subsample         = 0.7,
            colsample_bytree  = 0.6,
            reg_alpha         = 0.2,
            reg_lambda        = 2.0,
            n_jobs            = -1,
            random_state      = 456,
            verbosity         = 0,
        )
        m3.fit(
            X_tr, y_tr_enc,
            sample_weight = sw_amplified,
            eval_set      = [(X_va, y_va_enc)],
            early_stopping_rounds = 150,
            verbose       = 50,
        )
        models.append(m3)
        model_names.append("XGBoost")
        print(f"  → Best iteration: {m3.best_iteration}")

    # -- Ensemble Predictions --------------------------------------------------
    print("\n[5/5] Combining ensemble predictions ...")
    
    # Get probability predictions from all models
    all_probs = []
    for i, model in enumerate(models):
        try:
            probs = model.predict_proba(X_va)
        except:
            # Fallback if predict_proba doesn't work with custom objective
            preds_raw = model.predict(X_va)
            if preds_raw.ndim == 2:
                probs = preds_raw  # Already probabilities
            else:
                # Convert class predictions to one-hot probabilities
                probs = np.zeros((len(preds_raw), num_class))
                probs[np.arange(len(preds_raw)), preds_raw.astype(int)] = 1.0
        
        all_probs.append(probs)
        preds = le.inverse_transform(np.argmax(probs, axis=1))
        acc = accuracy_score(y_va, preds)
        print(f"  {model_names[i]:20s} accuracy: {acc*100:.2f}%")
    
    # Soft voting (average probabilities)
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds_enc = np.argmax(ensemble_probs, axis=1)
    ensemble_preds = le.inverse_transform(ensemble_preds_enc)
    
    # Evaluate ensemble
    acc     = accuracy_score(y_va, ensemble_preds)
    bal_acc = balanced_accuracy_score(y_va, ensemble_preds)
    
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ENSEMBLE Val Accuracy   : {acc*100:.2f}%")
    print(f"  ENSEMBLE Balanced Acc   : {bal_acc*100:.2f}%")
    print(f"  Training time           : {elapsed:.0f}s")
    print(f"{'='*70}")

    print("\n  Classification Report:")
    print(classification_report(y_va, ensemble_preds,
                                target_names=CLASSES, digits=4))
    print("  Confusion Matrix:")
    print_cm(y_va, ensemble_preds, CLASSES)

    # -- Save ------------------------------------------------------------------
    print("\n  Saving artifacts ...")
    for i, model in enumerate(models):
        fname = f"ensemble_model{i+1}_{model_names[i].replace(' ', '_').lower()}.pkl"
        with open(ART_DIR / fname, "wb") as f:
            pickle.dump(model, f)
    
    with open(ART_DIR / "ensemble_le.pkl",      "wb") as f: pickle.dump(le,      f)
    with open(ART_DIR / "ensemble_imputer.pkl", "wb") as f: pickle.dump(imputer, f)
    with open(ART_DIR / "ensemble_var_sel.pkl", "wb") as f: pickle.dump(var_sel, f)
    with open(ART_DIR / "ensemble_mi_sel.pkl",  "wb") as f: pickle.dump(mi_sel,  f)
    with open(ART_DIR / "ensemble_clip.pkl",    "wb") as f: pickle.dump((p01, p99), f)

    import json as _json
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_va, ensemble_preds, labels=CLASSES, zero_division=0)
    
    with open(ART_DIR / "ensemble_metrics.json", "w") as f:
        _json.dump({
            "ensemble_accuracy"          : float(acc),
            "ensemble_balanced_accuracy" : float(bal_acc),
            "n_models"                   : len(models),
            "model_names"                : model_names,
            "n_features"                 : int(X_tr.shape[1]),
            "n_train": int(len(X_tr_r)), "n_val": int(len(X_va_r)),
            "individual_models": {
                name: {
                    "accuracy": float(accuracy_score(y_va, le.inverse_transform(np.argmax(all_probs[i], axis=1))))
                } for i, name in enumerate(model_names)
            },
            "per_class": {
                cls: {"precision": float(prec[i]), "recall": float(rec[i]),
                      "f1": float(f1[i]), "support": int(sup[i])}
                for i, cls in enumerate(CLASSES)
            }
        }, f, indent=2)

    print(f"  Artifacts saved to {ART_DIR}")
    print(f"\n  Done. Ensemble val accuracy = {acc*100:.2f}%")
    print(f"\n  🎯 Try running on FULL dataset for best results:")
    print(f"     python scripts/train_ensemble_max_accuracy.py")


if __name__ == "__main__":
    main()
