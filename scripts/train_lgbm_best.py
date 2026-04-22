# -*- coding: utf-8 -*-
"""
PTB-XL LightGBM Best Accuracy Script
=====================================
Target: 80%+ validation accuracy on 5-class superclass classification.
Uses LightGBM with DART booster + careful tuning.
Runtime: ~15-30 min (far less than RF which took 27 min at 68%).

Usage:
    python3.14 train_lgbm_best.py
"""

from __future__ import annotations
import ast, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, balanced_accuracy_score)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_class_weight
import pickle

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("LightGBM not installed. Run: pip install lightgbm")

warnings.filterwarnings("ignore")

# ==============================================================================
# Paths -- auto-detected
# ==============================================================================
def _find_root() -> Path:
    candidates = [Path(__file__).resolve().parent,
                  Path(__file__).resolve().parent.parent]
    for c in candidates:
        if (c / "ptbxl_comprehensive_features").exists():
            return c
    return Path(__file__).resolve().parent.parent

def _find_db_csv(root: Path) -> Path:
    archive = root / "archive"
    if archive.exists():
        hits = list(archive.rglob("ptbxl_database.csv"))
        if hits:
            return hits[0]
    return root / "archive" / "ptbxl_database.csv"

ROOT     = _find_root()
FEAT_DIR = ROOT / "ptbxl_comprehensive_features"
DB_CSV   = _find_db_csv(ROOT)
ART_DIR  = ROOT / "ECG_Diag_pipeline" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

print(f"[paths] ROOT     = {ROOT}")
print(f"[paths] FEAT_DIR = {FEAT_DIR}")
print(f"[paths] DB_CSV   = {DB_CSV}")

# ==============================================================================
CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]

SCP_MAP = {
    "NORM": "NORM",
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


# ==============================================================================
# Data loading
# ==============================================================================
def get_superclass(scp_str: str):
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
        for feat_name, val in lead_feats.items():
            if feat_name in SKIP_KEYS:
                continue
            col = f"{lead_key}_{feat_name}"
            try:
                flat[col] = float(val) if val is not None else np.nan
            except (TypeError, ValueError):
                flat[col] = np.nan
    return flat


def load_features() -> pd.DataFrame:
    batch_files = sorted(FEAT_DIR.glob("batch_*_features.json"))
    print(f"  Found {len(batch_files)} batch files")
    records, t0, n_ok, n_fail = [], time.time(), 0, 0
    for bi, bf in enumerate(batch_files):
        try:
            batch = json.loads(bf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [!] {bf.name}: {e}")
            continue
        if isinstance(batch, dict):
            batch = [batch]
        for rec in batch:
            if not rec.get("success", False):
                n_fail += 1; continue
            rid  = rec.get("record_id")
            feat = rec.get("features")
            if rid is None or feat is None:
                n_fail += 1; continue
            flat = flatten_record(feat if isinstance(feat, dict) else {})
            flat["record_id"] = int(rid)
            records.append(flat)
            n_ok += 1
        if (bi + 1) % 50 == 0:
            print(f"  [{bi+1:>3}/{len(batch_files)}] ok={n_ok}  "
                  f"fail={n_fail}  {time.time()-t0:.0f}s", flush=True)
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
    for i, label in enumerate(labels):
        print(f"  {label:<10}" +
              "".join(f"{cm[i,j]:>{w}}" for j in range(len(labels))))


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 70)
    print("  PTB-XL LightGBM -- Best Accuracy Run")
    print("=" * 70)

    # 1. Load
    print("\n[1/4] Loading features ...")
    feat_df  = load_features()
    print("\n[2/4] Loading labels ...")
    label_df = load_labels()
    print(f"  Labels: {len(label_df)}")
    print(label_df["superclass"].value_counts().to_string(header=False))

    # 2. Merge
    merged = feat_df.merge(label_df, on="record_id", how="inner")
    print(f"\n  Merged: {len(merged)} records")

    # 3. Build feature matrix
    meta_cols = {"record_id", "superclass", "strat_fold", "record_path"}
    feat_cols = [c for c in merged.columns if c not in meta_cols]

    X_raw = merged[feat_cols].values.astype(np.float32)
    y     = merged["superclass"].values
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    strat_fold = merged["strat_fold"].values
    val_mask   = strat_fold >= 9
    X_tr_r, X_va_r = X_raw[~val_mask], X_raw[val_mask]
    y_tr,   y_va   = y[~val_mask],     y[val_mask]
    y_tr_e, y_va_e = y_enc[~val_mask], y_enc[val_mask]
    print(f"  Train: {len(X_tr_r)}  |  Val: {len(X_va_r)}")

    # 4. Preprocess
    print("\n[3/4] Preprocessing ...")
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr_r)
    X_va = imputer.transform(X_va_r)

    var_sel = VarianceThreshold(threshold=1e-6)
    X_tr = var_sel.fit_transform(X_tr)
    X_va = var_sel.transform(X_va)
    feat_cols_sel = [c for c, k in zip(feat_cols, var_sel.get_support()) if k]
    print(f"  Features after variance filter: {X_tr.shape[1]}")

    # Clip extreme values (outlier features hurt LGB)
    p1  = np.percentile(X_tr, 1,  axis=0)
    p99 = np.percentile(X_tr, 99, axis=0)
    X_tr = np.clip(X_tr, p1, p99)
    X_va = np.clip(X_va, p1, p99)

    # Sample weights
    cw_vals = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw_dict = dict(zip(np.unique(y_tr), cw_vals))
    sw_tr   = np.array([cw_dict[c] for c in y_tr])

    # 5. Train -- TWO LightGBM models with different settings, then ensemble
    print("\n[4/4] Training LightGBM models ...")
    t0 = time.time()

    # ---- Model 1: GBDT booster (fast, good baseline) -------------------------
    print("\n  [1/2] LightGBM GBDT booster ...")
    m1 = lgb.LGBMClassifier(
        boosting_type     = "gbdt",
        objective         = "multiclass",
        num_class         = len(le.classes_),
        metric            = "multi_logloss",
        learning_rate     = 0.03,
        n_estimators      = 1500,
        num_leaves        = 127,
        max_depth         = -1,
        min_child_samples = 15,
        subsample         = 0.8,
        subsample_freq    = 1,
        colsample_bytree  = 0.6,
        reg_alpha         = 0.05,
        reg_lambda        = 0.5,
        force_col_wise    = True,
        n_jobs            = -1,
        random_state      = 42,
        verbosity         = -1,
    )
    m1.fit(
        X_tr, y_tr_e,
        sample_weight = sw_tr,
        eval_set      = [(X_va, y_va_e)],
        callbacks     = [lgb.early_stopping(80, verbose=True),
                         lgb.log_evaluation(25)],
    )
    p1_enc  = m1.predict(X_va)
    p1_lbl  = le.inverse_transform(p1_enc)
    acc1    = accuracy_score(y_va, p1_lbl)
    bal1    = balanced_accuracy_score(y_va, p1_lbl)
    print(f"    GBDT  Val accuracy         : {acc1*100:.2f}%")
    print(f"    GBDT  Balanced val accuracy: {bal1*100:.2f}%")

    # ---- Model 2: DART booster (slower but often +2-4% on medical data) ------
    print("\n  [2/2] LightGBM DART booster (drop-out trees, ~15 min) ...")
    m2 = lgb.LGBMClassifier(
        boosting_type     = "dart",
        objective         = "multiclass",
        num_class         = len(le.classes_),
        metric            = "multi_logloss",
        learning_rate     = 0.05,
        n_estimators      = 600,     # DART does not support early stopping
        num_leaves        = 127,
        max_depth         = -1,
        min_child_samples = 15,
        subsample         = 0.8,
        subsample_freq    = 1,
        colsample_bytree  = 0.6,
        drop_rate         = 0.1,     # DART-specific: fraction of trees to drop
        skip_drop         = 0.5,     # probability of skipping drop
        reg_alpha         = 0.05,
        reg_lambda        = 0.5,
        force_col_wise    = True,
        n_jobs            = -1,
        random_state      = 42,
        verbosity         = -1,
    )
    # DART does not support early stopping -- use fixed n_estimators
    # Print progress via a custom callback every 25 rounds
    class PrintEvery:
        def __init__(self, every=25): self.every = every
        def __call__(self, env):
            if env.iteration % self.every == 0 or env.iteration == env.end_iteration - 1:
                val_loss = env.evaluation_result_list[0][2]
                print(f"    [DART {env.iteration:>4}]  val logloss = {val_loss:.5f}",
                      flush=True)
        def _get_tags(self): return {"is_valid_contain_train": False}

    m2.fit(
        X_tr, y_tr_e,
        sample_weight = sw_tr,
        eval_set      = [(X_va, y_va_e)],
        callbacks     = [lgb.log_evaluation(-1), PrintEvery(25)],
    )
    p2_enc = m2.predict(X_va)
    p2_lbl = le.inverse_transform(p2_enc)
    acc2   = accuracy_score(y_va, p2_lbl)
    bal2   = balanced_accuracy_score(y_va, p2_lbl)
    print(f"    DART  Val accuracy         : {acc2*100:.2f}%")
    print(f"    DART  Balanced val accuracy: {bal2*100:.2f}%")

    # ---- Ensemble: soft vote GBDT + DART ------------------------------------
    print("\n  Ensemble (GBDT + DART soft vote) ...")
    proba1 = m1.predict_proba(X_va)
    proba2 = m2.predict_proba(X_va)

    best_ens_acc   = 0.0
    best_ens_preds = None
    best_w         = (1, 1)
    for w1, w2 in [(1,1),(2,1),(1,2),(3,1),(1,3),(3,2),(2,3)]:
        avg  = (w1 * proba1 + w2 * proba2) / (w1 + w2)
        ep   = le.inverse_transform(np.argmax(avg, axis=1))
        ea   = accuracy_score(y_va, ep)
        if ea > best_ens_acc:
            best_ens_acc   = ea
            best_ens_preds = ep
            best_w         = (w1, w2)

    ens_bal = balanced_accuracy_score(y_va, best_ens_preds)
    print(f"    Best weights (gbdt, dart): {best_w}")
    print(f"    Ensemble val accuracy    : {best_ens_acc*100:.2f}%")
    print(f"    Ensemble balanced acc    : {ens_bal*100:.2f}%")

    # ---- Pick best ----------------------------------------------------------
    candidates = {
        "LGB_GBDT": (acc1,  p1_lbl),
        "LGB_DART": (acc2,  p2_lbl),
        "Ensemble" : (best_ens_acc, best_ens_preds),
    }
    best_name, (best_acc, final_preds) = max(
        candidates.items(), key=lambda x: x[1][0])

    print(f"\n{'='*70}")
    print(f"  BEST MODEL  : {best_name}")
    print(f"  Val Accuracy: {best_acc*100:.2f}%")
    print(f"{'='*70}")

    print("\n  Classification Report:")
    print(classification_report(y_va, final_preds,
                                target_names=CLASSES, digits=4))
    print("  Confusion Matrix:")
    print_cm(y_va, final_preds, CLASSES)

    elapsed = time.time() - t0
    print(f"\n  Total training time: {elapsed:.0f}s")

    # ---- Save ---------------------------------------------------------------
    print("\n  Saving artifacts ...")
    with open(ART_DIR / "lgbm_gbdt.pkl",      "wb") as f: pickle.dump(m1, f)
    with open(ART_DIR / "lgbm_dart.pkl",      "wb") as f: pickle.dump(m2, f)
    with open(ART_DIR / "lgbm_imputer.pkl",   "wb") as f: pickle.dump(imputer, f)
    with open(ART_DIR / "lgbm_var_sel.pkl",   "wb") as f: pickle.dump(var_sel, f)
    with open(ART_DIR / "lgbm_label_enc.pkl", "wb") as f: pickle.dump(le, f)
    with open(ART_DIR / "lgbm_clip.pkl",      "wb") as f: pickle.dump((p1, p99), f)
    with open(ART_DIR / "lgbm_feat_cols.txt", "w")  as f: f.write("\n".join(feat_cols_sel))

    import json as _json
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_va, final_preds, labels=CLASSES, zero_division=0)
    with open(ART_DIR / "lgbm_metrics.json", "w") as f:
        _json.dump({
            "best_model"            : best_name,
            "val_accuracy"          : float(best_acc),
            "val_balanced_accuracy" : float(balanced_accuracy_score(y_va, final_preds)),
            "gbdt_accuracy"         : float(acc1),
            "dart_accuracy"         : float(acc2),
            "ensemble_weights"      : best_w,
            "n_train" : int(len(X_tr)),
            "n_val"   : int(len(X_va)),
            "n_features": int(X_tr.shape[1]),
            "per_class": {
                cls: {"precision": float(prec[i]), "recall": float(rec[i]),
                      "f1": float(f1[i]), "support": int(sup[i])}
                for i, cls in enumerate(CLASSES)
            },
        }, f, indent=2)

    print(f"  Artifacts saved to {ART_DIR}")
    print(f"\n  Done. Best val accuracy = {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
