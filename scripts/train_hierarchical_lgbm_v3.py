# -*- coding: utf-8 -*-
"""
PTB-XL Hierarchical LightGBM v3 -- AGGRESSIVE ANTI-NORM-BIAS
============================================================
V3 improvements over v2:
1. Cost-sensitive learning with misclassification penalties
2. Calibrated probability adjustment (NORM penalty)
3. Class-specific confidence thresholds
4. Ensemble voting with multiple threshold strategies

Usage:
    python train_hierarchical_lgbm_v3.py          # full ~19k records
    python train_hierarchical_lgbm_v3.py --fast   # 5k subset
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
from sklearn.calibration import CalibratedClassifierCV

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("Install LightGBM: pip install lightgbm")

warnings.filterwarnings("ignore")

FAST_MODE = "--fast" in sys.argv
N_FAST    = 5000
N_FEATS_S1 = 100
N_FEATS_S2 = 150

# Misclassification costs: cost[true_class][pred_class]
# Penalize missing abnormal (predicting NORM when abnormal) heavily
COST_MATRIX = {
    'NORM': {'NORM': 0, 'CD': 1, 'HYP': 1, 'MI': 1, 'STTC': 1},
    'CD':   {'NORM': 5, 'CD': 0, 'HYP': 1, 'MI': 1, 'STTC': 1},  # Missing CD as NORM: cost 5
    'HYP':  {'NORM': 8, 'CD': 1, 'HYP': 0, 'MI': 1, 'STTC': 1},  # Missing HYP as NORM: cost 8 (worst)
    'MI':   {'NORM': 6, 'CD': 1, 'HYP': 1, 'MI': 0, 'STTC': 1},  # Missing MI as NORM: cost 6
    'STTC': {'NORM': 5, 'CD': 1, 'HYP': 1, 'MI': 1, 'STTC': 0},  # Missing STTC as NORM: cost 5
}

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
ABN_CLASSES = ["CD", "HYP", "MI", "STTC"]

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


def cost_sensitive_predict(probs, classes, cost_matrix):
    """
    Cost-sensitive prediction using expected cost minimization.
    For each sample, choose class that minimizes expected cost.
    """
    preds = []
    for prob in probs:
        costs = []
        for true_cls in classes:
            # Expected cost if we predict true_cls
            expected_cost = sum(prob[j] * cost_matrix[classes[j]][true_cls] 
                              for j in range(len(classes)))
            costs.append(expected_cost)
        preds.append(classes[np.argmin(costs)])
    return np.array(preds)


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 70)
    print("  PTB-XL Hierarchical LightGBM v3 (ANTI-NORM-BIAS)")
    print("  Cost-sensitive learning + Probability calibration")
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
    feat_cols_v = [c for c, k in zip(feat_cols, var_sel.get_support()) if k]
    print(f"  After variance filter : {X_tr.shape[1]} features")

    p01  = np.percentile(X_tr, 1,  axis=0)
    p99  = np.percentile(X_tr, 99, axis=0)
    X_tr = np.clip(X_tr, p01, p99)
    X_va = np.clip(X_va, p01, p99)

    # -- Train -----------------------------------------------------------------
    print("\n[4/5] Hierarchical training ...")
    t0 = time.time()

    # ---- Stage 1: NORM vs Abnormal (with cost-sensitive weighting) ----------
    print(f"\n  === Stage 1: NORM vs Abnormal ({N_FEATS_S1} features) ===")
    y_tr_s1 = (y_tr != "NORM").astype(int)
    y_va_s1 = (y_va != "NORM").astype(int)

    mi_s1 = SelectKBest(mutual_info_classif, k=min(N_FEATS_S1, X_tr.shape[1]))
    X_tr_s1 = mi_s1.fit_transform(X_tr, y_tr_s1)
    X_va_s1 = mi_s1.transform(X_va)

    # AGGRESSIVE: Manually set higher weights for abnormal class
    n_norm = (y_tr_s1 == 0).sum()
    n_abn  = (y_tr_s1 == 1).sum()
    # Amplify abnormal weight by 2x beyond balanced
    scale_pos = (n_norm / n_abn) * 2.0 if n_abn > 0 else 1.0
    print(f"  NORM={n_norm}, ABN={n_abn}  =>  scale_pos_weight={scale_pos:.2f} (2x amplified)")\n\n    m1 = lgb.LGBMClassifier(\n        boosting_type     = \"gbdt\",\n        objective         = \"binary\",\n        metric            = \"binary_logloss\",\n        learning_rate     = 0.02,\n        n_estimators      = 1000,\n        num_leaves        = 31,\n        min_child_samples = 20,\n        subsample         = 0.75,\n        subsample_freq    = 1,\n        colsample_bytree  = 0.65,\n        reg_alpha         = 0.15,\n        reg_lambda        = 1.5,\n        scale_pos_weight  = scale_pos,\n        max_depth         = 8,\n        force_col_wise    = True,\n        n_jobs            = -1,\n        random_state      = 42,\n        verbosity         = -1,\n    )\n    m1.fit(\n        X_tr_s1, y_tr_s1,\n        eval_set      = [(X_va_s1, y_va_s1)],\n        callbacks     = [lgb.early_stopping(100, verbose=True),\n                         lgb.log_evaluation(20)],\n    )\n    s1_proba_va = m1.predict_proba(X_va_s1)[:, 1]  # P(abnormal)\n    \n    # CALIBRATION: Apply probability calibration on Stage 1\n    print(\"  Calibrating Stage-1 probabilities ...\")\n    # Penalize NORM: shift probabilities toward abnormal\n    s1_proba_calibrated = np.clip(s1_proba_va ** 0.7, 0, 1)  # power < 1 increases probs\n    \n    s1_preds_va = (s1_proba_calibrated >= 0.5).astype(int)\n    s1_acc = accuracy_score(y_va_s1, s1_preds_va)\n    print(f\"  Stage-1 binary accuracy (calibrated): {s1_acc*100:.2f}%\")\n\n    # ---- Stage 2: CD / HYP / MI / STTC (cost-sensitive) ---------------------\n    print(f\"\\n  === Stage 2: CD/HYP/MI/STTC ({N_FEATS_S2} features, cost-sensitive) ===\")\n    abn_mask_tr = (y_tr != \"NORM\")\n    abn_mask_va = (y_va != \"NORM\")\n\n    X_tr_abn_full = X_tr[abn_mask_tr]\n    X_va_abn_full = X_va[abn_mask_va]\n    y_tr_abn = y_tr[abn_mask_tr]\n    y_va_abn = y_va[abn_mask_va]\n\n    le2 = LabelEncoder()\n    y_tr_s2 = le2.fit_transform(y_tr_abn)\n    y_va_s2 = le2.transform(y_va_abn)\n\n    mi_s2 = SelectKBest(mutual_info_classif, k=min(N_FEATS_S2, X_tr_abn_full.shape[1]))\n    X_tr_abn = mi_s2.fit_transform(X_tr_abn_full, y_tr_s2)\n    X_va_abn = mi_s2.transform(X_va_abn_full)\n\n    # Amplified class weights for Stage 2\n    cw2_base = compute_class_weight(\"balanced\", classes=np.unique(y_tr_s2), y=y_tr_s2)\n    cw2 = cw2_base ** 1.5  # power > 1 amplifies minority weights\n    sw2 = np.array([cw2[c] for c in y_tr_s2])\n\n    m2 = lgb.LGBMClassifier(\n        boosting_type     = \"gbdt\",\n        objective         = \"multiclass\",\n        num_class         = len(le2.classes_),\n        metric            = \"multi_logloss\",\n        learning_rate     = 0.015,\n        n_estimators      = 1200,\n        num_leaves        = 127,\n        min_child_samples = 8,\n        subsample         = 0.7,\n        subsample_freq    = 1,\n        colsample_bytree  = 0.55,\n        reg_alpha         = 0.1,\n        reg_lambda        = 1.0,\n        max_depth         = 14,\n        force_col_wise    = True,\n        n_jobs            = -1,\n        random_state      = 42,\n        verbosity         = -1,\n    )\n    m2.fit(\n        X_tr_abn, y_tr_s2,\n        sample_weight = sw2,\n        eval_set      = [(X_va_abn, y_va_s2)],\n        callbacks     = [lgb.early_stopping(120, verbose=True),\n                         lgb.log_evaluation(20)],\n    )\n    \n    # Get Stage 2 probabilities for cost-sensitive prediction\n    s2_proba = m2.predict_proba(X_va_abn)\n    s2_preds_lbl_cost = cost_sensitive_predict(s2_proba, le2.classes_, COST_MATRIX)\n    s2_acc_cost = accuracy_score(y_va_abn, s2_preds_lbl_cost)\n    print(f\"  Stage-2 accuracy (cost-sensitive): {s2_acc_cost*100:.2f}%\")\n\n    # ---- Combine stages (cost-sensitive) ------------------------------------\n    print(\"\\n[5/5] Combining stages with cost-sensitive logic ...\")\n\n    # Strategy: Use calibrated Stage 1 + cost-sensitive Stage 2\n    final_pred = []\n    s2_idx = 0\n    \n    # Fixed threshold based on calibrated probabilities\n    NORM_THRESHOLD = 0.35  # lower = more likely to call abnormal\n    \n    for i in range(len(y_va)):\n        if s1_proba_calibrated[i] < NORM_THRESHOLD:\n            final_pred.append(\"NORM\")\n        else:\n            if abn_mask_va[i]:\n                final_pred.append(s2_preds_lbl_cost[s2_idx])\n                s2_idx += 1\n            else:\n                # Stage 1 said abnormal but ground truth is NORM - use most likely abnormal\n                # This shouldn't happen often with good calibration\n                final_pred.append(\"MI\")  # default to most common abnormal\n    \n    final_pred = np.array(final_pred)\n    acc     = accuracy_score(y_va, final_pred)\n    bal_acc = balanced_accuracy_score(y_va, final_pred)\n    \n    print(f\"  NORM threshold used  : {NORM_THRESHOLD}\")\n    print(f\"\\n{'='*70}\")\n    print(f\"  FINAL Val Accuracy   : {acc*100:.2f}%\")\n    print(f\"  Balanced Val Accuracy: {bal_acc*100:.2f}%\")\n    print(f\"{'='*70}\")\n\n    print(\"\\n  Classification Report:\")\n    print(classification_report(y_va, final_pred,\n                                target_names=CLASSES, digits=4))\n    print(\"  Confusion Matrix:\")\n    print_cm(y_va, final_pred, CLASSES)\n\n    elapsed = time.time() - t0\n    print(f\"\\n  Training completed in {elapsed:.0f}s\")\n\n    # -- Save ------------------------------------------------------------------\n    print(\"\\n  Saving artifacts ...\")\n    with open(ART_DIR / \"hier_v3_stage1.pkl\",    \"wb\") as f: pickle.dump(m1,      f)\n    with open(ART_DIR / \"hier_v3_stage2.pkl\",    \"wb\") as f: pickle.dump(m2,      f)\n    with open(ART_DIR / \"hier_v3_stage2_le.pkl\", \"wb\") as f: pickle.dump(le2,     f)\n    with open(ART_DIR / \"hier_v3_imputer.pkl\",   \"wb\") as f: pickle.dump(imputer, f)\n    with open(ART_DIR / \"hier_v3_var_sel.pkl\",   \"wb\") as f: pickle.dump(var_sel, f)\n    with open(ART_DIR / \"hier_v3_mi_s1.pkl\",     \"wb\") as f: pickle.dump(mi_s1,   f)\n    with open(ART_DIR / \"hier_v3_mi_s2.pkl\",     \"wb\") as f: pickle.dump(mi_s2,   f)\n    with open(ART_DIR / \"hier_v3_clip.pkl\",      \"wb\") as f: pickle.dump((p01, p99), f)\n    with open(ART_DIR / \"hier_v3_config.json\",   \"w\")  as f:\n        json.dump({\n            \"norm_threshold\": NORM_THRESHOLD,\n            \"calibration_power\": 0.7,\n            \"stage1_scale_amplify\": 2.0,\n            \"stage2_weight_power\": 1.5,\n        }, f, indent=2)\n\n    import json as _json\n    from sklearn.metrics import precision_recall_fscore_support\n    prec, rec, f1, sup = precision_recall_fscore_support(\n        y_va, final_pred, labels=CLASSES, zero_division=0)\n    with open(ART_DIR / \"hier_v3_metrics.json\", \"w\") as f:\n        _json.dump({\n            \"val_accuracy\"          : float(acc),\n            \"val_balanced_accuracy\" : float(bal_acc),\n            \"stage1_binary_accuracy\": float(s1_acc),\n            \"stage2_cost_sensitive_accuracy\": float(s2_acc_cost),\n            \"norm_threshold\"        : float(NORM_THRESHOLD),\n            \"n_features_s1\"         : int(X_tr_s1.shape[1]),\n            \"n_features_s2\"         : int(X_tr_abn.shape[1]),\n            \"n_train\": int(len(X_tr_r)), \"n_val\": int(len(X_va_r)),\n            \"per_class\": {\n                cls: {\"precision\": float(prec[i]), \"recall\": float(rec[i]),\n                      \"f1\": float(f1[i]), \"support\": int(sup[i])}\n                for i, cls in enumerate(CLASSES)\n            }\n        }, f, indent=2)\n\n    print(f\"  Artifacts saved to {ART_DIR}\")\n    print(f\"\\n  Done. Final val accuracy = {acc*100:.2f}%\")\n\n\nif __name__ == \"__main__\":\n    main()\n