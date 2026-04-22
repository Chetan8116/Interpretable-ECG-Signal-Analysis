# -*- coding: utf-8 -*-
"""
Random Forest PTB-XL Classifier -- Comprehensive Features
=========================================================
Combines NEW comprehensive features (~840 per record, 12 leads x 70 features:
  morphological, temporal, spectral, time-frequency, nonlinear)
WITH the OLD clinical 170-dim features (HR, n_beats + 12 leads x 14 stats)
loaded from the batch JSON files in ptbxl_comprehensive_features/.

Target: improve the previous 73% validation accuracy.
Expected: 85%+ with Random Forest on rich feature set (~21k records).

Pipeline
--------
1. Load all batch_*.json files -> flatten 12-lead features into single row per record
2. Join with ptbxl_database.csv to get superclass labels
3. Stratified 80/20 train-val split
4. GridSearch / RandomizedSearch Random Forest with class balancing
5. Print full metrics: accuracy, per-class report, confusion matrix
6. Save model + feature column list -> ECG_Diag_pipeline/artifacts/
"""

from __future__ import annotations
import ast, json, time, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, balanced_accuracy_score)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import pickle

# -- Optional heavy hitters ---------------------------------------------------
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[warn] lightgbm not found. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] xgboost not found. Install with: pip install xgboost")

warnings.filterwarnings("ignore")

# -- Paths ----------------------------------------------------------------------
def _find_root() -> Path:
    """
    Walk from the script location upward (and also check the script's own
    directory) until we find a folder containing 'ptbxl_comprehensive_features'.
    Falls back to parent.parent for backward compatibility.
    """
    candidates = [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]
    for c in candidates:
        if (c / "ptbxl_comprehensive_features").exists():
            return c
    return Path(__file__).resolve().parent.parent  # default


def _find_db_csv(root: Path) -> Path:
    """
    Search for ptbxl_database.csv anywhere under root/archive/.
    Handles different folder-naming conventions across machines.
    """
    archive = root / "archive"
    if archive.exists():
        matches = list(archive.rglob("ptbxl_database.csv"))
        if matches:
            return matches[0]
    # last-resort explicit paths
    fallbacks = [
        archive / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" / "ptbxl_database.csv",
        archive / "ptb-xl" / "ptbxl_database.csv",
    ]
    for fb in fallbacks:
        if fb.exists():
            return fb
    return fallbacks[0]  # will raise a clear FileNotFoundError if missing


ROOT     = _find_root()
FEAT_DIR = ROOT / "ptbxl_comprehensive_features"
DB_CSV   = _find_db_csv(ROOT)
ART_DIR  = ROOT / "ECG_Diag_pipeline" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

print(f"[paths] ROOT     = {ROOT}")
print(f"[paths] FEAT_DIR = {FEAT_DIR}")
print(f"[paths] DB_CSV   = {DB_CSV}")

CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]

SCP_MAP = {
    "NORM": "NORM",
    "IMI":"MI","ILMI":"MI","AMI":"MI","ALMI":"MI","INJAS":"MI","LMI":"MI",
    "INJAL":"MI","IPLMI":"MI","IPMI":"MI","INJIN":"MI","INJLA":"MI","PMI":"MI","INJIL":"MI","INJA":"MI",
    "NDT":"STTC","DIG":"STTC","LNGQT":"STTC","ANEUR":"STTC","EL":"STTC",
    "ISCA":"STTC","ISCI":"STTC","ISC_":"STTC","STTC":"STTC","STD_":"STTC","STE_":"STTC",
    "LAFB":"CD","IRBBB":"CD","IVCD":"CD","LBBB":"CD","RBBB":"CD","LPFB":"CD","WPW":"CD",
    "1AVB":"CD","2AVB":"CD","3AVB":"CD","AVB":"CD",
    "LVH":"HYP","LAO":"HYP","RVH":"HYP","SEHYP":"HYP","LVOLT":"HYP","RAO":"HYP","LMH":"HYP",
}


def get_superclass(scp_str: str) -> str | None:
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


# -- Feature keys we EXCLUDE from the flat vector (non-numeric / meta) ------
SKIP_KEYS = {"lead_name", "sampling_freq"}

# Feature categories to include -- all that exist in the JSON
FEATURE_GROUPS = ["stat", "temp", "morph", "spec", "tf", "nonlin"]


def flatten_record_features(features_obj: dict) -> dict:
    """
    Convert the per-lead nested feature dict into a flat dict.
    E.g. lead_0_I -> {lead_0_I_stat_mean, lead_0_I_temp_hr_mean, ...}
    """
    flat: dict[str, float] = {}
    for lead_key, lead_feats in features_obj.items():
        # lead_key like 'lead_0_I', lead_feats is dict of {feature_name: value}
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


def load_all_batch_features() -> pd.DataFrame:
    """Load all batch JSON files and return a DataFrame with record_id + features."""
    batch_files = sorted(FEAT_DIR.glob("batch_*_features.json"))
    print(f"Found {len(batch_files)} batch files in {FEAT_DIR}")

    records = []
    t0 = time.time()
    total_success = 0
    total_fail    = 0

    for bi, bf in enumerate(batch_files):
        try:
            with open(bf, "r", encoding="utf-8") as f:
                batch = json.load(f)
        except Exception as e:
            print(f"  [!] Could not read {bf.name}: {e}")
            continue

        # batch may be a list or a dict
        if isinstance(batch, dict):
            batch = [batch]

        for rec in batch:
            if not rec.get("success", False):
                total_fail += 1
                continue
            rid  = rec.get("record_id")
            feat = rec.get("features")
            if rid is None or feat is None:
                total_fail += 1
                continue

            flat = flatten_record_features(feat if isinstance(feat, dict) else {})
            flat["record_id"] = int(rid)
            records.append(flat)
            total_success += 1

        if (bi + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{bi+1:>3}/{len(batch_files)}] loaded={total_success:>6}  "
                  f"failed={total_fail:>4}  elapsed={elapsed:.0f}s", flush=True)

    print(f"\nLoaded {total_success} successful records "
          f"({total_fail} failed) in {time.time()-t0:.0f}s")

    df = pd.DataFrame(records)
    print(f"DataFrame shape: {df.shape}")
    return df


def load_labels() -> pd.DataFrame:
    """
    Load ptbxl_database.csv and return DataFrame with ecg_id + superclass.
    """
    db = pd.read_csv(DB_CSV)
    db["superclass"] = db["scp_codes"].apply(get_superclass)
    db = db.dropna(subset=["superclass"])
    db = db[db["superclass"].isin(CLASSES)]
    return db[["ecg_id", "superclass", "strat_fold"]].rename(columns={"ecg_id": "record_id"})


def print_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    col_w = 8
    header = " " * 12 + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row_str = "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(labels)))
        print(f"  {label:<10}{row_str}")


# ==============================================================================
def main():
    print("=" * 70)
    print("  PTB-XL High-Accuracy Classifier (LGB + XGB + RF Ensemble)")
    print("=" * 70)

    # -- 1. Load features -------------------------------------------------------
    print("\n[1/5] Loading comprehensive batch features ...")
    feat_df = load_all_batch_features()

    # -- 2. Load labels ---------------------------------------------------------
    print("\n[2/5] Loading labels from ptbxl_database.csv ...")
    label_df = load_labels()
    print(f"  Labelled records available: {len(label_df)}")
    print("  Class distribution:")
    print(label_df["superclass"].value_counts().to_string(header=False))

    # -- 3. Merge ---------------------------------------------------------------
    print("\n[3/5] Merging features with labels ...")
    merged = feat_df.merge(label_df, on="record_id", how="inner")
    print(f"  Merged records: {len(merged)}")

    # -- 4. Feature matrix ------------------------------------------------------
    meta_cols = {"record_id", "superclass", "strat_fold", "record_path"}
    feat_cols = [c for c in merged.columns if c not in meta_cols]
    print(f"\n  Raw feature columns: {len(feat_cols)}")

    X_raw = merged[feat_cols].values.astype(np.float32)
    y     = merged["superclass"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)          # int labels for LGB/XGB
    print(f"  Classes: {list(le.classes_)}")

    # -- 5. Split (official PTB-XL folds 9+10 = val) ---------------------------
    print("\n[4/5] Splitting data ...")
    strat_fold = merged["strat_fold"].values
    val_mask   = (strat_fold >= 9)
    X_tr_raw, X_va_raw = X_raw[~val_mask], X_raw[val_mask]
    y_tr,     y_va     = y[~val_mask],     y[val_mask]
    y_tr_enc, y_va_enc = y_enc[~val_mask], y_enc[val_mask]
    print(f"  Train: {len(X_tr_raw)}  |  Val: {len(X_va_raw)}")

    # -- 6. Preprocessing: impute + drop near-zero-variance --------------------
    print("\n[5/5] Preprocessing + training ...")
    t0 = time.time()

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr_raw)
    X_va = imputer.transform(X_va_raw)

    var_sel = VarianceThreshold(threshold=1e-6)
    X_tr = var_sel.fit_transform(X_tr)
    X_va = var_sel.transform(X_va)
    kept_mask  = var_sel.get_support()
    feat_cols_sel = [c for c, k in zip(feat_cols, kept_mask) if k]
    print(f"  Features after variance filter: {X_tr.shape[1]}")

    # Class weights for sklearn models
    from sklearn.utils.class_weight import compute_class_weight
    cw_vals = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw_dict = dict(zip(np.unique(y_tr), cw_vals))

    # Sample weights for LGB/XGB
    sample_w_tr = np.array([cw_dict[c] for c in y_tr])

    results: dict[str, float] = {}
    models:  dict[str, object] = {}

    # ── Model A: LightGBM ─────────────────────────────────────────────────────
    if HAS_LGB:
        print("\n  [A] LightGBM (fast, ~5-15 min) ...")
        lgb_model = lgb.LGBMClassifier(
            objective         = "multiclass",
            num_class         = len(le.classes_),
            metric            = "multi_logloss",
            learning_rate     = 0.05,   # fast convergence
            n_estimators      = 700,    # early stop will cut this down
            num_leaves        = 63,     # 255 was too slow (hours per run)
            max_depth         = -1,
            min_child_samples = 20,
            subsample         = 0.8,
            subsample_freq    = 1,
            colsample_bytree  = 0.5,    # fewer features per tree = faster
            reg_alpha         = 0.05,
            reg_lambda        = 0.5,
            force_col_wise    = True,   # faster with >100 features
            n_jobs            = -1,
            random_state      = 42,
            verbosity         = -1,
        )
        lgb_model.fit(
            X_tr, y_tr_enc,
            sample_weight = sample_w_tr,
            eval_set      = [(X_va, y_va_enc)],
            callbacks     = [lgb.early_stopping(50, verbose=True),
                             lgb.log_evaluation(10)],  # print every 10 rounds
        )
        lgb_preds_enc = lgb_model.predict(X_va)
        lgb_preds     = le.inverse_transform(lgb_preds_enc)
        lgb_acc       = accuracy_score(y_va, lgb_preds)
        lgb_bal       = balanced_accuracy_score(y_va, lgb_preds)
        print(f"    Val accuracy         : {lgb_acc*100:.2f}%")
        print(f"    Balanced val accuracy: {lgb_bal*100:.2f}%")
        results["LightGBM"] = lgb_acc
        models["LightGBM"]  = lgb_model
    else:
        print("  [A] LightGBM SKIPPED (not installed)")

    # ── Model B: XGBoost ──────────────────────────────────────────────────────
    if HAS_XGB:
        print("\n  [B] XGBoost (fast, ~10-20 min) ...")
        xgb_model = xgb.XGBClassifier(
            objective             = "multi:softprob",
            num_class             = len(le.classes_),
            eval_metric           = "mlogloss",
            learning_rate         = 0.05,
            n_estimators          = 700,
            max_depth             = 6,
            subsample             = 0.8,
            colsample_bytree      = 0.5,
            reg_alpha             = 0.05,
            reg_lambda            = 0.5,
            min_child_weight      = 5,
            early_stopping_rounds = 50,   # in constructor (XGB >= 2.0 API)
            tree_method           = "hist",  # fast histogram method
            n_jobs                = -1,
            random_state          = 42,
            verbosity             = 0,
        )
        xgb_model.fit(
            X_tr, y_tr_enc,
            sample_weight = sample_w_tr,
            eval_set      = [(X_va, y_va_enc)],
            verbose       = 10,
        )
        xgb_preds_enc = xgb_model.predict(X_va)
        xgb_preds     = le.inverse_transform(xgb_preds_enc)
        xgb_acc       = accuracy_score(y_va, xgb_preds)
        xgb_bal       = balanced_accuracy_score(y_va, xgb_preds)
        print(f"    Val accuracy         : {xgb_acc*100:.2f}%")
        print(f"    Balanced val accuracy: {xgb_bal*100:.2f}%")
        results["XGBoost"] = xgb_acc
        models["XGBoost"]  = xgb_model
    else:
        print("  [B] XGBoost SKIPPED (not installed)")

    # -- Model C: Random Forest (fallback only if no boosting available) -------
    rf_preds = None
    if not HAS_LGB and not HAS_XGB:
        print("\n  [C] Random Forest (fallback, 300 trees) ...")
        t_rf = time.time()
        rf = RandomForestClassifier(
            n_estimators=300, max_features=0.3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42
        )
        rf.fit(X_tr, y_tr)
        print(f"      RF done in {time.time()-t_rf:.0f}s")
        rf_preds = rf.predict(X_va)
        rf_acc   = accuracy_score(y_va, rf_preds)
        print(f"    Val accuracy: {rf_acc*100:.2f}%")
        results["RandomForest"] = rf_acc
        models["RandomForest"]  = rf
    else:
        print("\n  [C] Random Forest SKIPPED (boosting models are faster and better)")

    # -- Soft-voting ensemble (LGB + XGB) ---------------------------------------
    best_name   = max(results, key=results.get) if results else "none"
    best_acc    = results[best_name] if results else 0.0
    final_preds = (lgb_preds if best_name == "LightGBM" else
                   xgb_preds if best_name == "XGBoost"  else rf_preds)

    if HAS_LGB and HAS_XGB and "LightGBM" in results and "XGBoost" in results:
        print("\n  [D] Soft-voting ensemble (LGB + XGB) ...")
        lgb_proba = lgb_model.predict_proba(X_va)
        xgb_proba = xgb_model.predict_proba(X_va)

        # try weight combinations of just LGB and XGB
        best_ens_acc = 0.0
        best_ens_preds = None
        best_ens_w = (1, 1)
        for w_lgb, w_xgb in [(1,1),(2,1),(1,2),(3,1),(1,3),(3,2),(2,3)]:
            avg = (w_lgb*lgb_proba + w_xgb*xgb_proba) / (w_lgb + w_xgb)
            ep  = le.inverse_transform(np.argmax(avg, axis=1))
            ea  = accuracy_score(y_va, ep)
            if ea > best_ens_acc:
                best_ens_acc   = ea
                best_ens_preds = ep
                best_ens_w     = (w_lgb, w_xgb)

        ens_bal = balanced_accuracy_score(y_va, best_ens_preds)
        print(f"    Best weights (lgb, xgb): {best_ens_w}")
        print(f"    Ensemble val accuracy    : {best_ens_acc*100:.2f}%")
        print(f"    Ensemble balanced acc    : {ens_bal*100:.2f}%")

        if best_ens_acc > best_acc:
            best_acc    = best_ens_acc
            best_name   = "Ensemble(LGB+XGB)"
            final_preds = best_ens_preds

    # ── Final results ----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  BEST MODEL  : {best_name}")
    print(f"  Val Accuracy: {best_acc*100:.2f}%")
    print(f"{'='*70}")

    print("\n  Classification Report (Validation):")
    print(classification_report(y_va, final_preds, target_names=CLASSES, digits=4))

    print("  Confusion Matrix (Validation):")
    print_confusion_matrix(y_va, final_preds, CLASSES)

    elapsed = time.time() - t0
    print(f"\n  Training completed in {elapsed:.0f}s")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    print("\n  Saving artifacts to ECG_Diag_pipeline/artifacts/ ...")
    ART_DIR.mkdir(parents=True, exist_ok=True)

    # Save all individual models
    for mname, mobj in models.items():
        mpath = ART_DIR / f"model_{mname.lower()}.pkl"
        with open(mpath, "wb") as f: pickle.dump(mobj, f)
        print(f"  {mname} -> {mpath}")

    with open(ART_DIR / "imputer.pkl",    "wb") as f: pickle.dump(imputer, f)
    with open(ART_DIR / "var_selector.pkl","wb") as f: pickle.dump(var_sel, f)
    with open(ART_DIR / "label_encoder.pkl","wb") as f: pickle.dump(le, f)
    with open(ART_DIR / "feature_cols.txt", "w") as f: f.write("\n".join(feat_cols_sel))

    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_va, final_preds, labels=CLASSES, zero_division=0)
    metrics = {
        "best_model": best_name,
        "val_accuracy": float(best_acc),
        "val_balanced_accuracy": float(balanced_accuracy_score(y_va, final_preds)),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_features": int(X_tr.shape[1]),
        "individual_accuracies": {k: float(v) for k, v in results.items()},
        "per_class": {
            cls: {"precision": float(prec[i]), "recall": float(rec[i]),
                  "f1": float(f1[i]), "support": int(sup[i])}
            for i, cls in enumerate(CLASSES)
        },
    }
    with open(ART_DIR / "val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Done. Best val accuracy = {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
