# -*- coding: utf-8 -*-
"""
PTB-XL XGBoost with Extreme Class Balancing
===========================================
Try XGBoost as alternative to LightGBM - sometimes handles imbalance better.

Usage:
    python train_xgboost_balanced.py --fast
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
    import xgboost as xgb
except ImportError:
    raise SystemExit("Install XGBoost: pip install xgboost")

warnings.filterwarnings("ignore")

FAST_MODE = "--fast" in sys.argv
N_FAST = 5000
N_FEATS = 175

# Paths
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

ROOT = _find_root()
FEAT_DIR = ROOT / "ptbxl_comprehensive_features"
DB_CSV = _find_db_csv(ROOT)
ART_DIR = ROOT / "ECG_Diag_pipeline" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

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

def get_superclass(scp_str):
    try:
        codes = ast.literal_eval(scp_str) if isinstance(scp_str, str) else scp_str
        scored = {}
        for code, conf in codes.items():
            sc = SCP_MAP.get(code)
            if sc:
                scored[sc] = scored.get(sc, 0) + conf
        return max(scored, key=scored.get) if scored else None
    except:
        return None

def flatten_record(features_obj: dict) -> dict:
    flat = {}
    for lead_key, lead_feats in features_obj.items():
        if not isinstance(lead_feats, dict):
            continue
        for fname, val in lead_feats.items():
            if fname in SKIP_KEYS:
                continue
            col = f"{lead_key}_{fname}"
            try:
                flat[col] = float(val) if val is not None else np.nan
            except:
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
            continue
        if isinstance(batch, dict):
            batch = [batch]
        for rec in batch:
            if not rec.get("success", False): continue
            rid = rec.get("record_id")
            feat = rec.get("features")
            if rid is None or feat is None: continue
            fl = flatten_record(feat if isinstance(feat, dict) else {})
            fl["record_id"] = int(rid)
            records.append(fl)
            n_ok += 1
        if (bi + 1) % 50 == 0:
            print(f"  [{bi+1:>3}/{len(batch_files)}] loaded={n_ok}  {time.time()-t0:.0f}s", flush=True)
    print(f"  Loaded {n_ok} records in {time.time()-t0:.0f}s")
    return pd.DataFrame(records)

def load_labels() -> pd.DataFrame:
    db = pd.read_csv(DB_CSV)
    db["superclass"] = db["scp_codes"].apply(get_superclass)
    db = db.dropna(subset=["superclass"])
    db = db[db["superclass"].isin(CLASSES)]
    return db[["ecg_id", "superclass", "strat_fold"]].rename(columns={"ecg_id": "record_id"})

def print_cm(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    w = 8
    print(" " * 12 + "".join(f"{l:>{w}}" for l in labels))
    for i, lbl in enumerate(labels):
        print(f"  {lbl:<10}" + "".join(f"{cm[i,j]:>{w}}" for j in range(len(labels))))

def main():
    print("=" * 70)
    print("  PTB-XL XGBoost with Extreme Balancing")
    print("=" * 70)

    print("\n[1/4] Loading features ...")
    feat_df = load_features()
    print("\n[2/4] Loading labels ...")
    label_df = load_labels()

    merged = feat_df.merge(label_df, on="record_id", how="inner")
    print(f"  Total merged records: {len(merged)}")
    print(merged["superclass"].value_counts().to_string(header=False))

    if FAST_MODE:
        from sklearn.model_selection import train_test_split as tts
        _, merged = tts(merged, test_size=N_FAST, stratify=merged["superclass"], random_state=42)
        print(f"\n  [FAST MODE] Using {len(merged)} records")
        print(merged["superclass"].value_counts().to_string(header=False))

    meta_cols = {"record_id", "superclass", "strat_fold", "record_path"}
    feat_cols = [c for c in merged.columns if c not in meta_cols]
    X_raw = merged[feat_cols].values.astype(np.float32)
    y = merged["superclass"].values

    strat_fold = merged["strat_fold"].values
    val_mask = strat_fold >= 9
    X_tr_r, X_va_r = X_raw[~val_mask], X_raw[val_mask]
    y_tr, y_va = y[~val_mask], y[val_mask]
    print(f"\n  Train: {len(X_tr_r)}  |  Val: {len(X_va_r)}")

    print("\n[3/4] Preprocessing ...")
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr_r)
    X_va = imputer.transform(X_va_r)

    var_sel = VarianceThreshold(threshold=1e-6)
    X_tr = var_sel.fit_transform(X_tr)
    X_va = var_sel.transform(X_va)
    print(f"  After variance filter : {X_tr.shape[1]} features")

    p01 = np.percentile(X_tr, 1, axis=0)
    p99 = np.percentile(X_tr, 99, axis=0)
    X_tr = np.clip(X_tr, p01, p99)
    X_va = np.clip(X_va, p01, p99)

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_va_enc = le.transform(y_va)

    print(f"  Selecting top {N_FEATS} features ...")
    mi_sel = SelectKBest(mutual_info_classif, k=min(N_FEATS, X_tr.shape[1]))
    X_tr = mi_sel.fit_transform(X_tr, y_tr_enc)
    X_va = mi_sel.transform(X_va)
    print(f"  Features selected: {X_tr.shape[1]}")

    # Extreme class weights
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw_dict = dict(zip(np.unique(y_tr), cw))
    sw = np.array([cw_dict[c] for c in y_tr]) ** 2.0  # Square for extreme weighting

    print("\n[4/4] Training XGBoost ...")
    t0 = time.time()

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        learning_rate=0.01,
        n_estimators=2000,
        max_depth=12,
        min_child_weight=2,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=0.3,
        reg_lambda=3.0,
        gamma=0.5,
        scale_pos_weight=None,  # Use sample_weight instead
        n_jobs=-1,
        random_state=42,
        verbosity=1,
    )

    model.fit(
        X_tr, y_tr_enc,
        sample_weight=sw,
        eval_set=[(X_va, y_va_enc)],
        early_stopping_rounds=200,
        verbose=50,
    )

    y_pred_enc = model.predict(X_va)
    y_pred = le.inverse_transform(y_pred_enc)
    
    acc = accuracy_score(y_va, y_pred)
    bal_acc = balanced_accuracy_score(y_va, y_pred)
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  XGBoost Val Accuracy   : {acc*100:.2f}%")
    print(f"  Balanced Val Accuracy  : {bal_acc*100:.2f}%")
    print(f"  Training time          : {elapsed:.0f}s")
    print(f"{'='*70}")

    print("\n  Classification Report:")
    print(classification_report(y_va, y_pred, target_names=CLASSES, digits=4))
    print("  Confusion Matrix:")
    print_cm(y_va, y_pred, CLASSES)

    print("\n  Saving artifacts ...")
    with open(ART_DIR / "xgb_model.pkl", "wb") as f: pickle.dump(model, f)
    with open(ART_DIR / "xgb_le.pkl", "wb") as f: pickle.dump(le, f)
    with open(ART_DIR / "xgb_imputer.pkl", "wb") as f: pickle.dump(imputer, f)
    with open(ART_DIR / "xgb_var_sel.pkl", "wb") as f: pickle.dump(var_sel, f)
    with open(ART_DIR / "xgb_mi_sel.pkl", "wb") as f: pickle.dump(mi_sel, f)
    with open(ART_DIR / "xgb_clip.pkl", "wb") as f: pickle.dump((p01, p99), f)

    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, sup = precision_recall_fscore_support(y_va, y_pred, labels=CLASSES, zero_division=0)
    with open(ART_DIR / "xgb_metrics.json", "w") as f:
        json.dump({
            "val_accuracy": float(acc),
            "val_balanced_accuracy": float(bal_acc),
            "n_features": int(X_tr.shape[1]),
            "per_class": {
                cls: {"precision": float(prec[i]), "recall": float(rec[i]),
                      "f1": float(f1[i]), "support": int(sup[i])}
                for i, cls in enumerate(CLASSES)
            }
        }, f, indent=2)

    print(f"  Done. XGBoost accuracy = {acc*100:.2f}%")

if __name__ == "__main__":
    main()
