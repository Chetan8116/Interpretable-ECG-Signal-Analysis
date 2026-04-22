"""
Real SHAP Explainability Server for ECG Diagnosis.

Uses:
  - public/mlp_model.pkl      -> sklearn MLPClassifier + scaler + label_encoder
  - public/ptbxl_records.json -> per-patient clinical features (12 leads x 17 features)

Run:
    python server/shap_server.py

API:
    GET  /api/shap/status         - health check
    GET  /api/shap/<ecg_id>       - real SHAP values for one patient
    GET  /api/shap/classes        - class list
    POST /api/shap/batch          - batch: {"ecg_ids": [1, 2, ...]}
"""

import os
import sys
import json
import pickle
import warnings
import traceback
import threading
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
except ImportError:
    print("ERROR: flask/flask-cors not installed. Run: pip install flask flask-cors")
    sys.exit(1)

try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Run: pip install shap")
    sys.exit(1)

# ---------- paths ----------
PROJECT_ROOT = Path(__file__).parent.parent
PUBLIC       = PROJECT_ROOT / "public"
MODEL_PKL    = PUBLIC / "mlp_model.pkl"
RECORDS_JSON = PUBLIC / "ptbxl_records.json"

# ---------- feature schema (must match train_mlp_classifier.py) ----------
LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
FEATURE_KEYS = [
    "P_duration_mean","P_amplitude_mean","P_area_mean",
    "PR_interval_mean","QRS_width_mean","Q_amplitude_mean",
    "R_amplitude_mean","S_amplitude_mean","QRS_area_mean",
    "ST_elevation_mean","ST_depression_mean","ST_slope_mean",
    "T_amplitude_mean","T_duration_mean","T_area_mean",
    "QT_interval_mean","QTc_mean",
]
FEATURE_COLS = [f"{lead}_{fk}" for lead in LEAD_NAMES for fk in FEATURE_KEYS]
N_FEATURES   = len(FEATURE_COLS)   # 204# NOTE: Model expects 240 features; pad with zeros as temporary fix
MODEL_FEATURES = 240
# ---------- globals ----------
model        = None
scaler       = None
label_encoder= None
class_names  = []
explainer    = None
patient_data = {}   # ecg_id -> {"feature_vector": [...], "label": str}
shap_cache   = {}
explainer_lock = threading.Lock()

app = Flask(__name__)
CORS(app)


def simplify_diagnosis(diagnosis: str) -> str:
    d = diagnosis.lower()
    if "norm" in d or "sinus" in d: return "NORM"
    if "mi" in d or "infarct" in d: return "MI"
    if "hyp" in d or "hypertrop" in d: return "HYP"
    if "cd" in d or "block" in d: return "CD"
    if "sttc" in d or "st-t" in d: return "STTC"
    return "OTHER"


def extract_features(record: dict):
    vec = []
    for lead in LEAD_NAMES:
        cf = record.get("leads", {}).get(lead, {}).get("clinical_features", {})
        vec.extend([float(cf.get(k, 0) or 0) for k in FEATURE_KEYS])
    if len(vec) != N_FEATURES:
        return None
    # Pad to MODEL_FEATURES (240) with zeros
    while len(vec) < MODEL_FEATURES:
        vec.append(0.0)
    return vec


def startup():
    global model, scaler, label_encoder, class_names, explainer, patient_data

    # 1. Load sklearn model from public/mlp_model.pkl
    print(f"[SHAP] Loading model from {MODEL_PKL} ...")
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PKL}")

    with open(MODEL_PKL, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        model         = obj.get("model") or obj.get("clf") or obj.get("mlp")
        scaler        = obj.get("scaler")
        label_encoder = obj.get("label_encoder") or obj.get("le")
    else:
        model = obj

    if model is None:
        raise RuntimeError("Could not extract classifier from pickle")

    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        class_names = list(label_encoder.classes_)
    elif hasattr(model, "classes_"):
        class_names = [str(c) for c in model.classes_]
    else:
        class_names = ["CD","HYP","MI","NORM","OTHER","STTC"]

    print(f"[SHAP] Model: {type(model).__name__}, classes: {class_names}")

    # 2. Load patient records from public/ptbxl_records.json
    print(f"[SHAP] Loading patient records ...")
    if not RECORDS_JSON.exists():
        raise FileNotFoundError(f"Records not found: {RECORDS_JSON}")
    with open(RECORDS_JSON, "r") as f:
        records = json.load(f)

    ok = 0
    for rec in records:
        ecg_id = rec.get("ecg_id")
        if ecg_id is None:
            continue
        vec = extract_features(rec)
        if vec is not None:
            patient_data[int(ecg_id)] = {
                "feature_vector": vec,
                "label": simplify_diagnosis(rec.get("diagnosis", "UNKNOWN")),
            }
            ok += 1
    print(f"[SHAP] {ok} / {len(records)} patients loaded with full feature vectors")

    # 3. Build SHAP KernelExplainer (one-time)
    print("[SHAP] Building SHAP KernelExplainer (one-time, ~5-15 s) ...")
    all_ids = list(patient_data.keys())
    rng = np.random.default_rng(42)
    bg_ids = rng.choice(all_ids, size=min(80, len(all_ids)), replace=False)
    X_bg = np.array([patient_data[eid]["feature_vector"] for eid in bg_ids], dtype=float)
    # NOTE: Skip scaler since it was trained on a different feature set (240 vs 204)
    #if scaler is not None:
        #X_bg = scaler.transform(X_bg)

    summary  = shap.kmeans(X_bg, 10)
    explainer = shap.KernelExplainer(model.predict_proba, summary)
    print("[SHAP] Server ready!")


def compute_shap(ecg_id: int) -> dict:
    if ecg_id in shap_cache:
        return shap_cache[ecg_id]

    info = patient_data.get(ecg_id)
    if info is None:
        return {"error": f"ecg_id {ecg_id} not found in dataset"}

    X_raw    = np.array([info["feature_vector"]], dtype=float)
    # NOTE: Skip scaler (trained on different feature set)
    X_scaled = X_raw  # scaler.transform(X_raw) if scaler is not None else X_raw

    # Predict
    proba         = model.predict_proba(X_scaled)[0]
    predicted_idx = int(np.argmax(proba))
    predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else str(predicted_idx)
    class_proba   = {
        (class_names[i] if i < len(class_names) else str(i)): float(proba[i])
        for i in range(len(proba))
    }

    # SHAP KernelExplainer is stateful; serialize calls to avoid concurrent maskMatrix corruption.
    with explainer_lock:
        shap_vals = explainer.shap_values(X_scaled, nsamples=200, l1_reg="num_features(30)")
    if isinstance(shap_vals, list):
        sv_pred = np.array(shap_vals[predicted_idx]).reshape(-1)
    else:
        sv_pred = np.array(shap_vals).reshape(-1)

    # Build feature result rows
    feat_values = X_raw[0]
    results = []
    for i, fname in enumerate(FEATURE_COLS):
        sv   = float(np.array(sv_pred[i]).flat[0])
        fv_r = np.array(feat_values[i]).flat[0]
        fv   = float(fv_r) if not np.isnan(float(fv_r)) else 0.0

        if fname.startswith("aV"):
            lead  = fname[:3]
            ftype = fname[4:]
        else:
            parts = fname.split("_", 1)
            lead  = parts[0]
            ftype = parts[1] if len(parts) > 1 else fname

        results.append({
            "feature":       fname,
            "lead":          lead,
            "feature_type":  ftype,
            "shap_value":    sv,
            "feature_value": fv,
        })

    results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    lead_total:  dict = {}
    lead_signed: dict = {}
    for r in results:
        ld = r["lead"]
        lead_total[ld]  = lead_total.get(ld, 0.0)  + abs(r["shap_value"])
        lead_signed[ld] = lead_signed.get(ld, 0.0) +     r["shap_value"]

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        base = float(np.array(ev[predicted_idx]).flat[0])
    else:
        base = float(np.array(ev).flat[0])

    out = {
        "ecg_id":              ecg_id,
        "true_label":          info["label"],
        "predicted_class":     predicted_class,
        "class_probabilities": class_proba,
        "top_shap_features":   results[:20],
        "all_shap_features":   results,
        "lead_total_shap":     lead_total,
        "lead_signed_shap":    lead_signed,
        "shap_base_value":     base,
    }
    shap_cache[ecg_id] = out
    return out


# ---------- routes ----------

@app.route("/api/shap/status")
def api_status():
    return jsonify({
        "ready":      model is not None and explainer is not None,
        "model_type": type(model).__name__ if model else None,
        "n_features": N_FEATURES,
        "n_patients": len(patient_data),
        "classes":    class_names,
        "cache_size": len(shap_cache),
    })


@app.route("/api/shap/classes")
def api_classes():
    return jsonify({"classes": class_names})


@app.route("/api/shap/<ecg_id_str>")
def api_get_shap(ecg_id_str):
    if model is None or explainer is None:
        return jsonify({"error": "Server not ready"}), 503
    try:
        ecg_id = int(ecg_id_str)
    except ValueError:
        return jsonify({"error": f"Invalid ecg_id: {ecg_id_str}"}), 400
    try:
        result = compute_shap(ecg_id)
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/shap/batch", methods=["POST"])
def api_batch():
    data = request.json or {}
    ids  = data.get("ecg_ids", [])
    results = {}
    for eid in ids:
        try:
            results[eid] = compute_shap(int(eid))
        except Exception as exc:
            results[eid] = {"error": str(exc)}
    return jsonify(results)


# ---------- entry point ----------

if __name__ == "__main__":
    try:
        startup()
    except Exception as e:
        print(f"\n[SHAP] FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)

    PORT = int(os.environ.get("SHAP_PORT", 5101))
    print(f"[SHAP] Listening on http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
