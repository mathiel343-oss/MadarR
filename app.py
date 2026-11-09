"""MADAR AI - model server.

Loads the four Random Forest pipelines (bin fill +15/+30 min, crowd +15/+30 min)
and exposes them to the dashboard over HTTP.

Run:   python server.py
Open:  http://localhost:8000
"""
import os
import warnings
from collections import UserList

import pandas as pd
import joblib
import sklearn.compose._column_transformer as _ct
from flask import Flask, jsonify, request, send_from_directory

warnings.filterwarnings("ignore")

# The models were saved with scikit-learn 1.6.1. Newer versions renamed an internal
# helper class, which breaks joblib.load. This shim restores it. If you install
# scikit-learn==1.6.1 (see requirements.txt) it is not needed and is skipped.
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(UserList):
        pass
    _ct._RemainderColsList = _RemainderColsList

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MADAR_MODEL_DIR", os.path.join(BASE, "madar_v4_models"))

ZONES = ["Mina_Camps", "Mina_Jamarat", "Mina_Main_Roads", "Mina_Service_Areas"]
FILL_FEATURES = ["fill_percent", "fill_rate_1h", "crowd_density", "temperature_c",
                 "capacity_liters", "hour", "day", "zone"]
CROWD_FEATURES = ["zone", "hour", "day", "temperature_c", "crowd_density",
                  "crowd_5m_ago", "crowd_15m_ago", "crowd_30m_ago"]

MODELS = {
    "fill_15": ("bin_fill_15m.joblib", FILL_FEATURES),
    "fill_30": ("bin_fill_30m.joblib", FILL_FEATURES),
    "crowd_15": ("crowd_15m.joblib", CROWD_FEATURES),
    "crowd_30": ("crowd_30m.joblib", CROWD_FEATURES),
}

PIPE = {}
for key, (fname, _) in MODELS.items():
    PIPE[key] = joblib.load(os.path.join(MODEL_DIR, fname))
    print(f"loaded {fname}")

app = Flask(__name__, static_folder=None)


def _num(row, name):
    """Return a float for a required numeric field, or raise a clear error."""
    if name not in row or row[name] is None:
        raise ValueError(f"missing field '{name}'")
    try:
        return float(row[name])
    except (TypeError, ValueError):
        raise ValueError(f"field '{name}' must be a number")


def _frame(rows, features):
    clean = []
    for i, r in enumerate(rows):
        try:
            z = r.get("zone")
            if z not in ZONES:
                raise ValueError(f"zone must be one of {ZONES}")
            rec = {"zone": z}
            for f in features:
                if f != "zone":
                    rec[f] = _num(r, f)
            clean.append(rec)
        except ValueError as e:
            raise ValueError(f"row {i}: {e}")
    return pd.DataFrame(clean, columns=features)


@app.get("/api/health")
def health():
    return jsonify(ok=True, models=list(PIPE.keys()), zones=ZONES)


@app.post("/api/predict")
def predict():
    """Body: {"bins": [ {id, zone, fill_percent, fill_rate_1h, crowd_density,
    temperature_c, capacity_liters, hour, day, crowd_5m_ago, crowd_15m_ago,
    crowd_30m_ago}, ... ]}
    Returns fill and crowd predictions for +15 and +30 minutes per bin."""
    body = request.get_json(silent=True) or {}
    bins = body.get("bins")
    if not isinstance(bins, list) or not bins:
        return jsonify(error="'bins' must be a non-empty list"), 400
    if len(bins) > 2000:
        return jsonify(error="too many bins (max 2000)"), 400
    try:
        out = [{"id": b.get("id")} for b in bins]
        for key, (_, feats) in MODELS.items():
            # crowd models need the history fields; fill models do not
            need = feats
            preds = PIPE[key].predict(_frame(bins, need))
            for o, p in zip(out, preds):
                o[key] = round(float(p), 2)
        return jsonify(predictions=out)
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.get("/")
def index():
    return send_from_directory(BASE, "dashboard.html")


@app.get("/<path:p>")
def static_files(p):
    if p.startswith("api/") or p.startswith("madar_v4_models"):
        return jsonify(error="not found"), 404
    return send_from_directory(BASE, p)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
