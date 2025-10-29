from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field, conlist
from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import multiprocess
from fastapi.responses import Response
import numpy as np, os
from .registry import load_model
from .train import train_and_register

app = FastAPI(title="Autopilot AI")

class IrisInput(BaseModel):
    sepal_length: float = Field(..., description="cm")
    sepal_width:  float = Field(..., description="cm")
    petal_length: float = Field(..., description="cm")
    petal_width:  float = Field(..., description="cm")

class BatchIris(BaseModel):
    items: conlist(IrisInput, min_length=1)

registry = CollectorRegistry()
if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
    multiprocess.MultiProcessCollector(registry)
PRED_COUNT = Counter("predictions_total", "Numero di predizioni", registry=registry)
CURRENT_VERSION = Gauge("model_version_info", "Versione attuale (label version)", ["version"], registry=registry)

try:
    PIPELINE, META = load_model()
except Exception:
    meta = train_and_register(use_optuna=True)
    PIPELINE, META = load_model(meta["version"])
CURRENT_VERSION.labels(META["version"]).set(1)

def predict_inner(x):
    pred = PIPELINE.predict(x)[0]
    probs = PIPELINE.predict_proba(x)[0].tolist() if hasattr(PIPELINE, "predict_proba") else None
    return int(pred), probs

@app.get("/health")
def health():
    return {"status": "ok", "model_version": META["version"]}

@app.get("/ready")
def ready():
    return {"ready": PIPELINE is not None, "model_version": META.get("version")}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(payload: IrisInput):
    x = np.array([[payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width]], dtype=float)
    yhat, probs = predict_inner(x)
    PRED_COUNT.inc()
    return {"version": META["version"], "pred": yhat, "probs": probs}

@app.post("/predict_batch")
def predict_batch(payload: BatchIris):
    xs = np.array([[i.sepal_length, i.sepal_width, i.petal_length, i.petal_width] for i in payload.items], dtype=float)
    preds = PIPELINE.predict(xs).tolist()
    PRED_COUNT.inc(len(preds))
    probs = PIPELINE.predict_proba(xs).tolist() if hasattr(PIPELINE, "predict_proba") else None
    return {"version": META["version"], "preds": preds, "probs": probs}

@app.post("/retrain")
def retrain():
    global PIPELINE, META
    meta = train_and_register(use_optuna=True)
    PIPELINE, META = load_model(meta["version"])
    CURRENT_VERSION.clear()
    CURRENT_VERSION.labels(META["version"]).set(1)
    return {"status": "trained", "meta": META}
