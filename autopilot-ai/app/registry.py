from pathlib import Path
import joblib
from .utils import REGISTRY_DIR, save_json, load_json, now_iso

def _model_dir(version: str) -> Path:
    return REGISTRY_DIR / version

def latest_version() -> str | None:
    versions = sorted([p.name for p in REGISTRY_DIR.iterdir() if p.is_dir()])
    return versions[-1] if versions else None

def save_model(version: str, pipeline, meta: dict):
    d = _model_dir(version)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, d / "model.joblib")
    save_json(meta, d / "meta.json")
    save_json({"version": version, "updated_at": now_iso()}, REGISTRY_DIR / "latest.json")

def load_model(version: str | None = None):
    if version is None:
        try:
            latest = load_json(REGISTRY_DIR / "latest.json")["version"]
        except FileNotFoundError:
            latest = latest_version()
        version = latest
    if version is None:
        raise FileNotFoundError("Nessun modello in registry.")
    d = _model_dir(version)
    pipeline = joblib.load(d / "model.joblib")
    meta = load_json(d / "meta.json")
    return pipeline, meta
