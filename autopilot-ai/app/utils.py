from pathlib import Path
import json
import datetime as dt

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
REGISTRY_DIR = BASE / "models"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

def now_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
