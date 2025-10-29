from __future__ import annotations
from pathlib import Path
import os, json, hashlib, datetime as dt
from .utils import BASE, DATA_DIR

PROJECT_PREFIX = "AUTOPILOT-AI"
KNOWLEDGE_DIR = BASE / "knowledge"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _rel_or_abs(p: Path) -> str:
    try:
        return str(p.relative_to(BASE))
    except ValueError:
        return str(p)

def _scan_path(p: Path) -> list[dict]:
    items = []
    if p.is_file():
        items.append(p)
    elif p.is_dir():
        for f in p.rglob("*"):
            if f.is_file():
                items.append(f)
    out = []
    for f in items:
        try:
            sz = f.stat().st_size
            out.append({
                "path": _rel_or_abs(f),
                "sha256": _sha256_file(f),
                "size_bytes": sz
            })
        except Exception as e:
            out.append({
                "path": _rel_or_abs(f),
                "error": str(e)
            })
    return out

def init_manifest(sources_env: str | None) -> dict:
    created_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    manifest = {
        "project": PROJECT_PREFIX,
        "created_at": created_at,
        "sources_env": sources_env,
        "sources": [],
        "artifacts": {
            "registry_dir": "models/",
            "reports_dir": f"knowledge/{PROJECT_PREFIX}_EVIDENCE.md"
        },
        "notes": "Manifest generato automaticamente durante il training."
    }
    paths = []
    if sources_env:
        paths = [Path(s.strip()).expanduser() for s in sources_env.split(";") if s.strip()]
    else:
        csv_path = DATA_DIR / "train.csv"
        if csv_path.exists():
            paths = [csv_path]
        else:
            manifest["sources"].append({
                "symbolic": "sklearn.datasets.load_iris",
                "license": "scikit-learn dataset license",
                "note": "Nessun file locale. Usato Iris di scikit-learn."
            })
    for p in paths:
        manifest["sources"].append({
            "root": _rel_or_abs(p),
            "files": _scan_path(p)
        })
    return manifest

def update_manifest_after_training(manifest: dict, meta: dict) -> dict:
    updated_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    manifest["updated_at"] = updated_at
    manifest.setdefault("train_runs", []).append({
        "version": meta.get("version"),
        "created_at": meta.get("created_at"),
        "metrics": meta.get("metrics"),
        "features": meta.get("features"),
        "params": meta.get("params"),
    })
    return manifest

def save_manifest(manifest: dict) -> Path:
    out = KNOWLEDGE_DIR / f"{PROJECT_PREFIX}_MANIFEST.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out

def write_canary_report(meta: dict) -> Path:
    out = KNOWLEDGE_DIR / f"{PROJECT_PREFIX}_CANARY_REPORT.md"
    txt = (
        f"# {PROJECT_PREFIX} — Canary Report\n"
        f"Versione: {meta.get('version')}  \n"
        f"Creato: {meta.get('created_at')}\n\n"
        "## Metriche (validation)\n"
        f"- Accuracy: {meta.get('metrics',{}).get('val_accuracy')}  \n"
        f"- F1 macro: {meta.get('metrics',{}).get('val_f1_macro')}\n\n"
        "## Note\n"
        "Rapporto generato automaticamente a fine training.\n"
    )
    out.write_text(txt, encoding="utf-8")
    return out
