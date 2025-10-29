from __future__ import annotations
from pathlib import Path
import pandas as pd

ALLOWED_EXT = {".csv", ".json"}
MAX_FILE_MB = 100

def _is_allowed_file(p: Path) -> bool:
    if p.name.startswith("."):
        return False
    if p.suffix.lower() not in ALLOWED_EXT:
        return False
    try:
        size_mb = p.stat().st_size / (1024 * 1024)
    except FileNotFoundError:
        return False
    return size_mb <= MAX_FILE_MB

def _read_one(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".json":
        try:
            return pd.read_json(p, lines=True)
        except ValueError:
            return pd.read_json(p)
    raise ValueError(f"Estensione non supportata: {p.suffix}")

def collect_frames(paths: list[str]) -> list[pd.DataFrame]:
    frames = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and _is_allowed_file(f):
                    frames.append(_read_one(f))
        elif p.is_file() and _is_allowed_file(p):
            frames.append(_read_one(p))
    return frames

def union_on_common_columns(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise FileNotFoundError("Nessun file valido trovato nelle sorgenti.")
    common = set(frames[0].columns)
    for df in frames[1:]:
        common &= set(df.columns)
    if not common:
        raise ValueError("Nessuna colonna in comune tra i file caricati.")
    common = list(common)
    return pd.concat([df[common] for df in frames], ignore_index=True)
