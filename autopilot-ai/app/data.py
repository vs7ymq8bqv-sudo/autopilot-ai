from __future__ import annotations
import os
import pandas as pd
from sklearn.datasets import load_iris
from .utils import DATA_DIR
from .ingest import collect_frames, union_on_common_columns

def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    sources = os.getenv("AUTOPILOT_DATA_PATHS")
    if sources:
        paths = [s.strip() for s in sources.split(";") if s.strip()]
        frames = collect_frames(paths)
        df = union_on_common_columns(frames)
        if "target" not in df.columns:
            raise ValueError("I dati aggregati devono contenere una colonna 'target'.")
        y = df["target"]
        X = df.drop(columns=["target"])
        return X, y

    csv_path = DATA_DIR / "train.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "target" not in df.columns:
            raise ValueError("Il CSV deve contenere una colonna 'target'.")
        y = df["target"]
        X = df.drop(columns=["target"])
        return X, y

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y
