from __future__ import annotations
from pathlib import Path
import pandas as pd

DTYPE_MAP = {
    "float": "float64",
    "int": "Int64",
    "string": "string",
    "bool": "boolean"
}

def _coerce_series(s: pd.Series, dtype: str, nullable: bool) -> pd.Series:
    if dtype == "float":
        return pd.to_numeric(s, errors="raise").astype("float64")
    if dtype == "int":
        ser = pd.to_numeric(s, errors="raise").astype("Int64")
        if not nullable and ser.isna().any():
            raise ValueError(f"Colonna {s.name}: valori nulli non permessi")
        return ser
    if dtype == "string":
        ser = s.astype("string")
        if not nullable and ser.isna().any():
            raise ValueError(f"Colonna {s.name}: valori nulli non permessi")
        return ser
    if dtype == "bool":
        mapping = {"true": True, "false": False, "1": True, "0": False}
        ser = s.map(lambda v: mapping.get(str(v).strip().lower(), v)).astype("boolean")
        if not nullable and ser.isna().any():
            raise ValueError(f"Colonna {s.name}: valori nulli non permessi")
        return ser
    return s

def validate_training_data(X: pd.DataFrame, y: pd.Series, dict_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    dd = pd.read_csv(dict_path)
    required = dd[dd["nullable"].astype(str).str.lower() == "false"]["name"].tolist()
    missing = [c for c in required if c != "target" and c not in X.columns]
    if missing:
        raise ValueError(f"Feature mancanti rispetto al data_dictionary: {missing}")
    for _, row in dd.iterrows():
        name = row["name"]
        dtype = str(row.get("dtype", "")).strip().lower()
        nullable = str(row.get("nullable", "true")).strip().lower() == "true"
        if name == "target":
            continue
        if name in X.columns and dtype in DTYPE_MAP:
            X[name] = _coerce_series(X[name], dtype, nullable)
    if "target" in dd["name"].values:
        trow = dd[dd["name"] == "target"].iloc[0]
        tdtype = str(trow.get("dtype", "")).strip().lower()
        tnullable = str(trow.get("nullable", "false")).strip().lower() == "true"
        if tdtype in DTYPE_MAP:
            y = _coerce_series(y.rename("target"), tdtype, tnullable)
    return X, y
