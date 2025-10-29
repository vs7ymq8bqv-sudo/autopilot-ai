from __future__ import annotations
import time, uuid, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import optuna
from .data import load_training_data
from .model import build_pipeline
from .registry import save_model
from .utils import now_iso, BASE
from .manifest_utils import init_manifest, update_manifest_after_training, save_manifest, write_canary_report
from .validation import validate_training_data

def train_and_register(use_optuna: bool = True, seed: int = 42) -> dict:
    X, y = load_training_data()
    dict_path = BASE / 'configs' / 'data_dictionary.csv'
    if dict_path.exists():
        X, y = validate_training_data(X, y, dict_path)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    def objective(trial):
        C = trial.suggest_float("C", 1e-2, 10.0, log=True)
        pipe = build_pipeline()
        pipe.set_params(clf__C=C)
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xva)
        return accuracy_score(yva, preds)

    if use_optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        best_C = study.best_params["C"]
    else:
        best_C = 1.0

    pipe = build_pipeline()
    pipe.set_params(clf__C=best_C)
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xva)
    acc = accuracy_score(yva, preds)
    f1 = f1_score(yva, preds, average="macro")

    version = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    meta = {
        "version": version,
        "created_at": now_iso(),
        "metrics": {"val_accuracy": float(acc), "val_f1_macro": float(f1)},
        "params": {"C": float(best_C)},
        "features": list(X.columns),
        "target_name": "target"
    }
    save_model(version, pipe, meta)

    manifest = init_manifest(os.getenv('AUTOPILOT_DATA_PATHS'))
    manifest = update_manifest_after_training(manifest, meta)
    save_manifest(manifest)
    write_canary_report(meta)
    return meta

if __name__ == "__main__":
    meta = train_and_register()
    print("Model registrato:", meta)
