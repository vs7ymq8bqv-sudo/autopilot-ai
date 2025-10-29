from app.train import train_and_register
from app.registry import load_model

def test_train_and_load():
    meta = train_and_register(use_optuna=False)
    pipe, meta2 = load_model(meta["version"])
    assert meta2["version"] == meta["version"]
