.PHONY: setup train serve test prod-up prod-down manifest
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
train:
	python app/train.py
serve:
	uvicorn app.serve:app --reload --port 8000
test:
	pytest -q
prod-up:
	docker compose --profile prod up --build -d
prod-down:
	docker compose --profile prod down
manifest:
	python -c "from app.manifest_utils import init_manifest, save_manifest; import os; m=init_manifest(os.getenv('AUTOPILOT_DATA_PATHS')); save_manifest(m); print('manifest scritto')"
