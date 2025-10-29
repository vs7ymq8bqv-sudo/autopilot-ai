# Autopilot AI (Hardened, Mobile-friendly)

Avvio rapido (Render/Cloud, 1 comando) e locale (uvicorn). Auto-train all'avvio, ingest multi-sorgente via `AUTOPILOT_DATA_PATHS`, manifest+canary, validazione dati, Docker hardening (non-root, read-only).

## Avvio più semplice (consigliato - Cloud)
1) Carica su GitHub (dal telefono).
2) Collega a Render.com → New Web Service → seleziona il repo → Deploy (Dockerfile incluso).
3) Test: `https://<tuo-servizio>/health` e `POST /predict`.

## Avvio locale veloce
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && uvicorn app.serve:app --host 0.0.0.0 --port 8000
```

## Endpoint
`/health`, `/ready`, `/metrics`, `/predict`, `/predict_batch`, `/retrain`.
