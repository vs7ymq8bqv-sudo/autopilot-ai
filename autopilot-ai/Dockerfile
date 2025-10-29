FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/metrics

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN addgroup --system app && adduser --system --ingroup app --uid 10001 app

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN mkdir -p /app/models && chown -R app:app /app
USER app:app

EXPOSE 8000
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.serve:app"]
