import multiprocessing, os
workers = int(os.getenv("WEB_CONCURRENCY", max(2, multiprocessing.cpu_count() // 2)))
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8000"
timeout = int(os.getenv("TIMEOUT", "60"))
graceful_timeout = 30
keepalive = 5
max_requests = 2000
max_requests_jitter = 200
loglevel = os.getenv("LOGLEVEL", "info")
accesslog = "-"
errorlog = "-"
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
