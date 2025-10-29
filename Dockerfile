# syntax=docker/dockerfile:1
FROM python:3.11-slim
WORKDIR /app
COPY . .
# Render espone la porta tramite la variabile $PORT
ENV PORT=10000
EXPOSE 10000
# Server HTTP minimale per far risultare vivo il servizio
CMD python -m http.server $PORT
