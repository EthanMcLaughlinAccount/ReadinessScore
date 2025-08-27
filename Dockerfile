# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Bring in the rest of the code (tolerates missing api/ or model/ dirs)
COPY . /app

# Ensure model dir exists even if no artifact is committed yet
RUN mkdir -p /app/model

# Optional: healthcheck endpoint (FastAPI route /healthz)
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-8000}/healthz || exit 1

# APP_MODULE can be overridden; we auto-detect api.main:app vs main:app
ENV APP_MODULE="api.main:app"

# Start command:
# - if /app/api/main.py exists, use api.main:app
# - else if /app/main.py exists, use main:app
# - bind to PORT if provided by platform, else 8000
CMD bash -lc '\
  MOD="${APP_MODULE}"; \
  if [ ! -f "/app/api/main.py" ] && [ -f "/app/main.py" ]; then MOD="main:app"; fi; \
  uvicorn "$MOD" --host 0.0.0.0 --port "${PORT:-8000}" --workers 2 \
'
