# syntax=docker/dockerfile:1

FROM python:3.11-slim

# ---------------------------------------------------------
# Python settings
# ---------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ---------------------------------------------------------
# System dependencies needed by Pillow
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Install inference dependencies
# ---------------------------------------------------------
COPY requirements-inference.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-inference.txt

# ---------------------------------------------------------
# Copy application code
# ---------------------------------------------------------
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# ---------------------------------------------------------
# Create non-root user
# ---------------------------------------------------------
RUN useradd -m appuser \
    && chown -R appuser:appuser /app

USER appuser

# ---------------------------------------------------------
# API port
# ---------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ---------------------------------------------------------
# Start FastAPI
# ---------------------------------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
