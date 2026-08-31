# Bandhu: Production Google Cloud Run Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HOST=0.0.0.0 \
    ENVIRONMENT=production

# Install system audio dependencies (ffmpeg, libsndfile, git, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data assets
COPY src/ ./src/
COPY data/ ./data/
COPY pyproject.toml .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Expose standard Cloud Run port
EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Start FastAPI server
CMD ["uvicorn", "bandhu.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
