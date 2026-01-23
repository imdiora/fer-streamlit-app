FROM python:3.11-slim

# OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install as package (includes dependencies)
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY config ./config
COPY docs ./docs
COPY scripts ./scripts
COPY models ./models

# Install
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir .

# Expose ports: 8000 API, 8501 UI
EXPOSE 8000 8501

# Default: run API
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
