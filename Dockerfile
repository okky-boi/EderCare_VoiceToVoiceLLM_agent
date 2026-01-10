# Multi-stage build untuk mengoptimalkan ukuran image
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    build-essential \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download model - menggunakan Hugging Face model
RUN mkdir -p /app/models && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF', \
    filename='qwen2.5-1.5b-instruct-q4_k_m.gguf', \
    local_dir='/app/models')"

# Final stage
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages dari builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Copy model dari builder
COPY --from=builder /app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf /app/models/

# Expose port
EXPOSE 8051

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run server
CMD ["python", "-m", "src.main", "--mode", "server", "--host", "0.0.0.0", "--port", "8051"]
