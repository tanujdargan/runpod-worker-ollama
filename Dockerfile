FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV RUNPOD_SERVERLESS=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker caching)
COPY src/requirements.txt /app/requirements.txt

# Install Python dependencies including uvloop for better performance
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvloop

# Copy application code
COPY src/ /app/

# Create necessary directories
RUN mkdir -p /app/logs
RUN mkdir -p /app/data

# Start Ollama and download MedGemma model
RUN ollama serve & \
    sleep 15 && \
    echo "Pulling MedGemma model..." && \
    ollama pull medgemma:27b && \
    echo "Model download complete" && \
    pkill ollama

# Expose ports
EXPOSE 8000 11434

# Create startup script for serverless
COPY src/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Health check for standalone mode
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - will be overridden by RunPod
CMD ["python", "handler.py"]