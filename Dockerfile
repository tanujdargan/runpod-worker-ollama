ARG OLLAMA_VERSION=0.11.10

# Use the official Ollama image which includes CUDA support
FROM ollama/ollama:${OLLAMA_VERSION}

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV RUNPOD_SERVERLESS=1

# System dependencies - install Python and required packages
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    gpg-agent \
    build-essential \
    apt-utils \
    && apt-get install --reinstall ca-certificates \
    && add-apt-repository --yes ppa:deadsnakes/ppa && apt update --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-lib2to3 \
    python3.11-gdbm \
    python3.11-tk \
    bash \
    curl \
    wget \
    git \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /work

# Copy application code
COPY src/ /work/

# Set default ollama models directory to /runpod-volume for RunPod compatibility
ENV OLLAMA_MODELS="/runpod-volume"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt && \
    chmod +x /work/start.sh

# Create necessary directories
RUN mkdir -p /work/logs && \
    mkdir -p /work/data

# Expose ports
EXPOSE 8000 11434

# Health check for standalone mode
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for RunPod serverless
CMD ["python", "handler.py"]