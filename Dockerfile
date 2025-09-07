FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

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

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/

# Create necessary directories
RUN mkdir -p /app/logs
RUN mkdir -p /app/data

# Download and preload MedGemma model
RUN ollama serve & \
    sleep 15 && \
    echo "Pulling MedGemma model..." && \
    ollama pull medgemma:27b && \
    echo "Model download complete" && \
    pkill ollama

# Expose ports
EXPOSE 8000 11434

# Create startup script
COPY src/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/app/start.sh"]