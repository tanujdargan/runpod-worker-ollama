ARG OLLAMA_VERSION=0.9.0

# Use an official base${OLLAMA_VERSION} image with your desired version
FROM ollama/ollama:${OLLAMA_VERSION}

ENV PYTHONUNBUFFERED=1

# Set up the working directory
WORKDIR /

RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    gpg-agent \
    build-essential apt-utils \
    && apt-get install --reinstall ca-certificates \
    && add-apt-repository --yes ppa:deadsnakes/ppa && apt update --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-lib2to3 \
    python3.11-gdbm \
    python3.11-tk \
    curl \
    pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /work

# Add my src as /work
ADD ./src /work

# Set defaut ollama models directory to /runpod-volume where runpod will mount the volume by default
ENV OLLAMA_MODELS="/runpod-volume"

# Install runpod and its dependencies
RUN pip install -r requirements.txt && \
    chmod +x start.sh

# Set the entrypoint
ENTRYPOINT ["/bin/sh", "-c", "/work/start.sh"]