#!/bin/bash
# Unified container startup script
# Starts both Ollama and FastAPI services

set -e

echo " Starting Pear Care Unified Container..."

# Function to cleanup on exit
cleanup() {
    echo "Shutting down services..."
    kill $OLLAMA_PID $FASTAPI_PID 2>/dev/null || true
    wait
    echo "Cleanup complete"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Start Ollama in background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo " Waiting for Ollama to start..."
sleep 10

# Test Ollama connection
echo "Testing Ollama connection..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "Ollama is ready"
        break
    fi
    echo "Waiting for Ollama... ($i/30)"
    sleep 2
done

# Verify MedGemma model is available
echo "Verifying MedGemma model..."
if curl -s http://localhost:11434/api/tags | grep -q "medgemma:27b"; then
    echo " MedGemma model is available"
else
    echo " MedGemma model not found, attempting to pull..."
    ollama pull medgemma:27b
fi

# Set environment variables for production
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export WORKERS=${UVICORN_WORKERS:-4}
export LOG_LEVEL=${LOG_LEVEL:-info}

# Start FastAPI application
echo " Starting FastAPI application on ${HOST}:${PORT}..."
python -m uvicorn app.main:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --log-level $LOG_LEVEL \
    --access-log \
    --timeout-keep-alive 5 \
    --timeout-graceful-shutdown 30 &

FASTAPI_PID=$!

echo " Services started:"
echo "   - Ollama: http://localhost:11434"
echo "   - FastAPI: http://${HOST}:${PORT}"
echo "   - Health check: http://${HOST}:${PORT}/health"

# Wait for either process to exit
wait -n

# If we reach here, one of the processes has exited
echo "  A service has exited, shutting down..."