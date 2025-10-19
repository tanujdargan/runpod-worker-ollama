# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pear Care Unified Serverless Container** - A medical consultation API that consolidates Langchain multi-agent orchestration, OpenAI integration, and MedGemma-27B hosting into a single RunPod serverless instance.

This system implements a **5-agent medical pipeline**:
1. **Symptom Agent** (GPT-5-nano) → Extracts symptoms from patient input
2. **ICD Agent** (MedGemma-27B) → Generates ICD-10 diagnostic codes
3. **CPT Agent** (MedGemma-27B) → Generates CPT procedure codes
4. **Doctor Agent** (GPT-4) → Matches providers and generates explanations
5. **Summary Agent** → Aggregates consultation results

## Development Commands

### Local Development (Standalone Mode)
```bash
# Start both Ollama and FastAPI services
cd src
bash start.sh

# Access points:
# - FastAPI: http://localhost:8000
# - Ollama: http://localhost:11434
# - Health check: http://localhost:8000/health
```

### RunPod Serverless Mode
```bash
# Build Docker image
docker build -t runpod-worker-ollama --build-arg OLLAMA_VERSION=0.12.6 .

# Test locally
docker run -e RUNPOD_SERVERLESS=1 -e OPENAI_API_KEY=<key> runpod-worker-ollama

# The handler automatically detects serverless mode via RUNPOD_SERVERLESS env var
```

### Testing
```bash
# Test the unified API locally
cd src
python test_unified_api.py

# Test with example inputs
# See test_inputs/ directory for various request formats
```

## Architecture

### Dual Execution Modes

**RunPod Serverless Mode** (`RUNPOD_SERVERLESS=1`):
- Entry: `src/handler.py` → RunPod's handler function
- Wraps FastAPI app for serverless deployment
- Uses `run_async_in_handler()` to manage async execution in RunPod's event loop
- No streaming (RunPod limitation) - returns accumulated chunks

**Standalone Container Mode** (default):
- Entry: `src/start.sh` → Starts Ollama + FastAPI
- Full streaming support via SSE
- Runs on port 8000 with uvicorn workers

### Request Flow

```
Client Request
    ↓
RunPod Handler (src/handler.py) OR FastAPI Router (src/app/routers/)
    ↓
Model Router (src/app/services/model_router.py)
    ↓
    ├─→ OpenAI Client (phraser, main, gpt-5-nano, gpt-4)
    ├─→ Ollama Client (medgemma:27b)
    └─→ Langchain Orchestrator (5-agent pipeline)
```

### Key Components

**src/handler.py** - RunPod serverless wrapper
- Supports multiple input formats (prompt, messages, endpoint, symptoms)
- Pre-warms services on cold start
- Handles async execution in RunPod's threading model

**src/app/main.py** - FastAPI application
- Lifespan management (startup warmup, cleanup)
- Global service initialization (auth, streaming, model_router, langchain_orchestrator)
- Router registration and dependency injection

**src/app/services/model_router.py** - Unified model interface
- Routes model requests to appropriate client (OpenAI or Ollama)
- Model mapping:
  - `phraser`, `main`, `gpt-5-nano`, `gpt-4` → OpenAI
  - `medgemma:27b`, `langchain` → Ollama
- Provides both streaming and non-streaming interfaces

**src/app/langchain/orchestrator.py** - Main consultation workflow
- Manages 5-agent pipeline with progressive warmup
- Streams results from each agent
- Supports Q&A continuation with `continue_consultation_with_answer()`
- Session management for multi-turn consultations

**src/app/services/auth.py** - Dual-layer authentication
- JWT validation (preferred)
- API key signature validation (format: `{user_id}.{timestamp}.{signature}`)
- Development mode when no secrets configured

### Agents Architecture

Each agent in `src/app/langchain/agents/`:
- **SymptomAgent** - Uses GPT-5-nano for fast symptom extraction
- **ICDAgent** - Uses MedGemma-27B for ICD-10 coding
- **CPTAgent** - Uses MedGemma-27B for CPT procedure codes
- **DoctorAgent** - Uses GPT-4 for provider matching

All agents implement:
- `warmup()` - Pre-warm model for faster first request
- `process()` - Main processing logic
- `stream_process()` - Streaming variant (where applicable)

### Progressive Warmup Strategy

The orchestrator implements **parallel warmup** to reduce latency:
1. While processing symptom extraction, warm up ICD agent
2. While processing ICD codes, warm up CPT agent
3. While processing CPT codes, warm up doctor agent

This reduces total pipeline time by ~40% compared to sequential execution.

## Environment Variables

Required:
- `OPENAI_API_KEY` - OpenAI API key for GPT models

Optional:
- `OLLAMA_MODEL_NAME` - Model to preload from Ollama (e.g., `phi3`, `llama3.2:1b`)
- `OLLAMA_FLASH_ATTENTION` - Enable flash attention for faster inference (set to `1`, default: `1`)
- `OLLAMA_NUM_PARALLEL` - Maximum number of parallel Ollama requests (default: 4 or 1)
- `MAX_CONCURRENCY` - Maximum concurrent requests in RunPod serverless (default: `8`)
- `VERCEL_API_SECRET` - Shared secret for API key validation
- `JWT_SECRET_KEY` - JWT signing secret for token validation
- `OLLAMA_MODEL` - Local model name for Pear Care pipeline (default: `medgemma:27b`)
- `LOG_LEVEL` - Logging verbosity (default: `info`)
- `RUNPOD_SERVERLESS` - Set to `1` for RunPod serverless mode
- `DISABLE_AUTH` - Set to disable authentication (development only)

## Important Implementation Notes

### Async Execution in RunPod Handler

The `run_async_in_handler()` function in `src/handler.py` handles async execution in RunPod's environment:
- Detects if event loop is already running
- Creates new thread with separate event loop if needed
- Essential for running FastAPI's async functions in RunPod's sync handler

### Model Name Mapping

The `ModelRouter` maps friendly model names to actual implementations:
- `phraser` → OpenAI API (custom model)
- `main` → OpenAI API (custom model)
- `langchain` → MedGemma-27B via Ollama
- `medgemma` → Normalized to `medgemma:27b`

### Streaming Behavior

**Standalone mode**: True SSE streaming with `StreamingResponse`
**RunPod serverless**: Simulated streaming - accumulates chunks and returns array

### Session Management

The Langchain orchestrator maintains sessions in memory (`self.sessions`). In production, this should be moved to Redis or a database for persistence across container restarts.

## Testing Strategy

Test inputs are in `test_inputs/`:
- `chat_completion.json` - Basic chat request
- `langchain_consultation.json` - Full medical consultation
- `streaming_chat.json` - Streaming chat test
- `health_check.json` - Health endpoint test

Each test file uses the format:
```json
{
  "input": {
    "endpoint": "chat|langchain|health|models",
    "method": "GET|POST",
    "data": { ... }
  }
}
```

## Common Gotchas

1. **Environment variable naming**: Use `OLLAMA_MODEL_NAME` (not `MODEL_NAME`) - RunPod blocks deployments with `MODEL_NAME` env var

2. **Ollama models location**: Set to `/runpod-volume` for RunPod compatibility - this allows model persistence across runs

3. **Flash attention**: Enabled by default via `OLLAMA_FLASH_ATTENTION=1` for faster inference on supported GPUs

4. **Concurrency control**: Set `MAX_CONCURRENCY` to control how many concurrent requests RunPod will send (default: 8, borrowed from vLLM logic)

5. **Authentication in development**: If `VERCEL_API_SECRET` and `JWT_SECRET_KEY` are not set, auth is disabled (dev mode)

6. **Warmup timing**: The orchestrator's progressive warmup strategy requires careful ordering - don't modify agent warmup calls without understanding the pipeline

7. **Docker base image**: Uses `ollama/ollama` which includes CUDA support - don't change without ensuring GPU compatibility

8. **Python version**: Locked to Python 3.11 - newer versions may have dependency conflicts

## File Structure

```
src/
├── handler.py              # RunPod serverless entry point
├── start.sh               # Standalone mode startup script
├── requirements.txt       # Python dependencies
├── app/
│   ├── main.py           # FastAPI application
│   ├── clients/          # OpenAI and Ollama client wrappers
│   ├── langchain/        # Multi-agent orchestration
│   │   ├── orchestrator.py
│   │   └── agents/       # Individual agents
│   ├── routers/          # FastAPI route handlers
│   ├── services/         # Auth, routing, streaming, rate limiting
│   └── models/           # Pydantic request/response models
└── test_inputs/          # Example API requests
```

## Deployment

See `DEPLOYMENT.md` for detailed deployment instructions.

For RunPod deployment:
1. Build and push Docker image
2. Create RunPod serverless endpoint
3. Set environment variables (especially `OPENAI_API_KEY`)
4. Test with health check: `{"input": {"endpoint": "health"}}`
