# Pear Care Unified Serverless Container

A unified serverless container that consolidates Langchain multi-agent orchestration, OpenAI integration, and MedGemma-27B hosting into a single RunPod serverless instance.

## Architecture Overview

```
Client Application
    ↓
Vercel Dashboard (Auth Layer 1)
    ↓ API Key Validation & Request Routing
RunPod Unified Container (Auth Layer 2)
    ↓
FastAPI Gateway
    ↓
Model Router
    ├── Langchain Orchestrator (5-Agent Pipeline)
    └── Direct Model Access (Chat Completions)
        ├── OpenAI Client (GPT-5)
        └── Local Ollama (MedGemma-27B)
    ↓
Streaming Manager (SSE/WebSocket)
    ↓
Token Streaming Output
```

## Features

- **Unified API**: Single endpoint for all model interactions
- **Application-Level Streaming**: Bypass RunPod limitations with custom SSE streaming
- **Dual Authentication**: Vercel dashboard + container-level validation
- **Model Agnostic Routing**: Unified interface for OpenAI and local models
- **5-Agent Medical Pipeline**: Symptom extraction → ICD coding → CPT coding → Provider matching → Summary
- **Real-time Streaming**: Token-level streaming for all endpoints
- **Rate Limiting**: Container-level rate limiting and usage tracking

## Quick Start

### Development with Docker Compose

1. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

2. **Build and run:**
   ```bash
   docker-compose up --build
   ```

3. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```

### Production Deployment

1. **Build Docker image:**
   ```bash
   docker build -t pear-care-unified .
   ```

2. **Deploy to RunPod:**
   ```bash
   # Use the RunPod template configuration from implementation_plan.md
   ```

## API Endpoints

### Chat Completions (OpenAI Compatible)

```http
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer your_api_key

{
  "model": "phraser|main|langchain",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": true,
  "max_tokens": 1024
}
```

### Langchain Medical Consultation

```http
POST /v1/langchain/consultation
Content-Type: application/json
Authorization: Bearer your_api_key

{
  "symptoms": "Severe headache for 3 days",
  "patient_data": {
    "age": 35,
    "gender": "Female",
    "location": "Boston, MA"
  },
  "stream": true
}
```

### Streaming Updates

```http
GET /v1/langchain/consultation/{session_id}/stream
Accept: text/event-stream
Authorization: Bearer your_api_key
```

## Environment Variables

### Required
- `OPENAI_API_KEY`: OpenAI API key
- `VERCEL_API_SECRET`: Shared secret for Vercel authentication
- `JWT_SECRET_KEY`: JWT signing secret

### Optional
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name (default: medgemma:27b)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `LOG_LEVEL`: Logging level (default: info)

## Model Routing

The container automatically routes requests to appropriate models:

- **OpenAI Models**: `phraser`, `main`, `gpt-5-nano`, `gpt-4`
- **Local Models**: `medgemma:27b`, `langchain`

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Available Models
```bash
curl http://localhost:8000/models
```

### Manual Warmup
```bash
curl http://localhost:8000/warmup
```

## Development

### Running Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama separately:**
   ```bash
   ollama serve
   ollama pull medgemma:27b
   ```

3. **Run the application:**
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Testing

```bash
# Test chat completion
curl -X POST http://localhost:8000/v1/chat/test

# Test Langchain consultation
curl -X POST http://localhost:8000/v1/langchain/test
```

## Container Specifications

### Requirements
- **GPU**: NVIDIA RTX A5000 or better (for MedGemma-27B)
- **Memory**: 32GB RAM minimum
- **Storage**: 100GB for models and dependencies
- **CPU**: 8+ cores for FastAPI concurrency

### Performance Targets
- **Response Time**: < 2s for first token
- **Throughput**: 100+ concurrent requests
- **Uptime**: 99.9% availability
- **Streaming Latency**: < 100ms between tokens

## Security

### Authentication Flow
1. **Layer 1**: Vercel dashboard validates user and generates API key
2. **Layer 2**: Container validates API key signature and checks rate limits
3. **Request Processing**: Authenticated requests are processed and routed

### Rate Limiting
- Container-level rate limiting with sliding window
- Configurable limits per user/API key
- Concurrent request limiting

## Troubleshooting

### Common Issues

1. **Ollama not responding:**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   ```

2. **Model not found:**
   ```bash
   # Pull the model manually
   ollama pull medgemma:27b
   ```

3. **Authentication errors:**
   ```bash
   # Check environment variables
   echo $OPENAI_API_KEY
   echo $VERCEL_API_SECRET
   ```

### Logs
```bash
# View container logs
docker logs pear-care-unified

# View specific service logs
docker exec pear-care-unified tail -f /app/logs/app.log
```

## Migration from Existing System

This container replaces the separate RunPod and Vercel deployments with a unified solution while maintaining backward compatibility with existing API endpoints.

### Migration Steps
1. Deploy unified container alongside existing system
2. Route small percentage of traffic to test
3. Gradually increase traffic percentage
4. Full cutover once validated
5. Decommission old endpoints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## License

Proprietary - Pear Care Medical Systems
