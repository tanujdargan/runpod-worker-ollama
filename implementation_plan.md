# Pear Care API - Unified Serverless Architecture Implementation Plan

## Executive Summary

This document outlines the complete architecture and implementation plan for consolidating the Pear Care API into a unified serverless solution. The goal is to combine Langchain multi-agent orchestration, OpenAI integration, and MedGemma-27B hosting into a single RunPod serverless container while maintaining a Vercel-hosted dashboard for authentication and management.

## Current Architecture Analysis

### Existing Components
1. **Langchain Multi-Agent System** (Python-based)
   - 5-stage pipeline: Intake (GPT-5) → ICD Coding (MedGemma) → CPT Coding (MedGemma) → Provider Matching → Summary (GPT-5)
   - Structured JSON output with medical coding
   - Agent orchestration via LangChain Expression Language (LCEL)

2. **Vercel Dashboard & API Gateway**
   - Auth0 authentication for admin access
   - API key generation and management (Supabase-backed)
   - Request routing and analytics
   - TypeScript/Next.js frontend

3. **Current Model Integration**
   - OpenAI GPT-5 via direct API calls
   - MedGemma-27B via RunPod serverless (Ollama-based)
   - Limited streaming capabilities on RunPod

4. **Authentication & Authorization**
   - Two-tier system: Admin dashboard + API key validation
   - Rate limiting and usage tracking
   - Model access control per API key

### Current Limitations
- **Streaming Issues**: RunPod serverless doesn't support proper token streaming
- **Architecture Complexity**: Multiple deployment targets (Vercel + RunPod)
- **Cost Overhead**: External API calls for model routing
- **Deployment Complexity**: Separate codebases for different components

## Proposed Unified Architecture

### High-Level Architecture

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

### Core Design Principles

1. **Single Container Deployment**: Everything runs in one RunPod serverless instance
2. **Application-Level Streaming**: Bypass RunPod limitations with custom streaming
3. **Dual Authentication**: Vercel dashboard + container-level validation
4. **Model Agnostic Routing**: Unified interface for all model types
5. **Scalable Architecture**: Easy to add new models or agents

## Technical Implementation Plan

### Phase 1: Core Container Development

#### 1.1 FastAPI Application Structure
```
unified-container/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── routers/
│   │   ├── chat.py            # Chat completion endpoints
│   │   ├── langchain.py       # Langchain consultation endpoints
│   │   └── health.py          # Health check endpoints
│   ├── services/
│   │   ├── auth.py            # Authentication service
│   │   ├── streaming.py       # Streaming management
│   │   ├── model_router.py    # Model routing logic
│   │   └── rate_limiter.py    # Rate limiting service
│   ├── clients/
│   │   ├── openai_client.py   # OpenAI API client
│   │   └── ollama_client.py   # Local Ollama client
│   ├── langchain/
│   │   ├── orchestrator.py    # Agent orchestrator
│   │   ├── agents/            # Individual agent implementations
│   │   └── prompts/           # System prompts
│   └── models/
│       ├── requests.py        # Request/response models
│       └── responses.py       # Streaming response models
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

#### 1.2 Key FastAPI Endpoints
```python
# Primary endpoints to implement
POST /v1/chat/completions        # OpenAI-compatible chat endpoint
POST /v1/langchain/consultation  # Multi-agent consultation
POST /v1/langchain/stream        # Streaming consultation
GET  /health                     # Health check
GET  /models                     # Available models
```

#### 1.3 Streaming Implementation Strategy
```python
# Application-level streaming for all endpoints
async def stream_chat_completion(request: ChatRequest):
    """Stream tokens from either OpenAI or local Ollama"""
    async for token in model_client.stream(request):
        yield f"data: {json.dumps(token)}\n\n"

async def stream_langchain_consultation(request: ConsultationRequest):
    """Stream results as each agent completes"""
    async for stage_result in langchain_pipeline.stream(request):
        yield f"data: {json.dumps(stage_result)}\n\n"
```

### Phase 2: Model Integration

#### 2.1 OpenAI Integration
- Direct API calls for GPT-5 models
- Streaming support via OpenAI's native streaming
- Error handling and retry logic
- Token usage tracking

#### 2.2 Local Ollama Integration
```python
# Local Ollama client for MedGemma-27B
class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    async def stream_completion(self, prompt: str, model: str = "medgemma:27b"):
        """Stream tokens from local Ollama instance"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": True}
            ) as response:
                async for line in response.content:
                    yield json.loads(line)["response"]
```

#### 2.3 Langchain Agent Migration
- Port existing agent orchestrator to container
- Maintain 5-stage pipeline structure
- Add streaming capabilities between agents
- Preserve medical coding functionality

### Phase 3: Authentication & Security

#### 3.1 Two-Layer Authentication
```python
# Layer 1: Vercel Dashboard (existing)
# - API key generation
# - Admin authentication
# - Usage analytics

# Layer 2: Container Authentication
async def validate_container_request(api_key: str, endpoint: str):
    """Validate requests at container level"""
    # Verify API key signature from Vercel
    # Check rate limits locally
    # Validate model access permissions
    return AuthResult(authorized=True, user_id="...", models=[...])
```

#### 3.2 Rate Limiting & Usage Tracking
```python
# Container-level rate limiting
class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
    
    async def check_rate_limit(self, api_key: str, limit: int = 100) -> bool:
        """Check if request is within rate limits"""
        now = time.time()
        user_requests = self.requests[api_key]
        # Remove old requests
        user_requests[:] = [req for req in user_requests if now - req < 3600]
        return len(user_requests) < limit
```

### Phase 4: Deployment Strategy

#### 4.1 Container Configuration
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Python dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Application code
COPY app/ /app/app/

# Download models
RUN ollama serve & \
    sleep 10 && \
    ollama pull medgemma:27b && \
    pkill ollama

EXPOSE 8000 11434

# Startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
```

#### 4.2 Startup Script
```bash
#!/bin/bash
# start.sh - Start both Ollama and FastAPI

# Start Ollama in background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
sleep 5

# Start FastAPI
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for any process to exit
wait -n

# Kill remaining processes
kill $OLLAMA_PID $FASTAPI_PID 2>/dev/null
```

#### 4.3 RunPod Deployment
```python
# RunPod template configuration
{
    "name": "pear-care-unified",
    "image": "your-registry/pear-care-unified:latest",
    "gpu": "NVIDIA RTX A5000",
    "memory": "32GB",
    "disk": "100GB",
    "ports": [8000],
    "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "VERCEL_API_SECRET": "${VERCEL_API_SECRET}",
        "LOG_LEVEL": "INFO"
    }
}
```

### Phase 5: Vercel Dashboard Integration

#### 5.1 Dashboard Modifications
- Update API endpoints to point to unified container
- Maintain existing authentication flow
- Add new model routing options
- Keep analytics and monitoring features

#### 5.2 Request Flow
```typescript
// Updated Vercel API routing
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // Validate API key (existing logic)
  const apiKey = await validateApiKey(req);
  
  // Route to unified container
  const containerUrl = process.env.RUNPOD_CONTAINER_URL;
  const response = await fetch(`${containerUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(req.body)
  });
  
  // Stream response back to client
  return streamResponse(response, res);
}
```

## Implementation Timeline

### Week 1-2: Container Foundation
- [ ] Set up FastAPI application structure
- [ ] Implement basic authentication and routing
- [ ] Create Dockerfile and test local deployment
- [ ] Port Langchain orchestrator to container

### Week 3-4: Model Integration
- [ ] Integrate OpenAI client with streaming
- [ ] Set up local Ollama with MedGemma-27B
- [ ] Implement unified model routing
- [ ] Add comprehensive error handling

### Week 5-6: Streaming & Testing
- [ ] Implement application-level streaming
- [ ] Add agent-by-agent streaming for Langchain
- [ ] Comprehensive testing of all endpoints
- [ ] Performance optimization

### Week 7-8: Deployment & Integration
- [ ] Deploy to RunPod serverless
- [ ] Update Vercel dashboard integration
- [ ] End-to-end testing
- [ ] Documentation and monitoring

## Technical Requirements

### Container Specifications
- **GPU**: NVIDIA RTX A5000 or better (for MedGemma-27B)
- **Memory**: 32GB RAM minimum
- **Storage**: 100GB for models and dependencies
- **CPU**: 8+ cores for FastAPI concurrency

### Dependencies
```txt
# Core framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# AI/ML libraries
langchain==0.1.0
langchain-core==0.1.0
langchain-openai==0.0.2
openai==1.3.0

# HTTP and streaming
aiohttp==3.9.0
sse-starlette==1.6.5
websockets==12.0

# Authentication and security
python-jose[cryptography]==3.3.0
python-multipart==0.0.6
bcrypt==4.1.2

# Utilities
python-dotenv==1.0.0
structlog==23.2.0
tenacity==8.2.3
```

### Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_key

# Authentication
VERCEL_API_SECRET=shared_secret_for_vercel_auth
JWT_SECRET_KEY=your_jwt_secret

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=medgemma:27b

# Application Configuration
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=10
RATE_LIMIT_PER_HOUR=1000

# Model Configuration
DEFAULT_MAX_TOKENS=4096
DEFAULT_TEMPERATURE=0.7
```

## API Documentation

### Chat Completions Endpoint
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

### Langchain Consultation Endpoint
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

### Streaming Response Format
```javascript
// Standard chat streaming
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"phraser","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

// Langchain agent streaming
data: {"stage":"intake","status":"completed","result":{"patient_intake":{"symptoms":["headache"],"severity":"severe"}}}
data: {"stage":"icd","status":"in_progress","message":"Analyzing symptoms for ICD codes..."}
data: {"stage":"icd","status":"completed","result":{"icd":{"codes":["G43.1"],"reasoning":"Migraine symptoms"}}}

// Final completion
data: [DONE]
```

## Success Metrics

### Performance Targets
- **Response Time**: < 2s for first token
- **Throughput**: 100+ concurrent requests
- **Uptime**: 99.9% availability
- **Streaming Latency**: < 100ms between tokens

### Functional Requirements
- [ ] Full Langchain pipeline functionality preserved
- [ ] Token-level streaming for all models
- [ ] OpenAI API compatibility maintained
- [ ] Authentication and rate limiting working
- [ ] Comprehensive error handling and logging

## Risk Mitigation

### Technical Risks
1. **GPU Memory Limits**: Monitor MedGemma-27B memory usage
2. **Cold Start Times**: Implement model preloading
3. **Streaming Reliability**: Add reconnection logic
4. **Rate Limiting**: Implement distributed rate limiting

### Deployment Risks
1. **RunPod Limitations**: Test extensively on target hardware
2. **Model Loading**: Ensure reliable model downloads
3. **Network Issues**: Add retry mechanisms
4. **Authentication**: Validate security thoroughly

## Migration Strategy

### Phase 1: Parallel Deployment
- Deploy unified container alongside existing system
- Route small percentage of traffic to test
- Monitor performance and functionality

### Phase 2: Gradual Migration
- Increase traffic percentage gradually
- Compare results with existing system
- Address any performance issues

### Phase 3: Full Cutover
- Route all traffic to unified container
- Decommission old RunPod endpoints
- Update documentation and monitoring

## Monitoring & Observability

### Key Metrics
- Request/response times per model
- Token generation rates
- Error rates by endpoint
- Resource utilization (GPU, RAM, CPU)
- Authentication success/failure rates

### Logging Strategy
```python
# Structured logging throughout application
import structlog

logger = structlog.get_logger()

logger.info("request_started", 
    endpoint="/v1/chat/completions",
    model="phraser",
    user_id="user_123",
    request_id="req_456"
)
```

### Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "ollama": await check_ollama_health(),
            "openai": await check_openai_health(),
            "langchain": await check_langchain_health()
        }
    }
```

## Conclusion

This unified architecture consolidates all Pear Care API functionality into a single, efficient, and scalable serverless container. By implementing application-level streaming and maintaining the Vercel dashboard as an authentication layer, we achieve:

1. **Simplified Architecture**: Single deployment target
2. **True Streaming**: Application-level control over token streaming
3. **Cost Optimization**: Local model hosting reduces API costs
4. **Enhanced Performance**: Reduced latency and improved throughput
5. **Easier Maintenance**: Single codebase for all functionality

The implementation plan provides a clear roadmap for migration while maintaining backward compatibility and ensuring a smooth transition from the current architecture.