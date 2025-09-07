# Pear Care Unified Container Deployment Guide

This guide covers deploying the unified Pear Care container to RunPod serverless infrastructure.

## Pre-deployment Checklist

### 1. Environment Setup
- [ ] OpenAI API key configured
- [ ] Vercel API secret configured  
- [ ] JWT secret key generated
- [ ] Container tested locally

### 2. Container Requirements
- [ ] NVIDIA RTX A5000 or better GPU
- [ ] 32GB RAM minimum
- [ ] 100GB storage for models
- [ ] CUDA 11.8+ support

## RunPod Deployment

### 1. Build and Push Container

```bash
# Build the container
docker build -t your-registry/pear-care-unified:latest .

# Push to your container registry
docker push your-registry/pear-care-unified:latest
```

### 2. RunPod Template Configuration

Create a new RunPod template with the following configuration:

```json
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
        "JWT_SECRET_KEY": "${JWT_SECRET_KEY}",
        "LOG_LEVEL": "INFO",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "medgemma:27b",
        "MAX_CONCURRENT_REQUESTS": "10",
        "RATE_LIMIT_PER_HOUR": "1000"
    },
    "containerDiskInGb": 100,
    "volumeInGb": 50,
    "volumeMountPath": "/workspace"
}
```

### 3. Serverless Endpoint Setup

1. **Create Endpoint:**
   - Go to RunPod Serverless
   - Click "New Endpoint"
   - Select your template
   - Set minimum/maximum workers
   - Configure idle timeout

2. **Endpoint Configuration:**
   ```json
   {
     "name": "pear-care-unified-endpoint",
     "template_id": "your-template-id",
     "min_workers": 0,
     "max_workers": 5,
     "idle_timeout": 300,
     "scale_job_queue": true,
     "flashboot": true
   }
   ```

### 4. Health Check Setup

Configure health checks for the endpoint:

```bash
# Health check URL
GET https://your-endpoint-id-direct.runpod.net/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "services": {
    "openai": "healthy",
    "ollama": "healthy",
    "langchain": "healthy"
  }
}
```

## Vercel Dashboard Integration

### 1. Update API Endpoints

Update your Vercel dashboard to point to the new unified container:

```typescript
// pages/api/chat/completions.ts
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // Validate API key (existing logic)
  const apiKey = await validateApiKey(req);
  
  // Route to unified container
  const containerUrl = process.env.RUNPOD_UNIFIED_ENDPOINT_URL;
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

### 2. Environment Variables

Add to your Vercel environment:

```bash
RUNPOD_UNIFIED_ENDPOINT_URL=https://your-endpoint-id-direct.runpod.net
RUNPOD_API_KEY=your-runpod-api-key
```

## Monitoring Setup

### 1. Health Monitoring

Set up monitoring for the container health:

```bash
# Monitoring script
#!/bin/bash
ENDPOINT_URL="https://your-endpoint-id-direct.runpod.net"

while true; do
    HEALTH=$(curl -s "${ENDPOINT_URL}/health" | jq -r '.status')
    
    if [ "$HEALTH" != "healthy" ]; then
        echo "$(date): Container unhealthy - $HEALTH"
        # Send alert
    else
        echo "$(date): Container healthy"
    fi
    
    sleep 30
done
```

### 2. Performance Metrics

Monitor key metrics:
- Response times per endpoint
- Token generation rates  
- Error rates
- GPU utilization
- Memory usage
- Request queue depth

### 3. Logging Setup

Configure structured logging:

```python
import structlog
import logging

# In production, send logs to external service
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = structlog.get_logger()
```

## Testing Deployment

### 1. Basic Functionality Test

```bash
# Test health
curl https://your-endpoint-id-direct.runpod.net/health

# Test chat completion
curl -X POST https://your-endpoint-id-direct.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "phraser",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Test Langchain consultation
curl -X POST https://your-endpoint-id-direct.runpod.net/v1/langchain/consultation \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "symptoms": "headache and nausea",
    "patient_data": {"age": 30, "gender": "female"}
  }'
```

### 2. Load Testing

```bash
# Install artillery for load testing
npm install -g artillery

# Create load test config
cat > load-test.yml << EOF
config:
  target: 'https://your-endpoint-id-direct.runpod.net'
  phases:
    - duration: 60
      arrivalRate: 5
  defaults:
    headers:
      Authorization: 'Bearer your-api-key'
      Content-Type: 'application/json'

scenarios:
  - name: 'Chat completion test'
    weight: 100
    flow:
      - post:
          url: '/v1/chat/completions'
          json:
            model: 'phraser'
            messages:
              - role: 'user'
                content: 'Test message'
            max_tokens: 50
EOF

# Run load test
artillery run load-test.yml
```

## Scaling Configuration

### 1. Auto-scaling Settings

Configure auto-scaling based on:
- Request queue depth
- Response time thresholds
- GPU utilization
- Memory usage

```json
{
  "scaling": {
    "min_workers": 1,
    "max_workers": 10,
    "scale_up_threshold": 5,
    "scale_down_threshold": 1,
    "scale_up_delay": 30,
    "scale_down_delay": 300
  }
}
```

### 2. Resource Optimization

- **Cold Start Optimization**: Use flashboot for faster startup
- **Model Caching**: Pre-load models in container image
- **Memory Management**: Monitor and optimize memory usage
- **Connection Pooling**: Configure optimal connection pool sizes

## Backup and Recovery

### 1. Container Image Backup

```bash
# Tag and backup container images
docker tag pear-care-unified:latest pear-care-unified:backup-$(date +%Y%m%d)
docker push your-registry/pear-care-unified:backup-$(date +%Y%m%d)
```

### 2. Configuration Backup

- RunPod template configurations
- Environment variables
- Vercel integration settings
- Monitoring configurations

## Security Considerations

### 1. Network Security
- Use HTTPS for all communications
- Implement proper CORS policies
- Validate all input data

### 2. Authentication Security
- Rotate API keys regularly
- Use strong JWT secrets
- Implement rate limiting

### 3. Container Security
- Use minimal base images
- Scan for vulnerabilities
- Keep dependencies updated

## Troubleshooting

### Common Issues

1. **Container fails to start:**
   ```bash
   # Check logs
   runpod logs your-endpoint-id
   
   # Check resource allocation
   # Ensure GPU/memory requirements met
   ```

2. **Ollama model loading fails:**
   ```bash
   # Check if model exists
   docker exec container ollama list
   
   # Re-pull model if needed
   docker exec container ollama pull medgemma:27b
   ```

3. **High response times:**
   ```bash
   # Check GPU utilization
   nvidia-smi
   
   # Monitor memory usage
   docker stats container-id
   ```

4. **Authentication errors:**
   ```bash
   # Verify environment variables
   echo $OPENAI_API_KEY
   echo $VERCEL_API_SECRET
   
   # Test API key validation
   curl -H "Authorization: Bearer test-key" /health
   ```

## Rollback Plan

In case of issues:

1. **Immediate rollback to previous container version**
2. **Route traffic back to separate services temporarily**
3. **Debug issues in staging environment**
4. **Gradual re-deployment with fixes**

## Support Contacts

- **Infrastructure**: DevOps team
- **Application**: Development team  
- **RunPod**: Support ticket system
- **Monitoring**: Monitoring team

## Documentation Links

- [RunPod Serverless Documentation](https://docs.runpod.io/serverless/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Ollama Documentation](https://ollama.ai/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
