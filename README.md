# Pear Care Unified Serverless Container

A unified serverless container for medical consultation API that consolidates Langchain multi-agent orchestration, OpenAI integration, and MedGemma-27B hosting into a single RunPod serverless instance.

[![RunPod](https://api.runpod.io/badge/tanujdargan/runpod-worker-ollama)](https://console.runpod.io/hub/tanujdargan/runpod-worker-ollama)

## Features

- **5-Agent Medical Pipeline**: Symptom extraction → ICD coding → CPT coding → Provider matching → Summary
- **Unified API**: Single endpoint for all model interactions (OpenAI + Local models)
- **Application-Level Streaming**: Bypass RunPod limitations with custom streaming
- **Dual Authentication**: Vercel dashboard + container-level validation
- **Model Agnostic Routing**: Unified interface for all model types

## Quick Start

### RunPod Serverless Deployment

1. **Create Endpoint**: Use this container on RunPod Serverless
2. **Set Environment Variables**: Configure your API keys (see below)
3. **Test Deployment**: Use the test inputs provided

### Environment Variables

| Variable Name | Description | Required | Default |
|---------------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes | - |
| `VERCEL_API_SECRET` | Shared secret for authentication | No | - |
| `JWT_SECRET_KEY` | JWT signing secret | No | - |
| `OLLAMA_MODEL` | Local model name | No | `medgemma:27b` |
| `LOG_LEVEL` | Logging level | No | `info` |

## Test Requests

See the [test_inputs](./test_inputs) directory for example test requests.

### Chat Completion Request
```json
{
  "input": {
    "endpoint": "chat",
    "method": "POST",
    "data": {
      "model": "phraser",
      "messages": [{"role": "user", "content": "Hello"}],
      "max_tokens": 100
    }
  }
}
```

### Langchain Consultation Request
```json
{
  "input": {
    "endpoint": "langchain",
    "method": "POST",
    "data": {
      "symptoms": "I have a severe headache and nausea",
      "patient_data": {"age": 30, "gender": "female"}
    }
  }
}
```

## Architecture

```
Client → RunPod Handler → Model Router → {OpenAI Client, Ollama Client, Langchain Orchestrator}
```

## Model Support

- **OpenAI Models**: `phraser`, `main`, `gpt-5-nano`
- **Local Models**: `medgemma:27b` (via Ollama)
- **Langchain**: Multi-agent medical consultation pipeline

## Licence

This project is licensed under the Creative Commons Attribution 4.0 International License. You are free to use, share, and adapt the material for any purpose, even commercially, under the following terms:

- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **Reference**: You must reference the original repository at [https://github.com/svenbrnn/runpod-worker-ollama](https://github.com/svenbrnn/runpod-worker-ollama).

For more details, see the [license](https://creativecommons.org/licenses/by/4.0/).