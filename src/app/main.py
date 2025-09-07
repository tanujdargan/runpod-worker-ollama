"""
FastAPI application entry point for Pear Care Unified Container
Consolidates Langchain multi-agent orchestration, OpenAI integration, and MedGemma-27B hosting
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .services.auth import AuthService, validate_api_key
from .services.streaming import StreamingManager
from .services.model_router import ModelRouter
from .services.rate_limiter import RateLimiter
from .clients.openai_client import OpenAIClient
from .clients.ollama_client import OllamaClient
from .langchain.orchestrator import LangchainOrchestrator
from .models.requests import ChatRequest, ConsultationRequest
from .models.responses import ChatResponse, ConsultationResponse

# Load environment variables
load_dotenv()

# Global services
auth_service: AuthService = None
streaming_manager: StreamingManager = None
model_router: ModelRouter = None
rate_limiter: RateLimiter = None
langchain_orchestrator: LangchainOrchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    global auth_service, streaming_manager, model_router, rate_limiter, langchain_orchestrator
    
    # Initialize services
    auth_service = AuthService()
    streaming_manager = StreamingManager()
    rate_limiter = RateLimiter()
    
    # Initialize clients
    openai_client = OpenAIClient()
    ollama_client = OllamaClient()
    
    # Initialize model router
    model_router = ModelRouter(openai_client, ollama_client)
    
    # Initialize Langchain orchestrator
    langchain_orchestrator = LangchainOrchestrator(openai_client, ollama_client)
    
    # Startup warmup
    await startup_warmup()
    
    # Set up router dependencies after services are initialized
    set_chat_dependencies(model_router, streaming_manager)
    set_langchain_dependencies(langchain_orchestrator, streaming_manager)
    
    yield
    
    # Cleanup
    await cleanup_services()

async def startup_warmup():
    """Minimal warmup for server startup"""
    print("🔄 Server startup warmup...")
    try:
        # Test OpenAI connection
        await model_router.openai_client.warmup()
        
        # Test Ollama connection
        await model_router.ollama_client.warmup()
        
        # Warmup Langchain agents
        await langchain_orchestrator.warmup()
        
        print("✅ Startup warmup complete")
    except Exception as e:
        print(f"⚠️  Startup warmup failed: {e}")

async def cleanup_services():
    """Cleanup all services"""
    try:
        if model_router:
            await model_router.cleanup()
        if langchain_orchestrator:
            await langchain_orchestrator.cleanup()
    except Exception as e:
        print(f"⚠️  Cleanup failed: {e}")

# Create FastAPI app
app = FastAPI(
    title="Pear Care Unified API",
    description="Unified serverless container for medical consultation API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for authentication
async def get_auth_result(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
):
    """Extract and validate API key from headers"""
    api_key = None
    
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
    elif x_api_key:
        api_key = x_api_key
    
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    auth_result = await auth_service.validate_request(api_key, "chat")
    if not auth_result.authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Check rate limits
    if not await rate_limiter.check_rate_limit(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return auth_result

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    try:
        # Check all services
        openai_healthy = await model_router.openai_client.health_check()
        ollama_healthy = await model_router.ollama_client.health_check()
        langchain_healthy = await langchain_orchestrator.health_check()
        
        return {
            "status": "healthy" if all([openai_healthy, ollama_healthy, langchain_healthy]) else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "openai": "healthy" if openai_healthy else "unhealthy",
                "ollama": "healthy" if ollama_healthy else "unhealthy", 
                "langchain": "healthy" if langchain_healthy else "unhealthy"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

# Get available models
@app.get("/models")
async def get_models(auth_result = Depends(get_auth_result)):
    """Get available models for the authenticated user"""
    return {
        "data": [
            {"id": "phraser", "object": "model", "created": int(time.time())},
            {"id": "main", "object": "model", "created": int(time.time())},
            {"id": "langchain", "object": "model", "created": int(time.time())},
            {"id": "medgemma:27b", "object": "model", "created": int(time.time())}
        ]
    }

# Manual warmup endpoint
@app.get("/warmup")
async def manual_warmup():
    """Manual warmup endpoint for load balancers"""
    await startup_warmup()
    return {"status": "warmed up", "timestamp": datetime.utcnow().isoformat()}

# Import routers
from .routers.chat import router as chat_router
from .routers.langchain import router as langchain_router

# Set up router dependencies
from .routers.chat import set_dependencies as set_chat_dependencies
from .routers.langchain import set_dependencies as set_langchain_dependencies

# Include routers
app.include_router(chat_router, prefix="/v1")
app.include_router(langchain_router, prefix="/v1")

# Dependencies are now set up in the lifespan function above

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
