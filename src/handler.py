"""
RunPod Serverless Handler for Pear Care Unified API
Wraps the FastAPI application for RunPod serverless deployment
"""

import runpod
import asyncio
import json
import time
import os
import sys
from typing import Dict, Any, Optional
import traceback
import threading
import uvloop

# Import our FastAPI app
from app.main import app
from app.services.auth import AuthService
from app.services.model_router import ModelRouter
from app.services.streaming import StreamingManager
from app.langchain.orchestrator import LangchainOrchestrator
from app.clients.openai_client import OpenAIClient
from app.clients.ollama_client import OllamaClient

# Global variables for services
services_initialized = False
model_router = None
langchain_orchestrator = None
auth_service = None

async def initialize_services():
    """Initialize all services once"""
    global services_initialized, model_router, langchain_orchestrator, auth_service
    
    if services_initialized:
        return
    
    try:
        print("🔄 Initializing Pear Care services...")
        
        # Initialize clients
        openai_client = OpenAIClient()
        ollama_client = OllamaClient()
        
        # Initialize services
        auth_service = AuthService()
        streaming_manager = StreamingManager()
        
        # Initialize model router
        model_router = ModelRouter(openai_client, ollama_client)
        
        # Initialize Langchain orchestrator
        langchain_orchestrator = LangchainOrchestrator(openai_client, ollama_client)
        
        # Warmup services
        await asyncio.gather(
            openai_client.warmup(),
            ollama_client.warmup(),
            langchain_orchestrator.warmup(),
            return_exceptions=True
        )
        
        services_initialized = True
        print("✅ Services initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        traceback.print_exc()
        raise

def handler(job):
    """
    Main RunPod handler function
    
    Expected job input format:
    {
        "input": {
            "endpoint": "chat|langchain|health",
            "method": "POST|GET",
            "data": {...},
            "headers": {...}
        }
    }
    """
    try:
        # Get job input
        job_input = job.get("input", {})
        
        if not job_input:
            return {"error": "No input provided"}
        
        # Extract request details
        endpoint = job_input.get("endpoint", "health")
        method = job_input.get("method", "GET")
        data = job_input.get("data", {})
        headers = job_input.get("headers", {})
        
        print(f"🚀 Processing {method} {endpoint} request")
        
        # Route to appropriate handler
        if endpoint == "health":
            return handle_health_check()
        elif endpoint == "chat":
            return asyncio.run(handle_chat_completion(data, headers))
        elif endpoint == "langchain":
            return asyncio.run(handle_langchain_consultation(data, headers))
        elif endpoint == "models":
            return handle_models_list()
        else:
            return {"error": f"Unknown endpoint: {endpoint}"}
    
    except Exception as e:
        print(f"❌ Handler error: {e}")
        traceback.print_exc()
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def handle_health_check():
    """Handle health check requests"""
    try:
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "handler": "ready",
                "initialized": services_initialized
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

async def handle_chat_completion(data: Dict, headers: Dict):
    """Handle chat completion requests"""
    try:
        # Initialize services if needed
        await initialize_services()
        
        # Validate authentication (simplified for RunPod)
        api_key = headers.get("authorization", "").replace("Bearer ", "")
        if not api_key and not os.getenv("DISABLE_AUTH"):
            return {"error": "Missing API key"}
        
        # Extract chat completion parameters
        model = data.get("model", "phraser")
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens")
        temperature = data.get("temperature", 0.7)
        stream = data.get("stream", False)
        
        if not messages:
            return {"error": "No messages provided"}
        
        print(f"🤖 Chat completion with model: {model}")
        
        if stream:
            # For streaming, we'll return a special response
            # RunPod doesn't support true streaming, so we simulate it
            response_chunks = []
            async for chunk_data in model_router.stream_chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                if chunk_data.startswith("data: "):
                    try:
                        chunk_json = json.loads(chunk_data[6:])
                        response_chunks.append(chunk_json)
                    except json.JSONDecodeError:
                        continue
                
                # Limit chunks for RunPod response size
                if len(response_chunks) >= 50:
                    break
            
            return {
                "stream": True,
                "chunks": response_chunks,
                "total_chunks": len(response_chunks)
            }
        else:
            # Non-streaming response
            response = await model_router.chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            return response
    
    except Exception as e:
        print(f"❌ Chat completion error: {e}")
        return {"error": str(e)}

async def handle_langchain_consultation(data: Dict, headers: Dict):
    """Handle Langchain consultation requests"""
    try:
        # Initialize services if needed
        await initialize_services()
        
        # Validate authentication (simplified for RunPod)
        api_key = headers.get("authorization", "").replace("Bearer ", "")
        if not api_key and not os.getenv("DISABLE_AUTH"):
            return {"error": "Missing API key"}
        
        # Extract consultation parameters
        symptoms = data.get("symptoms", "")
        patient_data = data.get("patient_data", {})
        session_id = data.get("session_id")
        
        if not symptoms:
            return {"error": "No symptoms provided"}
        
        print(f"🏥 Langchain consultation for symptoms: {symptoms[:50]}...")
        
        # Run consultation workflow
        consultation_results = []
        async for update in langchain_orchestrator.run_consultation(
            user_input=symptoms,
            patient_data=patient_data,
            session_id=session_id
        ):
            consultation_results.append(update)
            
            # Limit results for RunPod response size
            if len(consultation_results) >= 20:
                break
        
        return {
            "session_id": session_id,
            "consultation_updates": consultation_results,
            "total_updates": len(consultation_results)
        }
    
    except Exception as e:
        print(f"❌ Langchain consultation error: {e}")
        return {"error": str(e)}

def handle_models_list():
    """Handle models list request"""
    try:
        models = ["phraser", "main", "langchain", "medgemma:27b"]
        
        return {
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "pear-care"
                } for model in models
            ]
        }
    
    except Exception as e:
        return {"error": str(e)}

# Pre-warm the handler
def warm_handler():
    """Pre-warm the handler for faster cold starts"""
    try:
        print("🔥 Warming up handler...")
        
        # Test basic functionality
        test_job = {
            "input": {
                "endpoint": "health",
                "method": "GET"
            }
        }
        
        result = handler(test_job)
        print(f"✅ Handler warmed up: {result.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"⚠️ Handler warmup failed: {e}")

if __name__ == "__main__":
    # Set up asyncio event loop policy for better performance
    if sys.platform == 'linux':
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    # Pre-warm the handler
    warm_handler()
    
    # Start RunPod serverless
    print("🚀 Starting RunPod Serverless Handler")
    runpod.serverless.start({"handler": handler})