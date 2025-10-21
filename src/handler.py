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

# Concurrency control (from upstream vLLM logic)
DEFAULT_MAX_CONCURRENCY = 8
max_concurrency = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))

# Global variables for services
services_initialized = False
model_router = None
langchain_orchestrator = None
auth_service = None

async def run_async_in_handler(coro):
    """
    Helper function to run async code in the handler context
    Handles the case where an event loop is already running
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we get here, there's already a loop running
        # We need to run the coroutine in a thread pool
        import concurrent.futures
        import threading
        
        result = None
        exception = None
        
        def run_in_thread():
            nonlocal result, exception
            try:
                # Create a new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result = new_loop.run_until_complete(coro)
                new_loop.close()
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        if exception:
            raise exception
        return result
        
    except RuntimeError:
        # No running loop, we can use asyncio.run safely
        return asyncio.run(coro)

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
    
    Supports multiple input formats:
    1. Simple prompt: {"input": {"prompt": "Hello"}}
    2. Chat completion: {"input": {"messages": [...]}}
    3. Structured: {"input": {"endpoint": "chat", "data": {...}}}
    """
    try:
        # Get job input
        job_input = job.get("input", {})
        
        if not job_input:
            return {"error": "No input provided"}
        
        print(f"🚀 Processing RunPod request: {job_input}")
        
        # Handle simple prompt format (backward compatibility)
        if "prompt" in job_input:
            messages = [{"role": "user", "content": job_input["prompt"]}]
            data = {
                "model": job_input.get("model", "phraser"),
                "messages": messages,
                "max_tokens": job_input.get("max_tokens", 500),
                "temperature": job_input.get("temperature", 1)  # gpt-5-nano only supports temperature=1
            }
            # Skip auth for simple prompt format (backward compatibility)
            return run_async_in_handler(handle_chat_completion(data, {}, skip_auth=True))
        
        # Handle chat messages format
        elif "messages" in job_input:
            data = {
                "model": job_input.get("model", "phraser"),
                "messages": job_input["messages"],
                "max_tokens": job_input.get("max_tokens", 500),
                "temperature": job_input.get("temperature", 1),  # gpt-5-nano only supports temperature=1
                "stream": job_input.get("stream", False)
            }
            # Skip auth for direct message format (backward compatibility)
            return run_async_in_handler(handle_chat_completion(data, {}, skip_auth=True))
        
        # Handle structured format
        elif "endpoint" in job_input:
            endpoint = job_input.get("endpoint", "health")
            method = job_input.get("method", "GET")
            data = job_input.get("data", {})
            headers = job_input.get("headers", {})
            
            print(f"🚀 Processing {method} {endpoint} request")
            
            # Route to appropriate handler
            if endpoint == "health":
                return handle_health_check()
            elif endpoint == "chat":
                return run_async_in_handler(handle_chat_completion(data, headers))
            elif endpoint == "langchain":
                return run_async_in_handler(handle_langchain_consultation(data, headers))
            elif endpoint == "models":
                return handle_models_list()
            else:
                return {"error": f"Unknown endpoint: {endpoint}"}
        
        # Handle Langchain consultation format
        elif "symptoms" in job_input:
            data = {
                "symptoms": job_input["symptoms"],
                "patient_data": job_input.get("patient_data", {}),
                "session_id": job_input.get("session_id")
            }
            # Skip auth for direct symptoms format (backward compatibility)
            return run_async_in_handler(handle_langchain_consultation(data, {}, skip_auth=True))
        
        else:
            return {"error": "Invalid input format. Expected 'prompt', 'messages', 'endpoint', or 'symptoms'"}
    
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

async def handle_chat_completion(data: Dict, headers: Dict, skip_auth: bool = False):
    """Handle chat completion requests"""
    try:
        # Initialize services if needed
        await initialize_services()

        # Validate authentication (simplified for RunPod)
        # Skip auth if explicitly requested or if no secrets are configured
        if not skip_auth:
            api_key = headers.get("authorization", "").replace("Bearer ", "")
            vercel_secret = os.getenv("VERCEL_API_SECRET")
            jwt_secret = os.getenv("JWT_SECRET_KEY")

            # Only require auth if secrets are configured
            if (vercel_secret or jwt_secret) and not api_key:
                return {"error": "Missing API key"}
        
        # Extract chat completion parameters
        model = data.get("model", "phraser")
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens")
        temperature = data.get("temperature", 1)  # gpt-5-nano only supports temperature=1
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

async def handle_langchain_consultation(data: Dict, headers: Dict, skip_auth: bool = False):
    """Handle Langchain consultation requests"""
    try:
        # Initialize services if needed
        await initialize_services()

        # Validate authentication (simplified for RunPod)
        # Skip auth if explicitly requested or if no secrets are configured
        if not skip_auth:
            api_key = headers.get("authorization", "").replace("Bearer ", "")
            vercel_secret = os.getenv("VERCEL_API_SECRET")
            jwt_secret = os.getenv("JWT_SECRET_KEY")

            # Only require auth if secrets are configured
            if (vercel_secret or jwt_secret) and not api_key:
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

    # Start RunPod serverless with concurrency control
    print("🚀 Starting RunPod Serverless Handler")
    print(f"📊 Max concurrency: {max_concurrency}")
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": lambda x: max_concurrency,
        "return_aggregate_stream": True,
    })