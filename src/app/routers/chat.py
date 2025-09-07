"""
Chat completions router for OpenAI-compatible endpoints
"""

import json
import time
import uuid
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from ..services.auth import AuthService
from ..services.model_router import ModelRouter
from ..services.streaming import StreamingManager
from ..models.requests import ChatRequest, CompletionRequest
from ..models.responses import ChatResponse, CompletionResponse

router = APIRouter()

# These will be injected by the main app
model_router: ModelRouter = None
streaming_manager: StreamingManager = None

def set_dependencies(mr: ModelRouter, sm: StreamingManager):
    """Set dependencies (called from main app)"""
    global model_router, streaming_manager
    model_router = mr
    streaming_manager = sm

@router.post("/chat/completions")
async def chat_completions(
    request: ChatRequest,
    auth_result = Depends(lambda: None)  # Will be overridden by main app
):
    """
    Create chat completion (OpenAI-compatible)
    
    Supports both streaming and non-streaming responses.
    Routes to appropriate model client based on model name.
    """
    try:
        # Validate model availability
        if not model_router.is_model_available(request.model):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not available"
            )
        
        # Convert messages to list of dicts
        messages = [msg.dict() for msg in request.messages]
        
        if request.stream:
            # Return streaming response
            async def generate():
                async for chunk in model_router.stream_chat_completion(
                    messages=messages,
                    model=request.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    stop=request.stop
                ):
                    yield chunk
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers=streaming_manager.create_sse_headers()
            )
        else:
            # Return synchronous response
            response = await model_router.chat_completion(
                messages=messages,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
                top_p=request.top_p,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                stop=request.stop
            )
            
            return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/completions")
async def text_completions(
    request: CompletionRequest,
    auth_result = Depends(lambda: None)  # Will be overridden by main app
):
    """
    Create text completion (OpenAI-compatible)
    
    Converts text prompts to chat format and uses chat completion endpoint.
    """
    try:
        # Validate model availability
        if not model_router.is_model_available(request.model):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not available"
            )
        
        # Convert prompt to chat messages
        messages = [{"role": "user", "content": request.prompt}]
        
        if request.stream:
            # Return streaming response
            async def generate():
                async for chunk in model_router.stream_chat_completion(
                    messages=messages,
                    model=request.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    stop=request.stop
                ):
                    # Convert chat completion chunks to completion format
                    if chunk.startswith("data: "):
                        try:
                            chunk_data = json.loads(chunk[6:])
                            if "choices" in chunk_data and chunk_data["choices"]:
                                choice = chunk_data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    # Convert to completion format
                                    completion_chunk = {
                                        "id": chunk_data.get("id"),
                                        "object": "text_completion.chunk",
                                        "created": chunk_data.get("created"),
                                        "model": chunk_data.get("model"),
                                        "choices": [{
                                            "text": choice["delta"]["content"],
                                            "index": choice.get("index", 0),
                                            "finish_reason": choice.get("finish_reason")
                                        }]
                                    }
                                    yield f"data: {json.dumps(completion_chunk)}\n\n"
                        except (json.JSONDecodeError, KeyError):
                            yield chunk
                    else:
                        yield chunk
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers=streaming_manager.create_sse_headers()
            )
        else:
            # Return synchronous response
            chat_response = await model_router.chat_completion(
                messages=messages,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
                top_p=request.top_p,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                stop=request.stop
            )
            
            # Convert chat response to completion format
            completion_response = {
                "id": chat_response.get("id"),
                "object": "text_completion",
                "created": chat_response.get("created"),
                "model": chat_response.get("model"),
                "choices": [],
                "usage": chat_response.get("usage")
            }
            
            if "choices" in chat_response:
                for choice in chat_response["choices"]:
                    completion_choice = {
                        "text": choice.get("message", {}).get("content", ""),
                        "index": choice.get("index", 0),
                        "finish_reason": choice.get("finish_reason")
                    }
                    completion_response["choices"].append(completion_choice)
            
            return completion_response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/available")
async def get_available_models(auth_result = Depends(lambda: None)):
    """Get list of available models"""
    try:
        models = model_router.get_available_models()
        
        model_data = []
        for model in models:
            model_data.append({
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "pear-care",
                "permission": [{"allow_create_engine": True}]
            })
        
        return {"data": model_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}")
async def get_model_info(model_id: str, auth_result = Depends(lambda: None)):
    """Get information about a specific model"""
    try:
        if not model_router.is_model_available(model_id):
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "pear-care",
            "permission": [{"allow_create_engine": True}]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/test")
async def test_chat(auth_result = Depends(lambda: None)):
    """Test endpoint for chat functionality"""
    try:
        test_request = ChatRequest(
            model="phraser",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=50,
            temperature=0.1
        )
        
        messages = [msg.dict() for msg in test_request.messages]
        response = await model_router.chat_completion(
            messages=messages,
            model=test_request.model,
            max_tokens=test_request.max_tokens,
            temperature=test_request.temperature,
            stream=False
        )
        
        return {
            "status": "success",
            "test_response": response,
            "timestamp": time.time()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }
