"""
Ollama client for local MedGemma-27B model with streaming support
"""

import os
import json
import aiohttp
import asyncio
from typing import Dict, List, Optional, AsyncGenerator

class OllamaClient:
    """Local Ollama client for MedGemma-27B"""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "medgemma:27b")
        self.session = None
        
        # Configuration
        self.request_timeout = float(os.getenv("RUNPOD_REQUEST_TIMEOUT", "30.0"))
        self.max_retries = int(os.getenv("RUNPOD_MAX_RETRIES", "3"))
        self.connection_pool_size = int(os.getenv("RUNPOD_CONNECTION_POOL_SIZE", "100"))
        self.max_connections_per_host = int(os.getenv("RUNPOD_MAX_CONNECTIONS_PER_HOST", "10"))
        self.keepalive_timeout = int(os.getenv("RUNPOD_KEEPALIVE_TIMEOUT", "30"))
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            connector = aiohttp.TCPConnector(
                limit=self.connection_pool_size,
                limit_per_host=self.max_connections_per_host,
                keepalive_timeout=self.keepalive_timeout
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self.session
    
    async def warmup(self):
        """Warm up the Ollama client"""
        try:
            await self.generate("test", max_tokens=1)
            print("✅ Ollama client warmed up")
        except Exception as e:
            print(f"⚠️  Ollama warmup failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if Ollama is healthy"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text using Ollama
        
        Args:
            prompt: Input prompt
            model: Model name (default: configured model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response
            **kwargs: Additional parameters
            
        Returns:
            Generated text or stream generator
        """
        session = await self._get_session()
        
        if not model:
            model = self.model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature
            }
        }
        
        # Add max_tokens if specified (mapped to num_predict in Ollama)
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        # Add any additional options
        payload["options"].update(kwargs)
        
        if stream:
            return self._stream_generate(session, payload)
        else:
            return await self._generate_sync(session, payload)
    
    async def _generate_sync(self, session, payload) -> str:
        """Generate text synchronously"""
        async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            return result.get("response", "")
    
    async def _stream_generate(self, session, payload) -> AsyncGenerator[str, None]:
        """Stream text generation"""
        async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
            response.raise_for_status()
            
            async for line in response.content:
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if chunk.get("response"):
                            yield chunk["response"]
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    
    async def chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        stream: bool = False,
        **kwargs
    ) -> Dict:
        """
        Chat completion using Ollama chat API
        
        Args:
            messages: List of message dictionaries
            model: Model name
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            stream: Whether to stream
            **kwargs: Additional parameters
            
        Returns:
            Chat completion response or stream generator
        """
        session = await self._get_session()
        
        if not model:
            model = self.model
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        payload["options"].update(kwargs)
        
        if stream:
            return self._stream_chat_completion(session, payload)
        else:
            return await self._chat_completion_sync(session, payload)
    
    async def _chat_completion_sync(self, session, payload) -> Dict:
        """Synchronous chat completion"""
        async with session.post(f"{self.base_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            
            # Convert to OpenAI-like format
            return {
                "id": "ollama-" + str(hash(str(payload))),
                "object": "chat.completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": payload["model"],
                "choices": [{
                    "index": 0,
                    "message": result.get("message", {}),
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }
            }
    
    async def _stream_chat_completion(self, session, payload) -> AsyncGenerator[Dict, None]:
        """Stream chat completion"""
        async with session.post(f"{self.base_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            
            async for line in response.content:
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        
                        # Convert to OpenAI-like streaming format
                        if chunk.get("message"):
                            content = chunk["message"].get("content", "")
                            if content:
                                yield {
                                    "id": "ollama-stream",
                                    "object": "chat.completion.chunk",
                                    "created": int(asyncio.get_event_loop().time()),
                                    "model": payload["model"],
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None
                                    }]
                                }
                        
                        if chunk.get("done", False):
                            # Final chunk
                            yield {
                                "id": "ollama-stream",
                                "object": "chat.completion.chunk",
                                "created": int(asyncio.get_event_loop().time()),
                                "model": payload["model"],
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            break
                            
                    except json.JSONDecodeError:
                        continue
    
    async def stream_chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion in SSE format
        
        Yields:
            SSE-formatted strings for streaming response
        """
        try:
            async for chunk in await self.chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "ollama_error"
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
    
    async def stream_generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream a simple text response in SSE format"""
        try:
            async for token in await self.generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            ):
                # Wrap in OpenAI-like format for consistency
                chunk = {
                    "id": "ollama-generate",
                    "object": "text_completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": model or self.model,
                    "choices": [{
                        "text": token,
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "ollama_stream_error"
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
    
    async def cleanup(self):
        """Cleanup client resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
