"""
OpenAI client with streaming support and connection management
"""

import os
import asyncio
import aiohttp
from typing import Dict, List, Optional, AsyncGenerator
from openai import AsyncOpenAI
import json

class OpenAIClient:
    """OpenAI client with streaming and health checks"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        self.session = None
        
        # Configuration
        self.request_timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "30.0"))
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        self.connection_pool_size = int(os.getenv("OPENAI_CONNECTION_POOL_SIZE", "100"))
        self.max_connections_per_host = int(os.getenv("OPENAI_MAX_CONNECTIONS_PER_HOST", "10"))
        self.keepalive_timeout = int(os.getenv("OPENAI_KEEPALIVE_TIMEOUT", "30"))
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    async def _get_client(self):
        """Get or create OpenAI client"""
        if not self.client:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.request_timeout,
                max_retries=self.max_retries
            )
        return self.client
    
    async def warmup(self):
        """Warm up the OpenAI client"""
        try:
            client = await self._get_client()
            # Test with a minimal request to GPT-5-nano
            await client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            print("✅ OpenAI client warmed up")
        except Exception as e:
            print(f"⚠️  OpenAI warmup failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if OpenAI client is healthy"""
        try:
            client = await self._get_client()
            # Simple test request
            response = await client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "health"}],
                max_tokens=1
            )
            return True
        except Exception:
            return False
    
    async def chat_completion(
        self,
        messages: List[Dict],
        model: str = "gpt-5-nano",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Dict:
        """
        Create chat completion
        
        Args:
            messages: List of message dictionaries
            model: Model to use (default: gpt-5-nano)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional OpenAI parameters
            
        Returns:
            Chat completion response or generator for streaming
        """
        client = await self._get_client()
        
        # Map our model names to OpenAI models
        model_mapping = {
            "phraser": "gpt-5-nano",
            "main": "gpt-5-nano",
            "gpt-5-nano": "gpt-5-nano"
        }
        
        openai_model = model_mapping.get(model, model)
        
        if stream:
            return await self._stream_chat_completion(
                client, messages, openai_model, max_tokens, temperature, **kwargs
            )
        else:
            response = await client.chat.completions.create(
                model=openai_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.model_dump()
    
    async def _stream_chat_completion(
        self,
        client,
        messages: List[Dict],
        model: str,
        max_tokens: Optional[int],
        temperature: float,
        **kwargs
    ) -> AsyncGenerator[Dict, None]:
        """Stream chat completion responses"""
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                yield chunk.model_dump()
                
        except Exception as e:
            yield {
                "error": {
                    "message": str(e),
                    "type": "openai_error"
                }
            }
    
    async def stream_chat_completion(
        self,
        messages: List[Dict],
        model: str = "gpt-5-nano",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
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
                if "error" in chunk:
                    yield f"data: {json.dumps(chunk)}\n\n"
                    break
                
                yield f"data: {json.dumps(chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
    
    async def generate_response(
        self,
        prompt: str,
        model: str = "gpt-5-nano",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate a simple text response
        
        Args:
            prompt: Input prompt
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        if "choices" in response and response["choices"]:
            return response["choices"][0]["message"]["content"]
        else:
            raise ValueError("No response generated from OpenAI")
    
    async def stream_generate_response(
        self,
        prompt: str,
        model: str = "gpt-5-nano",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream a simple text response"""
        messages = [{"role": "user", "content": prompt}]
        
        async for chunk_data in self.stream_chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        ):
            yield chunk_data
    
    async def cleanup(self):
        """Cleanup client resources"""
        if self.client:
            await self.client.close()
            self.client = None
