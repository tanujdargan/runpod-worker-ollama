"""
Model router for unified interface to all model types
Routes requests to appropriate model client (OpenAI or Ollama)
"""

from typing import Dict, List, Optional, AsyncGenerator, Union
from ..clients.openai_client import OpenAIClient
from ..clients.ollama_client import OllamaClient

class ModelRouter:
    """Unified model routing service"""
    
    def __init__(self, openai_client: OpenAIClient, ollama_client: OllamaClient):
        self.openai_client = openai_client
        self.ollama_client = ollama_client
        
        # Model routing configuration
        self.model_routes = {
            # OpenAI models
            "phraser": "openai",
            "main": "openai", 
            "gpt-5-nano": "openai",
            "gpt-4": "openai",
            "gpt-3.5-turbo": "openai",
            
            # Ollama/Local models
            "medgemma:27b": "ollama",
            "medgemma": "ollama",
            "langchain": "ollama",  # Langchain uses MedGemma for ICD/CPT
        }
        
        # Default models for each service
        self.default_models = {
            "openai": "gpt-5-nano",
            "ollama": "medgemma:27b"
        }
    
    def get_client_for_model(self, model: str) -> tuple[Union[OpenAIClient, OllamaClient], str]:
        """
        Get the appropriate client and actual model name for a requested model
        
        Args:
            model: Requested model name
            
        Returns:
            Tuple of (client, actual_model_name)
        """
        route = self.model_routes.get(model, "openai")  # Default to OpenAI
        
        if route == "openai":
            return self.openai_client, model
        elif route == "ollama":
            # Map to actual Ollama model name
            if model in ["langchain", "medgemma"]:
                actual_model = "medgemma:27b"
            else:
                actual_model = model
            return self.ollama_client, actual_model
        else:
            raise ValueError(f"Unknown model route: {route}")
    
    async def chat_completion(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict, AsyncGenerator[Dict, None]]:
        """
        Route chat completion to appropriate client
        
        Args:
            messages: Chat messages
            model: Model name
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            stream: Whether to stream
            **kwargs: Additional parameters
            
        Returns:
            Chat completion response or stream generator
        """
        client, actual_model = self.get_client_for_model(model)
        
        return await client.chat_completion(
            messages=messages,
            model=actual_model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs
        )
    
    async def stream_chat_completion(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion in SSE format
        
        Args:
            messages: Chat messages
            model: Model name
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Yields:
            SSE-formatted strings
        """
        client, actual_model = self.get_client_for_model(model)
        
        async for chunk in client.stream_chat_completion(
            messages=messages,
            model=actual_model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        ):
            yield chunk
    
    async def generate_response(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate a simple text response
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        client, actual_model = self.get_client_for_model(model)
        
        if hasattr(client, 'generate_response'):
            return await client.generate_response(
                prompt=prompt,
                model=actual_model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        else:
            # Fallback to chat completion for clients without generate_response
            messages = [{"role": "user", "content": prompt}]
            response = await client.chat_completion(
                messages=messages,
                model=actual_model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            if "choices" in response and response["choices"]:
                return response["choices"][0]["message"]["content"]
            else:
                raise ValueError("No response generated")
    
    async def stream_generate_response(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream a simple text response in SSE format
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Yields:
            SSE-formatted strings
        """
        client, actual_model = self.get_client_for_model(model)
        
        if hasattr(client, 'stream_generate_response'):
            async for chunk in client.stream_generate_response(
                prompt=prompt,
                model=actual_model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            ):
                yield chunk
        else:
            # Fallback to streaming chat completion
            messages = [{"role": "user", "content": prompt}]
            async for chunk in client.stream_chat_completion(
                messages=messages,
                model=actual_model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            ):
                yield chunk
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.model_routes.keys())
    
    def is_model_available(self, model: str) -> bool:
        """Check if a model is available"""
        return model in self.model_routes
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all clients"""
        return {
            "openai": await self.openai_client.health_check(),
            "ollama": await self.ollama_client.health_check()
        }
    
    async def cleanup(self):
        """Cleanup all clients"""
        await self.openai_client.cleanup()
        await self.ollama_client.cleanup()
