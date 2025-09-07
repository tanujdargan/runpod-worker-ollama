"""
Authentication service for dual-layer validation
Layer 1: Vercel Dashboard (API key generation)
Layer 2: Container Authentication (API key validation)
"""

import os
import time
import hmac
import hashlib
import jwt
from typing import Optional, List
from pydantic import BaseModel

class AuthResult(BaseModel):
    """Authentication result"""
    authorized: bool
    user_id: Optional[str] = None
    models: List[str] = []
    rate_limit: int = 100
    error: Optional[str] = None

class AuthService:
    """Container-level authentication service"""
    
    def __init__(self):
        self.vercel_api_secret = os.getenv("VERCEL_API_SECRET")
        self.jwt_secret = os.getenv("JWT_SECRET_KEY")
        self.default_models = ["phraser", "main", "langchain", "medgemma:27b"]
        
        if not self.vercel_api_secret:
            print("⚠️  VERCEL_API_SECRET not set - API key validation disabled")
        if not self.jwt_secret:
            print("⚠️  JWT_SECRET_KEY not set - JWT validation disabled")
    
    async def validate_request(self, api_key: str, endpoint: str) -> AuthResult:
        """
        Validate requests at container level
        
        Args:
            api_key: API key from Authorization header or X-API-Key
            endpoint: Requested endpoint (for permission checking)
            
        Returns:
            AuthResult with authorization status and user details
        """
        try:
            # Skip validation if no secrets configured (development mode)
            if not self.vercel_api_secret and not self.jwt_secret:
                return AuthResult(
                    authorized=True,
                    user_id="dev_user",
                    models=self.default_models,
                    rate_limit=1000
                )
            
            # Try JWT validation first
            if self.jwt_secret:
                jwt_result = await self._validate_jwt(api_key)
                if jwt_result.authorized:
                    return jwt_result
            
            # Try API key signature validation
            if self.vercel_api_secret:
                signature_result = await self._validate_api_key_signature(api_key)
                if signature_result.authorized:
                    return signature_result
            
            return AuthResult(
                authorized=False,
                error="Invalid API key or token"
            )
            
        except Exception as e:
            return AuthResult(
                authorized=False,
                error=f"Authentication error: {str(e)}"
            )
    
    async def _validate_jwt(self, token: str) -> AuthResult:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                return AuthResult(
                    authorized=False,
                    error="Token expired"
                )
            
            return AuthResult(
                authorized=True,
                user_id=payload.get("user_id"),
                models=payload.get("models", self.default_models),
                rate_limit=payload.get("rate_limit", 100)
            )
            
        except jwt.InvalidTokenError as e:
            return AuthResult(
                authorized=False,
                error=f"Invalid JWT: {str(e)}"
            )
    
    async def _validate_api_key_signature(self, api_key: str) -> AuthResult:
        """
        Validate API key signature from Vercel dashboard
        Expected format: {user_id}.{timestamp}.{signature}
        """
        try:
            parts = api_key.split(".")
            if len(parts) != 3:
                return AuthResult(
                    authorized=False,
                    error="Invalid API key format"
                )
            
            user_id, timestamp_str, signature = parts
            
            # Check timestamp (not older than 24 hours for security)
            try:
                timestamp = int(timestamp_str)
                if time.time() - timestamp > 86400:  # 24 hours
                    return AuthResult(
                        authorized=False,
                        error="API key expired"
                    )
            except ValueError:
                return AuthResult(
                    authorized=False,
                    error="Invalid timestamp in API key"
                )
            
            # Verify signature
            expected_signature = self._generate_signature(user_id, timestamp_str)
            if not hmac.compare_digest(signature, expected_signature):
                return AuthResult(
                    authorized=False,
                    error="Invalid API key signature"
                )
            
            return AuthResult(
                authorized=True,
                user_id=user_id,
                models=self.default_models,
                rate_limit=100
            )
            
        except Exception as e:
            return AuthResult(
                authorized=False,
                error=f"API key validation error: {str(e)}"
            )
    
    def _generate_signature(self, user_id: str, timestamp: str) -> str:
        """Generate HMAC signature for API key"""
        message = f"{user_id}.{timestamp}"
        signature = hmac.new(
            self.vercel_api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature[:16]  # Use first 16 chars for shorter keys
    
    def generate_api_key(self, user_id: str) -> str:
        """
        Generate API key for user (utility function)
        This would typically be called by the Vercel dashboard
        """
        timestamp = str(int(time.time()))
        signature = self._generate_signature(user_id, timestamp)
        return f"{user_id}.{timestamp}.{signature}"
    
    def generate_jwt(self, user_id: str, models: List[str] = None, rate_limit: int = 100, expires_in: int = 3600) -> str:
        """
        Generate JWT token for user (utility function)
        This would typically be called by the Vercel dashboard
        """
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET_KEY not configured")
        
        payload = {
            "user_id": user_id,
            "models": models or self.default_models,
            "rate_limit": rate_limit,
            "iat": time.time(),
            "exp": time.time() + expires_in
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

# Convenience function for dependency injection
async def validate_api_key(api_key: str, endpoint: str = "chat") -> AuthResult:
    """Standalone function for validating API keys"""
    auth_service = AuthService()
    return await auth_service.validate_request(api_key, endpoint)
