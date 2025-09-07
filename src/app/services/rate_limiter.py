"""
Rate limiting service for API requests
Container-level rate limiting with sliding window
"""

import time
import asyncio
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_hour: int = 100
    requests_per_minute: int = 20
    concurrent_requests: int = 5

class RateLimiter:
    """Container-level rate limiting"""
    
    def __init__(self):
        # Storage for rate limiting data
        self.hourly_requests: Dict[str, List[float]] = defaultdict(list)
        self.minute_requests: Dict[str, List[float]] = defaultdict(list)
        self.concurrent_requests: Dict[str, int] = defaultdict(int)
        
        # Default limits
        self.default_limits = RateLimit()
        
        # Cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_old_requests()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_old_requests(self):
        """Remove old request timestamps"""
        current_time = time.time()
        
        # Cleanup hourly requests (older than 1 hour)
        for api_key in list(self.hourly_requests.keys()):
            requests = self.hourly_requests[api_key]
            requests[:] = [req for req in requests if current_time - req < 3600]
            if not requests:
                del self.hourly_requests[api_key]
        
        # Cleanup minute requests (older than 1 minute)
        for api_key in list(self.minute_requests.keys()):
            requests = self.minute_requests[api_key]
            requests[:] = [req for req in requests if current_time - req < 60]
            if not requests:
                del self.minute_requests[api_key]
    
    async def check_rate_limit(
        self, 
        api_key: str, 
        limits: Optional[RateLimit] = None
    ) -> bool:
        """
        Check if request is within rate limits
        
        Args:
            api_key: User's API key
            limits: Custom rate limits (uses default if None)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        if not limits:
            limits = self.default_limits
        
        current_time = time.time()
        
        # Check concurrent requests
        if self.concurrent_requests[api_key] >= limits.concurrent_requests:
            return False
        
        # Check minute rate limit
        minute_requests = self.minute_requests[api_key]
        minute_requests[:] = [req for req in minute_requests if current_time - req < 60]
        if len(minute_requests) >= limits.requests_per_minute:
            return False
        
        # Check hourly rate limit
        hourly_requests = self.hourly_requests[api_key]
        hourly_requests[:] = [req for req in hourly_requests if current_time - req < 3600]
        if len(hourly_requests) >= limits.requests_per_hour:
            return False
        
        # Record this request
        minute_requests.append(current_time)
        hourly_requests.append(current_time)
        
        return True
    
    async def start_request(self, api_key: str):
        """Mark start of a request (for concurrent tracking)"""
        self.concurrent_requests[api_key] += 1
    
    async def end_request(self, api_key: str):
        """Mark end of a request (for concurrent tracking)"""
        if self.concurrent_requests[api_key] > 0:
            self.concurrent_requests[api_key] -= 1
    
    async def get_rate_limit_status(self, api_key: str, limits: Optional[RateLimit] = None) -> Dict:
        """
        Get current rate limit status for an API key
        
        Returns:
            Dictionary with current usage and limits
        """
        if not limits:
            limits = self.default_limits
        
        current_time = time.time()
        
        # Clean up old requests
        minute_requests = self.minute_requests[api_key]
        minute_requests[:] = [req for req in minute_requests if current_time - req < 60]
        
        hourly_requests = self.hourly_requests[api_key]
        hourly_requests[:] = [req for req in hourly_requests if current_time - req < 3600]
        
        return {
            "requests_this_minute": len(minute_requests),
            "requests_this_hour": len(hourly_requests),
            "concurrent_requests": self.concurrent_requests[api_key],
            "limits": {
                "requests_per_minute": limits.requests_per_minute,
                "requests_per_hour": limits.requests_per_hour,
                "concurrent_requests": limits.concurrent_requests
            },
            "remaining": {
                "minute": max(0, limits.requests_per_minute - len(minute_requests)),
                "hour": max(0, limits.requests_per_hour - len(hourly_requests)),
                "concurrent": max(0, limits.concurrent_requests - self.concurrent_requests[api_key])
            }
        }
    
    async def reset_rate_limits(self, api_key: str):
        """Reset rate limits for an API key (admin function)"""
        if api_key in self.hourly_requests:
            del self.hourly_requests[api_key]
        if api_key in self.minute_requests:
            del self.minute_requests[api_key]
        if api_key in self.concurrent_requests:
            del self.concurrent_requests[api_key]
    
    async def cleanup(self):
        """Cleanup rate limiter resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
