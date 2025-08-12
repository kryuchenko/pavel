"""
Rate limiter for Google Play API calls.

Implements exponential backoff and per-app rate limiting.
"""

import time
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

from pavel.core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RateLimit:
    """Rate limit configuration per app"""
    requests_per_minute: int = 30  # Conservative limit
    burst_limit: int = 5          # Max burst requests
    cooldown_seconds: int = 60    # Cooldown after burst
    
class RateLimiter:
    """
    Rate limiter with per-app tracking and exponential backoff.
    
    Google Play scraper doesn't have official rate limits,
    but we implement conservative limits to avoid blocking.
    """
    
    def __init__(self, default_limit: Optional[RateLimit] = None):
        self.default_limit = default_limit or RateLimit()
        self.app_limits: Dict[str, RateLimit] = {}
        self.request_history: Dict[str, list] = {}
        self.burst_count: Dict[str, int] = {}
        self.last_cooldown: Dict[str, datetime] = {}
        
    def set_app_limit(self, app_id: str, limit: RateLimit):
        """Set custom rate limit for specific app"""
        self.app_limits[app_id] = limit
        logger.info(f"Set custom rate limit for {app_id}: {limit.requests_per_minute}/min")
        
    def _get_limit(self, app_id: str) -> RateLimit:
        """Get rate limit for app (custom or default)"""
        return self.app_limits.get(app_id, self.default_limit)
        
    def _cleanup_history(self, app_id: str, limit: RateLimit):
        """Remove old requests from history"""
        if app_id not in self.request_history:
            self.request_history[app_id] = []
            
        cutoff_time = time.time() - 60  # Last minute only
        self.request_history[app_id] = [
            req_time for req_time in self.request_history[app_id]
            if req_time > cutoff_time
        ]
        
    def _is_in_cooldown(self, app_id: str, limit: RateLimit) -> bool:
        """Check if app is in cooldown period"""
        if app_id not in self.last_cooldown:
            return False
            
        cooldown_end = self.last_cooldown[app_id] + timedelta(seconds=limit.cooldown_seconds)
        return datetime.now() < cooldown_end
        
    def _calculate_delay(self, app_id: str, limit: RateLimit) -> float:
        """Calculate delay needed before next request"""
        self._cleanup_history(app_id, limit)
        
        # Check cooldown
        if self._is_in_cooldown(app_id, limit):
            remaining = (self.last_cooldown[app_id] + 
                        timedelta(seconds=limit.cooldown_seconds) - 
                        datetime.now()).total_seconds()
            return max(0, remaining)
            
        recent_requests = len(self.request_history[app_id])
        
        # Check burst limit
        if recent_requests >= limit.burst_limit:
            # Trigger cooldown
            self.last_cooldown[app_id] = datetime.now()
            logger.warning(f"Burst limit exceeded for {app_id}, entering cooldown")
            return limit.cooldown_seconds
            
        # Check per-minute limit
        if recent_requests >= limit.requests_per_minute:
            # Wait until oldest request expires
            oldest_request = min(self.request_history[app_id])
            delay = 60 - (time.time() - oldest_request)
            return max(0, delay)
            
        # Add small jitter to avoid thundering herd
        jitter = random.uniform(0.1, 0.5)
        return jitter
        
    async def wait_if_needed(self, app_id: str) -> None:
        """Wait if rate limit requires delay"""
        limit = self._get_limit(app_id)
        delay = self._calculate_delay(app_id, limit)
        
        if delay > 0:
            logger.info(f"Rate limiting {app_id}: waiting {delay:.1f}s")
            await asyncio.sleep(delay)
            
        # Record this request
        if app_id not in self.request_history:
            self.request_history[app_id] = []
        self.request_history[app_id].append(time.time())
        
    def get_stats(self, app_id: str) -> Dict:
        """Get rate limiting stats for app"""
        limit = self._get_limit(app_id)
        self._cleanup_history(app_id, limit)
        
        recent_requests = len(self.request_history.get(app_id, []))
        is_cooldown = self._is_in_cooldown(app_id, limit)
        
        return {
            "app_id": app_id,
            "requests_last_minute": recent_requests,
            "limit_per_minute": limit.requests_per_minute,
            "burst_limit": limit.burst_limit,
            "in_cooldown": is_cooldown,
            "utilization": recent_requests / limit.requests_per_minute
        }