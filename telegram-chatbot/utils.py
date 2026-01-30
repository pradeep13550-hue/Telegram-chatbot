import json
import asyncio
import aiohttp
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
import redis.asyncio as redis
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

class RateLimitExceeded(Exception):
    pass

class APIError(Exception):
    pass

class ConversationManager(ABC):
    @abstractmethod
    async def save_conversation(self, user_id: str, messages: List[Dict]):
        pass
    
    @abstractmethod
    async def get_conversation(self, user_id: str) -> List[Dict]:
        pass
    
    @abstractmethod
    async def clear_conversation(self, user_id: str):
        pass
    
    @abstractmethod
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        pass

class RedisConversationManager(ConversationManager):
    def __init__(self, redis_url: str, max_history: int = 20):
        self.redis_url = redis_url
        self.max_history = max_history
        self.redis = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Redis connection"""
        if not self._initialized:
            try:
                self.redis = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis.ping()
                self._initialized = True
                logger.info("Redis connection established")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._initialized = False
                # Fallback to in-memory storage
                self._memory_storage = {}
    
    async def save_conversation(self, user_id: str, messages: List[Dict]):
        try:
            if self._initialized and self.redis:
                key = f"conversation:{user_id}"
                # Also save timestamp for each message
                enhanced_messages = []
                for msg in messages[-self.max_history:]:
                    enhanced_msg = msg.copy()
                    enhanced_msg['timestamp'] = datetime.now().isoformat()
                    enhanced_messages.append(enhanced_msg)
                
                await self.redis.setex(
                    key,
                    timedelta(hours=24),
                    json.dumps(enhanced_messages)
                )
                
                # Update user stats
                stats_key = f"stats:{user_id}"
                user_stats = await self.get_user_stats(user_id)
                user_stats['total_messages'] = user_stats.get('total_messages', 0) + 1
                user_stats['last_active'] = datetime.now().isoformat()
                await self.redis.setex(
                    stats_key,
                    timedelta(days=30),
                    json.dumps(user_stats)
                )
            else:
                # Fallback to in-memory storage
                if not hasattr(self, '_memory_storage'):
                    self._memory_storage = {}
                self._memory_storage[user_id] = messages[-self.max_history:]
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    async def get_conversation(self, user_id: str) -> List[Dict]:
        try:
            if self._initialized and self.redis:
                key = f"conversation:{user_id}"
                data = await self.redis.get(key)
                if data:
                    messages = json.loads(data)
                    # Remove timestamps for API compatibility
                    for msg in messages:
                        msg.pop('timestamp', None)
                    return messages
                return []
            else:
                # Fallback to in-memory storage
                return self._memory_storage.get(user_id, [])
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return []
    
    async def clear_conversation(self, user_id: str):
        try:
            if self._initialized and self.redis:
                key = f"conversation:{user_id}"
                await self.redis.delete(key)
            else:
                # Clear from memory
                self._memory_storage.pop(user_id, None)
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        try:
            if self._initialized and self.redis:
                stats_key = f"stats:{user_id}"
                data = await self.redis.get(stats_key)
                if data:
                    return json.loads(data)
            return {'total_messages': 0, 'last_active': None}
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {'total_messages': 0, 'last_active': None}

class MemoryConversationManager(ConversationManager):
    """Fallback conversation manager using memory"""
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.conversations = {}
        self.user_stats = {}
    
    async def save_conversation(self, user_id: str, messages: List[Dict]):
        self.conversations[user_id] = messages[-self.max_history:]
        self.user_stats[user_id] = self.user_stats.get(user_id, {'total_messages': 0})
        self.user_stats[user_id]['total_messages'] += 1
        self.user_stats[user_id]['last_active'] = datetime.now().isoformat()
    
    async def get_conversation(self, user_id: str) -> List[Dict]:
        return self.conversations.get(user_id, [])
    
    async def clear_conversation(self, user_id: str):
        self.conversations.pop(user_id, None)
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        return self.user_stats.get(user_id, {'total_messages': 0, 'last_active': None})

class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.session = None
    
    @asynccontextmanager
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        yield self.session
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> Optional[str]:
        """Generate response from DeepSeek API"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }
        
        try:
            async with self.get_session() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90)
                ) as response:
                    
                    if response.status == 429:
                        raise RateLimitExceeded("Rate limit exceeded")
                    elif response.status == 401:
                        raise APIError("Invalid API key")
                    elif response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API error: {response.status} - {error_text}")
                        raise APIError(f"API error: {response.status}")
                    
                    result = await response.json()
                    
                    if 'choices' not in result or not result['choices']:
                        logger.error(f"Unexpected API response: {result}")
                        return None
                    
                    return result["choices"][0]["message"]["content"]
                    
        except asyncio.TimeoutError:
            logger.error("DeepSeek API timeout")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            raise
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()

class RateLimiter:
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url
        self.redis = None
        self._initialized = False
        self._memory_counts = {}
    
    async def initialize(self):
        """Initialize Redis connection"""
        if self.redis_url:
            try:
                self.redis = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis.ping()
                self._initialized = True
                logger.info("Rate limiter Redis connection established")
            except Exception as e:
                logger.warning(f"Rate limiter Redis failed: {e}, using memory fallback")
                self._initialized = False
                self._memory_counts = {}
    
    async def check_rate_limit(self, user_id: str, limit: int = 10, window: int = 60) -> Tuple[bool, int]:
        """Check if user has exceeded rate limit"""
        current_time = datetime.now()
        minute_key = current_time.strftime("%Y%m%d%H%M")
        key = f"rate_limit:{user_id}:{minute_key}"
        
        try:
            if self._initialized and self.redis:
                # Using Redis
                current = await self.redis.get(key)
                if current is None:
                    await self.redis.setex(key, window, 1)
                    return True, 1
                
                current_count = int(current)
                if current_count >= limit:
                    return False, current_count
                
                await self.redis.incr(key)
                return True, current_count + 1
            else:
                # Using memory
                if key not in self._memory_counts:
                    self._memory_counts[key] = {
                        'count': 1,
                        'expiry': current_time + timedelta(seconds=window)
                    }
                    return True, 1
                
                if self._memory_counts[key]['expiry'] < current_time:
                    # Reset expired key
                    self._memory_counts[key] = {
                        'count': 1,
                        'expiry': current_time + timedelta(seconds=window)
                    }
                    return True, 1
                
                current_count = self._memory_counts[key]['count']
                if current_count >= limit:
                    return False, current_count
                
                self._memory_counts[key]['count'] += 1
                return True, current_count + 1
                
        except Exception as e:
            logger.error(f"Rate limit error: {e}")
            # Allow on error
            return True, 0
    
    def cleanup_expired(self):
        """Clean up expired memory entries"""
        current_time = datetime.now()
        expired_keys = [
            key for key, data in self._memory_counts.items()
            if data['expiry'] < current_time
        ]
        for key in expired_keys:
            del self._memory_counts[key]

class MessageProcessor:
    @staticmethod
    def format_messages_for_api(conversation_history: List[Dict], new_message: str) -> List[Dict]:
        """Format messages for DeepSeek API"""
        messages = conversation_history.copy()
        
        # Add system message if not present
        if not messages or messages[0].get("role") != "system":
            system_message = {
                "role": "system",
                "content": "You are DeepSeek AI, a helpful assistant created by DeepSeek. Respond helpfully and accurately."
            }
            messages.insert(0, system_message)
        
        messages.append({
            "role": "user",
            "content": new_message
        })
        return messages
    
    @staticmethod
    def truncate_conversation(messages: List[Dict], max_chars: int = 16000) -> List[Dict]:
        """Truncate conversation to fit character limit"""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        
        # Keep system message
        system_message = None
        if messages and messages[0].get("role") == "system":
            system_message = messages[0]
            messages = messages[1:]
        
        while total_chars > max_chars and len(messages) > 2:
            removed = messages.pop(1)  # Remove oldest non-system, non-last message
            total_chars -= len(removed.get("content", ""))
        
        if system_message:
            messages.insert(0, system_message)
        
        return messages
    
    @staticmethod
    def split_long_response(response: str, max_length: int = 4000) -> List[str]:
        """Split long responses into multiple messages"""
        if len(response) <= max_length:
            return [response]
        
        parts = []
        # Try to split at paragraph boundaries
        paragraphs = response.split('\n\n')
        current_part = ""
        
        for para in paragraphs:
            if len(current_part) + len(para) + 2 <= max_length:
                if current_part:
                    current_part += '\n\n' + para
                else:
                    current_part = para
            else:
                if current_part:
                    parts.append(current_part)
                if len(para) <= max_length:
                    current_part = para
                else:
                    # Paragraph too long, split by sentences
                    sentences = para.replace('. ', '.\n').split('\n')
                    for sent in sentences:
                        if len(current_part) + len(sent) + 1 <= max_length:
                            if current_part:
                                current_part += ' ' + sent
                            else:
                                current_part = sent
                        else:
                            parts.append(current_part)
                            current_part = sent
        
        if current_part:
            parts.append(current_part)
        
        return parts

class Analytics:
    @staticmethod
    def track_request(user_id: str, prompt_length: int, response_length: int, success: bool = True):
        """Track API request metrics"""
        # This would be integrated with a monitoring system
        logger.info(
            f"Request tracked - User: {user_id}, "
            f"Prompt: {prompt_length} chars, "
            f"Response: {response_length} chars, "
            f"Success: {success}"
        )
