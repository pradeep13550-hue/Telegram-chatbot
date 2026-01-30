from dataclasses import dataclass
from typing import Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class BotConfig:
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    
    # Bot settings
    ADMIN_IDS: List[int] = [int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x]
    MAX_TOKENS: int = 4096
    MAX_CONTEXT_LENGTH: int = 8000
    MAX_HISTORY_MESSAGES: int = 20
    
    # Redis configuration (for conversation history)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_ENABLED: bool = bool(os.getenv("REDIS_ENABLED", "true").lower() == "true")
    
    # Rate limiting
    REQUESTS_PER_MINUTE: int = 30
    MESSAGES_PER_USER_PER_MINUTE: int = 10
    
    # Model configuration
    MODEL_NAME: str = "deepseek-chat"
    TEMPERATURE: float = 0.7
    PRESENCE_PENALTY: float = 0.0
    FREQUENCY_PENALTY: float = 0.0
    
    @classmethod
    def validate(cls):
        if not cls.TELEGRAM_TOKEN:
            raise ValueError("TELEGRAM_TOKEN is not set in environment variables")
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is not set in environment variables")
        return cls()

config = BotConfig.validate()
