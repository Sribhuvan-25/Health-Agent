"""
Configuration Management for ExamBuilder Multi-Agent System
Centralized configuration with environment variable support
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management"""
    
    # API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "exambuilder-multi-agent")
    
    # VertexAI Configuration
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    # ExamBuilder API Configuration
    EXAMBUILDER_API_KEY: str = os.getenv("EXAMBUILDER_API_KEY", "FE0F8C82239FF183")
    EXAMBUILDER_API_SECRET: str = os.getenv("EXAMBUILDER_API_SECRET", "A227A6838F3D180A15E6D8ED")
    EXAMBUILDER_BASE_URL: str = os.getenv("EXAMBUILDER_BASE_URL", "https://instructor.exambuilder.com/v2")
    
    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # "openai", "gemini", or "vertexai"
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    LLM_MAX_TOKENS: Optional[int] = int(os.getenv("LLM_MAX_TOKENS", "1000")) if os.getenv("LLM_MAX_TOKENS") else None
    
    # Session Configuration
    SESSION_TIMEOUT_HOURS: int = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
    MAX_SESSIONS: int = int(os.getenv("MAX_SESSIONS", "1000"))
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_TRACING: bool = os.getenv("ENABLE_TRACING", "True").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        required_vars = [
            "EXAMBUILDER_API_KEY", 
            "EXAMBUILDER_API_SECRET"
        ]
        
        # Validate LLM provider API key
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            required_vars.append("OPENAI_API_KEY")
        elif cls.LLM_PROVIDER == "gemini" and not cls.GOOGLE_API_KEY:
            required_vars.append("GOOGLE_API_KEY")
        elif cls.LLM_PROVIDER == "vertexai" and not cls.GOOGLE_CLOUD_PROJECT:
            required_vars.append("GOOGLE_CLOUD_PROJECT")
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)"""
        print("🔧 ExamBuilder Multi-Agent Configuration:")
        print(f"   LLM Model: {cls.LLM_MODEL}")
        print(f"   Temperature: {cls.LLM_TEMPERATURE}")
        print(f"   Base URL: {cls.EXAMBUILDER_BASE_URL}")
        print(f"   Session Timeout: {cls.SESSION_TIMEOUT_HOURS} hours")
        print(f"   Max Sessions: {cls.MAX_SESSIONS}")
        print(f"   Debug Mode: {cls.DEBUG}")
        print(f"   Tracing Enabled: {cls.ENABLE_TRACING}")

# Global config instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config 