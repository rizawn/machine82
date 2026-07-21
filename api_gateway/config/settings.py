import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///trinity.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Absolute paths to component directories
    MLRL01_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "MLRL01"))
    MLRL02_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "MLRL02"))
    
    # Storage settings
    ARTIFACTS_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    
    # External services
    OLLAMA_HOST: str = "http://localhost:11434"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure artifacts directory exists
os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)
