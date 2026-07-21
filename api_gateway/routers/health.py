import httpx
import redis
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from models.database import get_db
from config.settings import settings

router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
def health_check(db: Session = Depends(get_db)):
    status = {
        "status": "healthy",
        "postgres": "unreachable",
        "redis": "unreachable",
        "ollama": "unreachable"
    }
    
    # 1. Check Postgres/SQLite connection
    try:
        db.execute(text("SELECT 1"))
        status["postgres"] = "healthy"
    except Exception as e:
        status["status"] = "unhealthy"
        status["postgres"] = f"unhealthy: {e}"
        
    # 2. Check Redis connection
    try:
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        status["redis"] = "healthy"
    except Exception as e:
        status["status"] = "unhealthy"
        status["redis"] = f"unhealthy: {e}"
        
    # 3. Check Ollama service availability
    try:
        resp = httpx.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            status["ollama"] = "healthy"
        else:
            status["ollama"] = f"unhealthy: status_code={resp.status_code}"
    except Exception as e:
        status["ollama"] = f"unhealthy: {e}"
        
    return status
