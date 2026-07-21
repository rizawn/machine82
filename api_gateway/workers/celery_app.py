from celery import Celery
import redis
from config.settings import settings

# Detect if Redis is running, fallback to SQLite otherwise
broker_url = settings.REDIS_URL
backend_url = settings.REDIS_URL
try:
    r = redis.from_url(settings.REDIS_URL)
    r.ping()
    print("[CELERY] Connected to Redis. Using Redis broker.")
except Exception:
    broker_url = "sqla+sqlite:///celery_broker.db"
    backend_url = "db+sqlite:///celery_results.db"
    print("[CELERY] Redis is offline. Falling back to SQLite broker.")

celery_app = Celery(
    "trinity",
    broker=broker_url,
    backend=backend_url,
    include=[
        "workers.muscle_tasks",
        "workers.brain_tasks"
    ]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Route tasks to specific queues
    task_routes={
        "workers.muscle_tasks.*": {"queue": "muscle"},
        "workers.brain_tasks.*": {"queue": "brain"},
    },
    worker_max_tasks_per_child=10,
    worker_max_memory_per_child=512000  # 512MB limit per worker
)
