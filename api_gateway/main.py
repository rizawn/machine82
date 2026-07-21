import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from routers import health, experiments, training, metrics, chat, system
from models.database import engine, Base
from websocket.broadcaster import manager, redis_listener

# Auto-create tables (for Postgres/local database)
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background Redis pub/sub listener on startup
    listener_task = asyncio.create_task(redis_listener())
    yield
    # Cancel background task on shutdown
    listener_task.cancel()
    try:
        await listener_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="The Trinity API Gateway",
    description="Unified API gateway for MLRL01, MLRL02, and MLRL03",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(system.router, prefix="/api")

@app.websocket("/ws/training/{job_id}")
async def websocket_training_logs(websocket: WebSocket, job_id: str):
    await manager.connect(job_id, websocket)
    try:
        while True:
            # Just keep the WebSocket channel alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(job_id, websocket)
    except Exception as e:
        print(f"[WS] WebSocket error: {e}")
        manager.disconnect(job_id, websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
