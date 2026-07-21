import asyncio
import json
import redis.asyncio as aioredis
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
from config.settings import settings

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)
        print(f"[WS] Client connected to job_id: {job_id}")

    def disconnect(self, job_id: str, websocket: WebSocket):
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        print(f"[WS] Client disconnected from job_id: {job_id}")

    async def broadcast_to_job(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            dead_connections = []
            for ws in list(self.active_connections[job_id]):
                try:
                    await ws.send_json(message)
                except Exception:
                    dead_connections.append(ws)
            for ws in dead_connections:
                self.disconnect(job_id, ws)

manager = ConnectionManager()

async def redis_listener():
    """Asynchronous background loop listening for logs on Redis and broadcasting via WebSockets."""
    print("[WS] Starting Redis pub/sub listener...")
    r = aioredis.from_url(settings.REDIS_URL)
    pubsub = r.pubsub()
    await pubsub.psubscribe("training_logs:*", "training_status:*")
    
    try:
        async for message in pubsub.listen():
            if message["type"] == "pmessage":
                channel = message["channel"].decode("utf-8") if isinstance(message["channel"], bytes) else message["channel"]
                data = message["data"].decode("utf-8") if isinstance(message["data"], bytes) else message["data"]
                
                parts = channel.split(":")
                if len(parts) >= 2:
                    job_id = parts[1]
                    try:
                        payload = json.loads(data)
                        await manager.broadcast_to_job(job_id, payload)
                    except json.JSONDecodeError:
                        await manager.broadcast_to_job(job_id, {"message": data})
    except asyncio.CancelledError:
        print("[WS] Redis pub/sub listener cancelled.")
    except Exception as e:
        print(f"[WS] Redis listener error: {e}")
    finally:
        await pubsub.close()
        await r.close()
