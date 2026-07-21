from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from uuid import UUID, uuid4
from typing import List, Optional
from models.database import get_db
from models import orm, schemas
from workers.brain_tasks import task_chat

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/sessions", response_model=schemas.ChatSessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(db: Session = Depends(get_db)):
    session = orm.ChatSession(id=uuid4(), title="New Chat Session")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

@router.get("/sessions", response_model=List[schemas.ChatSessionResponse])
def list_sessions(db: Session = Depends(get_db)):
    return db.query(orm.ChatSession).order_by(orm.ChatSession.created_at.desc()).all()

@router.get("/sessions/{session_id}", response_model=schemas.ChatSessionResponse)
def get_session(session_id: UUID, db: Session = Depends(get_db)):
    session = db.query(orm.ChatSession).filter(orm.ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session

@router.post("/sessions/{session_id}/message", response_model=schemas.ChatMessageResponse)
def send_message(
    session_id: UUID,
    payload: schemas.ChatMessageCreate,
    experiment_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db)
):
    session = db.query(orm.ChatSession).filter(orm.ChatSession.id == session_id).first()
    if not session:
        # Auto-create session if it doesn't exist
        session = orm.ChatSession(id=session_id, title=payload.content[:50])
        db.add(session)
        db.commit()
        
    # Check if this is the first message; if so, update session title
    msg_count = db.query(orm.ChatMessage).filter(orm.ChatMessage.session_id == session_id).count()
    if msg_count == 0:
        session.title = payload.content[:50]
        db.commit()
        
    # Execute the chat logic inline to return the AI message synchronously
    task_chat(
        str(session_id),
        payload.content,
        str(experiment_id) if experiment_id else None
    )
    
    # Query the newly inserted AI message (the latest one for this session)
    ai_msg = db.query(orm.ChatMessage)\
        .filter(orm.ChatMessage.session_id == session_id, orm.ChatMessage.role == "ai")\
        .order_by(orm.ChatMessage.id.desc())\
        .first()
        
    if not ai_msg:
        raise HTTPException(status_code=500, detail="Failed to get AI response")
        
    return ai_msg
