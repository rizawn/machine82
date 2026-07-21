
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime
from models.database import get_db
from models import orm, schemas
from workers.celery_app import celery_app
from workers import muscle_tasks, brain_tasks

router = APIRouter(prefix="/training", tags=["training"])

@router.post("/{experiment_id}/start", response_model=schemas.TrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_training_job(
    experiment_id: UUID,
    job_type: str = Query(..., pattern="^(ml|rl)$"),
    db: Session = Depends(get_db)
):
    # Check if experiment exists
    exp = db.query(orm.Experiment).filter(orm.Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    # Create database entry for job
    job = orm.TrainingJob(
        experiment_id=experiment_id,
        job_type=job_type,
        status="queued",
        progress_pct=0.0
    )
    db.add(job)
    db.flush()  # to get job.id
    
    # Enqueue task in Celery based on job_type
    celery_task_id = None
    if job_type == "ml":
        task = muscle_tasks.task_train_ml_models.delay(str(experiment_id), str(job.id))
        celery_task_id = task.id
    elif job_type == "rl":
        task = muscle_tasks.task_train_rl_agent.delay(str(experiment_id), str(job.id))
        celery_task_id = task.id
        
    # Update job with Celery task ID and update experiment status
    job.celery_task_id = celery_task_id
    exp.status = "running"
    db.commit()
    db.refresh(job)
    
    return job

@router.get("/jobs/{job_id}", response_model=schemas.TrainingJobResponse)
def get_job_status(job_id: UUID, db: Session = Depends(get_db)):
    job = db.query(orm.TrainingJob).filter(orm.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job

@router.post("/jobs/{job_id}/kill", response_model=schemas.TrainingJobResponse)
def kill_training_job(job_id: UUID, db: Session = Depends(get_db)):
    job = db.query(orm.TrainingJob).filter(orm.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
        
    if job.status in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Job has already finished")
        
    # Revoke Celery task
    if job.celery_task_id:
        celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGKILL")
        
    # Update database
    job.status = "failed"
    job.error_message = "Killed by user request"
    job.completed_at = datetime.utcnow()
    
    exp = db.query(orm.Experiment).filter(orm.Experiment.id == job.experiment_id).first()
    if exp:
        exp.status = "failed"
        
    db.commit()
    db.refresh(job)
    return job
