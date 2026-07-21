from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from models.database import get_db
from models import orm, schemas

router = APIRouter(prefix="/experiments", tags=["experiments"])

@router.post("", response_model=schemas.ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(payload: schemas.ExperimentCreate, db: Session = Depends(get_db)):
    # Create experiment
    experiment = orm.Experiment(
        name=payload.name,
        description=payload.description,
        status="created"
    )
    db.add(experiment)
    db.flush()  # to get experiment.id
    
    # Create config
    config_data = payload.config.model_dump() if payload.config else {}
    config = orm.ExperimentConfig(
        experiment_id=experiment.id,
        **config_data
    )
    db.add(config)
    db.commit()
    db.refresh(experiment)
    return experiment

@router.get("", response_model=List[schemas.ExperimentResponse])
def list_experiments(db: Session = Depends(get_db)):
    return db.query(orm.Experiment).order_by(orm.Experiment.created_at.desc()).all()

@router.get("/{id}", response_model=schemas.ExperimentResponse)
def get_experiment(id: UUID, db: Session = Depends(get_db)):
    exp = db.query(orm.Experiment).filter(orm.Experiment.id == id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(id: UUID, db: Session = Depends(get_db)):
    exp = db.query(orm.Experiment).filter(orm.Experiment.id == id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    db.delete(exp)
    db.commit()
    return None

@router.post("/{id}/clone", response_model=schemas.ExperimentResponse)
def clone_experiment(id: UUID, db: Session = Depends(get_db)):
    exp = db.query(orm.Experiment).filter(orm.Experiment.id == id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    cloned_exp = orm.Experiment(
        name=f"Copy of {exp.name}",
        description=exp.description,
        status="created"
    )
    db.add(cloned_exp)
    db.flush()
    
    if exp.config:
        # Clone configuration
        config_dict = {
            c.name: getattr(exp.config, c.name) 
            for c in exp.config.__table__.columns 
            if c.name not in ["id", "experiment_id"]
        }
        cloned_config = orm.ExperimentConfig(
            experiment_id=cloned_exp.id,
            **config_dict
        )
        db.add(cloned_config)
        
    db.commit()
    db.refresh(cloned_exp)
    return cloned_exp
