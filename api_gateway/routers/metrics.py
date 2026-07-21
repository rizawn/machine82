from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from uuid import UUID
import os
from models.database import get_db
from models import orm

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/experiment/{experiment_id}")
def get_experiment_metrics(experiment_id: UUID, db: Session = Depends(get_db)):
    # Fetch experiment config
    config = db.query(orm.ExperimentConfig).filter(orm.ExperimentConfig.experiment_id == experiment_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Experiment configuration not found")
        
    # Get all training jobs associated with this experiment
    jobs = db.query(orm.TrainingJob).filter(orm.TrainingJob.experiment_id == experiment_id).all()
    job_ids = [j.id for j in jobs]
    
    # Fetch results
    ml_results = db.query(orm.MLResult).filter(orm.MLResult.job_id.in_(job_ids)).all() if job_ids else []
    rl_results = db.query(orm.RLResult).filter(orm.RLResult.job_id.in_(job_ids)).all() if job_ids else []
    mc_results = db.query(orm.MonteCarloResult).filter(orm.MonteCarloResult.job_id.in_(job_ids)).all() if job_ids else []
    artifacts = db.query(orm.ModelArtifact).filter(orm.ModelArtifact.job_id.in_(job_ids)).all() if job_ids else []
    
    return {
        "experiment_id": str(experiment_id),
        "config": {
            "target_horizon": config.target_horizon,
            "target_method": config.target_method,
            "train_ratio": config.train_ratio,
            "embargo_bars": config.embargo_bars,
            "rl_algorithm": config.rl_algorithm,
            "rl_timesteps": config.rl_timesteps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_range": config.clip_range,
            "ent_coef": config.ent_coef,
            "lstm_hidden": config.lstm_hidden,
            "initial_capital": config.initial_capital,
            "fee_rate": config.fee_rate,
            "spread_cost": config.spread_cost,
            "slippage": config.slippage
        },
        "jobs": [
            {
                "id": str(j.id),
                "job_type": j.job_type,
                "status": j.status,
                "started_at": j.started_at,
                "completed_at": j.completed_at,
                "progress_pct": j.progress_pct,
                "error_message": j.error_message
            } for j in jobs
        ],
        "ml_results": [
            {
                "model_name": m.model_name,
                "accuracy": m.accuracy,
                "precision": m.precision_score,
                "recall": m.recall_score,
                "bt_return": m.bt_return,
                "bt_sharpe": m.bt_sharpe,
                "bt_max_dd": m.bt_max_dd,
                "bt_trades": m.bt_trades,
                "bt_costs": m.bt_costs
            } for m in ml_results
        ],
        "rl_results": [
            {
                "job_id": str(r.job_id),
                "total_return": r.total_return,
                "sharpe_ratio": r.sharpe_ratio,
                "sortino_ratio": r.sortino_ratio,
                "max_drawdown": r.max_drawdown,
                "calmar_ratio": r.calmar_ratio,
                "volatility": r.volatility,
                "n_trades": r.n_trades,
                "total_costs": r.total_costs,
                "killed": r.killed,
                "equity_curve": r.equity_curve,
                "trade_log": r.trade_log
            } for r in rl_results
        ],
        "monte_carlo_results": [
            {
                "method": m.method,
                "n_simulations": m.n_simulations,
                "mean_return": m.mean_return,
                "median_return": m.median_return,
                "prob_positive": m.prob_positive,
                "prob_ruin_10": m.prob_ruin_10,
                "prob_ruin_20": m.prob_ruin_20,
                "mean_max_dd": m.mean_max_dd,
                "worst_max_dd": m.worst_max_dd,
                "mean_sharpe": m.mean_sharpe
            } for m in mc_results
        ],
        "artifacts": [
            {
                "id": str(art.id),
                "job_id": str(art.job_id),
                "artifact_type": art.artifact_type,
                "filename": art.filename,
                "size_bytes": art.size_bytes,
                "created_at": art.created_at
            } for art in artifacts
        ]
    }

@router.get("/artifact/{artifact_id}/download")
def download_artifact(artifact_id: UUID, db: Session = Depends(get_db)):
    art = db.query(orm.ModelArtifact).filter(orm.ModelArtifact.id == artifact_id).first()
    if not art:
        raise HTTPException(status_code=404, detail="Artifact not found")
        
    if not os.path.exists(art.storage_path):
        raise HTTPException(status_code=404, detail="Physical file not found on disk")
        
    return FileResponse(
        path=art.storage_path,
        filename=art.filename,
        media_type="application/octet-stream"
    )
