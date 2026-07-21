import os
import sys
import json
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import redis
from celery.utils.log import get_task_logger

# Add necessary paths for imports at the end of sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from workers.celery_app import celery_app
from config.settings import settings
from models.database import SessionLocal
from models import orm
from MLRL01.api_interface import MLRL01Engine
from MLRL01.risk.risk_manager import RiskManager

logger = get_task_logger(__name__)

def update_job_status(db, job_id, status, error_msg=None, progress=None):
    job = db.query(orm.TrainingJob).filter(orm.TrainingJob.id == job_id).first()
    if job:
        job.status = status
        if status == "running" and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status in ["completed", "failed"]:
            job.completed_at = datetime.utcnow()
            
        if error_msg:
            job.error_message = error_msg
        if progress is not None:
            job.progress_pct = progress
            
        db.commit()
        
        # Publish event to Redis
        try:
            r = redis.from_url(settings.REDIS_URL)
            r.publish(
                f"training_status:{job_id}",
                json.dumps({
                    "job_id": str(job_id),
                    "status": status,
                    "progress_pct": job.progress_pct,
                    "error_message": error_msg
                })
            )
        except Exception as e:
            logger.error(f"Failed to publish status update to Redis: {e}")

@celery_app.task(name="workers.muscle_tasks.task_train_ml_models")
def task_train_ml_models(experiment_id, job_id):
    logger.info(f"Starting ML training task for job: {job_id}")
    db = SessionLocal()
    try:
        update_job_status(db, job_id, "running", progress=10.0)
        
        # Fetch experiment config
        config_obj = db.query(orm.ExperimentConfig).filter(orm.ExperimentConfig.experiment_id == experiment_id).first()
        if not config_obj:
            raise ValueError(f"No configuration found for experiment {experiment_id}")
            
        # Build engine config
        config = {
            "target_horizon": config_obj.target_horizon,
            "target_method": config_obj.target_method,
            "train_ratio": config_obj.train_ratio,
            "embargo_bars": config_obj.embargo_bars,
            "initial_capital": config_obj.initial_capital,
            "fee_rate": config_obj.fee_rate,
            "spread_cost": config_obj.spread_cost,
            "slippage": config_obj.slippage,
            "data_dir": "jupiter"
        }
        
        engine = MLRL01Engine(config)
        update_job_status(db, job_id, "running", progress=30.0)
        
        # Load and process data
        df = engine.load_and_prepare_data()
        update_job_status(db, job_id, "running", progress=50.0)
        
        # Train ML models
        results_df, predictions, close_test, dates_test = engine.train_ml_models(df)
        update_job_status(db, job_id, "running", progress=75.0)
        
        # Run backtest for each model
        bt_results = engine.run_backtest(predictions, close_test, dates_test)
        
        # Save results to database
        for _, row in results_df.iterrows():
            model_name = row["Model"]
            accuracy = float(row["Accuracy"])
            precision = float(row["Precision"])
            recall = float(row["Recall"])
            
            bt_stats = bt_results.get(model_name, {})
            
            ml_res = orm.MLResult(
                job_id=job_id,
                model_name=model_name,
                accuracy=accuracy,
                precision_score=precision,
                recall_score=recall,
                bt_return=bt_stats.get("total_return"),
                bt_sharpe=bt_stats.get("sharpe_ratio"),
                bt_max_dd=bt_stats.get("max_drawdown"),
                bt_trades=bt_stats.get("n_trades"),
                bt_costs=bt_stats.get("total_costs")
            )
            db.add(ml_res)
            
        update_job_status(db, job_id, "completed", progress=100.0)
        logger.info(f"ML training complete for job: {job_id}")
        
    except Exception as e:
        error_msg = f"ML task failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        update_job_status(db, job_id, "failed", error_msg=error_msg)
    finally:
        db.close()

@celery_app.task(name="workers.muscle_tasks.task_train_rl_agent")
def task_train_rl_agent(experiment_id, job_id):
    logger.info(f"Starting RL training task for job: {job_id}")
    db = SessionLocal()
    try:
        update_job_status(db, job_id, "running", progress=5.0)
        
        config_obj = db.query(orm.ExperimentConfig).filter(orm.ExperimentConfig.experiment_id == experiment_id).first()
        if not config_obj:
            raise ValueError(f"No configuration found for experiment {experiment_id}")
            
        config = {
            "target_horizon": config_obj.target_horizon,
            "target_method": config_obj.target_method,
            "train_ratio": config_obj.train_ratio,
            "embargo_bars": config_obj.embargo_bars,
            "rl_algorithm": config_obj.rl_algorithm,
            "rl_timesteps": config_obj.rl_timesteps,
            "learning_rate": config_obj.learning_rate,
            "batch_size": config_obj.batch_size,
            "gamma": config_obj.gamma,
            "gae_lambda": config_obj.gae_lambda,
            "clip_range": config_obj.clip_range,
            "ent_coef": config_obj.ent_coef,
            "lstm_hidden": config_obj.lstm_hidden,
            "initial_capital": config_obj.initial_capital,
            "fee_rate": config_obj.fee_rate,
            "spread_cost": config_obj.spread_cost,
            "slippage": config_obj.slippage,
            "data_dir": "jupiter"
        }
        
        engine = MLRL01Engine(config)
        update_job_status(db, job_id, "running", progress=10.0)
        
        df = engine.load_and_prepare_data()
        update_job_status(db, job_id, "running", progress=15.0)
        
        # Train agent with Redis progress callback
        rl_equity, rl_stats, rl_log, agent = engine.train_rl_agent(
            df, job_id=job_id, redis_url=settings.REDIS_URL
        )
        
        # Compute metrics
        metrics = RiskManager.compute_all_metrics(rl_equity)
        
        # Save RL Results
        rl_res = orm.RLResult(
            job_id=job_id,
            total_return=float(metrics["total_return"]),
            sharpe_ratio=float(metrics["sharpe_ratio"]),
            sortino_ratio=float(metrics["sortino_ratio"]),
            max_drawdown=float(metrics["max_drawdown"]),
            calmar_ratio=float(metrics["calmar_ratio"]),
            volatility=float(metrics["volatility"]),
            n_trades=int(rl_stats.get("total_trades", 0)),
            total_costs=float(rl_stats.get("total_costs", 0.0)),
            killed=bool(rl_stats.get("killed", False)),
            equity_curve=rl_equity.tolist(),
            trade_log=rl_log
        )
        db.add(rl_res)
        
        # Save agent zip artifact
        model_filename = f"ppo_agent_{job_id}.zip"
        model_path = os.path.join(settings.ARTIFACTS_DIR, model_filename)
        agent.save(model_path)
        
        artifact = orm.ModelArtifact(
            job_id=job_id,
            artifact_type="ppo_model",
            filename=model_filename,
            storage_path=model_path,
            size_bytes=os.path.getsize(model_path) if os.path.exists(model_path) else 0
        )
        db.add(artifact)
        
        # Proactively run Monte Carlo simulator on RL results if enabled
        if config.get("rl_timesteps", 100000) > 0:
            mc_results = engine.run_monte_carlo(rl_equity)
            for method, report in mc_results.items():
                mc_res = orm.MonteCarloResult(
                    job_id=job_id,
                    method=method,
                    n_simulations=report.get("n_simulations", 1000),
                    mean_return=report.get("mean_return"),
                    median_return=report.get("median_return"),
                    prob_positive=report.get("prob_positive"),
                    prob_ruin_10=report.get("prob_ruin_10pct"),
                    prob_ruin_20=report.get("prob_ruin_20pct"),
                    mean_max_dd=report.get("mean_max_dd"),
                    worst_max_dd=report.get("worst_max_dd"),
                    mean_sharpe=report.get("mean_sharpe")
                )
                db.add(mc_res)
                
        update_job_status(db, job_id, "completed", progress=100.0)
        logger.info(f"RL training complete for job: {job_id}")
        
    except Exception as e:
        error_msg = f"RL task failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        update_job_status(db, job_id, "failed", error_msg=error_msg)
    finally:
        db.close()
