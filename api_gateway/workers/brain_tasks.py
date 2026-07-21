import os
import sys
import json
import traceback
from datetime import datetime
from celery.utils.log import get_task_logger
import redis

# Add necessary paths for imports at the end of sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "MLRL02")))

from workers.celery_app import celery_app
from config.settings import settings
from models.database import SessionLocal
from models import orm

logger = get_task_logger(__name__)

def get_mlrl02_instance():
    """Dynamically imports and boots MLRL02 facade."""
    from core.mlrl02 import MLRL02
    
    # Initialize MLRL02 pointing to its workspace
    workspace = os.path.abspath(os.path.join(settings.MLRL02_PATH, "workspace"))
    system = MLRL02(workspace_dir=workspace, model="deepseek-r1:8b", verbose=False)
    system.boot(verbose=False)
    return system

@celery_app.task(name="workers.brain_tasks.task_analyze_results")
def task_analyze_results(experiment_id, job_id):
    logger.info(f"Starting MLRL02 analysis task for job: {job_id}")
    db = SessionLocal()
    try:
        # Check if job status can be updated
        job = db.query(orm.TrainingJob).filter(orm.TrainingJob.id == job_id).first()
        if job:
            job.status = "running"
            job.started_at = datetime.utcnow()
            db.commit()
            
        # Get experiment results
        rl_res = db.query(orm.RLResult).filter(orm.RLResult.job_id == job.id if job else False).first()
        ml_res = db.query(orm.MLResult).filter(orm.MLResult.job_id == job.id if job else False).all()
        config = db.query(orm.ExperimentConfig).filter(orm.ExperimentConfig.experiment_id == experiment_id).first()
        
        # Prepare context data for LLM
        context_data = {
            "experiment_id": str(experiment_id),
            "hyperparameters": {
                "rl_algorithm": config.rl_algorithm if config else None,
                "learning_rate": config.learning_rate if config else None,
                "gamma": config.gamma if config else None,
                "batch_size": config.batch_size if config else None,
                "entropy_coef": config.ent_coef if config else None
            },
            "rl_metrics": {
                "total_return": rl_res.total_return if rl_res else None,
                "sharpe_ratio": rl_res.sharpe_ratio if rl_res else None,
                "max_drawdown": rl_res.max_drawdown if rl_res else None,
                "volatility": rl_res.volatility if rl_res else None,
                "n_trades": rl_res.n_trades if rl_res else None,
                "killed": rl_res.killed if rl_res else False
            },
            "ml_metrics": [
                {
                    "model_name": m.model_name,
                    "accuracy": m.accuracy,
                    "precision": m.precision_score,
                    "recall": m.recall_score,
                    "bt_return": m.bt_return,
                    "bt_sharpe": m.bt_sharpe
                } for m in ml_res
            ]
        }
        
        # Boot MLRL02 and formulate prompt
        system = get_mlrl02_instance()
        prompt = f"""
        Analyze the following quantitative trading experiment results and provide a structured investment report.
        
        Experiment Configuration & Metrics Context:
        {json.dumps(context_data, indent=2)}
        
        Please structure your report as follows:
        1. **Executive Summary**: General performance assessment of RL and ML models.
        2. **Risk & Drawdown Analysis**: Assess the maximum drawdown, trade counts, volatility, and explain if the RL agent got killed.
        3. **ML vs RL Performance Comparison**: Evaluate the accuracy of the supervised models and compare their returns to the RL agent.
        4. **Hyperparameter Optimizations**: Suggest specific parameter changes based on performance (e.g. learning rate, batch size, entropy).
        """
        
        # Call ChatEngine
        response = system.chat_with_ai(prompt, use_agent=False)
        
        # Save analysis report as a model artifact (markdown file)
        report_filename = f"analysis_report_{job_id}.md"
        report_path = os.path.join(settings.ARTIFACTS_DIR, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(response)
            
        artifact = orm.ModelArtifact(
            job_id=job_id,
            artifact_type="report",
            filename=report_filename,
            storage_path=report_path,
            size_bytes=os.path.getsize(report_path)
        )
        db.add(artifact)
        
        # Save to ChromaDB for future chat contexts
        # Ingest document text manually into Chroma VectorStore
        if system.vector_store and system.vector_store.vectorstore:
            from langchain_core.documents import Document
            doc = Document(
                page_content=response,
                metadata={
                    "source": report_filename,
                    "experiment_id": str(experiment_id),
                    "job_id": str(job_id),
                    "type": "analysis_report",
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            system.vector_store.vectorstore.add_documents([doc])
            logger.info("Report successfully ingested into ChromaDB memory store.")
            
        if job:
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.progress_pct = 100.0
            db.commit()
            
    except Exception as e:
        error_msg = f"Brain analysis task failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        if job:
            job.status = "failed"
            job.error_message = error_msg
            db.commit()
    finally:
        db.close()

@celery_app.task(name="workers.brain_tasks.task_chat")
def task_chat(session_id, message_content, experiment_id=None):
    logger.info(f"Starting chat task for session: {session_id}")
    db = SessionLocal()
    try:
        # Load chat session
        session = db.query(orm.ChatSession).filter(orm.ChatSession.id == session_id).first()
        if not session:
            session = orm.ChatSession(id=session_id, title=message_content[:50])
            db.add(session)
            db.commit()
            
        # Insert user message
        user_msg = orm.ChatMessage(
            session_id=session_id,
            role="user",
            content=message_content
        )
        db.add(user_msg)
        db.commit()
        
        # Boot system
        system = get_mlrl02_instance()
        
        # Inject external PostgreSQL contexts if experiment_id provided
        external_context = ""
        if experiment_id:
            config = db.query(orm.ExperimentConfig).filter(orm.ExperimentConfig.experiment_id == experiment_id).first()
            rl_res = db.query(orm.RLResult).join(orm.TrainingJob).filter(orm.TrainingJob.experiment_id == experiment_id).first()
            if config and rl_res:
                external_context = f"\n[Experiment Context]\nAlgorithm: {config.rl_algorithm}\nTotal Return: {rl_res.total_return:+.2%}\nSharpe Ratio: {rl_res.sharpe_ratio:.3f}\nMax Drawdown: {rl_res.max_drawdown:.2%}\nTrades executed: {rl_res.n_trades}"
        
        # Append external context to the query or prompt
        final_query = message_content
        if external_context:
            final_query += f"\n\nContext regarding active experiment:\n{external_context}"
            
        # Execute chat
        response = system.chat_with_ai(final_query, use_agent=False)
        
        # Save AI message
        ai_msg = orm.ChatMessage(
            session_id=session_id,
            role="ai",
            content=response,
            metadata_json={"experiment_id": str(experiment_id) if experiment_id else None}
        )
        db.add(ai_msg)
        db.commit()
        
        # Publish response to redis chat channel
        try:
            r = redis.from_url(settings.REDIS_URL)
            r.publish(
                f"chat_session:{session_id}",
                json.dumps({
                    "session_id": str(session_id),
                    "role": "ai",
                    "content": response,
                    "created_at": datetime.utcnow().isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Redis chat broadcast failed: {e}")
            
    except Exception as e:
        logger.error(f"Chat task failed: {str(e)}\n{traceback.format_exc()}")
    finally:
        db.close()
