import uuid
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, ForeignKey, 
    Text, BigInteger, JSON, text, Uuid
)

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), default="created")  # created, queued, running, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    config = relationship("ExperimentConfig", uselist=False, back_populates="experiment", cascade="all, delete-orphan")
    jobs = relationship("TrainingJob", back_populates="experiment", cascade="all, delete-orphan")

class ExperimentConfig(Base):
    __tablename__ = "experiment_configs"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    experiment_id = Column(Uuid, ForeignKey("experiments.id", ondelete="CASCADE"), unique=True)
    
    # ML Config
    target_horizon = Column(Integer, default=5)
    target_method = Column(String(50), default="triple_barrier")
    train_ratio = Column(Float, default=0.80)
    embargo_bars = Column(Integer, default=60)
    
    # RL Config
    rl_algorithm = Column(String(50), default="RecurrentPPO")
    rl_timesteps = Column(Integer, default=100000)
    learning_rate = Column(Float, default=0.0003)
    batch_size = Column(Integer, default=64)
    gamma = Column(Float, default=0.99)
    gae_lambda = Column(Float, default=0.95)
    clip_range = Column(Float, default=0.2)
    ent_coef = Column(Float, default=0.01)
    lstm_hidden = Column(Integer, default=128)
    
    # Env Config
    initial_capital = Column(Float, default=100000.0)
    fee_rate = Column(Float, default=0.0001)
    spread_cost = Column(Float, default=0.0003)
    slippage = Column(Float, default=0.0002)

    experiment = relationship("Experiment", back_populates="config")

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    experiment_id = Column(Uuid, ForeignKey("experiments.id", ondelete="CASCADE"))
    celery_task_id = Column(String(255), nullable=True)
    job_type = Column(String(20), nullable=False)  # ml, rl, walk_forward, monte_carlo
    status = Column(String(20), default="queued")  # queued, running, completed, failed
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    progress_pct = Column(Float, default=0.0)

    experiment = relationship("Experiment", back_populates="jobs")
    ml_results = relationship("MLResult", back_populates="job", cascade="all, delete-orphan")
    rl_results = relationship("RLResult", back_populates="job", cascade="all, delete-orphan")
    logs = relationship("TrainingLog", back_populates="job", cascade="all, delete-orphan")
    artifacts = relationship("ModelArtifact", back_populates="job", cascade="all, delete-orphan")
    mc_results = relationship("MonteCarloResult", back_populates="job", cascade="all, delete-orphan")

class MLResult(Base):
    __tablename__ = "ml_results"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id = Column(Uuid, ForeignKey("training_jobs.id", ondelete="CASCADE"))
    model_name = Column(String(100), nullable=False)
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall_score = Column(Float, nullable=True)
    bt_return = Column(Float, nullable=True)
    bt_sharpe = Column(Float, nullable=True)
    bt_max_dd = Column(Float, nullable=True)
    bt_trades = Column(Integer, nullable=True)
    bt_costs = Column(Float, nullable=True)

    job = relationship("TrainingJob", back_populates="ml_results")

class RLResult(Base):
    __tablename__ = "rl_results"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id = Column(Uuid, ForeignKey("training_jobs.id", ondelete="CASCADE"))
    total_return = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    n_trades = Column(Integer, nullable=True)
    total_costs = Column(Float, nullable=True)
    killed = Column(Boolean, default=False)
    equity_curve = Column(JSON, nullable=True)  # list of floats
    trade_log = Column(JSON, nullable=True)     # list of dicts

    job = relationship("TrainingJob", back_populates="rl_results")

class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    job_id = Column(Uuid, ForeignKey("training_jobs.id", ondelete="CASCADE"))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    level = Column(String(10), default="INFO")
    message = Column(Text, nullable=False)
    metadata_json = Column(JSON, name="metadata", nullable=True)

    job = relationship("TrainingJob", back_populates="logs")

class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id = Column(Uuid, ForeignKey("training_jobs.id", ondelete="CASCADE"))
    artifact_type = Column(String(50), nullable=False)  # ppo_model, scaler, plot, report
    filename = Column(String(255), nullable=False)
    storage_path = Column(Text, nullable=False)
    size_bytes = Column(BigInteger, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("TrainingJob", back_populates="artifacts")

class MonteCarloResult(Base):
    __tablename__ = "monte_carlo_results"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id = Column(Uuid, ForeignKey("training_jobs.id", ondelete="CASCADE"))
    method = Column(String(50), nullable=False)  # block_bootstrap, stress_test, perturbation
    n_simulations = Column(Integer, default=1000)
    mean_return = Column(Float, nullable=True)
    median_return = Column(Float, nullable=True)
    prob_positive = Column(Float, nullable=True)
    prob_ruin_10 = Column(Float, nullable=True)
    prob_ruin_20 = Column(Float, nullable=True)
    mean_max_dd = Column(Float, nullable=True)
    worst_max_dd = Column(Float, nullable=True)
    mean_sharpe = Column(Float, nullable=True)

    job = relationship("TrainingJob", back_populates="mc_results")

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(Uuid, ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role = Column(String(10), nullable=False)  # user, ai
    content = Column(Text, nullable=False)
    metadata_json = Column(JSON, name="metadata", nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")
