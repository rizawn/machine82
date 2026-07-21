from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import List, Dict, Optional, Any

class ExperimentConfigBase(BaseModel):
    # ML Config
    target_horizon: int = Field(5, ge=1, le=20)
    target_method: str = Field("triple_barrier", pattern="^(triple_barrier|threshold|binary)$")
    train_ratio: float = Field(0.80, ge=0.5, le=0.95)
    embargo_bars: int = Field(60, ge=10, le=200)
    
    # RL Config
    rl_algorithm: str = Field("RecurrentPPO", pattern="^(PPO|RecurrentPPO)$")
    rl_timesteps: int = Field(100000, ge=10000, le=10000000)
    learning_rate: float = Field(0.0003, ge=1e-6, le=1e-2)
    batch_size: int = Field(64, description="Must be power of 2: 32, 64, 128, 256")
    gamma: float = Field(0.99, ge=0.9, le=0.999)
    gae_lambda: float = Field(0.95, ge=0.8, le=1.0)
    clip_range: float = Field(0.2, ge=0.1, le=0.4)
    ent_coef: float = Field(0.01, ge=0.0, le=0.1)
    lstm_hidden: int = Field(128, description="LSTM hidden size: 64, 128, 256")
    
    # Env Config
    initial_capital: float = Field(100000.0, ge=1000.0, le=10000000.0)
    fee_rate: float = Field(0.0001, ge=0.0, le=0.01)
    spread_cost: float = Field(0.0003, ge=0.0, le=0.01)
    slippage: float = Field(0.0002, ge=0.0, le=0.01)

class ExperimentConfigResponse(ExperimentConfigBase):
    id: UUID
    experiment_id: UUID

    class Config:
        from_attributes = True

class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    config: Optional[ExperimentConfigBase] = None

class ExperimentResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime
    config: Optional[ExperimentConfigResponse] = None

    class Config:
        from_attributes = True

class TrainingJobResponse(BaseModel):
    id: UUID
    experiment_id: UUID
    celery_task_id: Optional[str] = None
    job_type: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_pct: float

    class Config:
        from_attributes = True

class MLResultResponse(BaseModel):
    id: UUID
    job_id: UUID
    model_name: str
    accuracy: Optional[float] = None
    precision_score: Optional[float] = None
    recall_score: Optional[float] = None
    bt_return: Optional[float] = None
    bt_sharpe: Optional[float] = None
    bt_max_dd: Optional[float] = None
    bt_trades: Optional[int] = None
    bt_costs: Optional[float] = None

    class Config:
        from_attributes = True

class RLResultResponse(BaseModel):
    id: UUID
    job_id: UUID
    total_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    calmar_ratio: Optional[float] = None
    volatility: Optional[float] = None
    n_trades: Optional[int] = None
    total_costs: Optional[float] = None
    killed: bool
    equity_curve: Optional[List[float]] = None
    trade_log: Optional[List[Dict[str, Any]]] = None

    class Config:
        from_attributes = True

class MonteCarloResultResponse(BaseModel):
    id: UUID
    job_id: UUID
    method: str
    n_simulations: int
    mean_return: Optional[float] = None
    median_return: Optional[float] = None
    prob_positive: Optional[float] = None
    prob_ruin_10: Optional[float] = None
    prob_ruin_20: Optional[float] = None
    mean_max_dd: Optional[float] = None
    worst_max_dd: Optional[float] = None
    mean_sharpe: Optional[float] = None

    class Config:
        from_attributes = True

class ChatMessageCreate(BaseModel):
    content: str

class ChatMessageResponse(BaseModel):
    id: int
    session_id: UUID
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ChatSessionResponse(BaseModel):
    id: UUID
    title: Optional[str] = None
    created_at: datetime
    messages: List[ChatMessageResponse] = []

    class Config:
        from_attributes = True
