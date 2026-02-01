"""
State Trace Models using Pydantic for type-safe state management
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
import uuid


class ActionType(str, Enum):
    """Types of actions that can be taken in the world model"""
    MOVE = "move"
    INTERACT = "interact"
    OBSERVE = "observe"
    COMMUNICATE = "communicate"
    COMPUTE = "compute"


class StateTrace(BaseModel):
    """Immutable state trace for recording agent state transitions"""
    model_config = ConfigDict(frozen=True)
    
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state_vector: List[float] = Field(description="Dense state representation")
    action_taken: Optional[ActionType] = None
    reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorldState(BaseModel):
    """Represents the current state of the generative world model"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state_vector: List[float] = Field(description="Current state vector")
    predicted_next_states: List[List[float]] = Field(default_factory=list, description="Predicted future states")
    confidence_scores: List[float] = Field(default_factory=list, description="Confidence for each prediction")
    trace_history: List[StateTrace] = Field(default_factory=list)
    
    def add_trace(self, trace: StateTrace) -> None:
        """Add a new state trace to the history"""
        self.trace_history.append(trace)


class InstructionPacket(BaseModel):
    """LoRA weight instruction packet for edge agent injection"""
    model_config = ConfigDict(frozen=True)
    
    packet_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    lora_weights: Dict[str, List[float]] = Field(description="Layer-wise LoRA weight deltas")
    target_layers: List[str] = Field(description="Target layers for injection")
    adaptation_strength: float = Field(default=0.1, ge=0.0, le=1.0, description="Strength of adaptation")
    context: str = Field(default="", description="Context for this adaptation")
    priority: int = Field(default=5, ge=1, le=10, description="Priority level for injection")


class AuditRecord(BaseModel):
    """Record of an audit check by the Judicial Auditor"""
    model_config = ConfigDict(frozen=True)
    
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state_trace_id: str = Field(description="ID of the state trace being audited")
    anomaly_score: float = Field(ge=0.0, le=1.0, description="Anomaly detection score")
    reward_hacking_detected: bool = False
    adversarial_flags: List[str] = Field(default_factory=list)
    corrective_action: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class SimulationResult(BaseModel):
    """Result of a phantom path simulation"""
    simulation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    initial_state: List[float]
    simulated_trajectory: List[Tuple[List[float], ActionType, float]] = Field(
        description="List of (state, action, reward) tuples"
    )
    total_reward: float = 0.0
    success_probability: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
