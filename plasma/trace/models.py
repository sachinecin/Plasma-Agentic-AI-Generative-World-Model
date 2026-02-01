"""
Pydantic Models for State Tracking

Type-safe models for atomic spans and trace state.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class SpanStatus(str, Enum):
    """Status of an atomic span"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SpanMetadata(BaseModel):
    """Metadata for an atomic span"""
    span_id: str = Field(..., description="Unique identifier for the span")
    parent_id: Optional[str] = Field(None, description="Parent span ID if nested")
    name: str = Field(..., description="Human-readable span name")
    tags: Dict[str, str] = Field(default_factory=dict, description="Custom tags")
    
    class Config:
        frozen = False


class AtomicSpanData(BaseModel):
    """Data model for an atomic execution span"""
    span_id: str
    parent_id: Optional[str] = None
    name: str
    status: SpanStatus = SpanStatus.PENDING
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    state: Dict[str, Any] = Field(default_factory=dict)
    metadata: SpanMetadata
    error: Optional[str] = None
    
    class Config:
        use_enum_values = True


class TraceState(BaseModel):
    """Complete trace state with all spans"""
    trace_id: str = Field(..., description="Unique trace identifier")
    root_span_id: str = Field(..., description="Root span of the trace")
    spans: Dict[str, AtomicSpanData] = Field(
        default_factory=dict,
        description="All spans in the trace"
    )
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.ACTIVE
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        
    def add_span(self, span: AtomicSpanData) -> None:
        """Add a span to the trace"""
        self.spans[span.span_id] = span
        
    def get_span(self, span_id: str) -> Optional[AtomicSpanData]:
        """Get a span by ID"""
        return self.spans.get(span_id)
        
    def get_active_spans(self) -> List[AtomicSpanData]:
        """Get all active spans"""
        return [
            span for span in self.spans.values()
            if span.status == SpanStatus.ACTIVE
        ]
        
    def get_completed_spans(self) -> List[AtomicSpanData]:
        """Get all completed spans"""
        return [
            span for span in self.spans.values()
            if span.status == SpanStatus.COMPLETED
        ]


class StateSnapshot(BaseModel):
    """Snapshot of state at a point in time"""
    snapshot_id: str
    timestamp: datetime
    state_data: Dict[str, Any]
    span_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        frozen = True
