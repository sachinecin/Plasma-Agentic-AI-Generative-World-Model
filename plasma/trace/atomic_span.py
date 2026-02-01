"""
Atomic Span - Individual execution unit tracker
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from plasma.trace.models import AtomicSpanData, SpanMetadata, SpanStatus


class AtomicSpan:
    """
    Atomic span for tracking individual execution units
    
    Provides context manager interface for automatic span lifecycle management
    with Pydantic-based type safety.
    """
    
    def __init__(self,
                 span_id: str,
                 name: str,
                 parent_id: Optional[str] = None,
                 metadata: Optional[Dict[str, str]] = None):
        self.span_data = AtomicSpanData(
            span_id=span_id,
            parent_id=parent_id,
            name=name,
            status=SpanStatus.PENDING,
            start_time=datetime.now(),
            metadata=SpanMetadata(
                span_id=span_id,
                parent_id=parent_id,
                name=name,
                tags=metadata or {}
            )
        )
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if exc_type is not None:
            self.fail(str(exc_val))
        else:
            self.complete()
        return False
        
    def start(self) -> None:
        """Start the span"""
        self.span_data.status = SpanStatus.ACTIVE
        self.span_data.start_time = datetime.now()
        
    def complete(self) -> None:
        """Complete the span successfully"""
        self.span_data.status = SpanStatus.COMPLETED
        self.span_data.end_time = datetime.now()
        self._calculate_duration()
        
    def fail(self, error: str) -> None:
        """Fail the span with error"""
        self.span_data.status = SpanStatus.FAILED
        self.span_data.error = error
        self.span_data.end_time = datetime.now()
        self._calculate_duration()
        
    def cancel(self) -> None:
        """Cancel the span"""
        self.span_data.status = SpanStatus.CANCELLED
        self.span_data.end_time = datetime.now()
        self._calculate_duration()
        
    def update_state(self, state: Dict[str, Any]) -> None:
        """Update span state"""
        self.span_data.state.update(state)
        
    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to span metadata"""
        self.span_data.metadata.tags[key] = value
        
    def _calculate_duration(self) -> None:
        """Calculate span duration in milliseconds"""
        if self.span_data.end_time:
            delta = self.span_data.end_time - self.span_data.start_time
            self.span_data.duration_ms = delta.total_seconds() * 1000
            
    def get_data(self) -> AtomicSpanData:
        """Get the span data"""
        return self.span_data
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return self.span_data.model_dump()


class SpanFactory:
    """Factory for creating atomic spans"""
    
    def __init__(self):
        self.span_counter = 0
        
    def create_span(self,
                   name: str,
                   parent_id: Optional[str] = None,
                   metadata: Optional[Dict[str, str]] = None) -> AtomicSpan:
        """
        Create a new atomic span
        
        Args:
            name: Span name
            parent_id: Optional parent span ID
            metadata: Optional metadata tags
            
        Returns:
            New AtomicSpan instance
        """
        self.span_counter += 1
        span_id = f"span_{self.span_counter}"
        
        return AtomicSpan(
            span_id=span_id,
            name=name,
            parent_id=parent_id,
            metadata=metadata
        )
