"""
State Tracker - Main state tracking orchestrator with Pydantic
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from plasma.trace.models import TraceState, SpanStatus, StateSnapshot
from plasma.trace.atomic_span import AtomicSpan, SpanFactory


class StateTracker:
    """
    Pydantic-based state tracker for atomic spans
    
    Provides high-performance async state tracking with full type safety
    using Pydantic models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.span_factory = SpanFactory()
        self.traces: Dict[str, TraceState] = {}
        self.trace_counter = 0
        self._running = False
        
    async def start(self) -> None:
        """Start the state tracker"""
        self._running = True
        
    async def stop(self) -> None:
        """Stop the state tracker"""
        self._running = False
        
    def create_trace(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new trace
        
        Args:
            metadata: Optional trace metadata
            
        Returns:
            Trace ID
        """
        self.trace_counter += 1
        trace_id = f"trace_{self.trace_counter}"
        root_span_id = f"{trace_id}_root"
        
        trace = TraceState(
            trace_id=trace_id,
            root_span_id=root_span_id,
            start_time=datetime.now(),
            metadata=metadata or {}
        )
        
        self.traces[trace_id] = trace
        return trace_id
        
    async def create_span(self,
                        trace_id: str,
                        name: str,
                        parent_id: Optional[str] = None,
                        metadata: Optional[Dict[str, str]] = None) -> AtomicSpan:
        """
        Create a new span within a trace
        
        Args:
            trace_id: Trace to add span to
            name: Span name
            parent_id: Optional parent span
            metadata: Optional metadata
            
        Returns:
            New AtomicSpan
        """
        if not self._running:
            await self.start()
            
        trace = self.traces.get(trace_id)
        if not trace:
            raise ValueError(f"Trace {trace_id} not found")
            
        span = self.span_factory.create_span(name, parent_id, metadata)
        trace.add_span(span.get_data())
        
        return span
        
    async def update_span(self,
                        trace_id: str,
                        span: AtomicSpan) -> None:
        """
        Update a span in the trace
        
        Args:
            trace_id: Trace containing the span
            span: Updated span
        """
        trace = self.traces.get(trace_id)
        if not trace:
            raise ValueError(f"Trace {trace_id} not found")
            
        span_data = span.get_data()
        trace.spans[span_data.span_id] = span_data
        
    def get_trace(self, trace_id: str) -> Optional[TraceState]:
        """Get a trace by ID"""
        return self.traces.get(trace_id)
        
    def get_active_traces(self) -> List[TraceState]:
        """Get all active traces"""
        return [
            trace for trace in self.traces.values()
            if trace.status == SpanStatus.ACTIVE
        ]
        
    async def complete_trace(self, trace_id: str) -> None:
        """
        Mark a trace as completed
        
        Args:
            trace_id: Trace to complete
        """
        trace = self.traces.get(trace_id)
        if trace:
            trace.status = SpanStatus.COMPLETED
            trace.end_time = datetime.now()
            
    async def snapshot_state(self,
                           trace_id: str,
                           span_id: str,
                           state_data: Dict[str, Any]) -> StateSnapshot:
        """
        Create a state snapshot
        
        Args:
            trace_id: Trace ID
            span_id: Span ID
            state_data: State data to snapshot
            
        Returns:
            StateSnapshot
        """
        snapshot_id = f"{trace_id}_{span_id}_{datetime.now().timestamp()}"
        
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            state_data=state_data,
            span_id=span_id
        )
        
        return snapshot
        
    def get_trace_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked traces"""
        total_traces = len(self.traces)
        active_traces = len(self.get_active_traces())
        
        total_spans = sum(len(trace.spans) for trace in self.traces.values())
        
        return {
            "total_traces": total_traces,
            "active_traces": active_traces,
            "completed_traces": total_traces - active_traces,
            "total_spans": total_spans,
            "avg_spans_per_trace": total_spans / total_traces if total_traces > 0 else 0
        }
        
    async def export_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        Export trace as dictionary
        
        Args:
            trace_id: Trace to export
            
        Returns:
            Trace data as dict
        """
        trace = self.traces.get(trace_id)
        if not trace:
            raise ValueError(f"Trace {trace_id} not found")
            
        return trace.model_dump()
