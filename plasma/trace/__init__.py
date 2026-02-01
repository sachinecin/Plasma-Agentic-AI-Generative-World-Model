"""
Pydantic-based State Tracking for Atomic Spans

This module implements state tracking using Pydantic models for type-safe
atomic span management and execution tracing.
"""

from plasma.trace.state_tracker import StateTracker
from plasma.trace.atomic_span import AtomicSpan
from plasma.trace.models import TraceState, SpanMetadata

__all__ = ["StateTracker", "AtomicSpan", "TraceState", "SpanMetadata"]
