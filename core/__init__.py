"""
Core simulation engine for Plasma Agentic AI.

This module contains the fundamental simulation infrastructure that powers
the Training-Agent Disaggregation architecture.
"""

__version__ = "0.1.0"

from .simulation_engine import SimulationEngine
from .config import CoreConfig

__all__ = [
    "SimulationEngine",
    "CoreConfig",
]
