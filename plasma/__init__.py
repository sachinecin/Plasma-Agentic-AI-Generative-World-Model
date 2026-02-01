"""
Project Plasma - Next-gen Agentic AI with Generative World Model

A high-performance framework that replaces heavy RL training with Phantom-Path
Simulations and LoRA Distillation. Features a generative "World Model" for
task practice in a sandbox environment, with real-time edge adaptation.
"""

__version__ = "0.1.0"

from plasma.simulator import Simulator
from plasma.distiller import Distiller
from plasma.auditor import Auditor
from plasma.trace import StateTracker

__all__ = ["Simulator", "Distiller", "Auditor", "StateTracker"]
