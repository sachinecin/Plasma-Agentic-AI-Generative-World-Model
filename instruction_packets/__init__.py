"""
Instruction Packets - Lightweight agent logic for edge deployment.

This module contains the instruction packet compiler and runtime that enables
Training-Agent Disaggregation by creating tiny, deployable agents.
"""

__version__ = "0.1.0"

from .compiler import InstructionPacketCompiler
from .runtime import InstructionPacketRuntime
from .config import PacketConfig

__all__ = [
    "InstructionPacketCompiler",
    "InstructionPacketRuntime",
    "PacketConfig",
]
