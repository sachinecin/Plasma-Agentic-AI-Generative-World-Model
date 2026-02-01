"""
World Model - Generative environment for phantom-path simulations.

This module contains the generative models that simulate environments
without requiring real-world interaction.
"""

__version__ = "0.1.0"

from .generative_env import WorldModel
from .config import WorldModelConfig

__all__ = [
    "WorldModel",
    "WorldModelConfig",
]
