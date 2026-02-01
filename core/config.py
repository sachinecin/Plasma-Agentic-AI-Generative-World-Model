"""
Configuration module for the core simulation engine.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CoreConfig:
    """Configuration for core simulation engine."""
    
    # World model settings
    world_model_path: Optional[str] = None
    world_model_device: str = "cuda"
    
    # Simulation settings
    max_simulation_steps: int = 1000
    parallel_simulations: int = 4
    use_adversarial_audit: bool = True
    
    # Training settings
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Performance settings
    enable_mixed_precision: bool = True
    gradient_checkpointing: bool = False
