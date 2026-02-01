"""
Configuration for Instruction Packets.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PacketConfig:
    """Configuration for instruction packet compilation and deployment."""
    
    # Compilation settings
    target_device: str = "edge"  # "edge", "mobile", "server"
    compression_level: int = 5  # 1-10, higher = smaller but slower
    quantization: str = "int8"  # "float32", "float16", "int8", "int4"
    
    # Runtime settings
    max_inference_time_ms: float = 10.0
    batch_size: int = 1
    enable_caching: bool = True
    
    # LoRA settings
    lora_rank: int = 8
    lora_alpha: float = 16.0
    
    # Optimization settings
    optimize_for_latency: bool = True
    enable_pruning: bool = False
    pruning_threshold: float = 0.01
