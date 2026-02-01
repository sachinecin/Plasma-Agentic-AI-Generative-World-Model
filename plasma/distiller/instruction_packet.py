"""
Instruction Packet - Lightweight LoRA adaptation units
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class InstructionPacket:
    """
    Lightweight instruction packet for LoRA adaptation
    
    These packets contain minimal adaptation instructions that can be
    deployed to edge devices for real-time model updates.
    """
    packet_id: str
    timestamp: datetime
    lora_weights: Dict[str, Any]
    target_layers: List[str]
    priority: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert packet to dictionary format"""
        return {
            "packet_id": self.packet_id,
            "timestamp": self.timestamp.isoformat(),
            "lora_weights": self.lora_weights,
            "target_layers": self.target_layers,
            "priority": self.priority,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructionPacket":
        """Create packet from dictionary"""
        return cls(
            packet_id=data["packet_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            lora_weights=data["lora_weights"],
            target_layers=data["target_layers"],
            priority=data["priority"],
            metadata=data["metadata"]
        )
        
    def size_bytes(self) -> int:
        """Estimate packet size in bytes"""
        # Simplified size estimation
        import json
        return len(json.dumps(self.to_dict()).encode('utf-8'))


class InstructionPacketBuilder:
    """Builder for creating instruction packets"""
    
    def __init__(self):
        self.packet_count = 0
        
    def build_packet(self,
                    lora_weights: Dict[str, Any],
                    target_layers: List[str],
                    priority: int = 1,
                    metadata: Optional[Dict[str, Any]] = None) -> InstructionPacket:
        """
        Build a new instruction packet
        
        Args:
            lora_weights: LoRA adaptation weights
            target_layers: Model layers to apply adaptation
            priority: Packet priority (higher = more important)
            metadata: Optional metadata
            
        Returns:
            InstructionPacket instance
        """
        self.packet_count += 1
        
        return InstructionPacket(
            packet_id=f"packet_{self.packet_count}",
            timestamp=datetime.now(),
            lora_weights=lora_weights,
            target_layers=target_layers,
            priority=priority,
            metadata=metadata or {}
        )
