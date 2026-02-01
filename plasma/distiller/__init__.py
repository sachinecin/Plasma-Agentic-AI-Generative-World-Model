"""
LoRA 'Instruction-Packet' Injection Logic

This module implements lightweight instruction packet injection using LoRA
(Low-Rank Adaptation) for real-time edge adaptation.
"""

from plasma.distiller.lora_injector import LoRAInjector
from plasma.distiller.instruction_packet import InstructionPacket
from plasma.distiller.distiller import Distiller

__all__ = ["LoRAInjector", "InstructionPacket", "Distiller"]
