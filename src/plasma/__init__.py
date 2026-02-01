"""
Project Plasma - Core Package
A successor to Microsoft's Agent-Lightning with Generative World Models
"""

__version__ = "0.1.0"
__author__ = "Project Plasma Team"

from .phantom_path import PhantomPathSimulator
from .distiller import LoRADistiller
from .judicial_auditor import JudicialAuditor
from .plasma_agent import PlasmaAgent

__all__ = [
    "PhantomPathSimulator",
    "LoRADistiller",
    "JudicialAuditor",
    "PlasmaAgent",
]
