"""
Simulator - Main async simulation orchestrator
"""

import asyncio
from typing import Dict, Any, Optional, List
from plasma.simulator.world_model import WorldModel, WorldState
from plasma.simulator.phantom_path import PhantomPathGenerator, PhantomPath


class Simulator:
    """
    High-performance async simulator orchestrator
    
    Coordinates world model and phantom path generation with asyncio
    event loop for maximum throughput.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.world_model = WorldModel(config.get("world_model", {}))
        self.path_generator = PhantomPathGenerator(
            self.world_model, 
            config.get("path_generator", {})
        )
        self._running = False
        
    async def start(self) -> None:
        """Initialize and start the simulator"""
        await self.world_model.initialize()
        self._running = True
        
    async def stop(self) -> None:
        """Stop the simulator"""
        self._running = False
        
    async def simulate(self, 
                      initial_state: Dict[str, Any],
                      num_paths: int = 10,
                      path_length: int = 100) -> List[PhantomPath]:
        """
        Run simulation to generate phantom paths
        
        Args:
            initial_state: Starting state for simulations
            num_paths: Number of paths to generate
            path_length: Length of each path
            
        Returns:
            List of generated phantom paths
        """
        if not self._running:
            await self.start()
            
        paths = await self.path_generator.generate_batch(
            batch_size=num_paths,
            start_state=initial_state,
            length=path_length
        )
        
        return paths
        
    async def run_episode(self, 
                         initial_state: Dict[str, Any],
                         max_steps: int = 1000) -> PhantomPath:
        """
        Run a single episode simulation
        
        Args:
            initial_state: Starting state
            max_steps: Maximum steps per episode
            
        Returns:
            Generated phantom path for the episode
        """
        if not self._running:
            await self.start()
            
        path = await self.path_generator.generate_path(
            start_state=initial_state,
            length=max_steps
        )
        
        return path
