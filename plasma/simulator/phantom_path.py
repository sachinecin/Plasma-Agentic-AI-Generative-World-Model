"""
Phantom Path Generator - Creates simulated training trajectories
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PhantomPath:
    """Represents a simulated execution path"""
    path_id: str
    states: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    rewards: List[float]
    metadata: Dict[str, Any]


class PhantomPathGenerator:
    """
    Generates phantom paths for training without actual environment interaction
    
    Uses the world model to create diverse training scenarios that enable
    agents to learn policies without heavy RL training.
    """
    
    def __init__(self, world_model, config: Optional[Dict[str, Any]] = None):
        self.world_model = world_model
        self.config = config or {}
        self.path_count = 0
        
    async def generate_path(self, 
                          start_state: Dict[str, Any],
                          length: int = 10,
                          policy: Optional[Any] = None) -> PhantomPath:
        """
        Generate a single phantom path
        
        Args:
            start_state: Initial state for the path
            length: Number of steps in the path
            policy: Optional policy to guide action selection
            
        Returns:
            PhantomPath: Generated trajectory
        """
        self.path_count += 1
        path_id = f"phantom_{self.path_count}"
        
        states = [start_state]
        actions = []
        rewards = []
        
        current_state = start_state
        
        for _ in range(length):
            # Generate action (simplified - would use policy in practice)
            action = {"action_type": "step", "value": 1.0}
            actions.append(action)
            
            # Simulate next state using world model
            next_state_obj = await self.world_model.generate_state(action)
            next_state = next_state_obj.state_vector
            states.append(next_state)
            
            # Compute reward (simplified)
            reward = 1.0 if "success" in next_state else 0.0
            rewards.append(reward)
            
            current_state = next_state
            
        return PhantomPath(
            path_id=path_id,
            states=states,
            actions=actions,
            rewards=rewards,
            metadata={"length": length, "start": start_state}
        )
        
    async def generate_batch(self, 
                           batch_size: int,
                           start_state: Dict[str, Any],
                           length: int = 10) -> List[PhantomPath]:
        """
        Generate a batch of phantom paths in parallel
        
        Args:
            batch_size: Number of paths to generate
            start_state: Initial state for all paths
            length: Length of each path
            
        Returns:
            List of phantom paths
        """
        tasks = [
            self.generate_path(start_state, length)
            for _ in range(batch_size)
        ]
        
        paths = await asyncio.gather(*tasks)
        return paths
