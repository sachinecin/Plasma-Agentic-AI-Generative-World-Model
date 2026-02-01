"""
Phantom-Path Simulator - Generative World Model for Pre-emptive Rollout
Uses asyncio for zero-latency evolution
"""
import asyncio
import numpy as np
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime

from .state_traces import (
    WorldState,
    StateTrace,
    ActionType,
    SimulationResult,
)


class PhantomPathSimulator:
    """
    Generative World Model that simulates "phantom paths" for pre-emptive task rollout.
    Replaces heavy RL training with lightweight, generative forward simulation.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_space: List[ActionType] = None,
        horizon: int = 10,
        num_simulations: int = 5,
        temperature: float = 0.7,
    ):
        """
        Initialize the Phantom-Path Simulator
        
        Args:
            state_dim: Dimensionality of state vectors
            action_space: Available actions for simulation
            horizon: Number of steps to simulate ahead
            num_simulations: Number of parallel phantom paths to explore
            temperature: Temperature for stochastic rollouts (higher = more exploration)
        """
        self.state_dim = state_dim
        self.action_space = action_space or list(ActionType)
        self.horizon = horizon
        self.num_simulations = num_simulations
        self.temperature = temperature
        
        # Internal world model parameters (would be learned in production)
        self._transition_model = self._initialize_transition_model()
        self._reward_model = self._initialize_reward_model()
        self._current_world_state: Optional[WorldState] = None
        
    def _initialize_transition_model(self) -> Dict[str, Any]:
        """Initialize a simple transition model (placeholder for neural network)"""
        return {
            "weights": np.random.randn(self.state_dim, self.state_dim) * 0.01,
            "bias": np.zeros(self.state_dim),
        }
    
    def _initialize_reward_model(self) -> Dict[str, Any]:
        """Initialize a simple reward model (placeholder for neural network)"""
        return {
            "weights": np.random.randn(self.state_dim) * 0.01,
            "bias": 0.0,
        }
    
    def _predict_next_state(
        self, 
        current_state: List[float], 
        action: ActionType,
        stochastic: bool = True,
    ) -> List[float]:
        """
        Predict next state given current state and action.
        This is a placeholder - in production, would use a trained neural network.
        """
        state_array = np.array(current_state)
        
        # Simple linear transformation with action encoding
        action_encoding = self._encode_action(action)
        combined = state_array + action_encoding
        
        # Apply transition model
        next_state = (
            self._transition_model["weights"] @ combined + 
            self._transition_model["bias"]
        )
        
        # Add stochasticity for exploration
        if stochastic:
            noise = np.random.randn(self.state_dim) * self.temperature * 0.1
            next_state += noise
        
        # Normalize to prevent explosion
        next_state = next_state / (np.linalg.norm(next_state) + 1e-8)
        
        return next_state.tolist()
    
    def _encode_action(self, action: ActionType) -> np.ndarray:
        """Encode action as a vector"""
        encoding = np.zeros(self.state_dim)
        action_idx = self.action_space.index(action)
        # One-hot-like encoding distributed across state dimensions
        encoding[action_idx * (self.state_dim // len(self.action_space))] = 1.0
        return encoding
    
    def _predict_reward(self, state: List[float], action: ActionType) -> float:
        """
        Predict reward for state-action pair.
        This is a placeholder - in production, would use a trained neural network.
        """
        state_array = np.array(state)
        reward = (
            self._reward_model["weights"] @ state_array + 
            self._reward_model["bias"]
        )
        return float(np.tanh(reward))  # Bound reward between -1 and 1
    
    async def simulate_phantom_path(
        self,
        initial_state: List[float],
        policy: Optional[Callable[[List[float]], ActionType]] = None,
    ) -> SimulationResult:
        """
        Simulate a single phantom path asynchronously.
        
        Args:
            initial_state: Starting state for simulation
            policy: Optional policy function to select actions. If None, uses random policy.
            
        Returns:
            SimulationResult containing the simulated trajectory
        """
        trajectory = []
        current_state = initial_state.copy()
        total_reward = 0.0
        
        for step in range(self.horizon):
            # Select action
            if policy is not None:
                action = policy(current_state)
            else:
                # Random policy for exploration
                import random
                action = random.choice(self.action_space)
            
            # Predict reward and next state
            reward = self._predict_reward(current_state, action)
            next_state = self._predict_next_state(current_state, action)
            
            # Record trajectory
            trajectory.append((current_state.copy(), action, reward))
            total_reward += reward
            
            # Move to next state
            current_state = next_state
            
            # Allow other coroutines to run (zero-latency evolution)
            await asyncio.sleep(0)
        
        # Calculate success probability based on trajectory quality
        success_probability = (total_reward + self.horizon) / (2 * self.horizon)
        success_probability = max(0.0, min(1.0, success_probability))
        
        # Calculate quality score (variance-adjusted return)
        rewards = [r for _, _, r in trajectory]
        quality_score = (np.mean(rewards) + 1.0) / 2.0  # Normalize to [0, 1]
        
        return SimulationResult(
            initial_state=initial_state,
            simulated_trajectory=trajectory,
            total_reward=total_reward,
            success_probability=success_probability,
            quality_score=quality_score,
            metadata={
                "horizon": self.horizon,
                "temperature": self.temperature,
            }
        )
    
    async def pre_emptive_rollout(
        self,
        initial_state: List[float],
        policy: Optional[Callable[[List[float]], ActionType]] = None,
    ) -> List[SimulationResult]:
        """
        Perform pre-emptive rollout by simulating multiple phantom paths in parallel.
        Uses asyncio for concurrent execution with zero-latency evolution.
        
        Args:
            initial_state: Starting state for all simulations
            policy: Optional policy function to select actions
            
        Returns:
            List of simulation results, one per phantom path
        """
        # Create multiple parallel simulations
        simulation_tasks = [
            self.simulate_phantom_path(initial_state, policy)
            for _ in range(self.num_simulations)
        ]
        
        # Run all simulations concurrently
        results = await asyncio.gather(*simulation_tasks)
        
        # Update world state with predictions
        self._update_world_state(initial_state, results)
        
        return results
    
    def _update_world_state(
        self, 
        initial_state: List[float], 
        simulations: List[SimulationResult]
    ) -> None:
        """Update internal world state based on simulation results"""
        predicted_states = [sim.simulated_trajectory[0][0] for sim in simulations]
        confidence_scores = [sim.success_probability for sim in simulations]
        
        self._current_world_state = WorldState(
            state_vector=initial_state,
            predicted_next_states=predicted_states,
            confidence_scores=confidence_scores,
        )
    
    def get_best_phantom_path(
        self, 
        results: List[SimulationResult]
    ) -> SimulationResult:
        """
        Select the best phantom path based on quality score and success probability.
        
        Args:
            results: List of simulation results
            
        Returns:
            The best simulation result
        """
        if not results:
            raise ValueError("No simulation results provided")
        
        # Score each result by a combination of quality and success probability
        scores = [
            r.quality_score * 0.6 + r.success_probability * 0.4
            for r in results
        ]
        
        best_idx = np.argmax(scores)
        return results[best_idx]
    
    def extract_state_traces(
        self, 
        simulation: SimulationResult
    ) -> List[StateTrace]:
        """
        Extract state traces from a simulation for logging and analysis.
        
        Args:
            simulation: A simulation result
            
        Returns:
            List of StateTrace objects
        """
        traces = []
        for i, (state, action, reward) in enumerate(simulation.simulated_trajectory):
            trace = StateTrace(
                state_vector=state,
                action_taken=action,
                reward=reward,
                metadata={
                    "simulation_id": simulation.simulation_id,
                    "step": i,
                    "timestamp": simulation.timestamp.isoformat(),
                }
            )
            traces.append(trace)
        
        return traces
    
    def get_world_state(self) -> Optional[WorldState]:
        """Get the current world state"""
        return self._current_world_state
    
    def update_transition_model(self, new_weights: Dict[str, Any]) -> None:
        """
        Update the transition model with new weights (for online learning).
        This allows fluid, self-correcting logic.
        """
        self._transition_model.update(new_weights)
    
    def update_reward_model(self, new_weights: Dict[str, Any]) -> None:
        """
        Update the reward model with new weights (for online learning).
        This allows fluid, self-correcting logic.
        """
        self._reward_model.update(new_weights)
