"""
PlasmaAgent - Main integration layer for Project Plasma
Combines Phantom-Path Simulator, LoRA Distiller, and Judicial Auditor
"""
import asyncio
import numpy as np
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timezone

from .phantom_path import PhantomPathSimulator
from .distiller import LoRADistiller
from .judicial_auditor import JudicialAuditor
from .state_traces import (
    ActionType,
    StateTrace,
    SimulationResult,
    InstructionPacket,
    AuditRecord,
)


class PlasmaAgent:
    """
    Project Plasma - The next-gen evolution of Agent-Lightning.
    
    Integrates:
    - Phantom-Path Simulator for pre-emptive rollout
    - LoRA Distiller for real-time edge adaptation
    - Judicial Auditor for reward-hacking prevention
    
    Features fluid, self-correcting logic with zero-latency evolution via asyncio.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_space: Optional[List[ActionType]] = None,
        simulation_horizon: int = 10,
        num_simulations: int = 5,
        enable_auditing: bool = True,
        enable_distillation: bool = True,
    ):
        """
        Initialize the Plasma Agent
        
        Args:
            state_dim: Dimensionality of state vectors
            action_space: Available actions
            simulation_horizon: Steps to simulate ahead
            num_simulations: Number of parallel phantom paths
            enable_auditing: Enable judicial auditing
            enable_distillation: Enable LoRA distillation
        """
        # Initialize core components
        self.simulator = PhantomPathSimulator(
            state_dim=state_dim,
            action_space=action_space,
            horizon=simulation_horizon,
            num_simulations=num_simulations,
        )
        
        self.distiller = LoRADistiller() if enable_distillation else None
        self.auditor = JudicialAuditor() if enable_auditing else None
        
        # Agent state
        self.state_dim = state_dim
        self.current_state: Optional[List[float]] = None
        self.enable_auditing = enable_auditing
        self.enable_distillation = enable_distillation
        
        # Execution history
        self._execution_history: List[Dict[str, Any]] = []
        self._background_tasks: List[asyncio.Task] = []
        
    def initialize_state(self, initial_state: Optional[List[float]] = None) -> None:
        """
        Initialize agent state.
        
        Args:
            initial_state: Initial state vector. If None, uses random initialization.
        """
        if initial_state is not None:
            self.current_state = initial_state
        else:
            # Random initialization
            state = np.random.randn(self.state_dim)
            self.current_state = (state / np.linalg.norm(state)).tolist()
    
    async def think(
        self,
        policy: Optional[Callable[[List[float]], ActionType]] = None,
    ) -> SimulationResult:
        """
        "Think" by simulating phantom paths and selecting the best one.
        This is the core cognitive loop.
        
        Args:
            policy: Optional policy for action selection
            
        Returns:
            The best simulation result
        """
        if self.current_state is None:
            self.initialize_state()
        
        # Run pre-emptive rollout
        simulations = await self.simulator.pre_emptive_rollout(
            self.current_state,
            policy,
        )
        
        # Audit simulations if enabled
        if self.enable_auditing and self.auditor:
            audit_tasks = [
                self.auditor.audit_simulation(sim)
                for sim in simulations
            ]
            audit_results = await asyncio.gather(*audit_tasks)
            
            # Filter out heavily flagged simulations
            valid_simulations = []
            for sim, audits in zip(simulations, audit_results):
                flagged_count = sum(1 for a in audits if a.reward_hacking_detected)
                if flagged_count < len(audits) // 2:  # Less than half flagged
                    valid_simulations.append(sim)
            
            if not valid_simulations:
                # All simulations flagged - use original but with reduced confidence
                valid_simulations = simulations
        else:
            valid_simulations = simulations
        
        # Select best phantom path
        best_simulation = self.simulator.get_best_phantom_path(valid_simulations)
        
        return best_simulation
    
    async def act(
        self,
        simulation: SimulationResult,
        execute_first_action: bool = True,
    ) -> StateTrace:
        """
        Act based on a simulation result.
        
        Args:
            simulation: Simulation to base action on
            execute_first_action: If True, executes the first action from trajectory
            
        Returns:
            StateTrace of the executed action
        """
        if not simulation.simulated_trajectory:
            raise ValueError("Simulation has no trajectory")
        
        # Get first action from best trajectory
        first_state, first_action, predicted_reward = simulation.simulated_trajectory[0]
        
        if execute_first_action:
            # Execute action (in production, would interact with environment)
            actual_reward = predicted_reward + np.random.randn() * 0.1  # Add noise
            
            # Update current state
            self.current_state = first_state
            
            # Create state trace
            trace = StateTrace(
                state_vector=first_state,
                action_taken=first_action,
                reward=actual_reward,
                metadata={
                    "simulation_id": simulation.simulation_id,
                    "predicted_reward": predicted_reward,
                }
            )
            
            # Audit the trace if enabled
            if self.enable_auditing and self.auditor:
                audit_record = await self.auditor.audit_state_trace(trace)
                
                # Apply correction if needed
                if audit_record.reward_hacking_detected:
                    trace = await self.auditor.apply_correction(audit_record, trace)
            
            return trace
        else:
            # Just return a trace without execution
            return StateTrace(
                state_vector=first_state,
                action_taken=first_action,
                reward=predicted_reward,
            )
    
    async def learn(
        self,
        simulation: SimulationResult,
        priority: int = 5,
    ) -> Optional[InstructionPacket]:
        """
        Learn from a simulation by distilling it into a LoRA instruction packet.
        
        Args:
            simulation: Simulation to learn from
            priority: Priority for the instruction packet
            
        Returns:
            Generated instruction packet, or None if distillation disabled
        """
        if not self.enable_distillation or not self.distiller:
            return None
        
        # Distill simulation
        packet = self.distiller.distill_from_simulation(simulation, priority)
        
        # Audit packet if enabled
        if self.enable_auditing and self.auditor:
            audit_record = await self.auditor.audit_instruction_packet(packet)
            
            if audit_record.reward_hacking_detected:
                # Scale down weights if flagged
                scaled_weights = {}
                for layer, weights in packet.lora_weights.items():
                    scaled = (np.array(weights) * 0.5).tolist()
                    scaled_weights[layer] = scaled
                
                packet = InstructionPacket(
                    lora_weights=scaled_weights,
                    target_layers=packet.target_layers,
                    adaptation_strength=packet.adaptation_strength * 0.5,
                    context=f"{packet.context} (scaled)",
                    priority=packet.priority,
                )
        
        # Enqueue for injection
        await self.distiller.enqueue_packet(packet)
        
        return packet
    
    async def evolve(
        self,
        steps: int = 10,
        policy: Optional[Callable[[List[float]], ActionType]] = None,
    ) -> List[StateTrace]:
        """
        Evolve the agent through multiple think-act-learn cycles.
        Uses asyncio for zero-latency evolution.
        
        Args:
            steps: Number of evolution steps
            policy: Optional policy for action selection
            
        Returns:
            List of state traces from evolution
        """
        traces = []
        
        for step in range(steps):
            # Think: simulate phantom paths
            simulation = await self.think(policy)
            
            # Act: execute best action
            trace = await self.act(simulation)
            traces.append(trace)
            
            # Learn: distill into LoRA packet
            packet = await self.learn(simulation)
            
            # Record execution
            self._execution_history.append({
                "step": step,
                "simulation_id": simulation.simulation_id,
                "trace_id": trace.trace_id,
                "packet_id": packet.packet_id if packet else None,
                "reward": trace.reward,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            # Allow other coroutines to run
            await asyncio.sleep(0)
        
        return traces
    
    async def start_continuous_evolution(
        self,
        policy: Optional[Callable[[List[float]], ActionType]] = None,
        agent_interface: Optional[Any] = None,
    ) -> None:
        """
        Start continuous evolution in the background.
        Runs think-act-learn loop continuously with zero-latency.
        
        Args:
            policy: Optional policy for action selection
            agent_interface: Interface to edge agent for LoRA injection
        """
        # Start continuous injection loop if distiller enabled
        if self.enable_distillation and self.distiller:
            injection_task = asyncio.create_task(
                self.distiller.continuous_injection_loop(agent_interface)
            )
            self._background_tasks.append(injection_task)
        
        # Start evolution loop
        async def evolution_loop():
            while True:
                try:
                    # One evolution step
                    await self.evolve(steps=1, policy=policy)
                    
                    # Brief sleep to prevent tight loop
                    await asyncio.sleep(0.1)
                except Exception as e:
                    print(f"Evolution error: {e}")
                    await asyncio.sleep(1.0)
        
        evolution_task = asyncio.create_task(evolution_loop())
        self._background_tasks.append(evolution_task)
    
    async def stop_continuous_evolution(self) -> None:
        """Stop all background evolution tasks"""
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        stats = {
            "execution_history_size": len(self._execution_history),
            "current_state_norm": float(np.linalg.norm(self.current_state)) if self.current_state else None,
            "simulator": {
                "world_state": self.simulator.get_world_state(),
            },
        }
        
        if self.distiller:
            stats["distiller"] = self.distiller.get_statistics()
        
        if self.auditor:
            stats["auditor"] = self.auditor.get_audit_statistics()
        
        return stats
    
    def get_execution_history(
        self, 
        count: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self._execution_history[-count:]
    
    def reset(self) -> None:
        """Reset agent state"""
        self.initialize_state()
        self._execution_history.clear()
        
        if self.distiller:
            self.distiller.clear_queue()
