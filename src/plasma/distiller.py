"""
LoRA Distiller - Real-time injection of LoRA weights (Instruction-Packets) into edge agents
"""
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import deque

from .state_traces import InstructionPacket, StateTrace, SimulationResult


class LoRADistiller:
    """
    Distiller that generates and injects real-time LoRA weight instruction packets
    into edge agents for rapid adaptation without full model retraining.
    """
    
    def __init__(
        self,
        layer_names: List[str] = None,
        max_packet_queue: int = 100,
        distillation_rate: float = 0.1,
        adaptation_strength: float = 0.1,
    ):
        """
        Initialize the LoRA Distiller
        
        Args:
            layer_names: Names of model layers that can receive LoRA injections
            max_packet_queue: Maximum number of packets to queue
            distillation_rate: Rate at which knowledge is distilled into LoRA weights
            adaptation_strength: Default strength of LoRA adaptations
        """
        self.layer_names = layer_names or [
            "attention.q_proj",
            "attention.k_proj",
            "attention.v_proj",
            "attention.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        self.max_packet_queue = max_packet_queue
        self.distillation_rate = distillation_rate
        self.adaptation_strength = adaptation_strength
        
        # Packet queue for async injection
        self._packet_queue: deque[InstructionPacket] = deque(maxlen=max_packet_queue)
        self._injection_history: List[InstructionPacket] = []
        
        # Statistics
        self._packets_generated = 0
        self._packets_injected = 0
        
    def distill_from_simulation(
        self,
        simulation: SimulationResult,
        priority: int = 5,
    ) -> InstructionPacket:
        """
        Distill a simulation result into a LoRA instruction packet.
        Extracts behavioral patterns and converts them to weight adjustments.
        
        Args:
            simulation: Simulation result to distill
            priority: Priority level for the packet (1-10)
            
        Returns:
            InstructionPacket ready for injection
        """
        # Extract key behavioral patterns from trajectory
        trajectory = simulation.simulated_trajectory
        
        # Calculate weight deltas based on trajectory quality
        lora_weights = {}
        for layer_name in self.layer_names:
            # Generate LoRA weights based on simulation outcomes
            # In production, this would use gradient-based distillation
            weight_delta = self._compute_weight_delta(
                trajectory,
                simulation.quality_score,
                layer_name,
            )
            lora_weights[layer_name] = weight_delta
        
        # Determine target layers based on action types in trajectory
        target_layers = self._select_target_layers(trajectory)
        
        # Create instruction packet
        packet = InstructionPacket(
            lora_weights=lora_weights,
            target_layers=target_layers,
            adaptation_strength=self.adaptation_strength * simulation.success_probability,
            context=f"Distilled from simulation {simulation.simulation_id[:8]}",
            priority=priority,
        )
        
        self._packets_generated += 1
        return packet
    
    def _compute_weight_delta(
        self,
        trajectory: List[tuple],
        quality_score: float,
        layer_name: str,
    ) -> List[float]:
        """
        Compute LoRA weight delta for a specific layer.
        This is a simplified version - production would use proper gradient distillation.
        """
        # Extract state patterns
        states = [state for state, _, _ in trajectory]
        rewards = [reward for _, _, reward in trajectory]
        
        # Compute aggregate features
        avg_state = np.mean(states, axis=0)
        reward_trend = np.polyfit(range(len(rewards)), rewards, deg=1)[0]
        
        # Generate weight delta based on trajectory characteristics
        # Use a small rank for LoRA (e.g., rank 8)
        rank = 8
        weight_delta = (
            avg_state[:rank] * quality_score * self.distillation_rate +
            np.ones(rank) * reward_trend * 0.01
        ).tolist()
        
        return weight_delta
    
    def _select_target_layers(self, trajectory: List[tuple]) -> List[str]:
        """
        Select which layers should receive LoRA weights based on trajectory.
        """
        # Analyze action types in trajectory
        actions = [action for _, action, _ in trajectory]
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Select layers based on dominant action types
        # This is a heuristic - in production, would use learned layer selection
        if action_counts:
            dominant_action = max(action_counts, key=action_counts.get)
            if dominant_action.value in ["move", "interact"]:
                return self.layer_names[:4]  # Focus on attention layers
            elif dominant_action.value in ["compute", "observe"]:
                return self.layer_names[4:]  # Focus on MLP layers
        
        # Default: target all layers
        return self.layer_names
    
    def distill_from_traces(
        self,
        traces: List[StateTrace],
        priority: int = 5,
    ) -> InstructionPacket:
        """
        Distill state traces into a LoRA instruction packet.
        Useful for online learning from real agent experiences.
        
        Args:
            traces: List of state traces to distill
            priority: Priority level for the packet
            
        Returns:
            InstructionPacket ready for injection
        """
        if not traces:
            raise ValueError("No traces provided for distillation")
        
        # Compute aggregate metrics
        avg_reward = np.mean([t.reward for t in traces])
        states = [t.state_vector for t in traces]
        
        # Generate LoRA weights
        lora_weights = {}
        for layer_name in self.layer_names:
            rank = 8
            avg_state = np.mean(states, axis=0)
            weight_delta = (
                avg_state[:rank] * (avg_reward + 1.0) / 2.0 * self.distillation_rate
            ).tolist()
            lora_weights[layer_name] = weight_delta
        
        packet = InstructionPacket(
            lora_weights=lora_weights,
            target_layers=self.layer_names,
            adaptation_strength=self.adaptation_strength,
            context=f"Distilled from {len(traces)} traces",
            priority=priority,
        )
        
        self._packets_generated += 1
        return packet
    
    async def enqueue_packet(self, packet: InstructionPacket) -> None:
        """
        Enqueue an instruction packet for asynchronous injection.
        
        Args:
            packet: Instruction packet to enqueue
        """
        self._packet_queue.append(packet)
        await asyncio.sleep(0)  # Yield to event loop
    
    async def inject_to_edge_agent(
        self,
        packet: InstructionPacket,
        agent_interface: Optional[Any] = None,
    ) -> bool:
        """
        Inject a LoRA instruction packet into an edge agent.
        This is an async operation for zero-latency injection.
        
        Args:
            packet: Instruction packet to inject
            agent_interface: Interface to the edge agent (if available)
            
        Returns:
            True if injection successful, False otherwise
        """
        try:
            # Simulate injection delay
            await asyncio.sleep(0.01)
            
            # In production, this would:
            # 1. Serialize the LoRA weights
            # 2. Send to edge agent via network/IPC
            # 3. Agent applies weights to specified layers
            # 4. Confirm injection
            
            if agent_interface is not None:
                # Would call agent_interface.apply_lora_weights(packet)
                pass
            
            # Record successful injection
            self._injection_history.append(packet)
            self._packets_injected += 1
            
            return True
            
        except Exception as e:
            # Log error but don't crash
            print(f"Injection failed: {e}")
            return False
    
    async def continuous_injection_loop(
        self,
        agent_interface: Optional[Any] = None,
        batch_size: int = 5,
    ) -> None:
        """
        Continuous loop that injects packets from the queue to the edge agent.
        Processes packets in priority order.
        
        Args:
            agent_interface: Interface to the edge agent
            batch_size: Number of packets to process per batch
        """
        while True:
            # Process available packets
            packets_to_inject = []
            
            # Get packets up to batch size
            while len(packets_to_inject) < batch_size and self._packet_queue:
                packets_to_inject.append(self._packet_queue.popleft())
            
            if packets_to_inject:
                # Sort by priority (higher priority first)
                packets_to_inject.sort(key=lambda p: p.priority, reverse=True)
                
                # Inject packets concurrently
                injection_tasks = [
                    self.inject_to_edge_agent(packet, agent_interface)
                    for packet in packets_to_inject
                ]
                await asyncio.gather(*injection_tasks)
            
            # Sleep briefly to avoid busy-waiting
            await asyncio.sleep(0.1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get distiller statistics"""
        return {
            "packets_generated": self._packets_generated,
            "packets_injected": self._packets_injected,
            "queue_size": len(self._packet_queue),
            "injection_history_size": len(self._injection_history),
            "distillation_rate": self.distillation_rate,
            "adaptation_strength": self.adaptation_strength,
        }
    
    def get_recent_injections(self, count: int = 10) -> List[InstructionPacket]:
        """Get the most recent injection history"""
        return self._injection_history[-count:]
    
    def clear_queue(self) -> None:
        """Clear the packet queue"""
        self._packet_queue.clear()
