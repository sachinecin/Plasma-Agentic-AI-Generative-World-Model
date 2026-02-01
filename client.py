"""
Client component for Training-Agent Disaggregation (TA) Architecture.

This client represents the agent side that communicates with the World Model server.
Following Agent-Lightning's TA design, this is lightweight and can run on edge devices
using instruction packets instead of full models.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json

import httpx
from websockets import connect as ws_connect
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Models
# ============================================================================

class ClientConfig(BaseModel):
    """Configuration for the agent client."""
    agent_id: str
    server_url: str = "http://localhost:8000"
    ws_url: str = "ws://localhost:8000"
    timeout: float = 30.0
    retry_attempts: int = 3


class TaskRequest(BaseModel):
    """Task to be learned by the agent."""
    task_name: str
    task_description: str
    num_simulations: int = 1000
    lora_rank: int = 8
    learning_rate: float = 1e-4


# ============================================================================
# Agent Client
# ============================================================================

class PlasmaAgentClient:
    """
    Lightweight agent client for Training-Agent Disaggregation.
    
    This client:
    1. Sends task descriptions to the World Model server
    2. Receives simulated training data (phantom paths)
    3. Downloads compiled instruction packets
    4. Deploys packets locally for real-time inference
    """
    
    def __init__(self, config: ClientConfig):
        """Initialize the agent client."""
        self.config = config
        self.http_client: Optional[httpx.AsyncClient] = None
        self.ws_connection = None
        self.instruction_packet: Optional[Dict] = None
        self.local_cache_dir = Path.home() / ".plasma_agent" / config.agent_id
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized agent client: {config.agent_id}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Connect to the World Model server."""
        logger.info(f"Connecting to server at {self.config.server_url}")
        self.http_client = httpx.AsyncClient(
            base_url=self.config.server_url,
            timeout=self.config.timeout
        )
        
        # Verify server is healthy
        try:
            response = await self.http_client.get("/health")
            response.raise_for_status()
            health = response.json()
            logger.info(f"Server health: {health['status']}")
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the World Model server."""
        logger.info("Disconnecting from server")
        
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
    
    async def connect_websocket(self):
        """Establish WebSocket connection for real-time updates."""
        if self.ws_connection:
            return
        
        ws_url = f"{self.config.ws_url}/ws/{self.config.agent_id}"
        logger.info(f"Connecting WebSocket to {ws_url}")
        
        try:
            self.ws_connection = await ws_connect(ws_url)
            logger.info("WebSocket connected")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    # ========================================================================
    # Simulation Methods
    # ========================================================================
    
    async def request_simulation(
        self,
        task_description: str,
        num_trajectories: int = 10,
        adversarial_audit: bool = True
    ) -> Dict[str, Any]:
        """
        Request phantom-path simulation from the World Model server.
        
        This is the first step in TA design - instead of learning from real
        interactions, the agent requests simulations from the generative model.
        """
        logger.info(f"Requesting simulation for task: {task_description}")
        
        payload = {
            "agent_id": self.config.agent_id,
            "task_description": task_description,
            "num_trajectories": num_trajectories,
            "adversarial_audit": adversarial_audit
        }
        
        response = await self.http_client.post("/simulate", json=payload)
        response.raise_for_status()
        result = response.json()
        
        logger.info(f"Simulation started: {result['simulation_id']}")
        return result
    
    async def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """Check the status of a running simulation."""
        response = await self.http_client.get(f"/simulate/{simulation_id}")
        response.raise_for_status()
        return response.json()
    
    async def wait_for_simulation(
        self,
        simulation_id: str,
        poll_interval: float = 2.0,
        max_wait: float = 300.0
    ) -> Dict[str, Any]:
        """Wait for simulation to complete."""
        logger.info(f"Waiting for simulation {simulation_id}")
        elapsed = 0.0
        
        while elapsed < max_wait:
            status = await self.get_simulation_status(simulation_id)
            
            if status['status'] == 'complete':
                logger.info("Simulation complete")
                return status
            elif status['status'] == 'failed':
                raise RuntimeError(f"Simulation failed: {status.get('error', 'Unknown error')}")
            
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        raise TimeoutError(f"Simulation timed out after {max_wait} seconds")
    
    # ========================================================================
    # Training Methods
    # ========================================================================
    
    async def start_training(self, task: TaskRequest) -> Dict[str, Any]:
        """
        Start LoRA distillation training on the server.
        
        The server uses phantom-path simulations to train a LoRA adapter,
        avoiding the need for real-world data collection or heavy on-device training.
        """
        logger.info(f"Starting training for task: {task.task_name}")
        
        payload = {
            "agent_id": self.config.agent_id,
            "task_name": task.task_name,
            "num_simulations": task.num_simulations,
            "lora_rank": task.lora_rank,
            "learning_rate": task.learning_rate,
            "use_phantom_paths": True
        }
        
        response = await self.http_client.post("/train", json=payload)
        response.raise_for_status()
        result = response.json()
        
        logger.info(f"Training started: {result['status']}")
        return result
    
    async def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history for this agent."""
        response = await self.http_client.get(f"/train/history/{self.config.agent_id}")
        response.raise_for_status()
        return response.json()['history']
    
    # ========================================================================
    # Instruction Packet Methods
    # ========================================================================
    
    async def compile_instruction_packet(
        self,
        trained_lora_path: str,
        target_device: str = "edge"
    ) -> Dict[str, Any]:
        """
        Request compilation of trained LoRA into instruction packet.
        
        This is the final step of TA design - converting the trained adapter
        into a tiny, deployable packet for edge inference.
        """
        logger.info("Requesting instruction packet compilation")
        
        payload = {
            "agent_id": self.config.agent_id,
            "trained_lora_path": trained_lora_path,
            "target_device": target_device
        }
        
        response = await self.http_client.post("/instruction-packet/compile", json=payload)
        response.raise_for_status()
        result = response.json()
        
        logger.info(f"Instruction packet compiled: {result['status']}")
        return result
    
    async def download_instruction_packet(self, save_path: Optional[Path] = None) -> Path:
        """
        Download compiled instruction packet to local device.
        
        The packet can then be loaded and used for real-time inference
        without needing the full World Model.
        """
        if save_path is None:
            save_path = self.local_cache_dir / "instruction_packet.bin"
        
        logger.info(f"Downloading instruction packet to {save_path}")
        
        response = await self.http_client.get(f"/instruction-packet/download/{self.config.agent_id}")
        response.raise_for_status()
        
        # Save the packet
        save_path.write_bytes(response.content)
        logger.info(f"Instruction packet saved to {save_path}")
        
        return save_path
    
    async def load_instruction_packet(self, packet_path: Optional[Path] = None):
        """
        Load instruction packet for local inference.
        
        This enables edge deployment - the agent can now execute tasks
        using the lightweight packet instead of querying the full model.
        """
        if packet_path is None:
            packet_path = self.local_cache_dir / "instruction_packet.bin"
        
        if not packet_path.exists():
            raise FileNotFoundError(f"Instruction packet not found: {packet_path}")
        
        logger.info(f"Loading instruction packet from {packet_path}")
        
        # TODO: Implement actual packet loading
        # self.instruction_packet = load_packet(packet_path)
        
        logger.info("Instruction packet loaded and ready for inference")
    
    # ========================================================================
    # Inference Methods
    # ========================================================================
    
    async def execute_task(self, observation: Any) -> Any:
        """
        Execute task using loaded instruction packet.
        
        This is real-time edge inference - no server communication needed.
        """
        if self.instruction_packet is None:
            raise RuntimeError("No instruction packet loaded. Call load_instruction_packet() first.")
        
        # TODO: Implement actual inference
        # action = self.instruction_packet.forward(observation)
        # return action
        
        logger.warning("Instruction packet inference not yet implemented")
        return None
    
    # ========================================================================
    # High-Level Workflow
    # ========================================================================
    
    async def train_and_deploy(self, task: TaskRequest) -> Path:
        """
        Complete workflow: simulate -> train -> compile -> download.
        
        This is the full TA pipeline:
        1. Request phantom-path simulations
        2. Train LoRA on simulated data
        3. Compile into instruction packet
        4. Download for edge deployment
        """
        logger.info(f"Starting full training pipeline for: {task.task_name}")
        
        # Step 1: Request simulation
        sim_result = await self.request_simulation(
            task_description=task.task_description,
            num_trajectories=task.num_simulations // 10
        )
        sim_id = sim_result['simulation_id']
        
        # Step 2: Wait for simulation
        await self.wait_for_simulation(sim_id)
        
        # Step 3: Start training
        train_result = await self.start_training(task)
        
        # TODO: Wait for training to complete (needs training job ID tracking)
        logger.info("Waiting for training to complete...")
        await asyncio.sleep(5)  # Placeholder
        
        # Step 4: Compile instruction packet
        lora_path = f"/tmp/lora_{self.config.agent_id}_{task.task_name}"
        packet_result = await self.compile_instruction_packet(lora_path)
        
        # Step 5: Download packet
        packet_path = await self.download_instruction_packet()
        
        logger.info(f"Training pipeline complete! Packet saved to {packet_path}")
        return packet_path


# ============================================================================
# Example Usage
# ============================================================================

async def example_workflow():
    """Example of using the agent client."""
    
    # Configure the agent
    config = ClientConfig(
        agent_id="agent_001",
        server_url="http://localhost:8000",
        ws_url="ws://localhost:8000"
    )
    
    # Create and connect client
    async with PlasmaAgentClient(config) as client:
        
        # Define a task
        task = TaskRequest(
            task_name="navigate_maze",
            task_description="Navigate through a maze to reach the goal",
            num_simulations=500,
            lora_rank=8,
            learning_rate=1e-4
        )
        
        # Run full training pipeline
        packet_path = await client.train_and_deploy(task)
        
        # Load the instruction packet
        await client.load_instruction_packet(packet_path)
        
        # Now ready for real-time inference!
        logger.info("Agent is ready for deployment!")


async def example_simulation_only():
    """Example of just running simulations."""
    
    config = ClientConfig(agent_id="test_agent")
    
    async with PlasmaAgentClient(config) as client:
        # Request a simulation
        result = await client.request_simulation(
            task_description="Pick up a red cube and place it on the blue platform",
            num_trajectories=20,
            adversarial_audit=True
        )
        
        # Wait for completion
        final_status = await client.wait_for_simulation(result['simulation_id'])
        
        print(f"Simulation complete: {final_status}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Run example workflow."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "simulate":
        asyncio.run(example_simulation_only())
    else:
        asyncio.run(example_workflow())


if __name__ == "__main__":
    main()
