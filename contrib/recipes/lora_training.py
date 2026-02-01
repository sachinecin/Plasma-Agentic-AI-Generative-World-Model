"""
LoRA Training Recipe

This recipe demonstrates how to train a LoRA adapter using phantom-path
simulations and compile it into an instruction packet for edge deployment.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.simulation_engine import SimulationEngine
from instruction_packets.compiler import InstructionPacketCompiler
from instruction_packets.runtime import InstructionPacketRuntime
from instruction_packets.config import PacketConfig


async def main():
    """
    Complete LoRA training and deployment workflow:
    1. Initialize components
    2. Train LoRA adapter from simulations
    3. Compile into instruction packet
    4. Deploy to edge runtime
    """
    print("=== LoRA Training & Deployment Recipe ===\n")
    
    # Step 1: Setup
    print("Step 1: Initializing components...")
    sim_engine = SimulationEngine()
    packet_compiler = InstructionPacketCompiler()
    packet_runtime = InstructionPacketRuntime()
    print("✓ Components initialized\n")
    
    # Step 2: Train LoRA adapter
    print("Step 2: Training LoRA adapter...")
    agent_id = "demo_agent"
    task_name = "object_manipulation"
    
    training_result = await sim_engine.train_lora(
        agent_id=agent_id,
        task_name=task_name,
        num_simulations=1000,
        lora_config={
            'rank': 8,
            'lr': 1e-4,
            'alpha': 16.0
        }
    )
    
    print(f"✓ Training complete!")
    print(f"  - Checkpoint: {training_result['checkpoint_path']}")
    print(f"  - LoRA rank: {training_result['lora_rank']}")
    print(f"  - Training steps: {training_result['training_steps']}")
    print(f"  - Final loss: {training_result['final_loss']}\n")
    
    # Step 3: Compile instruction packet
    print("Step 3: Compiling instruction packet...")
    packet_info = await packet_compiler.compile(
        lora_path=training_result['checkpoint_path'],
        agent_id=agent_id,
        target_device="edge"
    )
    
    print(f"✓ Compilation complete!")
    print(f"  - Packet size: {packet_info['packet_size_bytes']:,} bytes")
    print(f"  - Compression ratio: {packet_info['compression_ratio']:.1f}x")
    print(f"  - Quantization: {packet_info['quantization']}")
    print(f"  - Target device: {packet_info['target_device']}\n")
    
    # Step 4: Save and load packet
    print("Step 4: Deploying to edge runtime...")
    packet_path = Path(packet_info['packet_path'])
    packet_compiler.save_packet(agent_id, packet_path)
    
    # Load into runtime
    success = packet_runtime.load_packet(packet_path)
    
    if success:
        print("✓ Packet deployed successfully!")
        stats = packet_runtime.get_statistics()
        print(f"  - Agent ID: {stats['agent_id']}")
        print(f"  - Packet loaded: {stats['packet_loaded']}")
        print(f"  - Ready for inference\n")
    else:
        print("✗ Failed to deploy packet\n")
        return
    
    # Step 5: Demo inference
    print("Step 5: Running sample inference...")
    print("  (Note: Actual inference requires a trained model)")
    print("  - Inference count: 0")
    print("  - Latency: <10ms (target)")
    print("  - Memory: <100KB (target)\n")
    
    print("=== Recipe Complete! ===")
    print("\nKey Takeaways:")
    print("- LoRA adapters are trained from phantom-path simulations")
    print("- Compilation optimizes for edge deployment (quantization, compression)")
    print("- Instruction packets are <100KB and run offline")
    print("- This enables Training-Agent Disaggregation (TA) at scale")


if __name__ == "__main__":
    asyncio.run(main())
