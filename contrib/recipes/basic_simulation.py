"""
Basic Simulation Recipe

This recipe demonstrates how to use the Plasma Agentic AI system
to run basic phantom-path simulations following the Training-Agent
Disaggregation (TA) design pattern.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.simulation_engine import SimulationEngine
from core.config import CoreConfig
from world_model.generative_env import WorldModel
from world_model.config import WorldModelConfig


async def main():
    """
    Basic simulation workflow:
    1. Initialize World Model
    2. Create Simulation Engine
    3. Generate phantom-path trajectories
    4. Analyze results
    """
    print("=== Basic Phantom-Path Simulation ===\n")
    
    # Step 1: Configure and initialize World Model
    print("Step 1: Initializing World Model...")
    world_model_config = WorldModelConfig(
        model_type="diffusion",
        hidden_dim=512,
        num_layers=8
    )
    world_model = WorldModel(config=world_model_config)
    print("✓ World Model initialized\n")
    
    # Step 2: Create Simulation Engine
    print("Step 2: Creating Simulation Engine...")
    core_config = CoreConfig(
        max_simulation_steps=100,
        parallel_simulations=4,
        use_adversarial_audit=True
    )
    sim_engine = SimulationEngine(world_model=world_model, config=core_config)
    print("✓ Simulation Engine ready\n")
    
    # Step 3: Define task and generate trajectories
    print("Step 3: Generating phantom-path trajectories...")
    task_description = "Navigate through a maze to reach the goal position"
    num_trajectories = 10
    
    trajectories = await sim_engine.generate_trajectories(
        task=task_description,
        num_trajectories=num_trajectories,
        adversarial=True
    )
    print(f"✓ Generated {len(trajectories)} trajectories\n")
    
    # Step 4: Analyze results
    print("Step 4: Analyzing trajectories...")
    for i, traj in enumerate(trajectories[:3]):  # Show first 3
        print(f"  Trajectory {i+1}:")
        print(f"    - ID: {traj['trajectory_id']}")
        print(f"    - Audited: {traj.get('audited', False)}")
        print(f"    - Audit Score: {traj.get('audit_score', 'N/A')}")
        print(f"    - Total Reward: {traj['total_reward']}")
        print(f"    - Steps: {traj['steps']}")
    
    if len(trajectories) > 3:
        print(f"  ... and {len(trajectories) - 3} more\n")
    
    # Step 5: Show statistics
    print("\nStep 5: Simulation Statistics")
    stats = sim_engine.get_statistics()
    print(f"  Total simulations: {stats['total_simulations']}")
    print(f"  Cached trajectories: {stats['cached_trajectories']}")
    print(f"  World Model loaded: {stats['world_model_loaded']}")
    
    print("\n=== Simulation Complete! ===")
    print("\nKey Takeaways:")
    print("- Phantom-path simulations enable learning without real-world interaction")
    print("- Adversarial auditing prevents reward hacking")
    print("- Trajectories can be cached and reused for training")
    print("- This is the foundation of Training-Agent Disaggregation (TA)")


if __name__ == "__main__":
    asyncio.run(main())
