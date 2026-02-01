"""
Full Client-Server Recipe

This recipe demonstrates the complete Training-Agent Disaggregation (TA)
workflow using the client-server architecture.

NOTE: This requires server.py to be running separately.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client import PlasmaAgentClient, ClientConfig, TaskRequest


async def main():
    """
    Complete client-server TA workflow:
    1. Connect to World Model server
    2. Request phantom-path simulations
    3. Train LoRA adapter on server
    4. Download instruction packet
    5. Deploy for edge inference
    """
    print("=== Client-Server TA Workflow ===\n")
    print("NOTE: Ensure server.py is running on localhost:8000\n")
    
    # Step 1: Configure and connect client
    print("Step 1: Connecting to World Model server...")
    config = ClientConfig(
        agent_id="client_demo",
        server_url="http://localhost:8000",
        ws_url="ws://localhost:8000"
    )
    
    try:
        async with PlasmaAgentClient(config) as client:
            print("✓ Connected to server\n")
            
            # Step 2: Request simulation
            print("Step 2: Requesting phantom-path simulation...")
            sim_result = await client.request_simulation(
                task_description="Solve a navigation puzzle",
                num_trajectories=20,
                adversarial_audit=True
            )
            print(f"✓ Simulation started: {sim_result['simulation_id']}\n")
            
            # Step 3: Wait for simulation
            print("Step 3: Waiting for simulation to complete...")
            final_status = await client.wait_for_simulation(
                sim_result['simulation_id'],
                poll_interval=2.0,
                max_wait=60.0
            )
            print(f"✓ Simulation complete: {final_status['status']}\n")
            
            # Step 4: Start training
            print("Step 4: Starting LoRA training on server...")
            task = TaskRequest(
                task_name="navigation_puzzle",
                task_description="Solve a navigation puzzle",
                num_simulations=500,
                lora_rank=8,
                learning_rate=1e-4
            )
            train_result = await client.start_training(task)
            print(f"✓ Training initiated: {train_result['status']}\n")
            
            # Step 5: Compile packet
            print("Step 5: Compiling instruction packet...")
            packet_result = await client.compile_instruction_packet(
                trained_lora_path=f"/tmp/lora_{config.agent_id}_{task.task_name}",
                target_device="edge"
            )
            print(f"✓ Packet compiled: {packet_result['status']}")
            print(f"  - Download URL: {packet_result['download_url']}\n")
            
            # Step 6: Get training history
            print("Step 6: Checking training history...")
            history = await client.get_training_history()
            print(f"✓ Found {len(history)} training sessions")
            for i, session in enumerate(history[-3:], 1):  # Last 3
                print(f"  {i}. {session['task_name']} - {session['status']}")
            print()
            
            print("=== Workflow Complete! ===")
            print("\nKey Takeaways:")
            print("- Client sends task descriptions to server")
            print("- Server runs simulations in World Model")
            print("- Training happens server-side (no local GPU needed)")
            print("- Client downloads lightweight instruction packets")
            print("- This is true Training-Agent Disaggregation!")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure server.py is running: python server.py")
        print("2. Check server is accessible at http://localhost:8000")
        print("3. Verify no firewall blocking the connection")


if __name__ == "__main__":
    asyncio.run(main())
