"""
Example usage of Project Plasma
Demonstrates the key features and components
"""
import asyncio
import numpy as np
from plasma import PlasmaAgent
from plasma.state_traces import ActionType


async def basic_example():
    """Basic example showing think-act-learn cycle"""
    print("=== Basic Plasma Agent Example ===\n")
    
    # Initialize agent
    agent = PlasmaAgent(
        state_dim=128,
        simulation_horizon=5,
        num_simulations=3,
        enable_auditing=True,
        enable_distillation=True,
    )
    
    # Initialize state
    agent.initialize_state()
    print(f"✓ Agent initialized with state dimension: {agent.state_dim}")
    
    # Think: simulate phantom paths
    print("\n1. THINKING - Simulating phantom paths...")
    simulation = await agent.think()
    print(f"   • Simulated {len(simulation.simulated_trajectory)} steps")
    print(f"   • Total reward: {simulation.total_reward:.3f}")
    print(f"   • Success probability: {simulation.success_probability:.3f}")
    print(f"   • Quality score: {simulation.quality_score:.3f}")
    
    # Act: execute best action
    print("\n2. ACTING - Executing best action...")
    trace = await agent.act(simulation)
    print(f"   • Action taken: {trace.action_taken.value if trace.action_taken else 'None'}")
    print(f"   • Reward received: {trace.reward:.3f}")
    print(f"   • State trace ID: {trace.trace_id[:8]}...")
    
    # Learn: distill into LoRA packet
    print("\n3. LEARNING - Distilling into LoRA packet...")
    packet = await agent.learn(simulation, priority=8)
    if packet:
        print(f"   • Packet ID: {packet.packet_id[:8]}...")
        print(f"   • Target layers: {len(packet.target_layers)}")
        print(f"   • Adaptation strength: {packet.adaptation_strength:.3f}")
        print(f"   • Priority: {packet.priority}")
    
    # Get statistics
    print("\n4. STATISTICS")
    stats = agent.get_statistics()
    print(f"   • Execution history: {stats['execution_history_size']} steps")
    if 'distiller' in stats:
        print(f"   • Packets generated: {stats['distiller']['packets_generated']}")
        print(f"   • Packets injected: {stats['distiller']['packets_injected']}")
    if 'auditor' in stats:
        print(f"   • Total audits: {stats['auditor']['total_audits']}")
        print(f"   • Corrections applied: {stats['auditor']['corrections_applied']}")


async def evolution_example():
    """Example showing multi-step evolution"""
    print("\n\n=== Evolution Example ===\n")
    
    agent = PlasmaAgent(
        state_dim=64,
        simulation_horizon=3,
        num_simulations=2,
    )
    
    print("Starting evolution for 5 steps...")
    traces = await agent.evolve(steps=5)
    
    print(f"\n✓ Completed {len(traces)} evolution steps")
    print("\nReward progression:")
    for i, trace in enumerate(traces):
        print(f"  Step {i+1}: {trace.reward:.3f} (action: {trace.action_taken.value if trace.action_taken else 'None'})")
    
    total_reward = sum(t.reward for t in traces)
    avg_reward = total_reward / len(traces)
    print(f"\nTotal reward: {total_reward:.3f}")
    print(f"Average reward: {avg_reward:.3f}")


async def custom_policy_example():
    """Example with custom policy"""
    print("\n\n=== Custom Policy Example ===\n")
    
    # Define a simple policy that prefers COMPUTE actions
    def compute_policy(state):
        # Simple heuristic: prefer COMPUTE if state norm is high
        state_norm = np.linalg.norm(state)
        if state_norm > 0.5:
            return ActionType.COMPUTE
        else:
            return ActionType.OBSERVE
    
    agent = PlasmaAgent(state_dim=32)
    agent.initialize_state()
    
    print("Using custom policy that prefers COMPUTE actions...")
    simulation = await agent.think(policy=compute_policy)
    
    print("\nActions in best trajectory:")
    for i, (state, action, reward) in enumerate(simulation.simulated_trajectory):
        print(f"  Step {i+1}: {action.value} (reward: {reward:.3f})")


async def auditing_example():
    """Example showing judicial auditing"""
    print("\n\n=== Judicial Auditing Example ===\n")
    
    agent = PlasmaAgent(
        state_dim=64,
        enable_auditing=True,
    )
    
    print("Running evolution with judicial oversight...")
    traces = await agent.evolve(steps=10)
    
    stats = agent.get_statistics()
    auditor_stats = stats.get('auditor', {})
    
    print(f"\n✓ Completed with auditing enabled")
    print(f"   • Total audits: {auditor_stats.get('total_audits', 0)}")
    print(f"   • Corrections applied: {auditor_stats.get('corrections_applied', 0)}")
    print(f"   • Detection breakdown: {auditor_stats.get('detections', {})}")
    
    if agent.auditor:
        recent_audits = agent.auditor.get_recent_audits(count=3, flagged_only=True)
        if recent_audits:
            print(f"\n   Recent flagged audits:")
            for audit in recent_audits:
                print(f"     - Anomaly score: {audit.anomaly_score:.3f}")
                print(f"       Flags: {', '.join(audit.adversarial_flags)}")
                print(f"       Corrective action: {audit.corrective_action}")


async def distillation_example():
    """Example showing LoRA distillation"""
    print("\n\n=== LoRA Distillation Example ===\n")
    
    agent = PlasmaAgent(
        state_dim=64,
        enable_distillation=True,
    )
    
    print("Running with LoRA distillation enabled...")
    
    # Evolve and distill
    for i in range(3):
        simulation = await agent.think()
        trace = await agent.act(simulation)
        packet = await agent.learn(simulation, priority=i+1)
        
        print(f"\nStep {i+1}:")
        print(f"  • Reward: {trace.reward:.3f}")
        if packet:
            print(f"  • Generated LoRA packet (priority {packet.priority})")
            print(f"  • Target layers: {packet.target_layers[:2]}...")
    
    if agent.distiller:
        stats = agent.distiller.get_statistics()
        print(f"\nDistillation summary:")
        print(f"  • Packets generated: {stats['packets_generated']}")
        print(f"  • Queue size: {stats['queue_size']}")
        print(f"  • Distillation rate: {stats['distillation_rate']}")


async def main():
    """Run all examples"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           PROJECT PLASMA - USAGE EXAMPLES                  ║")
    print("║  Next-gen Agent-Lightning with Generative World Models     ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    try:
        await basic_example()
        await evolution_example()
        await custom_policy_example()
        await auditing_example()
        await distillation_example()
        
        print("\n\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
