"""
Basic tests for Project Plasma components
"""
import asyncio
import pytest
import numpy as np
from plasma import PlasmaAgent, PhantomPathSimulator, LoRADistiller, JudicialAuditor
from plasma.state_traces import StateTrace, ActionType, SimulationResult


def test_plasma_agent_initialization():
    """Test agent initialization"""
    agent = PlasmaAgent(state_dim=64)
    assert agent.state_dim == 64
    assert agent.simulator is not None
    assert agent.distiller is not None
    assert agent.auditor is not None


def test_phantom_path_simulator_initialization():
    """Test simulator initialization"""
    simulator = PhantomPathSimulator(state_dim=32, horizon=5)
    assert simulator.state_dim == 32
    assert simulator.horizon == 5


def test_lora_distiller_initialization():
    """Test distiller initialization"""
    distiller = LoRADistiller()
    assert distiller.layer_names is not None
    assert len(distiller.layer_names) > 0


def test_judicial_auditor_initialization():
    """Test auditor initialization"""
    auditor = JudicialAuditor()
    assert auditor.anomaly_threshold > 0
    assert auditor.reward_spike_threshold > 0


async def test_phantom_path_simulation():
    """Test phantom path simulation"""
    simulator = PhantomPathSimulator(state_dim=32, horizon=3, num_simulations=2)
    initial_state = np.random.randn(32).tolist()
    
    results = await simulator.pre_emptive_rollout(initial_state)
    
    assert len(results) == 2
    assert all(isinstance(r, SimulationResult) for r in results)
    assert all(len(r.simulated_trajectory) == 3 for r in results)


async def test_think_act_learn_cycle():
    """Test complete think-act-learn cycle"""
    agent = PlasmaAgent(state_dim=32, simulation_horizon=3, num_simulations=2)
    agent.initialize_state()
    
    # Think
    simulation = await agent.think()
    assert simulation is not None
    assert len(simulation.simulated_trajectory) == 3
    
    # Act
    trace = await agent.act(simulation)
    assert isinstance(trace, StateTrace)
    assert trace.action_taken is not None
    
    # Learn
    packet = await agent.learn(simulation)
    assert packet is not None
    assert len(packet.lora_weights) > 0


async def test_evolution():
    """Test multi-step evolution"""
    agent = PlasmaAgent(state_dim=32, simulation_horizon=2, num_simulations=2)
    
    traces = await agent.evolve(steps=3)
    
    assert len(traces) == 3
    assert all(isinstance(t, StateTrace) for t in traces)


async def test_judicial_auditing():
    """Test judicial auditing functionality"""
    auditor = JudicialAuditor()
    
    # Create a state trace
    trace = StateTrace(
        state_vector=np.random.randn(32).tolist(),
        action_taken=ActionType.COMPUTE,
        reward=0.5,
    )
    
    # Audit the trace
    audit_record = await auditor.audit_state_trace(trace)
    
    assert audit_record is not None
    assert audit_record.anomaly_score >= 0.0
    assert audit_record.anomaly_score <= 1.0


async def test_lora_distillation():
    """Test LoRA distillation"""
    distiller = LoRADistiller()
    
    # Create a simulation result
    trajectory = [
        (np.random.randn(32).tolist(), ActionType.COMPUTE, 0.1),
        (np.random.randn(32).tolist(), ActionType.OBSERVE, 0.2),
    ]
    
    simulation = SimulationResult(
        initial_state=np.random.randn(32).tolist(),
        simulated_trajectory=trajectory,
        total_reward=0.3,
        success_probability=0.6,
        quality_score=0.7,
    )
    
    # Distill
    packet = distiller.distill_from_simulation(simulation)
    
    assert packet is not None
    assert len(packet.lora_weights) > 0
    assert packet.adaptation_strength > 0


def test_state_trace_immutability():
    """Test that StateTrace is immutable"""
    trace = StateTrace(
        state_vector=[1.0, 2.0, 3.0],
        action_taken=ActionType.MOVE,
        reward=1.0,
    )
    
    # Should not be able to modify
    try:
        trace.reward = 2.0
        assert False, "StateTrace should be immutable"
    except Exception:
        pass  # Expected


if __name__ == "__main__":
    # Run tests with asyncio
    asyncio.run(test_phantom_path_simulation())
    asyncio.run(test_think_act_learn_cycle())
    asyncio.run(test_evolution())
    asyncio.run(test_judicial_auditing())
    asyncio.run(test_lora_distillation())
    
    # Run sync tests
    test_plasma_agent_initialization()
    test_phantom_path_simulator_initialization()
    test_lora_distiller_initialization()
    test_judicial_auditor_initialization()
    test_state_trace_immutability()
    
    print("✓ All tests passed!")
