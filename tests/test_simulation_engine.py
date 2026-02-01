"""
Basic tests for the core simulation engine.
"""

import pytest
from core.simulation_engine import SimulationEngine
from core.config import CoreConfig


def test_simulation_engine_initialization():
    """Test that the simulation engine can be initialized."""
    engine = SimulationEngine()
    assert engine is not None
    assert engine.simulation_count == 0
    assert engine.trajectory_cache == []


def test_simulation_engine_with_config():
    """Test simulation engine with configuration."""
    config = CoreConfig(
        max_simulation_steps=500,
        parallel_simulations=2,
        use_adversarial_audit=True
    )
    engine = SimulationEngine(config=config)
    assert engine.config == config


@pytest.mark.asyncio
async def test_generate_trajectories():
    """Test trajectory generation."""
    engine = SimulationEngine()
    trajectories = await engine.generate_trajectories(
        task="Test task",
        num_trajectories=5,
        adversarial=True
    )
    
    assert len(trajectories) == 5
    assert engine.simulation_count == 5
    
    # Check trajectory structure
    for traj in trajectories:
        assert 'trajectory_id' in traj
        assert 'task' in traj
        assert 'states' in traj
        assert 'actions' in traj
        assert 'rewards' in traj
        assert 'audited' in traj
        assert traj['audited'] is True


@pytest.mark.asyncio
async def test_train_lora():
    """Test LoRA training."""
    engine = SimulationEngine()
    result = await engine.train_lora(
        agent_id="test_agent",
        task_name="test_task",
        num_simulations=100,
        lora_config={'rank': 8, 'lr': 1e-4}
    )
    
    assert result['agent_id'] == "test_agent"
    assert result['task_name'] == "test_task"
    assert result['status'] == 'completed'
    assert 'checkpoint_path' in result


def test_get_statistics():
    """Test getting simulation statistics."""
    engine = SimulationEngine()
    stats = engine.get_statistics()
    
    assert 'total_simulations' in stats
    assert 'cached_trajectories' in stats
    assert 'world_model_loaded' in stats
    assert stats['total_simulations'] == 0
