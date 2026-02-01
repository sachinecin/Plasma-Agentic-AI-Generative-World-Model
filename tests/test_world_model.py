"""
Tests for World Model.
"""

import pytest
from world_model.generative_env import WorldModel
from world_model.config import WorldModelConfig


def test_world_model_initialization():
    """Test World Model initialization."""
    model = WorldModel()
    assert model is not None
    assert model.is_loaded is False


def test_world_model_with_config():
    """Test World Model with configuration."""
    config = WorldModelConfig(
        model_type="diffusion",
        hidden_dim=512,
        num_layers=8
    )
    model = WorldModel(config=config)
    assert model.config == config


def test_rollout():
    """Test phantom-path rollout generation."""
    model = WorldModel()
    trajectory = model.rollout(
        task_embedding="test_task",
        max_steps=100
    )
    
    assert 'states' in trajectory
    assert 'actions' in trajectory
    assert 'rewards' in trajectory
    assert 'done' in trajectory


def test_visual_error_prediction():
    """Test visual error prediction."""
    model = WorldModel()
    errors = model.visual_error_prediction(
        state=None,
        action=None,
        task_context=None
    )
    
    assert 'collision_risk' in errors
    assert 'goal_violation' in errors
    assert 'unsafe_state' in errors
    assert 'reward_hacking_detected' in errors


def test_adversarial_audit():
    """Test adversarial audit."""
    model = WorldModel()
    trajectory = {'states': [], 'actions': [], 'rewards': []}
    
    audit_report = model.adversarial_audit(trajectory)
    
    assert 'passed' in audit_report
    assert 'confidence' in audit_report
    assert 'flags' in audit_report
    assert 'anomaly_score' in audit_report
