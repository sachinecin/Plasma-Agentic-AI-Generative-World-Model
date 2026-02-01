# Project Plasma - Quickstart Guide

## Introduction

Project Plasma is a next-generation AI agent system that replaces heavy reinforcement learning with:
- **Phantom-Path Simulation**: Think ahead by simulating possible futures
- **LoRA Distillation**: Learn by extracting tiny weight updates
- **Judicial Auditing**: Stay safe with adversarial oversight

## Installation (5 minutes)

```bash
# Clone the repository
git clone https://github.com/sachinecin/Plasma-Agentic-AI-Generative-World-Model.git
cd Plasma-Agentic-AI-Generative-World-Model

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from plasma import PlasmaAgent; print('✓ Plasma installed!')"
```

## Your First Agent (2 minutes)

Create `my_agent.py`:

```python
import asyncio
from plasma import PlasmaAgent

async def main():
    # Create agent
    agent = PlasmaAgent(
        state_dim=64,           # State vector size
        simulation_horizon=5,   # Look 5 steps ahead
        num_simulations=3,      # Try 3 different paths
    )
    
    # Initialize with random state
    agent.initialize_state()
    print("✓ Agent initialized")
    
    # Think: Simulate possible futures
    simulation = await agent.think()
    print(f"✓ Simulated {len(simulation.simulated_trajectory)} steps")
    print(f"  Success probability: {simulation.success_probability:.2%}")
    
    # Act: Take the best action
    trace = await agent.act(simulation)
    print(f"✓ Executed action: {trace.action_taken.value}")
    print(f"  Reward: {trace.reward:.3f}")
    
    # Learn: Extract LoRA weights
    packet = await agent.learn(simulation)
    print(f"✓ Generated LoRA packet for {len(packet.target_layers)} layers")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python my_agent.py
```

Expected output:
```
✓ Agent initialized
✓ Simulated 5 steps
  Success probability: 52%
✓ Executed action: compute
  Reward: 0.142
✓ Generated LoRA packet for 7 layers
```

## Key Concepts

### 1. Think-Act-Learn Loop

The core cycle of every Plasma agent:

```python
# THINK: Simulate multiple possible futures in parallel
simulation = await agent.think()

# ACT: Execute the best action from the best future
trace = await agent.act(simulation)

# LEARN: Distill the experience into LoRA weights
packet = await agent.learn(simulation)
```

### 2. Phantom Paths

Instead of learning through trial-and-error, Plasma **simulates** multiple possible futures:

```python
# Agent simulates 5 different paths in parallel
agent = PlasmaAgent(num_simulations=5)

# Each path explores a different sequence of actions
# The best path is selected automatically
simulation = await agent.think()
```

### 3. LoRA Distillation

Learning happens by extracting **tiny weight updates** (LoRA packets) that can be injected into edge agents:

```python
# Distill simulation into LoRA weights
packet = agent.learn(simulation)

# Packet contains:
# - Layer-wise weight deltas
# - Target layers for injection
# - Adaptation strength
# - Priority for scheduling
```

### 4. Judicial Auditing

Every action and weight update is monitored for anomalies:

```python
# Auditing happens automatically
agent = PlasmaAgent(enable_auditing=True)

# Check audit statistics
stats = agent.get_statistics()
print(f"Corrections applied: {stats['auditor']['corrections_applied']}")
```

## Common Patterns

### Multi-Step Evolution

```python
# Evolve for 10 steps
traces = await agent.evolve(steps=10)

# Calculate total reward
total_reward = sum(t.reward for t in traces)
print(f"Total reward: {total_reward:.2f}")
```

### Custom Policy

```python
from plasma.state_traces import ActionType

def my_policy(state):
    """Custom decision-making logic"""
    state_energy = sum(abs(x) for x in state)
    
    if state_energy > 1.0:
        return ActionType.COMPUTE
    else:
        return ActionType.OBSERVE

# Use your policy
simulation = await agent.think(policy=my_policy)
```

### Background Evolution

```python
# Start continuous evolution
await agent.start_continuous_evolution()

# Do other work...
await asyncio.sleep(60)

# Stop when done
await agent.stop_continuous_evolution()
```

## Next Steps

### Run Examples
```bash
python examples.py
```

This runs 5 comprehensive examples showing:
- Basic think-act-learn cycle
- Multi-step evolution
- Custom policy usage
- Judicial auditing
- LoRA distillation

### Run Tests
```bash
pip install pytest
PYTHONPATH=src pytest tests/
```

### Read Documentation
- [API Reference](API.md) - Complete API documentation
- [README](README.md) - Architecture and technical details

## Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'plasma'`

**Solution**: Make sure to set PYTHONPATH:
```bash
export PYTHONPATH=/path/to/Plasma-Agentic-AI-Generative-World-Model/src:$PYTHONPATH
```

Or install as package:
```bash
pip install -e .
```

---

**Problem**: Simulations give low quality scores

**Solution**: Increase simulation horizon and number of simulations:
```python
agent = PlasmaAgent(
    simulation_horizon=20,    # Look further ahead
    num_simulations=10,       # Try more paths
)
```

---

**Problem**: Too many audit corrections

**Solution**: Adjust audit thresholds:
```python
from plasma import JudicialAuditor

auditor = JudicialAuditor(
    anomaly_threshold=0.8,        # Higher = less sensitive
    reward_spike_threshold=3.0,   # Higher = less sensitive
)
```

## Tips for Best Results

1. **Start small**: Begin with `state_dim=32` and `simulation_horizon=5`
2. **Tune gradually**: Increase complexity as you understand the system
3. **Monitor statistics**: Use `agent.get_statistics()` to track performance
4. **Custom policies**: Implement domain-specific logic in your policy function
5. **Batch operations**: Process multiple decisions in parallel when possible

## Getting Help

- Check [examples.py](examples.py) for working code
- Read [API.md](API.md) for detailed API reference
- Open an issue on GitHub for bugs or questions

## What's Next?

You now know the basics of Project Plasma! Try:
- Implementing a custom policy for your domain
- Adjusting simulation parameters for your use case
- Integrating with a real environment
- Extending the world model with learned components

Happy coding! 🚀
