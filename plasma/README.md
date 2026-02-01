# Plasma Core

This directory contains the core framework code for the Plasma Agentic AI Generative World Model.

## Structure

The plasma package implements the main components of the framework:

- **World Model**: Generative models for task simulation
- **Phantom-Path Simulations**: Lightweight training alternatives to heavy RL
- **LoRA Distillation**: Efficient model adaptation
- **Instruction Packets**: Real-time edge adaptation modules
- **Visual Error Learning**: Learning from visual feedback
- **Adversarial Auditing**: Reward hacking prevention

## Usage

```python
from plasma import WorldModel, PhantomPath

# Initialize the world model
model = WorldModel()

# Run phantom-path simulations
simulator = PhantomPath(model)
results = simulator.run_simulation()
```

## Development

This is the main package that will be distributed on PyPI. Keep dependencies minimal and ensure backward compatibility.
