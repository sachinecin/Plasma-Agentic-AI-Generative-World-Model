# Quick Start Guide

Get up and running with Plasma Agentic AI in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/sachinecin/Plasma-Agentic-AI-Generative-World-Model.git
cd Plasma-Agentic-AI-Generative-World-Model

# Install the package
pip install -e .
```

## Option 1: Run Basic Examples (No Server Needed)

These examples demonstrate the core concepts without requiring a running server:

### Example 1: Basic Simulation

```bash
python contrib/recipes/basic_simulation.py
```

This demonstrates:
- ✅ Phantom-path trajectory generation
- ✅ Adversarial auditing
- ✅ Simulation statistics

### Example 2: LoRA Training

```bash
python contrib/recipes/lora_training.py
```

This demonstrates:
- ✅ LoRA adapter training
- ✅ Instruction packet compilation
- ✅ Edge deployment workflow

## Option 2: Full Client-Server Workflow

This demonstrates the complete Training-Agent Disaggregation (TA) design.

### Step 1: Start the Server

In one terminal:

```bash
python server.py
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Run the Client

In another terminal:

```bash
python contrib/recipes/client_server_demo.py
```

This demonstrates:
- ✅ Client-server communication
- ✅ Remote simulation requests
- ✅ Server-side training
- ✅ Instruction packet download

### Step 3: Test the API

Test the REST API with curl:

```bash
# Health check
curl http://localhost:8000/health

# Request a simulation
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "demo_agent",
    "task_description": "Navigate a maze",
    "num_trajectories": 10
  }'
```

## Option 3: Interactive Python

Use the library interactively:

```python
import asyncio
from core import SimulationEngine, CoreConfig
from world_model import WorldModel, WorldModelConfig

async def demo():
    # Create World Model
    world_model = WorldModel(
        config=WorldModelConfig(model_type="diffusion")
    )
    
    # Create Simulation Engine
    sim_engine = SimulationEngine(
        world_model=world_model,
        config=CoreConfig(max_simulation_steps=100)
    )
    
    # Generate trajectories
    trajectories = await sim_engine.generate_trajectories(
        task="Pick up red cube",
        num_trajectories=5,
        adversarial=True
    )
    
    print(f"Generated {len(trajectories)} trajectories!")
    for i, traj in enumerate(trajectories):
        print(f"  Trajectory {i+1}: {traj['trajectory_id']}")

# Run the demo
asyncio.run(demo())
```

## Next Steps

### Learn the Architecture

Read the architecture documentation:
```bash
cat ARCHITECTURE.md
```

### Explore the Code

Key files to explore:
- `server.py` - World Model server with REST API
- `client.py` - Agent client for TA design
- `core/simulation_engine.py` - Core simulation logic
- `world_model/generative_env.py` - Generative World Model
- `instruction_packets/compiler.py` - Packet compilation

### Run Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Customize Configuration

Edit configuration in Python:

```python
from core.config import CoreConfig

config = CoreConfig(
    max_simulation_steps=1000,      # Longer simulations
    parallel_simulations=8,          # More parallel sims
    use_adversarial_audit=True      # Enable safety checks
)
```

### Add Your Own Recipe

Create a new recipe:

```bash
# Copy a template
cp contrib/recipes/basic_simulation.py contrib/recipes/my_recipe.py

# Edit it
nano contrib/recipes/my_recipe.py

# Run it
python contrib/recipes/my_recipe.py
```

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Make sure you installed the package
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Server Won't Start

If port 8000 is already in use:
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9

# Or change the port in server.py
```

### Missing Dependencies

If you get module not found errors:
```bash
# Install all dependencies
pip install -e ".[dev,contrib]"
```

## Getting Help

- 📖 Read the [README](README.md)
- 🏗️ Check [ARCHITECTURE.md](ARCHITECTURE.md)
- 🤝 See [CONTRIBUTING.md](CONTRIBUTING.md)
- 🐛 Open an issue on GitHub
- 💬 Start a discussion

## What's Next?

Now that you're up and running, you can:

1. **Train your own agent**: Modify recipes to train on custom tasks
2. **Deploy to edge**: Test instruction packets on edge devices
3. **Contribute**: Add your own recipes or improvements
4. **Research**: Experiment with different World Model architectures

Happy coding! 🚀
