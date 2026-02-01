# Plasma-Agentic-AI-Generative-World-Model

Next-gen evolution of Agent Lightning that replaces heavy RL training with Phantom-Path Simulations and LoRA Distillation. It uses a generative "World Model" to practice tasks in a sandbox, deploying tiny "instruction packets" for real-time edge adaptation. It features visual error-learning and adversarial auditing to prevent reward hacking.

## Architecture Overview

This project follows Microsoft's **Agent-Lightning** architectural patterns with **Training-Agent Disaggregation (TA)** design. The system separates the heavy generative World Model from lightweight instruction packets that can be deployed to edge devices.

### Key Components

```
├── core/                      # Simulation engine and core infrastructure
│   ├── simulation_engine.py   # Phantom-path trajectory generation
│   └── config.py              # Core configuration
│
├── world_model/               # Generative environment (server-side)
│   ├── generative_env.py      # World Model for simulations
│   └── config.py              # World Model configuration
│
├── instruction_packets/       # Agent logic (edge-deployable)
│   ├── compiler.py            # Compile LoRA to instruction packets
│   ├── runtime.py             # Edge inference runtime
│   └── config.py              # Packet configuration
│
├── contrib/                   # Community recipes and examples
│   └── recipes/               # Example implementations
│
├── server.py                  # World Model server (TA design)
└── client.py                  # Agent client (TA design)
```

## Training-Agent Disaggregation (TA) Design

The TA architecture separates concerns between training and deployment:

**Server Side (Heavy):**
- Hosts the generative World Model
- Runs phantom-path simulations
- Trains LoRA adapters
- Compiles instruction packets

**Client Side (Lightweight):**
- Sends task descriptions to server
- Downloads instruction packets (~100KB)
- Runs real-time inference on edge devices
- No GPU or server connection needed for inference

This design enables:
- **Efficient Training**: Simulations in generative environment (no real-world data needed)
- **Edge Deployment**: Tiny instruction packets run on resource-constrained devices
- **Scalability**: One server supports many edge agents
- **Safety**: Adversarial auditing prevents reward hacking

## Installation

### Basic Installation

```bash
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### With Community Recipes

```bash
pip install -e ".[contrib]"
```

## Quick Start

### 1. Start the World Model Server

```bash
python server.py
```

The server will start on `http://localhost:8000` with the following endpoints:

- `GET /health` - Health check
- `POST /simulate` - Request phantom-path simulation
- `POST /train` - Start LoRA training
- `POST /instruction-packet/compile` - Compile instruction packet
- `WS /ws/{client_id}` - WebSocket for real-time updates

### 2. Use the Agent Client

```python
import asyncio
from client import PlasmaAgentClient, ClientConfig, TaskRequest

async def train_agent():
    # Configure client
    config = ClientConfig(
        agent_id="my_agent",
        server_url="http://localhost:8000"
    )
    
    # Connect and train
    async with PlasmaAgentClient(config) as client:
        # Define task
        task = TaskRequest(
            task_name="navigate_maze",
            task_description="Navigate through a maze to reach the goal",
            num_simulations=1000,
            lora_rank=8
        )
        
        # Run full training pipeline
        packet_path = await client.train_and_deploy(task)
        
        # Load instruction packet for edge inference
        await client.load_instruction_packet(packet_path)
        
        # Now ready for real-time inference!
        print("Agent deployed and ready!")

asyncio.run(train_agent())
```

### 3. Run Example Recipes

```bash
# Basic simulation example
python contrib/recipes/basic_simulation.py

# LoRA training example
python contrib/recipes/lora_training.py

# Full client-server workflow (requires server running)
python contrib/recipes/client_server_demo.py
```

## Key Features

### 1. Phantom-Path Simulations

Instead of learning from real-world interactions, agents practice in a generative World Model:

```python
from core.simulation_engine import SimulationEngine

sim_engine = SimulationEngine(world_model=my_world_model)
trajectories = await sim_engine.generate_trajectories(
    task="Pick up red cube",
    num_trajectories=100,
    adversarial=True  # Enable adversarial auditing
)
```

### 2. LoRA Distillation

Train tiny adapters instead of full models:

```python
training_result = await sim_engine.train_lora(
    agent_id="agent_001",
    task_name="object_manipulation",
    num_simulations=1000,
    lora_config={'rank': 8, 'lr': 1e-4}
)
```

### 3. Instruction Packet Compilation

Compile trained LoRA into deployable packets:

```python
from instruction_packets.compiler import InstructionPacketCompiler

compiler = InstructionPacketCompiler()
packet_info = await compiler.compile(
    lora_path="./checkpoints/agent_001_lora.pt",
    agent_id="agent_001",
    target_device="edge"
)
# Result: ~100KB packet ready for deployment
```

### 4. Edge Inference

Run inference on edge devices without server connection:

```python
from instruction_packets.runtime import InstructionPacketRuntime

runtime = InstructionPacketRuntime()
runtime.load_packet("./packets/agent_001_packet.bin")

# Real-time inference (<10ms latency)
action = runtime.forward(observation)
```

### 5. Adversarial Auditing

Prevent reward hacking with built-in adversarial auditing:

```python
trajectories = await sim_engine.generate_trajectories(
    task="Reach goal without breaking rules",
    adversarial=True  # Flags suspicious patterns
)
# Trajectories include audit scores and flags
```

## API Documentation

### Server API

**Health Check**
```bash
curl http://localhost:8000/health
```

**Request Simulation**
```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_001",
    "task_description": "Navigate maze",
    "num_trajectories": 10,
    "adversarial_audit": true
  }'
```

**Start Training**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_001",
    "task_name": "maze_navigation",
    "num_simulations": 1000,
    "lora_rank": 8,
    "learning_rate": 0.0001
  }'
```

### Client API

See `client.py` for the full Python client API.

## Configuration

### Core Configuration

```python
from core.config import CoreConfig

config = CoreConfig(
    world_model_path="./checkpoints/world_model.pt",
    max_simulation_steps=1000,
    parallel_simulations=4,
    use_adversarial_audit=True
)
```

### World Model Configuration

```python
from world_model.config import WorldModelConfig

config = WorldModelConfig(
    model_type="diffusion",
    hidden_dim=512,
    num_layers=8,
    image_size=256
)
```

### Instruction Packet Configuration

```python
from instruction_packets.config import PacketConfig

config = PacketConfig(
    target_device="edge",
    compression_level=5,
    quantization="int8",
    lora_rank=8
)
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
isort .
```

### Type Checking

```bash
mypy core/ world_model/ instruction_packets/
```

## Contributing

We welcome contributions! Please see `contrib/README.md` for guidelines on:

- Adding new recipes
- Contributing custom environments
- Sharing trained models
- Reporting issues

## Architecture Comparison

### Traditional RL Training
```
Agent → Environment → Rollouts → Training (GPU-intensive)
                                      ↓
                                 Full Model (100s of MB)
                                      ↓
                                  Deployment
```

### Plasma Agentic AI (TA Design)
```
Task Description → World Model Server → Phantom-Path Simulations
                                              ↓
                                        LoRA Training
                                              ↓
                                     Instruction Packet (~100KB)
                                              ↓
                                      Edge Deployment
```

**Benefits:**
- ✅ No real-world data collection needed
- ✅ 1000x smaller deployment packages
- ✅ <10ms inference latency
- ✅ Offline edge inference
- ✅ Adversarial safety guarantees

## Citation

If you use this project in your research, please cite:

```bibtex
@software{plasma_agentic_ai,
  title={Plasma Agentic AI Generative World Model},
  author={Plasma Agentic AI Team},
  year={2024},
  url={https://github.com/sachinecin/Plasma-Agentic-AI-Generative-World-Model}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project is inspired by Microsoft's Agent-Lightning architecture and builds upon the Training-Agent Disaggregation (TA) design pattern.

