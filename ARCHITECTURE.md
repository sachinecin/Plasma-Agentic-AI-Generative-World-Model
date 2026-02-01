# Architecture Documentation

## Training-Agent Disaggregation (TA) Design

This document describes the architectural patterns used in the Plasma Agentic AI Generative World Model, following Microsoft's Agent-Lightning Training-Agent Disaggregation (TA) design.

## Overview

The TA design separates the heavy generative World Model (server-side) from lightweight instruction packets (edge-deployable), enabling efficient training and deployment.

## Components

### 1. Core Simulation Engine (`/core`)

**Purpose**: Orchestrates phantom-path simulations and LoRA training.

**Key Classes**:
- `SimulationEngine`: Generates trajectories using the World Model
- `CoreConfig`: Configuration for simulation parameters

**Responsibilities**:
- Generate phantom-path trajectories
- Run adversarial audits on trajectories
- Train LoRA adapters from simulated data
- Track simulation statistics

### 2. World Model (`/world_model`)

**Purpose**: Generative environment for simulations (runs server-side).

**Key Classes**:
- `WorldModel`: The generative model that simulates environments
- `WorldModelConfig`: Model architecture configuration

**Responsibilities**:
- Predict next states given current state and action
- Generate complete trajectory rollouts
- Visual error prediction (what could go wrong)
- Adversarial auditing to prevent reward hacking
- Fine-tuning on real-world data

**Key Innovation**: Replaces real-world data collection with generative simulations.

### 3. Instruction Packets (`/instruction_packets`)

**Purpose**: Lightweight agent logic for edge deployment.

**Key Classes**:
- `InstructionPacketCompiler`: Compiles LoRA to deployable packets
- `InstructionPacketRuntime`: Executes packets on edge devices
- `PacketConfig`: Configuration for compilation and deployment

**Responsibilities**:
- Compile trained LoRA adapters into tiny packets (~100KB)
- Apply quantization and compression
- Optimize for target devices (edge, mobile, server)
- Provide fast inference runtime (<10ms latency)
- Enable offline operation (no server needed)

**Key Innovation**: Enables edge AI with minimal resources.

### 4. Server (`server.py`)

**Purpose**: World Model server implementing TA server-side.

**Architecture**: FastAPI-based REST and WebSocket server.

**Endpoints**:
```
GET  /health                           - Health check
GET  /status                           - Detailed status
POST /simulate                         - Request phantom-path simulation
GET  /simulate/{simulation_id}         - Get simulation status
POST /train                            - Start LoRA training
GET  /train/history/{agent_id}         - Get training history
POST /instruction-packet/compile       - Compile instruction packet
GET  /instruction-packet/download/{id} - Download packet
WS   /ws/{client_id}                   - WebSocket for real-time updates
```

**Key Features**:
- Asynchronous operation with asyncio
- Background task processing for simulations
- WebSocket support for streaming updates
- Graceful startup and shutdown
- Connection pooling for multiple clients

### 5. Client (`client.py`)

**Purpose**: Agent client implementing TA client-side.

**Architecture**: Async Python client using httpx and websockets.

**Key Methods**:
```python
async def request_simulation()          # Request server-side simulation
async def wait_for_simulation()         # Wait for simulation completion
async def start_training()              # Start LoRA training on server
async def compile_instruction_packet()  # Request packet compilation
async def download_instruction_packet() # Download compiled packet
async def load_instruction_packet()     # Load packet for inference
async def execute_task()                # Edge inference using packet
async def train_and_deploy()            # Complete workflow
```

**Key Features**:
- Async/await for efficient I/O
- Context manager for resource management
- Retry logic and error handling
- Local caching of instruction packets
- Offline inference after deployment

### 6. Community Contributions (`/contrib`)

**Purpose**: Community recipes and examples.

**Structure**:
```
contrib/
├── recipes/
│   ├── basic_simulation.py      # Simple simulation example
│   ├── lora_training.py         # LoRA training tutorial
│   └── client_server_demo.py    # Full TA workflow demo
├── environments/                # Custom environment definitions
├── models/                      # Alternative model architectures
└── experiments/                 # Research experiments
```

## Workflow

### Complete TA Pipeline

```
1. Client Request
   ├─ Task Description → Server
   └─ Parameters (num_sims, lora_rank, etc.)

2. Server-Side Simulation
   ├─ World Model generates phantom-paths
   ├─ Adversarial audit on trajectories
   └─ Cache trajectories

3. Server-Side Training
   ├─ LoRA adapter training
   ├─ Training on simulated data
   └─ Save LoRA checkpoint

4. Packet Compilation
   ├─ Load LoRA weights
   ├─ Quantization (float32 → int8)
   ├─ Compression (pruning, encoding)
   └─ Device-specific optimization

5. Client Deployment
   ├─ Download instruction packet (~100KB)
   ├─ Load into runtime
   └─ Ready for edge inference

6. Edge Inference (Offline)
   ├─ No server connection needed
   ├─ <10ms latency
   └─ Minimal memory footprint
```

## Design Principles

### 1. Separation of Concerns

- **Training** (server-side): Heavy computation, generative models, simulations
- **Inference** (client-side): Lightweight, fast, offline-capable

### 2. Efficiency

- **Training**: Use simulations instead of real-world data collection
- **Deployment**: 1000x smaller than full models
- **Inference**: Sub-10ms latency on edge devices

### 3. Safety

- **Adversarial Auditing**: Detect reward hacking in simulations
- **Visual Error Prediction**: Anticipate failure modes
- **Validation**: Test in sandbox before real deployment

### 4. Scalability

- **One Server**: Supports many edge agents
- **Parallel Simulations**: Multiple trajectories simultaneously
- **Distributed Training**: Future: multiple GPU nodes

### 5. Modularity

- **Pluggable Models**: Swap World Model architectures
- **Custom Environments**: Add new task domains
- **Flexible Configuration**: Easy parameter tuning

## Configuration

### Environment Variables

```bash
# Server configuration
WORLD_MODEL_SERVER_HOST=0.0.0.0
WORLD_MODEL_SERVER_PORT=8000

# World Model settings
WORLD_MODEL_CHECKPOINT_PATH=./checkpoints/world_model.pt
WORLD_MODEL_DEVICE=cuda

# Training settings
LORA_RANK=8
LORA_ALPHA=16
LEARNING_RATE=1e-4

# Packet compilation
TARGET_DEVICE=edge
QUANTIZATION=int8
COMPRESSION_LEVEL=5
```

### Configuration Files

All modules support configuration via Python dataclasses:

```python
# Core configuration
from core.config import CoreConfig
config = CoreConfig(
    max_simulation_steps=1000,
    parallel_simulations=4,
    use_adversarial_audit=True
)

# World Model configuration
from world_model.config import WorldModelConfig
config = WorldModelConfig(
    model_type="diffusion",
    hidden_dim=512,
    num_layers=8
)

# Packet configuration
from instruction_packets.config import PacketConfig
config = PacketConfig(
    target_device="edge",
    quantization="int8",
    compression_level=5
)
```

## Testing

### Unit Tests

Located in `/tests`:

- `test_simulation_engine.py` - Core simulation tests
- `test_world_model.py` - World Model tests
- `test_instruction_packets.py` - Packet compiler/runtime tests

Run with:
```bash
pytest tests/
```

### Integration Tests

Test the complete TA workflow:

```bash
# Terminal 1: Start server
python server.py

# Terminal 2: Run client tests
python contrib/recipes/client_server_demo.py
```

## Performance Characteristics

### Server (World Model)

- **Memory**: ~4-8GB (depends on model size)
- **GPU**: Recommended (CUDA capable)
- **Simulation Speed**: ~100 trajectories/second
- **Training Time**: Minutes to hours (depends on task complexity)

### Client (Instruction Packet)

- **Packet Size**: ~100KB (after compression)
- **Memory**: <100MB runtime footprint
- **Inference Latency**: <10ms per decision
- **Device**: CPU only (no GPU needed)

## Future Enhancements

1. **Multi-Agent Coordination**: Multiple agents sharing World Model
2. **Federated Learning**: Update World Model from real-world feedback
3. **Continual Learning**: Agents adapt to new tasks over time
4. **Model Zoo**: Pre-trained World Models for common domains
5. **Hardware Acceleration**: ONNX/TensorRT optimization
6. **Mobile Deployment**: iOS/Android SDKs

## References

- Microsoft Agent-Lightning architecture
- Training-Agent Disaggregation (TA) design pattern
- LoRA: Low-Rank Adaptation of Large Language Models
- World Models for reinforcement learning
