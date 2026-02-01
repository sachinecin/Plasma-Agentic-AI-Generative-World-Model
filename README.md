# Plasma Agentic AI - Generative World Model

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A next-gen evolution of Agent Lightning that replaces heavy RL training with Phantom-Path Simulations and LoRA Distillation. It uses a generative "World Model" to practice tasks in a sandbox, deploying tiny "instruction packets" for real-time edge adaptation. It features visual error-learning and adversarial auditing to prevent reward hacking.

## 🚀 Features

- **🎮 Phantom-Path Simulation**: Generative world model creates diverse training scenarios without real environment interaction
- **⚡ LoRA Distillation**: Lightweight instruction packets for real-time edge adaptation
- **🛡️ Adversarial Auditing**: Prevents reward hacking with systematic judicial review
- **📊 Pydantic State Tracking**: Type-safe atomic span management with full execution tracing
- **🔄 Asyncio Event Loop**: High-performance concurrent operations throughout the framework

## 📁 Repository Structure

```
plasma/
├── simulator/          # Generative World Model for Phantom-Paths
│   ├── world_model.py          # State transition generation
│   ├── phantom_path.py         # Training trajectory creation
│   └── simulator.py            # Main async orchestrator
├── distiller/          # LoRA 'Instruction-Packet' injection logic
│   ├── instruction_packet.py   # Lightweight adaptation units
│   ├── lora_injector.py        # Real-time weight injection
│   └── distiller.py            # Distillation orchestrator
├── auditor/            # Adversarial Judicial Review
│   ├── adversarial_checker.py  # Exploit detection
│   ├── review_engine.py        # Systematic analysis
│   └── auditor.py              # Complete audit orchestration
└── trace/              # Pydantic-based state tracking for Atomic Spans
    ├── models.py               # Pydantic models (TraceState, SpanMetadata)
    ├── atomic_span.py          # Execution unit tracker
    └── state_tracker.py        # Main tracking orchestrator

scripts/
├── train.py            # Training entry point
├── eval.py             # Evaluation entry point
└── example.py          # End-to-end workflow example
```

## 🔧 Installation

### Using UV (Recommended)

UV is a fast Python package installer and resolver. Install it first if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install Project Plasma:

```bash
# Clone the repository
git clone https://github.com/sachinecin/Plasma-Agentic-AI-Generative-World-Model.git
cd Plasma-Agentic-AI-Generative-World-Model

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# For development
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/sachinecin/Plasma-Agentic-AI-Generative-World-Model.git
cd Plasma-Agentic-AI-Generative-World-Model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# For development
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Run the Example Workflow

```bash
python scripts/example.py
```

This demonstrates the complete Project Plasma workflow:
1. Phantom path generation with the simulator
2. Adversarial audit of generated paths
3. LoRA distillation of instruction packets
4. Packet deployment
5. State tracking throughout

### Training

```bash
python scripts/train.py --iterations 10 --num-paths 5 --path-length 20
```

Options:
- `--iterations`: Number of training iterations (default: 10)
- `--num-paths`: Number of phantom paths per iteration (default: 5)
- `--path-length`: Length of each phantom path (default: 20)

### Evaluation

```bash
python scripts/eval.py --episodes 10 --max-steps 100
```

Options:
- `--episodes`: Number of evaluation episodes (default: 10)
- `--max-steps`: Maximum steps per episode (default: 100)

## 💻 Usage Example

```python
import asyncio
from plasma import Simulator, Distiller, Auditor, StateTracker

async def main():
    # Initialize components
    simulator = Simulator()
    distiller = Distiller()
    auditor = Auditor()
    tracker = StateTracker()
    
    # Start all components
    await asyncio.gather(
        simulator.start(),
        distiller.start(),
        auditor.start(),
        tracker.start()
    )
    
    # Create trace
    trace_id = tracker.create_trace()
    
    # Generate phantom paths
    paths = await simulator.simulate(
        initial_state={"step": 0},
        num_paths=10,
        path_length=100
    )
    
    # Audit paths
    audit = await auditor.audit_paths(paths)
    
    if audit["passed"]:
        # Distill and deploy
        packets = await distiller.distill_from_paths(paths)
        results = await distiller.deploy_packets(packets)
        print(f"Deployed {results['deployed']} packets")
    
    # Complete trace
    await tracker.complete_trace(trace_id)
    
    # Cleanup
    await asyncio.gather(
        simulator.stop(),
        distiller.stop(),
        auditor.stop(),
        tracker.stop()
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## 📚 Documentation

For detailed API documentation and usage examples, see [DOCUMENTATION.md](DOCUMENTATION.md).

## 🧪 Development

### Running Tests

```bash
pytest tests/
pytest --cov=plasma tests/  # With coverage
```

### Code Formatting

```bash
black plasma/ scripts/ tests/
isort plasma/ scripts/ tests/
```

### Linting

```bash
ruff check plasma/ scripts/ tests/
```

### Type Checking

```bash
mypy plasma/
```

## 🏗️ Architecture Highlights

### Asyncio-Based Concurrency
All core components use asyncio for maximum performance:
- Parallel phantom path generation
- Concurrent packet injection
- Simultaneous auditing and tracking

### Pydantic Type Safety
Full type safety with Pydantic models:
- Runtime validation
- Automatic schema generation
- Easy serialization/deserialization

### Modular Design
Clean separation of concerns:
- **Simulator**: Generates training scenarios
- **Distiller**: Extracts and deploys knowledge
- **Auditor**: Ensures safety and integrity
- **Tracker**: Monitors execution

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. Code is formatted with Black
2. Tests pass with pytest
3. Type hints are included
4. Documentation is updated

## 🔗 Links

- [Documentation](DOCUMENTATION.md)
- [GitHub Repository](https://github.com/sachinecin/Plasma-Agentic-AI-Generative-World-Model)
