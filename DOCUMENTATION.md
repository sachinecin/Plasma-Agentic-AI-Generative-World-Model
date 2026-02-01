# Project Plasma Documentation

## High-Performance Python Repository Structure

This repository implements a next-generation Agentic AI framework using Phantom-Path Simulations and LoRA Distillation.

## Architecture

### Core Modules

#### 1. `plasma/simulator/` - Generative World Model
The simulator module provides phantom path generation for training without heavy RL:

- **WorldModel**: Generates realistic state transitions
- **PhantomPathGenerator**: Creates diverse training trajectories
- **Simulator**: Main async orchestrator for simulations

```python
from plasma.simulator import Simulator

simulator = Simulator(config)
await simulator.start()
paths = await simulator.simulate(initial_state, num_paths=10, path_length=100)
```

#### 2. `plasma/distiller/` - LoRA Instruction-Packet Injection
Lightweight adaptation using LoRA for real-time edge deployment:

- **InstructionPacket**: Minimal adaptation units
- **LoRAInjector**: Real-time weight injection
- **Distiller**: Distillation orchestrator

```python
from plasma.distiller import Distiller

distiller = Distiller(config)
await distiller.start()
packets = await distiller.distill_from_paths(phantom_paths)
results = await distiller.deploy_packets(packets, model)
```

#### 3. `plasma/auditor/` - Adversarial Judicial Review
Prevents reward hacking and ensures model integrity:

- **AdversarialChecker**: Detects exploits and violations
- **ReviewEngine**: Systematic behavior analysis
- **Auditor**: Complete audit orchestration

```python
from plasma.auditor import Auditor

auditor = Auditor(config)
await auditor.start()
audit_result = await auditor.audit_paths(paths)
print(audit_result['overall_status'])  # APPROVED or REJECTED
```

#### 4. `plasma/trace/` - Pydantic State Tracking
Type-safe atomic span tracking with Pydantic models:

- **AtomicSpan**: Individual execution unit tracker
- **StateTracker**: Main tracking orchestrator
- **TraceState, SpanMetadata**: Pydantic models

```python
from plasma.trace import StateTracker

tracker = StateTracker(config)
await tracker.start()

trace_id = tracker.create_trace(metadata={"type": "training"})
span = await tracker.create_span(trace_id, "operation_name")

async with span:
    # Your operation here
    span.update_state({"key": "value"})
```

### Scripts

#### Training (`scripts/train.py`)
Main training entry point with async event loop:

```bash
python scripts/train.py --iterations 10 --num-paths 5 --path-length 20
```

Or using the installed command:
```bash
plasma-train --iterations 10 --num-paths 5 --path-length 20
```

#### Evaluation (`scripts/eval.py`)
Evaluation entry point for model testing:

```bash
python scripts/eval.py --episodes 10 --max-steps 100
```

Or using the installed command:
```bash
plasma-eval --episodes 10 --max-steps 100
```

## Installation

### Using UV (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Development

### Running Tests

```bash
pytest tests/
pytest --cov=plasma tests/  # With coverage
```

### Code Quality

```bash
# Format code
black plasma/ scripts/ tests/
isort plasma/ scripts/ tests/

# Lint
ruff check plasma/ scripts/ tests/

# Type check
mypy plasma/
```

## Features

### Asyncio-Based Event Loop
All core components use asyncio for high-performance concurrent operations:
- Parallel phantom path generation
- Concurrent packet injection
- Simultaneous auditing and tracking

### Pydantic Type Safety
Full type safety with Pydantic models for state tracking:
- Runtime validation
- Schema generation
- Easy serialization

### Modular Architecture
Clean separation of concerns:
- Simulator: Path generation
- Distiller: Knowledge extraction
- Auditor: Safety verification
- Tracker: Execution monitoring

## Example Workflow

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

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure:
1. Code is formatted with Black
2. Tests pass with pytest
3. Type hints are included
4. Documentation is updated
