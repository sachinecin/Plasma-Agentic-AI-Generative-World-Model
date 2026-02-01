# Project Plasma - Agentic AI with Generative World Models

A next-gen evolution of Microsoft's Agent-Lightning that replaces heavy RL training with **Phantom-Path Simulations** and **LoRA Distillation**. Uses a generative "World Model" to practice tasks in a sandbox, deploying tiny "instruction packets" for real-time edge adaptation with adversarial auditing to prevent reward hacking.

## 🚀 Key Features

### 1. **Phantom-Path Simulator** 
- Generative World Model for pre-emptive task rollout
- Simulates multiple "phantom paths" in parallel using asyncio
- Zero-latency evolution with concurrent execution
- Replaces heavy RL training with lightweight forward simulation

### 2. **LoRA Distiller**
- Generates real-time LoRA weight instruction packets
- Injects tiny adaptations into edge agents without full retraining
- Priority-based packet queue for efficient deployment
- Distills behavioral patterns from simulations into weight deltas

### 3. **Judicial Auditor**
- Prevents reward-hacking via adversarial oversight
- Multiple detection strategies:
  - Reward spike detection
  - Pattern anomaly detection
  - Distribution shift monitoring
  - Adversarial probing
- Self-correcting logic with automatic remediation
- Continuous monitoring of simulations and instruction packets

### 4. **Fluid Architecture**
- Built with Python, Pydantic for type-safe state traces
- Asyncio throughout for zero-latency evolution
- Modular design with clean separation of concerns
- Production-ready foundations with extensible components

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/sachinecin/Plasma-Agentic-AI-Generative-World-Model.git
cd Plasma-Agentic-AI-Generative-World-Model

# Install dependencies
pip install -r requirements.txt

# Install as package (optional)
pip install -e .
```

## 🎯 Quick Start

```python
import asyncio
from plasma import PlasmaAgent

async def main():
    # Initialize the Plasma Agent
    agent = PlasmaAgent(
        state_dim=128,
        simulation_horizon=10,
        num_simulations=5,
        enable_auditing=True,
        enable_distillation=True,
    )
    
    # Initialize state
    agent.initialize_state()
    
    # Think: Simulate phantom paths
    simulation = await agent.think()
    print(f"Success probability: {simulation.success_probability:.2f}")
    
    # Act: Execute best action
    trace = await agent.act(simulation)
    print(f"Action: {trace.action_taken}, Reward: {trace.reward:.3f}")
    
    # Learn: Distill into LoRA packet
    packet = await agent.learn(simulation)
    print(f"Generated LoRA packet with {len(packet.target_layers)} target layers")
    
    # Evolve: Run multiple cycles
    traces = await agent.evolve(steps=10)
    print(f"Completed {len(traces)} evolution steps")

if __name__ == "__main__":
    asyncio.run(main())
```

## 📚 Core Components

### PlasmaAgent
Main integration layer combining all components:
```python
agent = PlasmaAgent(
    state_dim=128,              # Dimensionality of state vectors
    simulation_horizon=10,       # Steps to simulate ahead
    num_simulations=5,           # Number of parallel phantom paths
    enable_auditing=True,        # Enable judicial oversight
    enable_distillation=True,    # Enable LoRA distillation
)
```

### PhantomPathSimulator
```python
from plasma import PhantomPathSimulator

simulator = PhantomPathSimulator(
    state_dim=128,
    horizon=10,
    num_simulations=5,
    temperature=0.7,
)

# Pre-emptive rollout
results = await simulator.pre_emptive_rollout(initial_state)
best_path = simulator.get_best_phantom_path(results)
```

### LoRADistiller
```python
from plasma import LoRADistiller

distiller = LoRADistiller(
    adaptation_strength=0.1,
    distillation_rate=0.1,
)

# Distill simulation into LoRA weights
packet = distiller.distill_from_simulation(simulation)
await distiller.inject_to_edge_agent(packet)
```

### JudicialAuditor
```python
from plasma import JudicialAuditor

auditor = JudicialAuditor(
    anomaly_threshold=0.7,
    reward_spike_threshold=2.0,
)

# Audit a state trace
audit_record = await auditor.audit_state_trace(trace)
if audit_record.reward_hacking_detected:
    corrected_trace = await auditor.apply_correction(audit_record, trace)
```

## 🎓 Examples

Run the comprehensive examples:
```bash
python examples.py
```

Includes:
- Basic think-act-learn cycle
- Multi-step evolution
- Custom policy usage
- Judicial auditing demonstration
- LoRA distillation showcase

## 🏗️ Architecture

```
Project Plasma
├── Phantom-Path Simulator
│   ├── Generative World Model
│   ├── Pre-emptive Rollout Engine
│   └── Parallel Simulation (asyncio)
│
├── LoRA Distiller
│   ├── Simulation → LoRA Weights
│   ├── Instruction Packet Queue
│   └── Edge Agent Injection
│
├── Judicial Auditor
│   ├── Anomaly Detection
│   ├── Reward-Hacking Prevention
│   ├── Adversarial Oversight
│   └── Self-Correcting Logic
│
└── State Traces (Pydantic)
    ├── WorldState
    ├── StateTrace
    ├── InstructionPacket
    ├── AuditRecord
    └── SimulationResult
```

## 🔬 Technical Details

### State Traces
All state transitions are tracked using immutable Pydantic models:
- **StateTrace**: Records individual state transitions with actions and rewards
- **WorldState**: Maintains current world model predictions
- **InstructionPacket**: LoRA weight deltas for edge injection
- **AuditRecord**: Audit results with anomaly scores and flags
- **SimulationResult**: Complete phantom path trajectory

### Zero-Latency Evolution
- Asyncio throughout for concurrent execution
- Parallel phantom path simulation
- Non-blocking LoRA injection queue
- Continuous background evolution mode

### Self-Correcting Logic
- Online statistics tracking for reward distribution
- Pattern memory for anomaly detection
- Automatic correction of flagged behaviors
- Adaptive thresholds based on history

## 🛡️ Security & Safety

The Judicial Auditor provides multiple layers of protection:
1. **Reward Spike Detection**: Statistical outlier detection
2. **Pattern Anomaly Detection**: Deviation from historical patterns
3. **Distribution Shift**: Monitors state space exploration
4. **Adversarial Probing**: Tests robustness to perturbations
5. **Packet Validation**: Inspects LoRA weights before injection

## 🚧 Development Status

**Current Version**: 0.1.0 (Alpha)

This is a foundational implementation demonstrating the core architecture. Production deployment would require:
- Trained neural network models for world model and reward prediction
- Real edge agent integration interfaces
- Distributed execution framework
- Comprehensive test suite
- Performance optimization

## 📄 License

This project is open source. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! This is a foundational architecture that can be extended in many directions:
- Neural network implementations for world models
- Real-world task environments
- Advanced auditing strategies
- Distributed execution
- Visualization tools

## 📞 Contact

For questions or collaboration opportunities, please open an issue on GitHub.
