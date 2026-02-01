# Project Plasma - API Reference

## Table of Contents
- [PlasmaAgent](#plasmaagent)
- [PhantomPathSimulator](#phantompathsimulator)
- [LoRADistiller](#loradistiller)
- [JudicialAuditor](#judicialauditor)
- [State Traces](#state-traces)

---

## PlasmaAgent

The main integration layer that combines all components of Project Plasma.

### Constructor

```python
PlasmaAgent(
    state_dim: int = 128,
    action_space: Optional[List[ActionType]] = None,
    simulation_horizon: int = 10,
    num_simulations: int = 5,
    enable_auditing: bool = True,
    enable_distillation: bool = True,
)
```

**Parameters:**
- `state_dim`: Dimensionality of state vectors (default: 128)
- `action_space`: Available actions (default: all ActionTypes)
- `simulation_horizon`: Number of steps to simulate ahead (default: 10)
- `num_simulations`: Number of parallel phantom paths (default: 5)
- `enable_auditing`: Enable judicial oversight (default: True)
- `enable_distillation`: Enable LoRA distillation (default: True)

### Methods

#### `initialize_state(initial_state: Optional[List[float]] = None)`
Initialize agent state vector.

#### `async think(policy: Optional[Callable] = None) -> SimulationResult`
Simulate phantom paths and select the best one.

**Returns:** Best simulation result

#### `async act(simulation: SimulationResult, execute_first_action: bool = True) -> StateTrace`
Execute action based on simulation result.

**Returns:** State trace of executed action

#### `async learn(simulation: SimulationResult, priority: int = 5) -> Optional[InstructionPacket]`
Distill simulation into LoRA instruction packet.

**Returns:** Generated instruction packet

#### `async evolve(steps: int = 10, policy: Optional[Callable] = None) -> List[StateTrace]`
Evolve through multiple think-act-learn cycles.

**Returns:** List of state traces from evolution

#### `async start_continuous_evolution(policy: Optional[Callable] = None, agent_interface: Optional[Any] = None)`
Start continuous evolution in background.

#### `async stop_continuous_evolution()`
Stop all background evolution tasks.

#### `get_statistics() -> Dict[str, Any]`
Get comprehensive agent statistics.

#### `get_execution_history(count: int = 10) -> List[Dict[str, Any]]`
Get recent execution history.

#### `reset()`
Reset agent state.

---

## PhantomPathSimulator

Generative World Model for pre-emptive rollout simulation.

### Constructor

```python
PhantomPathSimulator(
    state_dim: int = 128,
    action_space: List[ActionType] = None,
    horizon: int = 10,
    num_simulations: int = 5,
    temperature: float = 0.7,
)
```

**Parameters:**
- `state_dim`: Dimensionality of state vectors
- `action_space`: Available actions for simulation
- `horizon`: Number of steps to simulate ahead
- `num_simulations`: Number of parallel phantom paths to explore
- `temperature`: Temperature for stochastic rollouts (higher = more exploration)

### Methods

#### `async simulate_phantom_path(initial_state: List[float], policy: Optional[Callable] = None) -> SimulationResult`
Simulate a single phantom path asynchronously.

**Returns:** SimulationResult containing the simulated trajectory

#### `async pre_emptive_rollout(initial_state: List[float], policy: Optional[Callable] = None) -> List[SimulationResult]`
Perform pre-emptive rollout by simulating multiple phantom paths in parallel.

**Returns:** List of simulation results

#### `get_best_phantom_path(results: List[SimulationResult]) -> SimulationResult`
Select the best phantom path based on quality and success probability.

**Returns:** Best simulation result

#### `extract_state_traces(simulation: SimulationResult) -> List[StateTrace]`
Extract state traces from a simulation.

**Returns:** List of StateTrace objects

#### `get_world_state() -> Optional[WorldState]`
Get the current world state.

#### `update_transition_model(new_weights: Dict[str, Any])`
Update transition model weights for online learning.

#### `update_reward_model(new_weights: Dict[str, Any])`
Update reward model weights for online learning.

---

## LoRADistiller

Real-time generation and injection of LoRA weight instruction packets.

### Constructor

```python
LoRADistiller(
    layer_names: List[str] = None,
    max_packet_queue: int = 100,
    distillation_rate: float = 0.1,
    adaptation_strength: float = 0.1,
)
```

**Parameters:**
- `layer_names`: Names of model layers for LoRA injection
- `max_packet_queue`: Maximum number of packets to queue
- `distillation_rate`: Rate at which knowledge is distilled
- `adaptation_strength`: Default strength of adaptations

### Methods

#### `distill_from_simulation(simulation: SimulationResult, priority: int = 5) -> InstructionPacket`
Distill a simulation result into a LoRA instruction packet.

**Returns:** InstructionPacket ready for injection

#### `distill_from_traces(traces: List[StateTrace], priority: int = 5) -> InstructionPacket`
Distill state traces into a LoRA instruction packet.

**Returns:** InstructionPacket

#### `async enqueue_packet(packet: InstructionPacket)`
Enqueue an instruction packet for asynchronous injection.

#### `async inject_to_edge_agent(packet: InstructionPacket, agent_interface: Optional[Any] = None) -> bool`
Inject a LoRA instruction packet into an edge agent.

**Returns:** True if successful

#### `async continuous_injection_loop(agent_interface: Optional[Any] = None, batch_size: int = 5)`
Continuous loop that injects packets from the queue.

#### `get_statistics() -> Dict[str, Any]`
Get distiller statistics.

#### `get_recent_injections(count: int = 10) -> List[InstructionPacket]`
Get most recent injection history.

#### `clear_queue()`
Clear the packet queue.

---

## JudicialAuditor

Adversarial oversight system for reward-hacking prevention.

### Constructor

```python
JudicialAuditor(
    anomaly_threshold: float = 0.7,
    reward_spike_threshold: float = 2.0,
    pattern_memory_size: int = 1000,
    adversarial_probes: int = 5,
)
```

**Parameters:**
- `anomaly_threshold`: Threshold for flagging anomalous behavior (0-1)
- `reward_spike_threshold`: Z-score threshold for reward spikes
- `pattern_memory_size`: Number of past patterns to remember
- `adversarial_probes`: Number of adversarial checks to run

### Methods

#### `async audit_state_trace(trace: StateTrace, context: Optional[Dict[str, Any]] = None) -> AuditRecord`
Audit a single state trace for anomalies and reward-hacking.

**Returns:** AuditRecord with audit results

#### `async apply_correction(audit_record: AuditRecord, trace: StateTrace) -> StateTrace`
Apply corrective action to a state trace.

**Returns:** Corrected state trace

#### `async audit_simulation(simulation: SimulationResult) -> List[AuditRecord]`
Audit an entire simulation for systemic reward-hacking.

**Returns:** List of audit records for flagged issues

#### `async audit_instruction_packet(packet: InstructionPacket) -> AuditRecord`
Audit a LoRA instruction packet before injection.

**Returns:** AuditRecord with audit results

#### `get_audit_statistics() -> Dict[str, Any]`
Get audit statistics and detection counts.

#### `get_recent_audits(count: int = 10, flagged_only: bool = False) -> List[AuditRecord]`
Get recent audit records.

---

## State Traces

Type-safe Pydantic models for state management.

### ActionType (Enum)
```python
class ActionType(str, Enum):
    MOVE = "move"
    INTERACT = "interact"
    OBSERVE = "observe"
    COMMUNICATE = "communicate"
    COMPUTE = "compute"
```

### StateTrace
Immutable state trace recording agent state transitions.

**Fields:**
- `trace_id: str` - Unique identifier
- `timestamp: datetime` - Timestamp of trace
- `state_vector: List[float]` - Dense state representation
- `action_taken: Optional[ActionType]` - Action taken
- `reward: float` - Reward received
- `metadata: Dict[str, Any]` - Additional metadata

### WorldState
Current state of the generative world model.

**Fields:**
- `state_id: str` - Unique identifier
- `timestamp: datetime` - Timestamp
- `state_vector: List[float]` - Current state vector
- `predicted_next_states: List[List[float]]` - Predicted future states
- `confidence_scores: List[float]` - Confidence for each prediction
- `trace_history: List[StateTrace]` - Historical traces

### InstructionPacket
LoRA weight instruction packet for edge agent injection.

**Fields:**
- `packet_id: str` - Unique identifier
- `timestamp: datetime` - Timestamp
- `lora_weights: Dict[str, List[float]]` - Layer-wise LoRA weight deltas
- `target_layers: List[str]` - Target layers for injection
- `adaptation_strength: float` - Strength of adaptation (0-1)
- `context: str` - Context for this adaptation
- `priority: int` - Priority level (1-10)

### AuditRecord
Record of an audit check by the Judicial Auditor.

**Fields:**
- `audit_id: str` - Unique identifier
- `timestamp: datetime` - Timestamp
- `state_trace_id: str` - ID of trace being audited
- `anomaly_score: float` - Anomaly detection score (0-1)
- `reward_hacking_detected: bool` - Whether reward-hacking was detected
- `adversarial_flags: List[str]` - List of detected issues
- `corrective_action: Optional[str]` - Recommended correction
- `details: Dict[str, Any]` - Additional details

### SimulationResult
Result of a phantom path simulation.

**Fields:**
- `simulation_id: str` - Unique identifier
- `timestamp: datetime` - Timestamp
- `initial_state: List[float]` - Starting state
- `simulated_trajectory: List[Tuple[List[float], ActionType, float]]` - Trajectory as (state, action, reward) tuples
- `total_reward: float` - Cumulative reward
- `success_probability: float` - Probability of success (0-1)
- `quality_score: float` - Quality score (0-1)
- `metadata: Dict[str, Any]` - Additional metadata

---

## Usage Examples

### Basic Usage
```python
import asyncio
from plasma import PlasmaAgent

async def main():
    agent = PlasmaAgent()
    agent.initialize_state()
    
    simulation = await agent.think()
    trace = await agent.act(simulation)
    packet = await agent.learn(simulation)

asyncio.run(main())
```

### Custom Policy
```python
def my_policy(state):
    # Custom logic
    return ActionType.COMPUTE

simulation = await agent.think(policy=my_policy)
```

### Continuous Evolution
```python
await agent.start_continuous_evolution(policy=my_policy)
# ... do other work ...
await agent.stop_continuous_evolution()
```

### Standalone Components
```python
from plasma import PhantomPathSimulator, LoRADistiller, JudicialAuditor

simulator = PhantomPathSimulator(state_dim=64)
distiller = LoRADistiller()
auditor = JudicialAuditor()

# Use independently
results = await simulator.pre_emptive_rollout(initial_state)
packet = distiller.distill_from_simulation(results[0])
audit = await auditor.audit_instruction_packet(packet)
```
