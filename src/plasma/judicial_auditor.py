"""
Judicial Auditor - Prevents reward-hacking via adversarial oversight
Implements self-correcting logic and adversarial monitoring
"""
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict

from .state_traces import StateTrace, AuditRecord, SimulationResult, InstructionPacket


class JudicialAuditor:
    """
    Adversarial oversight system that detects and prevents reward-hacking.
    Uses multiple detection strategies and self-correcting mechanisms.
    """
    
    def __init__(
        self,
        anomaly_threshold: float = 0.7,
        reward_spike_threshold: float = 2.0,
        pattern_memory_size: int = 1000,
        adversarial_probes: int = 5,
    ):
        """
        Initialize the Judicial Auditor
        
        Args:
            anomaly_threshold: Threshold for flagging anomalous behavior (0-1)
            reward_spike_threshold: Z-score threshold for reward spikes
            pattern_memory_size: Number of past patterns to remember
            adversarial_probes: Number of adversarial checks to run
        """
        self.anomaly_threshold = anomaly_threshold
        self.reward_spike_threshold = reward_spike_threshold
        self.pattern_memory_size = pattern_memory_size
        self.adversarial_probes = adversarial_probes
        
        # Memory systems
        self._reward_history: List[float] = []
        self._pattern_memory: List[Dict[str, Any]] = []
        self._audit_history: List[AuditRecord] = []
        
        # Statistics for reward distribution
        self._reward_mean = 0.0
        self._reward_std = 1.0
        
        # Detection counters
        self._detections: Dict[str, int] = defaultdict(int)
        self._corrections_applied = 0
        
    async def audit_state_trace(
        self,
        trace: StateTrace,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """
        Audit a single state trace for anomalies and reward-hacking.
        
        Args:
            trace: State trace to audit
            context: Additional context for the audit
            
        Returns:
            AuditRecord with audit results
        """
        adversarial_flags = []
        anomaly_score = 0.0
        
        # Check 1: Reward spike detection
        reward_spike_detected, spike_score = self._detect_reward_spike(trace.reward)
        if reward_spike_detected:
            adversarial_flags.append("reward_spike")
            anomaly_score = max(anomaly_score, spike_score)
        
        # Check 2: Pattern anomaly detection
        pattern_anomaly, pattern_score = await self._detect_pattern_anomaly(trace)
        if pattern_anomaly:
            adversarial_flags.append("pattern_anomaly")
            anomaly_score = max(anomaly_score, pattern_score)
        
        # Check 3: State distribution shift
        distribution_shift, dist_score = self._detect_distribution_shift(trace.state_vector)
        if distribution_shift:
            adversarial_flags.append("distribution_shift")
            anomaly_score = max(anomaly_score, dist_score)
        
        # Check 4: Adversarial probes
        adversarial_detected, adv_score = await self._run_adversarial_probes(trace)
        if adversarial_detected:
            adversarial_flags.append("adversarial_behavior")
            anomaly_score = max(anomaly_score, adv_score)
        
        # Determine if reward-hacking detected
        reward_hacking_detected = (
            len(adversarial_flags) >= 2 or 
            anomaly_score >= self.anomaly_threshold
        )
        
        # Determine corrective action
        corrective_action = None
        if reward_hacking_detected:
            corrective_action = self._determine_corrective_action(adversarial_flags)
            self._corrections_applied += 1
        
        # Update internal state
        self._update_reward_statistics(trace.reward)
        self._update_pattern_memory(trace)
        
        # Create audit record
        audit_record = AuditRecord(
            state_trace_id=trace.trace_id,
            anomaly_score=anomaly_score,
            reward_hacking_detected=reward_hacking_detected,
            adversarial_flags=adversarial_flags,
            corrective_action=corrective_action,
            details={
                "reward": trace.reward,
                "action": trace.action_taken.value if trace.action_taken else None,
                "timestamp": trace.timestamp.isoformat(),
                "context": context or {},
            }
        )
        
        self._audit_history.append(audit_record)
        
        # Update detection counters
        for flag in adversarial_flags:
            self._detections[flag] += 1
        
        return audit_record
    
    def _detect_reward_spike(self, reward: float) -> Tuple[bool, float]:
        """
        Detect anomalous reward spikes using statistical methods.
        
        Returns:
            Tuple of (spike_detected, anomaly_score)
        """
        if len(self._reward_history) < 10:
            return False, 0.0
        
        # Calculate z-score
        z_score = abs((reward - self._reward_mean) / (self._reward_std + 1e-8))
        
        # Detect spike
        spike_detected = z_score > self.reward_spike_threshold
        anomaly_score = min(1.0, z_score / (self.reward_spike_threshold * 2))
        
        return spike_detected, anomaly_score
    
    async def _detect_pattern_anomaly(
        self, 
        trace: StateTrace
    ) -> Tuple[bool, float]:
        """
        Detect anomalous patterns using memory of past behaviors.
        
        Returns:
            Tuple of (anomaly_detected, anomaly_score)
        """
        if len(self._pattern_memory) < 10:
            return False, 0.0
        
        # Extract pattern features
        current_pattern = self._extract_pattern_features(trace)
        
        # Compare with historical patterns
        distances = []
        for past_pattern in self._pattern_memory[-100:]:
            distance = self._pattern_distance(current_pattern, past_pattern)
            distances.append(distance)
        
        # Detect anomaly if current pattern is far from all past patterns
        min_distance = min(distances) if distances else 0
        anomaly_score = min(1.0, min_distance / 2.0)
        anomaly_detected = anomaly_score > self.anomaly_threshold
        
        await asyncio.sleep(0)  # Yield to event loop
        
        return anomaly_detected, anomaly_score
    
    def _extract_pattern_features(self, trace: StateTrace) -> Dict[str, Any]:
        """Extract pattern features from a state trace"""
        state_array = np.array(trace.state_vector)
        return {
            "state_norm": float(np.linalg.norm(state_array)),
            "state_mean": float(np.mean(state_array)),
            "state_std": float(np.std(state_array)),
            "reward": trace.reward,
            "action": trace.action_taken.value if trace.action_taken else None,
        }
    
    def _pattern_distance(
        self, 
        pattern1: Dict[str, Any], 
        pattern2: Dict[str, Any]
    ) -> float:
        """Calculate distance between two patterns"""
        distance = 0.0
        
        # Compare numerical features
        for key in ["state_norm", "state_mean", "state_std", "reward"]:
            if key in pattern1 and key in pattern2:
                distance += abs(pattern1[key] - pattern2[key])
        
        # Compare action (categorical)
        if pattern1.get("action") != pattern2.get("action"):
            distance += 0.5
        
        return distance
    
    def _detect_distribution_shift(
        self, 
        state_vector: List[float]
    ) -> Tuple[bool, float]:
        """
        Detect if state distribution has shifted significantly.
        
        Returns:
            Tuple of (shift_detected, shift_score)
        """
        state_array = np.array(state_vector)
        
        # Calculate state statistics
        state_norm = np.linalg.norm(state_array)
        state_mean = np.mean(state_array)
        
        # Check for abnormal values
        if state_norm > 10.0 or abs(state_mean) > 5.0:
            shift_score = min(1.0, state_norm / 20.0)
            return True, shift_score
        
        return False, 0.0
    
    async def _run_adversarial_probes(
        self, 
        trace: StateTrace
    ) -> Tuple[bool, float]:
        """
        Run adversarial probes to detect reward-hacking.
        Tests if small perturbations lead to unrealistic reward changes.
        
        Returns:
            Tuple of (adversarial_detected, adversarial_score)
        """
        adversarial_scores = []
        
        for _ in range(self.adversarial_probes):
            # Create small perturbation
            state_array = np.array(trace.state_vector)
            perturbation = np.random.randn(len(state_array)) * 0.01
            perturbed_state = state_array + perturbation
            
            # Check if perturbation causes unrealistic changes
            # (In production, would use the reward model)
            reward_sensitivity = abs(trace.reward) * np.linalg.norm(perturbation)
            
            if reward_sensitivity > 0.1:
                adversarial_scores.append(reward_sensitivity)
            
            await asyncio.sleep(0)
        
        if adversarial_scores:
            avg_score = np.mean(adversarial_scores)
            adversarial_detected = avg_score > 0.15
            return adversarial_detected, min(1.0, avg_score)
        
        return False, 0.0
    
    def _determine_corrective_action(
        self, 
        adversarial_flags: List[str]
    ) -> str:
        """
        Determine what corrective action should be taken.
        This implements self-correcting logic.
        """
        if "reward_spike" in adversarial_flags:
            return "clip_reward"
        elif "pattern_anomaly" in adversarial_flags:
            return "reset_to_baseline"
        elif "distribution_shift" in adversarial_flags:
            return "normalize_state"
        elif "adversarial_behavior" in adversarial_flags:
            return "reject_action"
        else:
            return "increase_oversight"
    
    async def apply_correction(
        self,
        audit_record: AuditRecord,
        trace: StateTrace,
    ) -> StateTrace:
        """
        Apply corrective action to a state trace.
        Returns corrected state trace.
        
        Args:
            audit_record: Audit record with corrective action
            trace: Original state trace
            
        Returns:
            Corrected state trace
        """
        if not audit_record.corrective_action:
            return trace
        
        corrected_metadata = trace.metadata.copy()
        corrected_metadata["corrected"] = True
        corrected_metadata["correction_type"] = audit_record.corrective_action
        
        if audit_record.corrective_action == "clip_reward":
            # Clip reward to reasonable range
            corrected_reward = max(-1.0, min(1.0, trace.reward))
            return StateTrace(
                trace_id=trace.trace_id,
                timestamp=datetime.utcnow(),
                state_vector=trace.state_vector,
                action_taken=trace.action_taken,
                reward=corrected_reward,
                metadata=corrected_metadata,
            )
        
        elif audit_record.corrective_action == "normalize_state":
            # Normalize state vector
            state_array = np.array(trace.state_vector)
            normalized = state_array / (np.linalg.norm(state_array) + 1e-8)
            return StateTrace(
                trace_id=trace.trace_id,
                timestamp=datetime.utcnow(),
                state_vector=normalized.tolist(),
                action_taken=trace.action_taken,
                reward=trace.reward,
                metadata=corrected_metadata,
            )
        
        await asyncio.sleep(0)
        return trace
    
    async def audit_simulation(
        self,
        simulation: SimulationResult,
    ) -> List[AuditRecord]:
        """
        Audit an entire simulation for systemic reward-hacking.
        
        Args:
            simulation: Simulation result to audit
            
        Returns:
            List of audit records for flagged issues
        """
        audit_records = []
        
        # Check overall trajectory for suspicious patterns
        rewards = [reward for _, _, reward in simulation.simulated_trajectory]
        
        # Check 1: Monotonically increasing rewards (potential exploitation)
        if len(rewards) > 2:
            is_monotonic = all(rewards[i] <= rewards[i+1] for i in range(len(rewards)-1))
            if is_monotonic and rewards[-1] - rewards[0] > 1.0:
                audit_records.append(AuditRecord(
                    state_trace_id=simulation.simulation_id,
                    anomaly_score=0.8,
                    reward_hacking_detected=True,
                    adversarial_flags=["monotonic_reward_growth"],
                    corrective_action="reduce_simulation_confidence",
                    details={"rewards": rewards},
                ))
        
        # Check 2: Unrealistic success probability
        if simulation.success_probability > 0.95 and simulation.total_reward < 0:
            audit_records.append(AuditRecord(
                state_trace_id=simulation.simulation_id,
                anomaly_score=0.9,
                reward_hacking_detected=True,
                adversarial_flags=["inconsistent_success_probability"],
                corrective_action="recalculate_probability",
                details={"success_probability": simulation.success_probability},
            ))
        
        await asyncio.sleep(0)
        return audit_records
    
    async def audit_instruction_packet(
        self,
        packet: InstructionPacket,
    ) -> AuditRecord:
        """
        Audit a LoRA instruction packet before injection.
        Prevents malicious or corrupted weight injections.
        
        Args:
            packet: Instruction packet to audit
            
        Returns:
            AuditRecord with audit results
        """
        adversarial_flags = []
        anomaly_score = 0.0
        
        # Check for abnormally large weight magnitudes
        for layer_name, weights in packet.lora_weights.items():
            weight_array = np.array(weights)
            weight_norm = np.linalg.norm(weight_array)
            
            if weight_norm > 5.0:
                adversarial_flags.append(f"large_weights_{layer_name}")
                anomaly_score = max(anomaly_score, min(1.0, weight_norm / 10.0))
        
        # Check adaptation strength
        if packet.adaptation_strength > 0.5:
            adversarial_flags.append("high_adaptation_strength")
            anomaly_score = max(anomaly_score, packet.adaptation_strength)
        
        reward_hacking_detected = anomaly_score > self.anomaly_threshold
        corrective_action = "scale_down_weights" if reward_hacking_detected else None
        
        audit_record = AuditRecord(
            state_trace_id=packet.packet_id,
            anomaly_score=anomaly_score,
            reward_hacking_detected=reward_hacking_detected,
            adversarial_flags=adversarial_flags,
            corrective_action=corrective_action,
            details={
                "packet_id": packet.packet_id,
                "adaptation_strength": packet.adaptation_strength,
                "target_layers": packet.target_layers,
            }
        )
        
        await asyncio.sleep(0)
        return audit_record
    
    def _update_reward_statistics(self, reward: float) -> None:
        """Update running statistics for reward distribution"""
        self._reward_history.append(reward)
        
        # Keep only recent history
        if len(self._reward_history) > self.pattern_memory_size:
            self._reward_history.pop(0)
        
        # Update mean and std
        if len(self._reward_history) > 1:
            self._reward_mean = np.mean(self._reward_history)
            self._reward_std = np.std(self._reward_history)
    
    def _update_pattern_memory(self, trace: StateTrace) -> None:
        """Update pattern memory with new trace"""
        pattern = self._extract_pattern_features(trace)
        self._pattern_memory.append(pattern)
        
        # Keep only recent patterns
        if len(self._pattern_memory) > self.pattern_memory_size:
            self._pattern_memory.pop(0)
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit statistics and detection counts"""
        return {
            "total_audits": len(self._audit_history),
            "detections": dict(self._detections),
            "corrections_applied": self._corrections_applied,
            "reward_mean": self._reward_mean,
            "reward_std": self._reward_std,
            "pattern_memory_size": len(self._pattern_memory),
        }
    
    def get_recent_audits(
        self, 
        count: int = 10,
        flagged_only: bool = False,
    ) -> List[AuditRecord]:
        """Get recent audit records"""
        audits = self._audit_history[-count:]
        if flagged_only:
            audits = [a for a in audits if a.reward_hacking_detected]
        return audits
