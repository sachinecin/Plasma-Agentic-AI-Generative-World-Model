"""
Adversarial Checker - Detects reward hacking and policy issues
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ViolationType(Enum):
    """Types of violations that can be detected"""
    REWARD_HACKING = "reward_hacking"
    POLICY_DEVIATION = "policy_deviation"
    SAFETY_VIOLATION = "safety_violation"
    DISTRIBUTION_SHIFT = "distribution_shift"


@dataclass
class Violation:
    """Represents a detected violation"""
    violation_type: ViolationType
    severity: float  # 0.0 to 1.0
    description: str
    evidence: Dict[str, Any]
    timestamp: float


class AdversarialChecker:
    """
    Adversarial checker to detect reward hacking and policy issues
    
    Uses adversarial testing to identify potential exploits and ensure
    model behavior aligns with intended objectives.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.detected_violations: List[Violation] = []
        self.check_count = 0
        
    async def check_reward_hacking(self,
                                  paths: List[Any],
                                  expected_rewards: Optional[List[float]] = None) -> List[Violation]:
        """
        Check for reward hacking patterns in paths
        
        Args:
            paths: Phantom paths to check
            expected_rewards: Optional expected reward range
            
        Returns:
            List of detected violations
        """
        violations = []
        
        for i, path in enumerate(paths):
            # Check for suspicious reward patterns
            if hasattr(path, 'rewards') and path.rewards:
                avg_reward = sum(path.rewards) / len(path.rewards)
                
                # Detect abnormally high rewards (potential gaming)
                if avg_reward > 0.95:
                    violation = Violation(
                        violation_type=ViolationType.REWARD_HACKING,
                        severity=0.8,
                        description=f"Path {i} shows suspiciously high rewards",
                        evidence={
                            "path_id": getattr(path, 'path_id', f'path_{i}'),
                            "avg_reward": avg_reward,
                            "rewards": path.rewards[:10]  # Sample
                        },
                        timestamp=self.check_count
                    )
                    violations.append(violation)
                    
        self.detected_violations.extend(violations)
        self.check_count += 1
        
        return violations
        
    async def check_policy_consistency(self,
                                      paths: List[Any],
                                      baseline_policy: Optional[Any] = None) -> List[Violation]:
        """
        Check if paths show consistent policy behavior
        
        Args:
            paths: Paths to analyze
            baseline_policy: Optional baseline for comparison
            
        Returns:
            List of detected violations
        """
        violations = []
        
        # Check for erratic action patterns
        for i, path in enumerate(paths):
            if hasattr(path, 'actions') and len(path.actions) > 2:
                # Simplified check for action diversity
                unique_actions = len(set(str(a) for a in path.actions))
                if unique_actions < len(path.actions) * 0.3:
                    violation = Violation(
                        violation_type=ViolationType.POLICY_DEVIATION,
                        severity=0.6,
                        description=f"Path {i} shows low action diversity",
                        evidence={
                            "path_id": getattr(path, 'path_id', f'path_{i}'),
                            "unique_actions": unique_actions,
                            "total_actions": len(path.actions)
                        },
                        timestamp=self.check_count
                    )
                    violations.append(violation)
                    
        self.detected_violations.extend(violations)
        self.check_count += 1
        
        return violations
        
    async def check_safety_constraints(self,
                                      paths: List[Any],
                                      constraints: Optional[Dict[str, Any]] = None) -> List[Violation]:
        """
        Check if paths violate safety constraints
        
        Args:
            paths: Paths to check
            constraints: Safety constraints to verify
            
        Returns:
            List of violations
        """
        violations = []
        
        # Placeholder for safety checks
        # In practice, would check domain-specific constraints
        
        return violations
        
    async def run_full_check(self, paths: List[Any]) -> Dict[str, Any]:
        """
        Run all adversarial checks on paths
        
        Args:
            paths: Paths to audit
            
        Returns:
            Complete audit results
        """
        reward_violations = await self.check_reward_hacking(paths)
        policy_violations = await self.check_policy_consistency(paths)
        safety_violations = await self.check_safety_constraints(paths)
        
        all_violations = reward_violations + policy_violations + safety_violations
        
        return {
            "total_violations": len(all_violations),
            "reward_hacking": len(reward_violations),
            "policy_issues": len(policy_violations),
            "safety_issues": len(safety_violations),
            "violations": all_violations,
            "passed": len(all_violations) == 0
        }
