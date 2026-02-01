"""
Review Engine - Systematic judicial review of model behavior
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ReviewReport:
    """Comprehensive review report"""
    review_id: str
    timestamp: float
    passed: bool
    score: float  # 0.0 to 1.0
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]


class ReviewEngine:
    """
    Systematic review engine for model behavior analysis
    
    Provides structured judicial review process to evaluate model
    performance, safety, and alignment with objectives.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.review_count = 0
        self.review_history: List[ReviewReport] = []
        
    async def review_paths(self,
                          paths: List[Any],
                          criteria: Optional[Dict[str, Any]] = None) -> ReviewReport:
        """
        Review phantom paths against criteria
        
        Args:
            paths: Paths to review
            criteria: Review criteria
            
        Returns:
            ReviewReport with findings
        """
        self.review_count += 1
        criteria = criteria or {}
        
        findings = []
        score = 1.0
        
        # Analyze path quality
        if paths:
            avg_length = sum(len(getattr(p, 'states', [])) for p in paths) / len(paths)
            
            findings.append({
                "metric": "path_length",
                "value": avg_length,
                "status": "good" if avg_length > 5 else "poor"
            })
            
            if avg_length < 5:
                score -= 0.2
                
        # Check path diversity
        if len(paths) > 1:
            findings.append({
                "metric": "diversity",
                "value": len(paths),
                "status": "good"
            })
        else:
            score -= 0.3
            findings.append({
                "metric": "diversity",
                "value": len(paths),
                "status": "poor"
            })
            
        recommendations = []
        if score < 0.8:
            recommendations.append("Increase path diversity")
        if avg_length < 10:
            recommendations.append("Generate longer paths for better learning")
            
        passed = score >= criteria.get("min_score", 0.7)
        
        report = ReviewReport(
            review_id=f"review_{self.review_count}",
            timestamp=self.review_count,
            passed=passed,
            score=max(0.0, score),
            findings=findings,
            recommendations=recommendations,
            metadata={"path_count": len(paths)}
        )
        
        self.review_history.append(report)
        return report
        
    async def review_adaptations(self,
                                packets: List[Any],
                                criteria: Optional[Dict[str, Any]] = None) -> ReviewReport:
        """
        Review LoRA adaptation packets
        
        Args:
            packets: Instruction packets to review
            criteria: Review criteria
            
        Returns:
            ReviewReport for adaptations
        """
        self.review_count += 1
        
        findings = []
        score = 1.0
        
        # Check packet count
        findings.append({
            "metric": "packet_count",
            "value": len(packets),
            "status": "good" if packets else "empty"
        })
        
        if not packets:
            score = 0.0
            
        # Check packet sizes (simplified)
        if packets:
            avg_size = sum(getattr(p, 'size_bytes', lambda: 100)() for p in packets) / len(packets)
            findings.append({
                "metric": "avg_packet_size",
                "value": avg_size,
                "status": "good" if avg_size < 10000 else "large"
            })
            
        passed = score >= 0.7
        
        report = ReviewReport(
            review_id=f"review_{self.review_count}",
            timestamp=self.review_count,
            passed=passed,
            score=score,
            findings=findings,
            recommendations=["Optimize packet size"] if score < 1.0 else [],
            metadata={"packet_count": len(packets)}
        )
        
        self.review_history.append(report)
        return report
        
    def get_review_history(self) -> List[ReviewReport]:
        """Get all review reports"""
        return self.review_history.copy()
        
    def get_review_stats(self) -> Dict[str, Any]:
        """Get statistics about reviews"""
        if not self.review_history:
            return {"total": 0, "passed": 0, "failed": 0, "avg_score": 0.0}
            
        passed = sum(1 for r in self.review_history if r.passed)
        avg_score = sum(r.score for r in self.review_history) / len(self.review_history)
        
        return {
            "total": len(self.review_history),
            "passed": passed,
            "failed": len(self.review_history) - passed,
            "avg_score": avg_score
        }
