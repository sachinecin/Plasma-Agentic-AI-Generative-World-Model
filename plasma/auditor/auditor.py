"""
Auditor - Main adversarial judicial review orchestrator
"""

import asyncio
from typing import Dict, Any, List, Optional
from plasma.auditor.adversarial_checker import AdversarialChecker, Violation
from plasma.auditor.review_engine import ReviewEngine, ReviewReport


class Auditor:
    """
    Adversarial judicial review orchestrator with async event loop
    
    Coordinates adversarial checking and systematic review to prevent
    reward hacking and ensure model integrity.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.checker = AdversarialChecker(config.get("checker", {}))
        self.review_engine = ReviewEngine(config.get("review_engine", {}))
        self._running = False
        
    async def start(self) -> None:
        """Start the auditor"""
        self._running = True
        
    async def stop(self) -> None:
        """Stop the auditor"""
        self._running = False
        
    async def audit_paths(self,
                         paths: List[Any],
                         criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete audit of phantom paths
        
        Args:
            paths: Paths to audit
            criteria: Audit criteria
            
        Returns:
            Comprehensive audit results
        """
        if not self._running:
            await self.start()
            
        # Run adversarial checks
        adversarial_results = await self.checker.run_full_check(paths)
        
        # Run systematic review
        review_report = await self.review_engine.review_paths(paths, criteria)
        
        # Combine results
        passed = adversarial_results["passed"] and review_report.passed
        
        return {
            "passed": passed,
            "adversarial_check": adversarial_results,
            "review_report": review_report,
            "overall_status": "APPROVED" if passed else "REJECTED",
            "violations": adversarial_results["violations"],
            "recommendations": review_report.recommendations
        }
        
    async def audit_adaptations(self,
                               packets: List[Any],
                               criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Audit LoRA adaptation packets
        
        Args:
            packets: Instruction packets to audit
            criteria: Audit criteria
            
        Returns:
            Audit results for adaptations
        """
        if not self._running:
            await self.start()
            
        # Review packets
        review_report = await self.review_engine.review_adaptations(packets, criteria)
        
        return {
            "passed": review_report.passed,
            "review_report": review_report,
            "overall_status": "APPROVED" if review_report.passed else "REJECTED",
            "score": review_report.score
        }
        
    async def continuous_audit(self,
                             paths: List[Any],
                             packets: List[Any],
                             interval: float = 1.0) -> Dict[str, Any]:
        """
        Continuous auditing with periodic checks
        
        Args:
            paths: Paths to audit
            packets: Packets to audit
            interval: Check interval in seconds
            
        Returns:
            Continuous audit results
        """
        if not self._running:
            await self.start()
            
        results = []
        
        # Run initial audit
        path_audit = await self.audit_paths(paths)
        packet_audit = await self.audit_adaptations(packets)
        
        results.append({
            "timestamp": 0,
            "path_audit": path_audit,
            "packet_audit": packet_audit
        })
        
        # Would continue monitoring in production
        # For now, return initial results
        
        return {
            "continuous_mode": True,
            "checks_performed": len(results),
            "latest_results": results[-1] if results else None,
            "all_passed": all(
                r["path_audit"]["passed"] and r["packet_audit"]["passed"]
                for r in results
            )
        }
        
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get overall audit statistics"""
        return {
            "checker_stats": {
                "total_violations": len(self.checker.detected_violations),
                "checks_performed": self.checker.check_count
            },
            "review_stats": self.review_engine.get_review_stats()
        }
