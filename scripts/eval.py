#!/usr/bin/env python3
"""
Evaluation Script - Main entry point for evaluating Project Plasma models

Uses asyncio event loop for high-performance evaluation.
"""

import asyncio
import argparse
from typing import Dict, Any, List
import sys

from plasma.simulator import Simulator
from plasma.auditor import Auditor
from plasma.trace import StateTracker


async def evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main evaluation loop using asyncio
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Evaluation results
    """
    print("🔍 Starting Project Plasma Evaluation")
    print("=" * 60)
    
    # Initialize components
    simulator = Simulator(config.get("simulator", {}))
    auditor = Auditor(config.get("auditor", {}))
    tracker = StateTracker(config.get("tracker", {}))
    
    # Start all components
    await asyncio.gather(
        simulator.start(),
        auditor.start(),
        tracker.start()
    )
    
    print("✓ All components initialized")
    
    # Create trace for evaluation
    trace_id = tracker.create_trace(metadata={"type": "evaluation"})
    print(f"✓ Created trace: {trace_id}")
    
    # Evaluation loop
    num_episodes = config.get("num_episodes", 10)
    results = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # Create span for this episode
        span = await tracker.create_span(
            trace_id=trace_id,
            name=f"episode_{episode}",
            metadata={"episode": str(episode)}
        )
        
        async with span:
            # Run episode
            print("  Running episode...")
            initial_state = {"episode": episode, "step": 0}
            path = await simulator.run_episode(
                initial_state=initial_state,
                max_steps=config.get("max_steps", 100)
            )
            
            # Calculate metrics
            total_reward = sum(path.rewards) if path.rewards else 0.0
            avg_reward = total_reward / len(path.rewards) if path.rewards else 0.0
            episode_length = len(path.states)
            
            print(f"  ✓ Episode completed")
            print(f"    Length: {episode_length} steps")
            print(f"    Total reward: {total_reward:.2f}")
            print(f"    Average reward: {avg_reward:.2f}")
            
            # Audit episode
            print("  Auditing episode...")
            audit_result = await auditor.audit_paths([path])
            print(f"  ✓ Audit: {audit_result['overall_status']}")
            
            # Update span state
            span.update_state({
                "episode_length": episode_length,
                "total_reward": total_reward,
                "avg_reward": avg_reward,
                "audit_passed": audit_result["passed"]
            })
            
            results.append({
                "episode": episode,
                "length": episode_length,
                "total_reward": total_reward,
                "avg_reward": avg_reward,
                "audit_passed": audit_result["passed"]
            })
    
    # Complete trace
    await tracker.complete_trace(trace_id)
    
    # Calculate overall statistics
    print("\n" + "=" * 60)
    print("📊 Evaluation Summary")
    print("=" * 60)
    
    avg_episode_length = sum(r["length"] for r in results) / len(results)
    avg_total_reward = sum(r["total_reward"] for r in results) / len(results)
    avg_reward_per_step = sum(r["avg_reward"] for r in results) / len(results)
    passed_audits = sum(1 for r in results if r["audit_passed"])
    
    print(f"Total episodes: {num_episodes}")
    print(f"Average episode length: {avg_episode_length:.1f} steps")
    print(f"Average total reward: {avg_total_reward:.2f}")
    print(f"Average reward per step: {avg_reward_per_step:.2f}")
    print(f"Passed audits: {passed_audits}/{num_episodes}")
    print(f"Audit pass rate: {(passed_audits / num_episodes * 100):.1f}%")
    
    tracker_stats = tracker.get_trace_stats()
    auditor_stats = auditor.get_audit_statistics()
    
    print(f"\nTotal spans tracked: {tracker_stats['total_spans']}")
    print(f"Total violations: {auditor_stats['checker_stats']['total_violations']}")
    
    # Stop components
    await asyncio.gather(
        simulator.stop(),
        auditor.stop(),
        tracker.stop()
    )
    
    print("\n✅ Evaluation complete!")
    
    return {
        "num_episodes": num_episodes,
        "avg_episode_length": avg_episode_length,
        "avg_total_reward": avg_total_reward,
        "avg_reward_per_step": avg_reward_per_step,
        "passed_audits": passed_audits,
        "audit_pass_rate": passed_audits / num_episodes,
        "results": results,
        "tracker_stats": tracker_stats,
        "auditor_stats": auditor_stats
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Project Plasma Evaluation")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode"
    )
    
    args = parser.parse_args()
    
    config = {
        "num_episodes": args.episodes,
        "max_steps": args.max_steps,
        "simulator": {},
        "auditor": {},
        "tracker": {}
    }
    
    # Run evaluation with asyncio
    try:
        results = asyncio.run(evaluate(config))
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
