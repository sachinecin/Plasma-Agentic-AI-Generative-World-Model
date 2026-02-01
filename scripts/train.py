#!/usr/bin/env python3
"""
Training Script - Main entry point for training Project Plasma

Uses asyncio event loop to coordinate simulator, distiller, auditor, and trace components.
"""

import asyncio
import argparse
from typing import Dict, Any
import sys

from plasma.simulator import Simulator
from plasma.distiller import Distiller
from plasma.auditor import Auditor
from plasma.trace import StateTracker


async def train(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main training loop using asyncio
    
    Args:
        config: Training configuration
        
    Returns:
        Training results
    """
    print("🚀 Starting Project Plasma Training")
    print("=" * 60)
    
    # Initialize components
    simulator = Simulator(config.get("simulator", {}))
    distiller = Distiller(config.get("distiller", {}))
    auditor = Auditor(config.get("auditor", {}))
    tracker = StateTracker(config.get("tracker", {}))
    
    # Start all components
    await asyncio.gather(
        simulator.start(),
        distiller.start(),
        auditor.start(),
        tracker.start()
    )
    
    print("✓ All components initialized")
    
    # Create trace for training
    trace_id = tracker.create_trace(metadata={"type": "training"})
    print(f"✓ Created trace: {trace_id}")
    
    # Training loop
    num_iterations = config.get("num_iterations", 10)
    results = []
    
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
        
        # Create span for this iteration
        span = await tracker.create_span(
            trace_id=trace_id,
            name=f"iteration_{iteration}",
            metadata={"iteration": str(iteration)}
        )
        
        async with span:
            # Generate phantom paths
            print("  Generating phantom paths...")
            initial_state = {"iteration": iteration, "step": 0}
            paths = await simulator.simulate(
                initial_state=initial_state,
                num_paths=config.get("num_paths", 5),
                path_length=config.get("path_length", 20)
            )
            print(f"  ✓ Generated {len(paths)} phantom paths")
            
            # Audit paths
            print("  Auditing phantom paths...")
            audit_result = await auditor.audit_paths(paths)
            print(f"  ✓ Audit: {audit_result['overall_status']}")
            
            if not audit_result["passed"]:
                print("  ⚠️  Audit failed, skipping distillation")
                continue
                
            # Distill instruction packets
            print("  Distilling instruction packets...")
            packets = await distiller.distill_from_paths(paths)
            print(f"  ✓ Created {len(packets)} instruction packets")
            
            # Audit packets
            print("  Auditing instruction packets...")
            packet_audit = await auditor.audit_adaptations(packets)
            print(f"  ✓ Packet audit: {packet_audit['overall_status']}")
            
            # Deploy if audit passed
            if packet_audit["passed"]:
                print("  Deploying instruction packets...")
                deploy_result = await distiller.deploy_packets(packets)
                print(f"  ✓ Deployed {deploy_result['deployed']}/{deploy_result['total']} packets")
            
            # Update span state
            span.update_state({
                "paths_generated": len(paths),
                "packets_created": len(packets),
                "audit_passed": audit_result["passed"]
            })
            
            results.append({
                "iteration": iteration,
                "audit_passed": audit_result["passed"],
                "packets_deployed": len(packets) if packet_audit["passed"] else 0
            })
    
    # Complete trace
    await tracker.complete_trace(trace_id)
    
    # Get final statistics
    print("\n" + "=" * 60)
    print("📊 Training Summary")
    print("=" * 60)
    
    tracker_stats = tracker.get_trace_stats()
    auditor_stats = auditor.get_audit_statistics()
    
    print(f"Total iterations: {num_iterations}")
    print(f"Successful audits: {sum(1 for r in results if r['audit_passed'])}")
    print(f"Total packets deployed: {sum(r['packets_deployed'] for r in results)}")
    print(f"Total spans tracked: {tracker_stats['total_spans']}")
    print(f"Total violations: {auditor_stats['checker_stats']['total_violations']}")
    
    # Stop components
    await asyncio.gather(
        simulator.stop(),
        distiller.stop(),
        auditor.stop(),
        tracker.stop()
    )
    
    print("\n✅ Training complete!")
    
    return {
        "iterations": num_iterations,
        "results": results,
        "tracker_stats": tracker_stats,
        "auditor_stats": auditor_stats
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Project Plasma Training")
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=5,
        help="Number of phantom paths per iteration"
    )
    parser.add_argument(
        "--path-length",
        type=int,
        default=20,
        help="Length of each phantom path"
    )
    
    args = parser.parse_args()
    
    config = {
        "num_iterations": args.iterations,
        "num_paths": args.num_paths,
        "path_length": args.path_length,
        "simulator": {},
        "distiller": {},
        "auditor": {},
        "tracker": {}
    }
    
    # Run training with asyncio
    try:
        results = asyncio.run(train(config))
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
