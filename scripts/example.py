"""
Example: End-to-End Project Plasma Workflow

This example demonstrates the complete workflow of Project Plasma,
showing how all components work together with asyncio.
"""

import asyncio
from plasma import Simulator, Distiller, Auditor, StateTracker


async def run_plasma_workflow():
    """
    Complete example workflow demonstrating all Project Plasma components
    """
    print("=" * 70)
    print("Project Plasma - End-to-End Example")
    print("=" * 70)
    
    # Configuration
    config = {
        "simulator": {"world_model": {"type": "generative"}},
        "distiller": {"injector": {"max_packets": 100}},
        "auditor": {"checker": {"strictness": "high"}},
        "tracker": {}
    }
    
    # Initialize all components
    print("\n1️⃣  Initializing components...")
    simulator = Simulator(config["simulator"])
    distiller = Distiller(config["distiller"])
    auditor = Auditor(config["auditor"])
    tracker = StateTracker(config["tracker"])
    
    # Start all components concurrently
    await asyncio.gather(
        simulator.start(),
        distiller.start(),
        auditor.start(),
        tracker.start()
    )
    print("   ✓ All components started")
    
    # Create execution trace
    print("\n2️⃣  Creating execution trace...")
    trace_id = tracker.create_trace(metadata={
        "workflow": "example",
        "version": "1.0"
    })
    print(f"   ✓ Trace created: {trace_id}")
    
    # Phase 1: Simulation
    print("\n3️⃣  Phase 1: Generating Phantom Paths")
    sim_span = await tracker.create_span(
        trace_id=trace_id,
        name="simulation_phase",
        metadata={"phase": "1"}
    )
    
    async with sim_span:
        initial_state = {
            "environment": "sandbox",
            "step": 0,
            "goal": "learn_optimal_policy"
        }
        
        print("   Simulating with 10 phantom paths...")
        paths = await simulator.simulate(
            initial_state=initial_state,
            num_paths=10,
            path_length=50
        )
        
        print(f"   ✓ Generated {len(paths)} phantom paths")
        print(f"   ✓ Average path length: {sum(len(p.states) for p in paths) / len(paths):.1f}")
        
        sim_span.update_state({
            "paths_generated": len(paths),
            "total_steps": sum(len(p.states) for p in paths)
        })
    
    # Phase 2: Adversarial Audit
    print("\n4️⃣  Phase 2: Adversarial Audit of Paths")
    audit_span = await tracker.create_span(
        trace_id=trace_id,
        name="audit_phase",
        metadata={"phase": "2"}
    )
    
    async with audit_span:
        print("   Running adversarial checks...")
        audit_result = await auditor.audit_paths(
            paths,
            criteria={"min_score": 0.7}
        )
        
        status = audit_result["overall_status"]
        print(f"   ✓ Audit Status: {status}")
        print(f"   ✓ Violations: {audit_result['adversarial_check']['total_violations']}")
        print(f"   ✓ Review Score: {audit_result['review_report'].score:.2f}")
        
        audit_span.update_state({
            "audit_passed": audit_result["passed"],
            "violations": audit_result["adversarial_check"]["total_violations"]
        })
    
    # Phase 3: Distillation (if audit passed)
    if audit_result["passed"]:
        print("\n5️⃣  Phase 3: LoRA Distillation")
        distill_span = await tracker.create_span(
            trace_id=trace_id,
            name="distillation_phase",
            metadata={"phase": "3"}
        )
        
        async with distill_span:
            print("   Distilling instruction packets from paths...")
            packets = await distiller.distill_from_paths(
                paths,
                target_layers=["layer_0", "layer_1", "layer_2"]
            )
            
            print(f"   ✓ Created {len(packets)} instruction packets")
            
            # Audit packets
            print("   Auditing instruction packets...")
            packet_audit = await auditor.audit_adaptations(packets)
            
            if packet_audit["passed"]:
                print("   Deploying packets...")
                deploy_result = await distiller.deploy_packets(packets)
                
                print(f"   ✓ Deployed {deploy_result['deployed']}/{deploy_result['total']} packets")
                print(f"   ✓ Total size: {deploy_result['stats']['total_size_bytes']} bytes")
                
                distill_span.update_state({
                    "packets_created": len(packets),
                    "packets_deployed": deploy_result["deployed"]
                })
            else:
                print("   ⚠️  Packet audit failed, skipping deployment")
    else:
        print("\n5️⃣  Phase 3: Skipped (audit failed)")
    
    # Complete trace
    print("\n6️⃣  Completing trace...")
    await tracker.complete_trace(trace_id)
    
    # Get final statistics
    print("\n7️⃣  Final Statistics")
    print("   " + "-" * 60)
    
    tracker_stats = tracker.get_trace_stats()
    print(f"   Total traces: {tracker_stats['total_traces']}")
    print(f"   Total spans: {tracker_stats['total_spans']}")
    print(f"   Avg spans per trace: {tracker_stats['avg_spans_per_trace']:.1f}")
    
    auditor_stats = auditor.get_audit_statistics()
    print(f"   Audits performed: {auditor_stats['review_stats']['total']}")
    print(f"   Violations detected: {auditor_stats['checker_stats']['total_violations']}")
    
    # Export trace for analysis
    trace_data = await tracker.export_trace(trace_id)
    print(f"   Trace exported: {len(trace_data['spans'])} spans")
    
    # Cleanup
    print("\n8️⃣  Shutting down components...")
    await asyncio.gather(
        simulator.stop(),
        distiller.stop(),
        auditor.stop(),
        tracker.stop()
    )
    print("   ✓ All components stopped")
    
    print("\n" + "=" * 70)
    print("✅ Workflow Complete!")
    print("=" * 70)


def main():
    """Entry point"""
    try:
        asyncio.run(run_plasma_workflow())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
