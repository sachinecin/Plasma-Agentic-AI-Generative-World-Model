"""
Tests for instruction packet compiler and runtime.
"""

import pytest
from pathlib import Path
from instruction_packets.compiler import InstructionPacketCompiler
from instruction_packets.runtime import InstructionPacketRuntime
from instruction_packets.config import PacketConfig


def test_compiler_initialization():
    """Test compiler initialization."""
    compiler = InstructionPacketCompiler()
    assert compiler is not None
    assert compiler.compiled_packets == {}


@pytest.mark.asyncio
async def test_compile_instruction_packet():
    """Test instruction packet compilation."""
    compiler = InstructionPacketCompiler()
    result = await compiler.compile(
        lora_path="/tmp/test_lora.pt",
        agent_id="test_agent",
        target_device="edge"
    )
    
    assert result['agent_id'] == "test_agent"
    assert result['target_device'] == "edge"
    assert result['status'] == 'compiled'
    assert 'packet_size_bytes' in result


def test_runtime_initialization():
    """Test runtime initialization."""
    runtime = InstructionPacketRuntime()
    assert runtime is not None
    assert runtime.loaded_packet is None
    assert runtime.inference_count == 0


def test_runtime_statistics():
    """Test runtime statistics."""
    runtime = InstructionPacketRuntime()
    stats = runtime.get_statistics()
    
    assert 'agent_id' in stats
    assert 'packet_loaded' in stats
    assert 'inference_count' in stats
    assert stats['packet_loaded'] is False
    assert stats['inference_count'] == 0


def test_runtime_forward_without_packet():
    """Test that forward fails without a loaded packet."""
    runtime = InstructionPacketRuntime()
    
    with pytest.raises(RuntimeError, match="No instruction packet loaded"):
        runtime.forward(observation=None)
