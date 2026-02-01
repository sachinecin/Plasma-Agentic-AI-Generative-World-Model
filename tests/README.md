# Tests

This directory contains the test suite for the Plasma framework.

## Test Structure

Tests are organized by module:

- **test_world_model.py**: World model tests
- **test_phantom_path.py**: Phantom-path simulation tests
- **test_lora_distillation.py**: LoRA distillation tests
- **test_instruction_packets.py**: Instruction packet tests
- **test_visual_learning.py**: Visual error learning tests
- **test_adversarial_auditing.py**: Adversarial auditing tests

## Running Tests

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_world_model.py
```

Run with coverage:
```bash
pytest --cov=plasma --cov-report=html
```

## Test Guidelines

When writing tests:
1. Use pytest framework
2. Follow naming convention: `test_*.py`
3. Include unit tests for core functionality
4. Add integration tests for complex workflows
5. Mock external dependencies
6. Aim for >80% code coverage

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Commits to main branch
- Scheduled daily builds
