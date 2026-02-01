# Contributing to Plasma Agentic AI

Thank you for your interest in contributing to the Plasma Agentic AI Generative World Model project!

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Plasma-Agentic-AI-Generative-World-Model.git
   cd Plasma-Agentic-AI-Generative-World-Model
   ```
3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

Follow the project structure:
- **Core changes**: Edit files in `/core`
- **World Model**: Edit files in `/world_model`
- **Instruction Packets**: Edit files in `/instruction_packets`
- **Examples/Recipes**: Add to `/contrib/recipes`

### 3. Write Tests

Add tests for your changes in `/tests`:

```python
# tests/test_your_feature.py
import pytest

def test_your_feature():
    # Your test code here
    assert True
```

### 4. Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=core --cov=world_model --cov=instruction_packets tests/

# Run specific test file
pytest tests/test_your_feature.py
```

### 5. Format Your Code

```bash
# Format with black
black .

# Sort imports
isort .

# Type checking (optional but recommended)
mypy core/ world_model/ instruction_packets/
```

### 6. Commit Your Changes

```bash
git add .
git commit -m "feat: Add your feature description"
```

Follow conventional commit format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `style:` - Code style changes
- `chore:` - Maintenance tasks

### 7. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Contributing Recipes

Community recipes are a great way to contribute! Add your recipe to `/contrib/recipes`:

1. **Create your recipe file**:
   ```bash
   touch contrib/recipes/my_recipe.py
   ```

2. **Write your recipe** with:
   - Clear docstring explaining the purpose
   - Step-by-step comments
   - Example usage
   - Expected output

3. **Add documentation** in `contrib/README.md`

4. **Test your recipe**:
   ```bash
   python contrib/recipes/my_recipe.py
   ```

### Recipe Template

```python
"""
My Recipe Name

This recipe demonstrates [what it does].
"""

import asyncio
from core import SimulationEngine
from world_model import WorldModel

async def main():
    """
    Recipe workflow:
    1. [Step 1 description]
    2. [Step 2 description]
    ...
    """
    print("=== My Recipe ===\n")
    
    # Your implementation here
    
    print("\n=== Complete! ===")

if __name__ == "__main__":
    asyncio.run(main())
```

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints where appropriate
- Max line length: 100 characters
- Use docstrings for classes and functions

Example:
```python
from typing import List, Dict, Any

def generate_trajectories(
    task: str,
    num_trajectories: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate phantom-path trajectories.
    
    Args:
        task: Description of the task to simulate
        num_trajectories: Number of trajectories to generate
    
    Returns:
        List of trajectory dictionaries
    """
    # Implementation here
    pass
```

### Docstring Format

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    More detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When input is invalid
    """
    pass
```

## Architecture Guidelines

### Adding New Features

When adding new features, consider:

1. **Separation of Concerns**: Does this belong in core, world_model, or instruction_packets?
2. **TA Design**: Does it maintain the Training-Agent Disaggregation pattern?
3. **Configurability**: Should this be configurable via config classes?
4. **Testing**: Can this be unit tested?
5. **Documentation**: Is the purpose clear from docstrings?

### Server Changes

When modifying `server.py`:

- Add new endpoints with proper type hints
- Include OpenAPI documentation
- Handle errors gracefully
- Log important operations
- Test with curl or httpx

### Client Changes

When modifying `client.py`:

- Maintain async/await pattern
- Handle network errors
- Add retry logic where appropriate
- Cache data when possible
- Document expected server behavior

## Testing Guidelines

### Unit Tests

- Test individual functions/methods
- Mock external dependencies
- Use pytest fixtures for setup
- Aim for >80% coverage

### Integration Tests

- Test complete workflows
- Use actual server/client communication
- Test error cases
- Document setup requirements

### Example Test

```python
import pytest
from core.simulation_engine import SimulationEngine

@pytest.mark.asyncio
async def test_generate_trajectories():
    """Test trajectory generation."""
    engine = SimulationEngine()
    trajectories = await engine.generate_trajectories(
        task="Test task",
        num_trajectories=5
    )
    
    assert len(trajectories) == 5
    for traj in trajectories:
        assert 'states' in traj
        assert 'actions' in traj
```

## Documentation

### Code Documentation

- Docstrings for all public APIs
- Inline comments for complex logic
- Type hints for function signatures

### User Documentation

- Update README.md for user-facing changes
- Update ARCHITECTURE.md for architectural changes
- Add examples to contrib/recipes for new features

## Pull Request Guidelines

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts

### PR Description

Include in your PR:

1. **What**: Brief description of changes
2. **Why**: Motivation for the changes
3. **How**: Implementation approach
4. **Testing**: How you tested the changes
5. **Screenshots**: If UI/output changes

Example:
```markdown
## What
Add support for multi-agent simulations

## Why
Users requested the ability to simulate multiple agents interacting

## How
- Extended SimulationEngine to handle multiple agent IDs
- Updated WorldModel to support multi-agent state
- Added coordination logic for agent interactions

## Testing
- Added unit tests for multi-agent scenarios
- Ran integration test with 3 agents
- Verified no performance degradation

## Screenshots
N/A
```

## Community

### Getting Help

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Discord**: [Coming soon]

### Code of Conduct

Be respectful and constructive. We welcome contributors of all skill levels.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue or discussion if you have questions about contributing!
