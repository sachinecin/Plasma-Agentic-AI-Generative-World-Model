# Repository Organization Commands

This document provides the commands used to organize the Plasma Agentic AI Generative World Model repository into a professional research structure, following the microsoft/agent-lightning pattern.

## Directory Creation Commands

```bash
# Create the main directory structure
mkdir -p plasma
mkdir -p examples
mkdir -p docs
mkdir -p tests
```

Or in a single command:
```bash
mkdir -p plasma examples docs tests
```

## File Movement Commands

Since the repository was initially empty (only README.md at root), no file movement was necessary. However, here are guidelines for organizing future files:

### Moving Python Source Files to /plasma
```bash
# Move core framework files
mv world_model.py plasma/
mv phantom_path.py plasma/
mv lora_distillation.py plasma/
mv instruction_packets.py plasma/
mv visual_learning.py plasma/
mv adversarial_auditing.py plasma/
```

### Moving Example Scripts to /examples
```bash
# Move logistics and demonstration scripts
mv logistics_optimization.py examples/
mv phantom_path_demo.py examples/
mv edge_deployment_demo.py examples/
mv *_example.py examples/
```

### Moving Documentation to /docs
```bash
# Move research papers and documentation
mv *.pdf docs/
mv *.tex docs/
mv design_*.md docs/
mv architecture_*.md docs/
mv research_papers/ docs/
```

### Moving Tests to /tests
```bash
# Move test files
mv test_*.py tests/
mv *_test.py tests/
mv conftest.py tests/
```

## Subdirectory Structure

### For /plasma (Core Package)
```bash
# Create submodules for better organization
mkdir -p plasma/models
mkdir -p plasma/simulations
mkdir -p plasma/adaptation
mkdir -p plasma/learning
mkdir -p plasma/auditing
```

### For /docs (Documentation)
```bash
# Create documentation categories
mkdir -p docs/papers
mkdir -p docs/design
mkdir -p docs/tutorials
mkdir -p docs/api
```

### For /examples (Examples)
```bash
# Create example categories
mkdir -p examples/logistics
mkdir -p examples/vision
mkdir -p examples/edge
```

## Current Structure

After reorganization, the repository structure is:

```
Plasma-Agentic-AI-Generative-World-Model/
├── README.md                 # Root readme with structure overview
├── plasma/                   # Core framework code
│   ├── README.md            # Package documentation
│   └── __init__.py          # Package initialization
├── examples/                 # Logistics and demonstrations
│   └── README.md            # Examples documentation
├── docs/                     # Research papers and docs
│   └── README.md            # Documentation guide
└── tests/                    # Test suite
    └── README.md            # Testing guidelines
```

## Best Practices

1. **Keep the root clean**: Only keep essential files (README.md, LICENSE, pyproject.toml, etc.) at the root
2. **Organize by function**: Core code in plasma/, demos in examples/, docs in docs/, tests in tests/
3. **Use README files**: Each directory has a README explaining its purpose
4. **Follow conventions**: Match the structure of established projects like microsoft/agent-lightning
5. **Version control**: Use .gitignore to exclude build artifacts, __pycache__, etc.

## Additional Setup Files

Consider adding these standard files to the root:

```bash
# Create standard repository files
touch .gitignore
touch LICENSE
touch pyproject.toml
touch requirements.txt
touch setup.py
```

## Git Operations

After organizing files:

```bash
# Stage all changes
git add .

# Commit the reorganization
git commit -m "Organize repository into professional research structure"

# Push to remote
git push origin main
```

## References

This structure is inspired by:
- microsoft/agent-lightning: https://github.com/microsoft/agent-lightning
- Python packaging best practices
- Research software engineering standards
