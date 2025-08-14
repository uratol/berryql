# Contributing to BerryQL

Thank you for your interest in contributing to BerryQL! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/uratol/berryql.git
   cd berryql
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,test]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **flake8**: Linting

Run these tools before committing:

```bash
# Format code
black berryql tests

# Sort imports
isort berryql tests

# Type checking
mypy berryql

# Linting
flake8 berryql tests
```

### Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=berryql --cov-report=html

# Run specific test file
pytest tests/test_factory.py

# Run with verbose output
pytest -v
```

### Documentation

- Use docstrings for all public functions and classes
- Follow Google-style docstring format
- Include type hints for all function parameters and return values
- Add examples to docstrings where helpful

Example:
```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2 with default value
        
    Returns:
        Description of what is returned
        
    Raises:
        ValueError: When param1 is invalid
        
    Examples:
        >>> example_function("test", 5)
        True
    """
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, concise commit messages
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Submit a pull request**
   - Provide a clear description of the changes
   - Reference any related issues
   - Include screenshots or examples if applicable

### Pull Request Guidelines

- Keep changes focused and atomic
- Write descriptive commit messages
- Include tests for new features
- Update documentation for API changes
- Ensure all CI checks pass

## Issue Reporting

When reporting issues, please include:

1. **Bug description**: Clear description of the problem
2. **Expected behavior**: What you expected to happen
3. **Actual behavior**: What actually happened
4. **Environment details**: Python version, dependency versions, OS
5. **Minimal reproduction**: Minimal code example that reproduces the issue
6. **Stack trace**: If applicable, include the full stack trace

## Feature Requests

For feature requests, please include:

1. **Use case**: Describe the problem you're trying to solve
2. **Proposed solution**: Your ideas for how to address it
3. **Alternatives**: Other solutions you've considered
4. **Implementation details**: Any technical considerations

## Code Organization

### Directory Structure

```
berryql/
├── berryql/           # Main package
│   ├── __init__.py    # Package exports
│   ├── factory.py     # Main BerryQL factory
│   ├── query_analyzer.py  # GraphQL query analysis
│   ├── input_types.py # GraphQL input type definitions
│   ├── input_converter.py  # Input conversion utilities
│   └── resolved_data_helper.py  # Data resolution helpers
├── tests/             # Test suite
├── docs/              # Documentation
├── examples/          # Usage examples
└── scripts/           # Development scripts
```

### Coding Standards

- Use type hints throughout the codebase
- Follow PEP 8 style guidelines (enforced by Black)
- Keep functions focused and single-purpose
- Use descriptive variable and function names
- Add docstrings to all public APIs
- Write tests for all new functionality

## Release Process

Releases are handled by maintainers:

1. Update version in `pyproject.toml` and `__init__.py`
2. Update `CHANGELOG.md` with release notes
3. Create a git tag for the release
4. Build and publish to PyPI

## Questions?

If you have questions about contributing, please:

1. Check existing issues and discussions
2. Create a new issue with the "question" label
3. Join our community discussions

Thank you for contributing to BerryQL!
