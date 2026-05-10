# Contributing to DERMA-Agent

Thank you for your interest in contributing to DERMA-Agent! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project and everyone participating in it is governed by our commitment to:
- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Prioritize patient safety and data privacy

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please:
1. Check if the issue already exists
2. Use the latest version
3. Provide detailed information:
   - Python version
   - Operating system
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and stack traces

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
1. Use a clear, descriptive title
2. Explain why this enhancement would be useful
3. Provide examples of how it would work
4. Consider the scope and impact

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- Virtual environment (recommended)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/your-username/DERMA-Agent.git
cd DERMA-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest flake8 black

# Run tests
pytest

# Run linting
flake8 .
```

## Project Structure

```
DERMA-Agent/
├── agents/           # Agent implementations
├── tools/            # Tools and utilities
├── notebooks/        # Jupyter notebooks
├── .github/          # GitHub workflows
├── tests/            # Test files (create if needed)
├── docs/             # Documentation (create if needed)
└── README.md         # Main documentation
```

## Coding Standards

### Python Style Guide

- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Maximum line length: 127 characters

### Example:

```python
def analyze_survival(df: pd.DataFrame, time_col: str = 'time') -> SurvivalResult:
    """
    Analyze survival data using Kaplan-Meier estimator.
    
    Args:
        df: DataFrame with survival data
        time_col: Column name for time values
        
    Returns:
        SurvivalResult with statistics
        
    Raises:
        ValueError: If required columns are missing
    """
    # Implementation here
    pass
```

### Documentation

- Update README.md for major features
- Add docstrings to all public APIs
- Comment complex logic
- Keep DERMA_AGENT_STATUS.md updated

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test
pytest tests/test_knowledge_fabric.py
```

### Writing Tests

- Add tests for new features
- Test edge cases
- Mock external APIs
- Keep tests fast and isolated

## Commit Messages

Use clear, descriptive commit messages:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

Examples:
```
feat(knowledge-fabric): add semantic search capability

fix(discovery-engine): handle missing columns gracefully
docs(readme): update installation instructions
```

## Areas for Contribution

### High Priority

1. **Additional Cancer Cohorts**: Integrate more TCGA projects
2. **ML Models**: Add deep learning survival models
3. **Data Sources**: cBioPortal, PubMed integration
4. **Tests**: Increase test coverage
5. **Documentation**: Tutorials and examples

### Medium Priority

1. **UI/UX**: Dashboard improvements
2. **Performance**: Optimize large dataset handling
3. **Pathology**: Better segmentation methods
4. **Caching**: Smarter cache strategies

### Low Priority

1. **Visualization**: More plot types
2. **Export**: Additional output formats
3. **Configuration**: More customization options

## Questions?

- Open an issue for discussion
- Check existing documentation
- Review closed issues for solutions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DERMA-Agent! 🚀
