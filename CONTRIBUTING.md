# Contributing to VajraCode

Thank you for your interest in contributing to VajraCode! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (Python version, OS, etc.)

### Suggesting Enhancements

We welcome suggestions for new features or improvements! Please open an issue describing:
- The enhancement you'd like to see
- Why it would be useful
- How it might be implemented

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add or update tests as needed
5. Ensure all tests pass (`pytest tests/`)
6. Update documentation if needed
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to your branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/your-username/VajraCode.git
cd VajraCode
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install in development mode:
```bash
pip install -e .
pip install pytest pytest-cov black flake8
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Add type hints where appropriate

Format code with black:
```bash
black src/ tests/ scripts/
```

Check style with flake8:
```bash
flake8 src/ tests/ scripts/
```

## Testing

- Write tests for all new functionality
- Ensure all existing tests pass
- Aim for good test coverage

Run tests:
```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Documentation

- Update README.md if you add new features
- Add docstrings following Google or NumPy style
- Update or create example notebooks if relevant
- Keep documentation clear and concise

## Questions?

Feel free to open an issue for any questions about contributing!
