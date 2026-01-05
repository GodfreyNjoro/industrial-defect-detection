# Contributing to Industrial Defect Detection

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/industrial-defect-detection.git
   cd industrial-defect-detection
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

2. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Making Changes

### Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting: `black src/`
- Use type hints where applicable
- Write docstrings for all functions and classes

### Testing

- Write tests for new features
- Ensure all tests pass: `pytest tests/`
- Aim for >80% code coverage

### Documentation

- Update README.md if adding new features
- Add docstrings to all functions and classes
- Update configuration examples if needed

## Commit Guidelines

Use clear and descriptive commit messages:

```
type(scope): short description

Longer description if needed

Fixes #issue_number
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(models): add ResNet backbone support
fix(training): resolve memory leak in data loader
docs(readme): update installation instructions
```

## Pull Request Process

1. **Update your fork:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests and checks:**
   ```bash
   pytest tests/
   flake8 src/
   black src/ --check
   mypy src/
   ```

3. **Push your changes:**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub:
   - Provide a clear title and description
   - Reference any related issues
   - Include screenshots for visual changes
   - Ensure all CI checks pass

5. **Address review comments:**
   - Make requested changes
   - Push updates to your branch
   - Respond to reviewer comments

## Code Review

All submissions require code review. We follow these principles:

- **Be respectful**: Provide constructive feedback
- **Be thorough**: Check for correctness, style, and documentation
- **Be timely**: Respond to reviews within a few days

## Reporting Bugs

Bugs are tracked as GitHub issues. When creating a bug report, include:

- **Clear title** and description
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, GPU/CPU)
- **Code snippets** or error messages
- **Screenshots** if applicable

## Feature Requests

Feature requests are welcome! Please provide:

- **Clear description** of the feature
- **Use case** and motivation
- **Possible implementation** approach
- **Any alternatives** you've considered

## Questions?

If you have questions, feel free to:

- Open a GitHub issue with the `question` label
- Start a discussion in GitHub Discussions
- Contact the maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:

- The project README
- Release notes
- The contributors list on GitHub

Thank you for contributing! 🎉
