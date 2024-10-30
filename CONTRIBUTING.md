# Contributing to Multi-GPU Object Detection with YOLO Family Models

First off, thank you for your interest in contributing to our project! 🎉 We welcome contributions from the community to help improve and expand this repository. By participating in this project, you agree to abide by our [Code of Conduct](#code-of-conduct).

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
  - [Reporting Issues](#reporting-issues)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Development Guidelines](#development-guidelines)
  - [Branching Strategy](#branching-strategy)
  - [Commit Messages](#commit-messages)
  - [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Resources](#resources)

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) to understand the expectations for participants in this project.

## How to Contribute

### Reporting Issues

If you encounter any bugs or have suggestions for improvements, please open an issue using the [issue tracker](https://github.com/memari-majid/Multi-GPU-Object-Detection/issues). When reporting a bug, include the following:

- A clear and descriptive title.
- A detailed description of the problem.
- Steps to reproduce the issue.
- Expected and actual behavior.
- Any relevant screenshots or logs.

### Submitting Pull Requests

We welcome pull requests! To ensure a smooth process, please follow these steps:

1. **Fork the Repository**
   Click the "Fork" button at the top right of this page to create your own copy of the repository.

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Make Your Changes**
   - Ensure your code adheres to the project's coding standards.
   - Update or add tests as necessary.

4. **Commit Your Changes**
   ```bash
   git commit -m "Add feature: YourFeatureName"
   ```

5. **Push to Your Fork**
   ```bash
   git push origin feature/YourFeatureName
   ```

6. **Open a Pull Request**
   Navigate to the original repository and click "Compare & pull request". Provide a clear description of your changes and reference any related issues.

## Development Guidelines

### Branching Strategy

- **Main Branch (`main`)**: Contains stable and production-ready code.
- **Development Branch (`develop`)**: Integrates features before they are merged into the main branch.
- **Feature Branches (`feature/*`)**: Used for developing new features or fixes.

### Commit Messages

Use clear and descriptive commit messages. Follow this format:

```
[type]: [short description]

[optional longer description]
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Other changes (build process, etc.)

## Testing

- Ensure all new features and fixes are accompanied by appropriate tests.
- Run existing tests to confirm that your changes do not break the project.
- Use `pytest` for running tests:
  ```bash
  pytest
  ```

## Documentation

- Update the `README.md` and other relevant documentation to reflect your changes.
- Ensure that any new features are well-documented.

## Resources

- [Contributing to Open Source](https://opensource.guide/how-to-contribute/)
- [GitHub Forking Workflow](https://guides.github.com/introduction/flow/)
- [PEP 8 – Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [Commit Message Guidelines](https://www.conventionalcommits.org/en/v1.0.0/)

---

Thank you for contributing to the project! Your efforts help make this repository better for everyone. 