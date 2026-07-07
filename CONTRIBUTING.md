# Contributing to Unified XAI Engine

Thank you for choosing to help improve the Unified XAI Engine! Below are guidelines to help you get started contributing to this open-source project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork this repository.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Unified-XAI-Engine-Dashboard.git
   ```
3. Set up a virtual environment and install packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
4. Install the pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a branch for your changes:
   ```bash
   git checkout -b feature/amazing-feature
   ```
2. Write clean code conforming to project style rules (Black formatter & Ruff linter).
3. Add pytest test cases under the `tests/` directory.
4. Run tests to ensure everything compiles:
   ```bash
   make test
   ```
5. Commit and push your changes, then submit a Pull Request.

## Pull Request Guidelines

* Provide a clear description of the problem solved or feature added.
* Ensure all tests pass in the GitHub Actions CI pipeline.
* Link to any corresponding open issue in your PR.
