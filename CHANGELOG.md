# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-07

### Added
- Created `pyproject.toml` containing default configuration settings for `black`, `ruff`, and `mypy`.
- Configured remote development support with `.devcontainer/devcontainer.json`.
- Set up automated API and user documentation using `mkdocs` and `mkdocs.yml`.
- Drafted a formal Model Governance and Audit document: `model_card.yaml`.
- Created a data provenance catalog: `data/DATA_README.md`.
- Implemented core demographic disparity tests for protected cohort features.
- Developed the FastAPI server serving predictions, local explanations (LIME, SHAP), and cohort audits.
- Developed glassmorphic Streamlit frontend to display attributions and divergence models.

### Fixed
- Resolved syntactical installation error in `.github/workflows/ci.yml` (`python -m pip upgrade pip` corrected to `--upgrade pip`).
