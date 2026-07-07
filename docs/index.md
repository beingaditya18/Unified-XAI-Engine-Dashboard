# 🛡️ Unified XAI Governance Engine

Welcome to the official developer documentation for the **Unified XAI Governance Engine**. This project is a model-agnostic governance and audit dashboard designed to inspect tabular machine learning models, run cohort-based demographic parity/fairness audits, and perform model prediction divergence analytics.

---

## 🎯 Project Core Objectives

1. **Global & Local Interpretability:** Deconstruct XGBoost and MLP neural network model predictions using SHAP (Shapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations).
2. **Algorithmic Fairness Audits:** Quantify and display selection disparities (Disparate Impact Ratio, demographic parity, equal opportunity difference) across protected groups like Sex and Race.
3. **Model Divergence Analytics:** Track samples where tree-based models and neural network models reach conflicting decisions to identify model boundary behavior and uncertainty.
4. **Production-ready API:** Expose model inferences, explanations, and fairness metrics via FastAPI endpoints.

---

## 🏗️ Getting Started

To install dependencies and train models locally:

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Run training
make train

# Run UI
make run-ui
```

For dockerized orchestration:
```bash
docker compose up --build
```
