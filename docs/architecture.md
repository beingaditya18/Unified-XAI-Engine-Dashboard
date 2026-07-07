# Architecture & Design

The **Unified XAI Governance Engine** is designed to decouple data processing, model training, explanation computation, and client interfaces.

## File Structure

```
├── config/                  # Hyperparameters and thresholds
├── src/
│   ├── data/                # Data pipelines (AdultIncomePipeline)
│   ├── models/              # Model managers (ModelService)
│   ├── fairness/            # Disparity trackers (FairnessAuditEngine)
│   └── api/                 # FastAPI REST routing
├── explainability/          # SHAP & LIME wrappers
├── dashboard/               # Streamlit application UI
```

## Data Flow Pipeline

1. **Preprocessing:** UCI Adult Income dataset is cleaned (treating missing `?` tokens by mode replacement and imputing numerical columns with median statistics).
2. **Feature Encoding:** High-cardinality categorical variables are encoded using serialized `LabelEncoder` objects.
3. **Training & Serialization:** A tree-based ensemble (`XGBoost`) and deep feedforward architecture (`MLP`) are trained, evaluated, and saved to `models/`.
4. **Explanation Generation:**
   - **SHAP Engine:** Employs KernelSHAP background summaries and tree paths to compute additive feature contributions.
   - **LIME Engine:** Computes localized linear surrogate models dynamically around the active query instances.
5. **Fairness Audits:** Evaluates demographic parity differences, equal opportunity rate differences, and disparate impact ratios on the validation split.
