import logging
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from explainability.lime_engine import LimeExplainerEngine
from explainability.shap_engine import ShapExplainerEngine
from src.data.data_pipeline import AdultIncomePipeline
from src.fairness.fairness_audit import FairnessAuditEngine
from src.models.model_service import ModelService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified XAI Engine REST API",
    description="Production-grade API endpoints for model inference, local SHAP/LIME explanations, and fairness auditing.",
    version="1.0.0",
)

# Initialize service classes
config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
data_pipeline = AdultIncomePipeline(config_path)
model_service = ModelService(config_path)
shap_engine = ShapExplainerEngine(config_path)
lime_engine = LimeExplainerEngine(config_path)
fairness_engine = FairnessAuditEngine(config_path)

# Initialize models and encoders
try:
    data_pipeline.load_encoders()
    model_service.load_models()
    logger.info("REST API dependencies loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load dependencies during startup: {e}. Models might need training.")


# --- PYDANTIC SCHEMAS ---
class InputInstance(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlwgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    class Config:
        allow_population_by_field_name = True


class BatchPredictionRequest(BaseModel):
    instances: list[InputInstance]
    model_type: str = Field(default="xgboost", description="Architectures: 'xgboost' or 'mlp'")


class ExplanationRequest(BaseModel):
    instance: InputInstance
    model_type: str = Field(default="xgboost", description="Architectures: 'xgboost' or 'mlp'")


# --- ENDPOINTS ---


@app.get("/health")
def health_check():
    """Returns application health and model loading readiness status."""
    ready = model_service.model_xgb is not None and model_service.model_nn is not None
    return {
        "status": "healthy" if ready else "degraded",
        "models_loaded": ready,
        "models": {
            "xgboost": model_service.model_xgb is not None,
            "neural_network": model_service.model_nn is not None,
        },
    }


@app.post("/predict")
def predict_endpoint(request: BatchPredictionRequest):
    """Processes batch inference. Returns classification decisions and probabilities."""
    try:
        # Convert Pydantic request to pandas DataFrame
        df_raw = pd.DataFrame([inst.dict(by_alias=True) for inst in request.instances])

        # Preprocess features using saved label encoders
        df_encoded = data_pipeline.transform_instance(df_raw)

        # Predict using requested model
        probas = model_service.predict_proba(request.model_type, df_encoded)
        preds = model_service.predict(request.model_type, df_encoded)

        results = []
        for i, (pred, proba) in enumerate(zip(preds, probas)):
            results.append(
                {
                    "index": i,
                    "prediction": "High Income (>50K)" if pred == 1 else "Low Income (<=50K)",
                    "label": int(pred),
                    "probability": float(proba[1]),
                }
            )

        return {"model_type": request.model_type, "results": results}

    except Exception as e:
        logger.error(f"Error handling prediction request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/shap")
def explain_shap_endpoint(request: ExplanationRequest):
    """Generates structured local SHAP force values for a single data point."""
    try:
        # Raw dataframe
        df_raw = pd.DataFrame([request.instance.dict(by_alias=True)])
        df_encoded = data_pipeline.transform_instance(df_raw)

        # Load complete training set context for KernelSHAP background if needed
        # We run the pipeline load_data step to retrieve data context
        df_all = data_pipeline.load_data()
        X_train_enc, _ = data_pipeline.fit_transform(df_all)

        model = model_service._get_model(request.model_type)
        explanation = shap_engine.explain_instance(
            model, df_encoded, X_train_enc, request.model_type
        )

        return {"model_type": request.model_type, "explanation": explanation}

    except Exception as e:
        logger.error(f"Error executing SHAP explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/lime")
def explain_lime_endpoint(request: ExplanationRequest):
    """Generates local LIME surrogate explanation coefficients."""
    try:
        df_raw = pd.DataFrame([request.instance.dict(by_alias=True)])
        df_encoded = data_pipeline.transform_instance(df_raw)

        df_all = data_pipeline.load_data()
        X_train_enc, _ = data_pipeline.fit_transform(df_all)

        # Build tabular explainer
        explainer = lime_engine.get_explainer(X_train_enc)
        predict_fn = model_service._get_model(request.model_type).predict_proba

        explanation = lime_engine.explain_instance(
            explainer=explainer, instance=df_encoded.iloc[0], predict_fn=predict_fn
        )

        return {"model_type": request.model_type, "explanation": explanation}

    except Exception as e:
        logger.error(f"Error executing LIME explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fairness")
def fairness_endpoint(model_type: str = "xgboost"):
    """Audits cohorts and returns gender and race disparity indicators on the test split."""
    try:
        # Load split test dataset if exists
        test_path = data_pipeline.config["data"]["test_path"]
        if not os.path.exists(test_path):
            # Fallback to run split
            _, X_test, _, y_test = data_pipeline.run_pipeline()
        else:
            df_test = pd.read_csv(test_path)
            X_test = df_test.drop(data_pipeline.target_col, axis=1)
            y_test = df_test[data_pipeline.target_col]

        preds = model_service.predict(model_type, X_test)

        audit_results = fairness_engine.calculate_fairness_metrics(X_test, y_test, preds)

        return {"model_type": model_type, "metrics": audit_results}

    except Exception as e:
        logger.error(f"Error processing fairness audit: {e}")
        raise HTTPException(status_code=500, detail=str(e))
