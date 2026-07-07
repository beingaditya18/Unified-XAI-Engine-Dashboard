import pytest

from explainability.lime_engine import LimeExplainerEngine
from explainability.shap_engine import ShapExplainerEngine
from src.data.data_pipeline import AdultIncomePipeline
from src.models.model_service import ModelService


@pytest.fixture
def core_deps():
    pipeline = AdultIncomePipeline("config/config.yaml")
    service = ModelService("config/config.yaml")

    pipeline.load_encoders()
    service.load_models()

    # Redirect to prevent corrupting production encoders
    pipeline.encoder_path = "tests/test_encoders.pkl"

    shap_eng = ShapExplainerEngine("config/config.yaml")
    lime_eng = LimeExplainerEngine("config/config.yaml")

    return pipeline, service, shap_eng, lime_eng


def test_shap_explanation(core_deps):
    pipeline, service, shap_eng, lime_eng = core_deps

    df_all = pipeline.load_data().head(10)
    X_enc, _ = pipeline.fit_transform(df_all)

    model = service._get_model("xgboost")

    # Calculate shap values
    shap_vals = shap_eng.calculate_shap_values(model, X_enc, X_enc, "xgboost")
    assert shap_vals.shape == X_enc.shape

    # Local explanation
    local_exp = shap_eng.explain_instance(model, X_enc.iloc[[0]], X_enc, "xgboost")
    assert "base_value" in local_exp
    assert "predictions" in local_exp
    assert len(local_exp["predictions"]) == X_enc.shape[1]

    # Cleanup test artifact
    import os
    if os.path.exists(pipeline.encoder_path):
        os.remove(pipeline.encoder_path)


def test_lime_explanation(core_deps):
    pipeline, service, shap_eng, lime_eng = core_deps

    df_all = pipeline.load_data().head(10)
    X_enc, _ = pipeline.fit_transform(df_all)

    explainer = lime_eng.get_explainer(X_enc)
    predict_fn = service._get_model("xgboost").predict_proba

    local_exp = lime_eng.explain_instance(explainer, X_enc.iloc[0], predict_fn)
    assert "intercept" in local_exp
    assert "predictions" in local_exp
    assert len(local_exp["predictions"]) <= lime_eng.num_features

    # Cleanup test artifact
    import os
    if os.path.exists(pipeline.encoder_path):
        os.remove(pipeline.encoder_path)
