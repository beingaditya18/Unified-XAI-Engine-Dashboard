import numpy as np
import pandas as pd
import pytest

from src.data.data_pipeline import AdultIncomePipeline
from src.models.model_service import ModelService


@pytest.fixture
def services():
    pipeline = AdultIncomePipeline("config/config.yaml")
    service = ModelService("config/config.yaml")

    # Run pipeline fit if not exists to construct encoders
    try:
        pipeline.load_encoders()
    except Exception:
        pipeline.run_pipeline()

    try:
        service.load_models()
    except Exception:
        from models.train_models import train_and_save

        train_and_save("config/config.yaml")
        service.load_models()

    # Redirect to prevent corrupting production encoders
    pipeline.encoder_path = "tests/test_encoders.pkl"

    return pipeline, service


def test_model_service_prediction(services):
    pipeline, service = services

    # Select sample
    df_all = pipeline.load_data().head(5)
    X_enc, _ = pipeline.fit_transform(df_all)

    # Predict xgboost
    probas = service.predict_proba("xgboost", X_enc)
    preds = service.predict("xgboost", X_enc)

    assert probas.shape == (5, 2)
    assert preds.shape == (5,)
    assert np.all(probas >= 0) and np.all(probas <= 1)

    # Cleanup test artifact
    import os

    if os.path.exists(pipeline.encoder_path):
        os.remove(pipeline.encoder_path)


def test_model_divergence(services):
    pipeline, service = services
    df_all = pipeline.load_data().head(200)  # larger sample to verify mismatch
    X_enc, _ = pipeline.fit_transform(df_all)

    divergent_df = service.get_divergent_instances(X_enc)

    # Assert return types
    assert isinstance(divergent_df, pd.DataFrame)
    if not divergent_df.empty:
        assert "pred_xgb" in divergent_df.columns
        assert "pred_nn" in divergent_df.columns
        assert np.any(divergent_df["pred_xgb"] != divergent_df["pred_nn"])

    # Cleanup test artifact
    import os

    if os.path.exists(pipeline.encoder_path):
        os.remove(pipeline.encoder_path)
