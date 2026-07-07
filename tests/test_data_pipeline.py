import os

import numpy as np
import pandas as pd
import pytest

from src.data.data_pipeline import AdultIncomePipeline


@pytest.fixture
def pipeline():
    p = AdultIncomePipeline("config/config.yaml")
    p.encoder_path = "tests/test_encoders.pkl"
    return p


def test_pipeline_config(pipeline):
    assert pipeline.config is not None
    assert "data" in pipeline.config
    assert "models" in pipeline.config


def test_clean_data(pipeline):
    # Dummy data with '?' character
    dummy_df = pd.DataFrame(
        {
            "age": [39, 50, np.nan],
            "workclass": ["State-gov", "?", "Private"],
            "income": ["<=50K", ">50K", "<=50K"],
        }
    )

    cleaned = pipeline.clean_data(dummy_df)

    # Assert '?' replaced
    assert "?" not in cleaned["workclass"].values
    # Mode of workclass ('State-gov' or 'Private') should replace the missing value
    assert cleaned["workclass"].isnull().sum() == 0
    # Median age should fill nan age
    assert cleaned["age"].isnull().sum() == 0


def test_fit_transform_and_saving(pipeline):
    df = pipeline.load_data().head(100)  # Small subset
    X_enc, y_enc = pipeline.fit_transform(df)

    assert X_enc.shape[0] == 100
    assert y_enc.shape[0] == 100
    assert os.path.exists(pipeline.encoder_path)

    # Check that encoders were populated
    assert len(pipeline.encoders) > 0

    # Verify inverse transform
    decoded = pipeline.inverse_transform_instance(X_enc.head(1))
    assert decoded.iloc[0]["sex"] in ["Male", "Female"]

    # Cleanup test artifact
    if os.path.exists(pipeline.encoder_path):
        os.remove(pipeline.encoder_path)
