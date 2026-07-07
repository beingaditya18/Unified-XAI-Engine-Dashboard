import numpy as np
import pandas as pd
import pytest

from src.fairness.fairness_audit import FairnessAuditEngine


@pytest.fixture
def fairness_engine():
    return FairnessAuditEngine("config/config.yaml")


def test_fairness_calculations(fairness_engine):
    # Construct mock dataset
    # sex column: 1=Male (privileged), 0=Female (unprivileged)
    # race column: 4=White (privileged), 2=Black (unprivileged)
    df_mock = pd.DataFrame(
        {
            "sex": [1, 1, 1, 1, 0, 0, 0, 0],
            "race": [4, 4, 4, 4, 2, 2, 2, 2],
            "income": [1, 1, 0, 0, 1, 0, 0, 0],  # actuals
        }
    )

    # Mock model predictions
    # Male: 3 positive, 1 negative -> 75% selection rate
    # Female: 1 positive, 3 negative -> 25% selection rate
    preds_mock = np.array([1, 1, 1, 0, 1, 0, 0, 0])

    results = fairness_engine.calculate_fairness_metrics(df_mock, df_mock["income"], preds_mock)

    assert "gender" in results
    gender_res = results["gender"]

    assert gender_res["privileged_selection_rate"] == 0.75
    assert gender_res["unprivileged_selection_rate"] == 0.25
    assert gender_res["disparate_impact_ratio"] == pytest.approx(0.25 / 0.75)  # 0.3333
    assert gender_res["demographic_parity_difference"] == 0.50
    assert gender_res["is_biased"] is True
