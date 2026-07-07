from fastapi.testclient import TestClient

from src.api.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data


def test_predict_endpoint():
    # Send request with single mock profile
    payload = {
        "instances": [
            {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        ],
        "model_type": "xgboost",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert "probability" in data["results"][0]


def test_explain_endpoints():
    payload = {
        "instance": {
            "age": 39,
            "workclass": "State-gov",
            "fnlwgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        },
        "model_type": "xgboost",
    }

    # Test LIME
    response = client.post("/explain/lime", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data
    assert "predictions" in data["explanation"]

    # Test SHAP
    response = client.post("/explain/shap", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data
    assert "predictions" in data["explanation"]


def test_fairness_endpoint():
    response = client.get("/fairness?model_type=xgboost")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "gender" in data["metrics"]
