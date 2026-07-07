import json
import logging
import os

import joblib
import xgboost as xgb
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from src.data.data_pipeline import AdultIncomePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_save(config_path: str = "config/config.yaml"):
    logger.info("🚀 Starting Model Training Pipeline...")

    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Run data pipeline
    pipeline = AdultIncomePipeline(config_path)
    X_train, X_test, y_train, y_test = pipeline.run_pipeline()

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # 1. Train XGBoost
    xgb_config = config["models"]["xgb"]
    xgb_params = xgb_config["params"]
    logger.info(f"📦 Training XGBoost with params: {xgb_params}")

    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train)
    joblib.dump(model_xgb, xgb_config["path"])
    logger.info(f"Serialized XGBoost model to {xgb_config['path']}")

    # 2. Train Neural Network (MLP)
    nn_config = config["models"]["nn"]
    nn_params = nn_config["params"]
    logger.info(f"🧠 Training Neural Network (MLP) with params: {nn_params}")

    model_nn = MLPClassifier(**nn_params)
    model_nn.fit(X_train, y_train)
    joblib.dump(model_nn, nn_config["path"])
    logger.info(f"Serialized MLP model to {nn_config['path']}")

    # 3. Evaluate and Save Metrics
    metrics = {}
    for name, model in [("xgboost", model_xgb), ("neural_network", model_nn)]:
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probas)

        metrics[name] = {"accuracy": float(acc), "f1_score": float(f1), "roc_auc": float(auc)}
        logger.info(
            f"{name.upper()} Metrics -> Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
        )

    metrics_path = "models/model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved evaluation metrics to {metrics_path}")
    logger.info("✅ All models trained and saved successfully!")


if __name__ == "__main__":
    train_and_save()
