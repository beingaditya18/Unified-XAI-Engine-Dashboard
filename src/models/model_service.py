import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.model_xgb = None
        self.model_nn = None
        self.xgb_path = self.config["models"]["xgb"]["path"]
        self.nn_path = self.config["models"]["nn"]["path"]

    def _load_config(self, config_path: str) -> dict:
        with open(config_path) as f:
            return yaml.safe_load(f)

    def load_models(self) -> None:
        """Loads XGBoost and MLP models from serialized .pkl storage."""
        if not os.path.exists(self.xgb_path) or not os.path.exists(self.nn_path):
            raise FileNotFoundError(
                f"Model files not found. Run training script first. "
                f"Expected locations:\n- XGB: {self.xgb_path}\n- NN: {self.nn_path}"
            )

        self.model_xgb = joblib.load(self.xgb_path)
        self.model_nn = joblib.load(self.nn_path)
        logger.info("Loaded XGBoost and MLP models successfully.")

    def predict(self, model_type: str, X: pd.DataFrame) -> np.ndarray:
        """Predicts binary outcomes (0 or 1) using selected model architecture."""
        model = self._get_model(model_type)
        return model.predict(X)

    def predict_proba(self, model_type: str, X: pd.DataFrame) -> np.ndarray:
        """Predicts model probabilities [P(<=50K), P(>50K)] using selected model."""
        model = self._get_model(model_type)
        return model.predict_proba(X)

    def evaluate_model(
        self, model_type: str, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float]:
        """Calculates evaluation metrics (accuracy, F1, ROC-AUC) on test data."""
        model = self._get_model(model_type)
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]

        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_score": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probas)),
        }

    def get_divergent_instances(self, X: pd.DataFrame, max_samples: int = 100) -> pd.DataFrame:
        """Identifies instances where XGBoost and Neural Network predictions contradict."""
        if self.model_xgb is None or self.model_nn is None:
            self.load_models()

        preds_xgb = self.model_xgb.predict(X)
        preds_nn = self.model_nn.predict(X)

        # Divergence indices: where XGB prediction != NN prediction
        divergent_mask = preds_xgb != preds_nn
        divergent_df = X[divergent_mask].copy()

        # Add labels to make it clear what each model predicted
        if not divergent_df.empty:
            divergent_df["pred_xgb"] = preds_xgb[divergent_mask]
            divergent_df["pred_nn"] = preds_nn[divergent_mask]

            # Predict probabilities for extra context
            prob_xgb = self.model_xgb.predict_proba(X[divergent_mask])[:, 1]
            prob_nn = self.model_nn.predict_proba(X[divergent_mask])[:, 1]
            divergent_df["prob_xgb"] = prob_xgb
            divergent_df["prob_nn"] = prob_nn

        return divergent_df.head(max_samples)

    def _get_model(self, model_type: str):
        if self.model_xgb is None or self.model_nn is None:
            self.load_models()

        if "xgb" in model_type.lower() or "tree" in model_type.lower():
            return self.model_xgb
        elif (
            "nn" in model_type.lower()
            or "mlp" in model_type.lower()
            or "neural" in model_type.lower()
        ):
            return self.model_nn
        else:
            raise ValueError(f"Unknown model architecture type requested: {model_type}")
