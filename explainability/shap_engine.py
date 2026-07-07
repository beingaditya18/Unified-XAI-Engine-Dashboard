import logging

import numpy as np
import pandas as pd
import shap
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShapExplainerEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.explainer_cache = {}

    def _load_config(self, config_path: str) -> dict:
        with open(config_path) as f:
            return yaml.safe_load(f)

    def get_explainer(self, model, X_train: pd.DataFrame, model_type: str):
        """Initializes and caches the correct SHAP explainer based on model architecture."""
        cache_key = id(model)
        if cache_key in self.explainer_cache:
            return self.explainer_cache[cache_key]

        is_tree = "tree" in model_type.lower() or "xgb" in model_type.lower()

        if is_tree:
            logger.info("Initializing TreeExplainer for Tree-based model.")
            explainer = shap.TreeExplainer(model)
        else:
            bg_size = self.config["explainability"]["shap"]["background_samples"]
            logger.info(
                f"Initializing KernelExplainer with {bg_size} summarized background samples."
            )
            background = shap.sample(X_train, bg_size)
            explainer = shap.KernelExplainer(model.predict_proba, background)

        self.explainer_cache[cache_key] = explainer
        return explainer

    def calculate_shap_values(
        self, model, X_input: pd.DataFrame, X_train: pd.DataFrame, model_type: str
    ) -> np.ndarray:
        """Calculates SHAP values matrix for a given dataset."""
        explainer = self.get_explainer(model, X_train, model_type)
        is_tree = "tree" in model_type.lower() or "xgb" in model_type.lower()

        if is_tree:
            shap_values = explainer.shap_values(X_input)
            # Standarize multi-class/binary list outputs to class 1 SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            shap_values = explainer.shap_values(X_input)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        return shap_values

    def explain_instance(
        self, model, instance: pd.DataFrame, X_train: pd.DataFrame, model_type: str
    ) -> dict:
        """Generates structured local explanation values (SHAP values + base value)."""
        explainer = self.get_explainer(model, X_train, model_type)
        is_tree = "tree" in model_type.lower() or "xgb" in model_type.lower()

        shap_values = explainer.shap_values(instance)

        if is_tree:
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray) and len(base_value) > 1:
                base_value = base_value[1]
        else:
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            base_value = explainer.expected_value[1]

        feature_names = instance.columns.tolist()
        feature_values = instance.iloc[0].tolist()
        shap_vals_list = shap_values[0].tolist()

        explanations = [
            {"feature": name, "value": float(val), "shap_value": float(sv)}
            for name, val, sv in zip(feature_names, feature_values, shap_vals_list)
        ]

        # Sort features by absolute SHAP value impact
        explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {"base_value": float(base_value), "predictions": explanations}
