import logging

import pandas as pd
import yaml
from lime import lime_tabular

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LimeExplainerEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.num_features = self.config["explainability"]["lime"]["num_features"]

    def _load_config(self, config_path: str) -> dict:
        with open(config_path) as f:
            return yaml.safe_load(f)

    def get_explainer(self, X_train: pd.DataFrame) -> lime_tabular.LimeTabularExplainer:
        """Initializes LIME Tabular Explainer using training set distribution."""
        logger.info("Initializing LimeTabularExplainer.")
        return lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=["<=50K", ">50K"],
            mode="classification",
            discretize_continuous=True,
        )

    def explain_instance(
        self, explainer: lime_tabular.LimeTabularExplainer, instance: pd.Series, predict_fn
    ) -> dict:
        """Generates local LIME explanation structured as a readable JSON/Dict mapping."""
        exp = explainer.explain_instance(
            data_row=instance.values, predict_fn=predict_fn, num_features=self.num_features
        )

        # Parse list of (feature_id/rule, weight)
        raw_list = exp.as_list()

        formatted_explanations = []
        for feature_rule, weight in raw_list:
            formatted_explanations.append({"rule": feature_rule, "weight": float(weight)})

        return {
            "intercept": float(exp.intercept[1]),
            "predictions": formatted_explanations,
            "raw_scores": [float(val) for val in exp.local_pred],
        }
