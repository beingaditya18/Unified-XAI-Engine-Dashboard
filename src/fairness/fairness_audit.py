import logging

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairnessAuditEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.disparity_threshold = self.config["explainability"]["fairness"]["disparity_threshold"]
        self.protected_attrs = self.config["explainability"]["fairness"]["protected_attributes"]

    def _load_config(self, config_path: str) -> dict:
        with open(config_path) as f:
            return yaml.safe_load(f)

    def calculate_fairness_metrics(
        self, X_test: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray
    ) -> dict[str, dict]:
        """Audits model predictions for demographic parity and equalized odds across multiple attributes."""
        results = {}

        for attr_name, attr_config in self.protected_attrs.items():
            col = attr_config["column"]
            priv_val = attr_config["privileged_value"]
            unpriv_val = attr_config["unprivileged_value"]

            if col not in X_test.columns:
                logger.warning(
                    f"Protected attribute column '{col}' not found in test dataset. Skipping."
                )
                continue

            # Masks for groups
            priv_mask = X_test[col] == priv_val
            unpriv_mask = X_test[col] == unpriv_val

            # Count sizes
            n_priv = int(priv_mask.sum())
            n_unpriv = int(unpriv_mask.sum())

            if n_priv == 0 or n_unpriv == 0:
                logger.warning(
                    f"Insufficient samples for protected attribute '{col}'. Privileged count: {n_priv}, Unprivileged count: {n_unpriv}."
                )
                continue

            # 1. Selection Rates (Positive class prediction rate)
            # Positive prediction is y_pred == 1 (>50K)
            priv_selection_rate = float(y_pred[priv_mask].mean())
            unpriv_selection_rate = float(y_pred[unpriv_mask].mean())

            # Disparate Impact Ratio / Demographic Parity Ratio (DPR)
            # Ideal is 1.0; standard rule is 80% (0.80) to 1.25.
            if priv_selection_rate > 0:
                disparate_impact_ratio = unpriv_selection_rate / priv_selection_rate
            else:
                disparate_impact_ratio = 1.0

            demographic_parity_diff = abs(priv_selection_rate - unpriv_selection_rate)

            # 2. Equal Opportunity & Equalized Odds (True Positive Rate and False Positive Rate)
            # TPR = TP / (TP + FN) = TP / P
            # FPR = FP / (FP + TN) = FP / N
            tpr_priv = self._calculate_rate(y_true[priv_mask], y_pred[priv_mask], rate_type="tpr")
            tpr_unpriv = self._calculate_rate(
                y_true[unpriv_mask], y_pred[unpriv_mask], rate_type="tpr"
            )

            fpr_priv = self._calculate_rate(y_true[priv_mask], y_pred[priv_mask], rate_type="fpr")
            fpr_unpriv = self._calculate_rate(
                y_true[unpriv_mask], y_pred[unpriv_mask], rate_type="fpr"
            )

            equal_opp_diff = abs(tpr_priv - tpr_unpriv)
            fpr_diff = abs(fpr_priv - fpr_unpriv)

            # Flag if metrics fail the threshold checks
            # Usually Disparate Impact Ratio should be between 0.8 and 1.25.
            # Demographic parity difference should be < threshold (e.g. 0.15)
            is_biased = (
                disparate_impact_ratio < 0.80
                or disparate_impact_ratio > 1.25
                or demographic_parity_diff > self.disparity_threshold
            )

            results[attr_name] = {
                "privileged_count": n_priv,
                "unprivileged_count": n_unpriv,
                "privileged_selection_rate": priv_selection_rate,
                "unprivileged_selection_rate": unpriv_selection_rate,
                "disparate_impact_ratio": disparate_impact_ratio,
                "demographic_parity_difference": demographic_parity_diff,
                "true_positive_rate_privileged": tpr_priv,
                "true_positive_rate_unprivileged": tpr_unpriv,
                "equal_opportunity_difference": equal_opp_diff,
                "false_positive_rate_difference": fpr_diff,
                "is_biased": bool(is_biased),
            }

        return results

    def _calculate_rate(self, y_true: pd.Series, y_pred: np.ndarray, rate_type: str) -> float:
        """Helper to compute True Positive Rate or False Positive Rate."""
        y_true_arr = y_true.to_numpy()

        if rate_type == "tpr":
            # True positives over actual positives
            pos_mask = y_true_arr == 1
            if pos_mask.sum() == 0:
                return 0.0
            return float((y_pred[pos_mask] == 1).mean())
        elif rate_type == "fpr":
            # False positives over actual negatives
            neg_mask = y_true_arr == 0
            if neg_mask.sum() == 0:
                return 0.0
            return float((y_pred[neg_mask] == 1).mean())
        else:
            return 0.0
