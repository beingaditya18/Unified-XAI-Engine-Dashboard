import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdultIncomePipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.encoders = {}
        self.categorical_cols = []
        self.numeric_cols = []
        self.target_col = self.config["data"]["target_column"]
        self.columns = self.config["data"]["columns"]
        self.encoder_path = self.config["models"]["xgb"]["encoder_path"]  # Shared encoder path

    def _load_config(self, config_path: str) -> dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def load_data(self) -> pd.DataFrame:
        """Loads dataset from local CSV or falls back to UCI Repository over HTTPS."""
        local_path = self.config["data"]["raw_path"]
        if os.path.exists(local_path):
            logger.info(f"Loading local dataset from {local_path}")
            df = pd.read_csv(local_path, names=self.columns, skipinitialspace=True)
        else:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            logger.warning(f"Local file {local_path} not found. Fetching from remote: {url}")
            df = pd.read_csv(url, names=self.columns, skipinitialspace=True)
            # Create data directory and save locally for future runs
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            df.to_csv(local_path, index=False, header=False)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values represented by '?' in UCI Adult dataset."""
        df_clean = df.copy()

        # Replace '?' with NaN
        df_clean = df_clean.replace("?", np.nan)

        # Simple imputation: fill numeric with median, categoricals with mode
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
            else:
                mode_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_val)

        return df_clean

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Fits label encoders to categorical columns and transforms the dataframe."""
        df_clean = self.clean_data(df)

        X = df_clean.drop(self.target_col, axis=1)
        y = df_clean[self.target_col]

        # Detect categorical and numerical columns
        self.categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        self.numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

        logger.info(f"Categorical features: {self.categorical_cols}")
        logger.info(f"Numerical features: {self.numeric_cols}")

        X_encoded = X.copy()
        for col in self.categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            self.encoders[col] = le

        # Fit target encoder
        target_le = LabelEncoder()
        y_encoded = pd.Series(target_le.fit_transform(y), name=self.target_col)
        self.encoders[self.target_col] = target_le

        # Save encoders for consistent inference downstream
        os.makedirs(os.path.dirname(self.encoder_path), exist_ok=True)
        joblib.dump(self.encoders, self.encoder_path)
        logger.info(f"Serialized {len(self.encoders)} encoders to {self.encoder_path}")

        return X_encoded, y_encoded

    def transform_instance(self, raw_instance: pd.DataFrame) -> pd.DataFrame:
        """Transforms a raw user instance using previously saved fitted encoders."""
        if not self.encoders:
            self.load_encoders()

        instance_encoded = raw_instance.copy()

        # Fill missing values
        for col in instance_encoded.columns:
            if col in self.categorical_cols:
                # Use encoder classes to check or fall back if value is novel
                le = self.encoders[col]
                val = instance_encoded[col].iloc[0]
                if val not in le.classes_:
                    logger.warning(
                        f"Unseen category '{val}' for column '{col}'. Imputing with default class."
                    )
                    instance_encoded[col] = le.classes_[0]
                instance_encoded[col] = le.transform(instance_encoded[col])
            else:
                instance_encoded[col] = pd.to_numeric(instance_encoded[col])

        return instance_encoded

    def inverse_transform_instance(self, encoded_instance: pd.DataFrame) -> pd.DataFrame:
        """Converts integer-encoded features back to their original string labels."""
        if not self.encoders:
            self.load_encoders()

        decoded = encoded_instance.copy()
        for col in decoded.columns:
            if col in self.encoders and col != self.target_col:
                le = self.encoders[col]
                decoded[col] = le.inverse_transform(decoded[col].astype(int))
        return decoded

    def load_encoders(self) -> None:
        """Loads fitted LabelEncoders from model artifact storage."""
        if os.path.exists(self.encoder_path):
            self.encoders = joblib.load(self.encoder_path)
            # Reconstruct list of categorical columns based on keys
            self.categorical_cols = [k for k in self.encoders.keys() if k != self.target_col]
            logger.info(f"Loaded encoders from {self.encoder_path}")
        else:
            raise FileNotFoundError(
                f"Encountered missing model encoders at: {self.encoder_path}. Run training first."
            )

    def run_pipeline(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Runs loading, cleaning, encoding, and returns train-test splits."""
        df = self.load_data()
        X_encoded, y_encoded = self.fit_transform(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded,
            y_encoded,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=y_encoded,
        )

        # Save split test set to disk if not exists
        test_path = self.config["data"]["test_path"]
        if not os.path.exists(test_path):
            test_df = X_test.copy()
            test_df[self.target_col] = y_test
            test_df.to_csv(test_path, index=False)
            logger.info(f"Saved split test set to {test_path}")

        return X_train, X_test, y_train, y_test
