"""
detect_predictor.py

Small library to load a saved detect regressor (pickle) and predict on a chosen row
from a CSV that matches the benchmark schema. Assumes 0-based row indexing.

Usage example (from another file):
    from detect_predictor import DetectPredictor

    dp = DetectPredictor(verbose=True)
    result = dp.predict_row(row_index=0)
    print(result["prediction"])
"""

from __future__ import annotations
import pickle
import os
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

DEFAULT_EXPECTED_FEATURES = [
    'num_speech', 'num_detect', 'num_predict', 'total_ops',
    'position', 'is_first', 'is_last',
    'peak_ram', 'peak_gpu', 'peak_gpu_memory', 'total_processing_time'
]


class ModelLoadError(Exception):
    pass


class PredictionError(Exception):
    pass


class DetectPredictor:
    def __init__(self, verbose: bool = False):
        """
        Instantiate and load model pickle.

        Args:
            verbose: if True, prints extra information during load/predict
        """
        base_dir = Path(__file__).resolve().parent  # e.g. src/
        # Resolve model and csv paths relative to project layout:
        # project_root/
        #   models/server1/detect_regressor_model.pkl
        #   data/server1.csv
        self.model_path: Path = base_dir.parent.parent / "models" / "server1" / "detect_regressor_model.pkl"
        self.csv_path: Path = base_dir.parent.parent / "data" / "server1.csv"
        self.verbose = verbose

        self.raw_pickle = None
        self.model = None  # estimator or Pipeline
        self.scaler = None
        self.feature_columns: Optional[List[str]] = None
        self.model_name: Optional[str] = None

        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise ModelLoadError(f"Model pickle not found: {self.model_path}")

        with open(self.model_path, "rb") as f:
            data = pickle.load(f)

        self.raw_pickle = data

        # If the pickle is an sklearn Pipeline or estimator saved directly
        try:
            from sklearn.pipeline import Pipeline
            if isinstance(data, Pipeline):
                self.model = data
                self.model_name = type(data).__name__
                if self.verbose:
                    print(f"[DetectPredictor] Loaded sklearn Pipeline from {self.model_path}")
                return
        except Exception:
            # sklearn might not be installed; we'll fallback to other checks
            pass

        # If the pickle is a dict containing keys
        if isinstance(data, dict):
            # expected keys: 'model', optional 'scaler', optional 'feature_columns', optional 'model_name'
            self.model = data.get('model')
            self.scaler = data.get('scaler', None)
            self.feature_columns = data.get('feature_columns', None)
            self.model_name = data.get('model_name', None) or (type(self.model).__name__ if self.model is not None else None)
            if self.verbose:
                print(f"[DetectPredictor] Loaded dict pickle. Model: {self.model_name}")
            if self.model is None:
                raise ModelLoadError(f"Pickle at {self.model_path} is a dict but contains no 'model' key.")
            return

        # Otherwise assume data is the model object
        self.model = data
        self.model_name = type(self.model).__name__
        if self.verbose:
            print(f"[DetectPredictor] Loaded model object of type {self.model_name}")

    def infer_feature_columns_from_df(self, df: pd.DataFrame) -> List[str]:
        """Infer feature columns to use from a dataframe (preserve ordering of DEFAULT_EXPECTED_FEATURES)."""
        cols = [c for c in DEFAULT_EXPECTED_FEATURES if c in df.columns]
        if cols:
            return cols
        # fallback: choose numeric columns excluding obvious non-features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        blacklist = {'index', 'Unnamed: 0', 'timestamp', 'combo', 'command'}
        cols = [c for c in numeric_cols if c not in blacklist]
        return cols

    def _prepare_input(self, df: pd.DataFrame, row_index: int, feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Extract a single row and return (X_array, feature_columns_used).
        Uses 0-based indexing for row_index.
        """
        if row_index < 0 or row_index >= len(df):
            raise PredictionError(f"row_index {row_index} out of range [0, {len(df)-1}]")

        # determine feature columns order
        if feature_columns is not None:
            feature_cols = feature_columns
        elif self.feature_columns is not None:
            feature_cols = self.feature_columns
        else:
            feature_cols = self.infer_feature_columns_from_df(df)

        if not feature_cols:
            raise PredictionError("No feature columns available to build input vector.")

        # check missing columns, attempt fallback
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            # fallback to inference
            inferred = self.infer_feature_columns_from_df(df)
            if self.verbose:
                print(f"[DetectPredictor] Warning: missing feature columns in CSV: {missing}. Falling back to inferred: {inferred}")
            feature_cols = inferred
            if not feature_cols:
                raise PredictionError("After fallback, no valid feature columns found in CSV.")

        row_series = df.iloc[row_index]
        # attempt to coerce numeric; missing -> fill 0
        X_raw = row_series.reindex(feature_cols).astype(float).fillna(0.0).values.reshape(1, -1)
        return X_raw, feature_cols

    def predict_row(self, row_index: int, feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Predict detect processing time for a selected row in self.csv_path.

        Args:
            row_index: 0-based row index to predict on
            feature_columns: optional list of feature columns to use (overrides stored/inferred ones)

        Returns:
            dict {
                'prediction': float,
                'model_name': str,
                'feature_columns': List[str],
                'feature_values': Dict[str, float],
            }
        """
        if not self.csv_path.exists():
            raise PredictionError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        if self.verbose:
            print(f"[DetectPredictor] CSV loaded: {self.csv_path} ({len(df)} rows)")

        X_raw, used_feature_cols = self._prepare_input(df, row_index, feature_columns)

        # Convert to the input shape expected by the saved model
        # If the saved object is a Pipeline, assume it handles preprocessing itself
        try:
            from sklearn.pipeline import Pipeline
            is_pipeline = isinstance(self.model, Pipeline)
        except Exception:
            is_pipeline = False

        # Decide whether to apply scaler (if present) — use heuristics only if scaler exists and model is not a pipeline
        X_for_pred = X_raw
        if not is_pipeline:
            if self.scaler is not None:
                clsname = type(self.model).__name__.lower()
                if any(tok in clsname for tok in ('linear', 'ridge', 'lasso', 'elastic', 'sgd', 'orthogonal')):
                    try:
                        X_for_pred = self.scaler.transform(X_raw)
                        if self.verbose:
                            print("[DetectPredictor] Applied saved scaler to input features.")
                    except Exception as e:
                        if self.verbose:
                            print("[DetectPredictor] Failed to apply scaler; using raw features. Error:", e)
                        X_for_pred = X_raw
                else:
                    if self.verbose and self.scaler is not None:
                        print("[DetectPredictor] Scaler present but not applied (model looks tree-based).")
                    X_for_pred = X_raw
            else:
                X_for_pred = X_raw
        else:
            X_for_pred = X_raw

        # Predict
        try:
            pred = self.model.predict(X_for_pred)
        except Exception as e:
            # if pipeline was saved and expects a DataFrame with column names, try that
            if is_pipeline:
                try:
                    Xdf = pd.DataFrame(X_raw, columns=used_feature_cols)
                    pred = self.model.predict(Xdf)
                except Exception:
                    raise PredictionError(f"Model prediction failed: {e}")
            else:
                raise PredictionError(f"Model prediction failed: {e}")

        prediction_value = float(np.ravel(pred)[0])

        # build readable feature-value map
        feature_values = {col: float(val) for col, val in zip(used_feature_cols, X_raw.ravel().tolist())}

        return {
            "prediction": prediction_value,
            "model_name": self.model_name or type(self.model).__name__,
            "feature_columns": used_feature_cols,
            "feature_values": feature_values,
        }
