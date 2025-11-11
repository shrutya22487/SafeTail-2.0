"""
DetectPredictor
---------------

Usage:
    from detect_predictor import DetectPredictor

    dp = DetectPredictor(model_path="/mnt/data/detect_regressor_model.pkl",
                         csv_path="/mnt/data/server1.csv")

    predicted_time = dp.predict_from_combination("sdp")   # combination string example
    print("Predicted detect processing time (s):", predicted_time)
"""

import pickle
from typing import Optional, Union
import pandas as pd
import numpy as np
import os
from pathlib import Path


def _first_element_if_listlike(value: Union[str, float, int, None]) -> str:
    """
    If value is a string that contains commas, return the first comma-separated element (stripped).
    For NaN/None return empty string.
    For non-string, convert to str and return.
    """
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(',')]
        return parts[0] if parts else ""
    return str(value)


class DetectPredictor:
    def __init__(self):
        """
        Loads the model and CSV.

        model_path: path to the pickle that contains {'model', 'scaler', 'feature_columns', 'model_name'}
        csv_path: path to CSV with at least a 'Combination' column and columns used by feature engineering.
        """
        base_dir = Path(__file__).resolve().parent
        model_path: Path = base_dir.parent.parent / "models" / "server1" / "detect_regressor_model.pkl"
        csv_path: Path = base_dir.parent.parent / "data" / "server1.csv"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # load pickle
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        # expected keys: model, scaler, feature_columns, model_name
        self.model = saved.get("model")
        self.scaler = saved.get("scaler", None)
        self.feature_columns = saved.get("feature_columns", None)
        self.model_name = saved.get("model_name", None)

        if self.model is None or self.feature_columns is None:
            raise ValueError("Pickle does not contain required keys ('model' and 'feature_columns').")

        # load csv
        self.df = pd.read_csv(csv_path)

        # normalize column names (strip)
        self.df.columns = [c.strip() for c in self.df.columns]

        # required columns (best effort)
        if "Combination" not in self.df.columns:
            raise ValueError("CSV must contain a 'Combination' column.")

    def _find_row_by_combination(self, combination: str) -> pd.Series:
        """
        Returns the first row (Series) that matches the combination string.
        Matching is case-insensitive and strips whitespace.
        """
        comb = str(combination).strip()
        # exact match (case-insensitive)
        mask = self.df["Combination"].astype(str).str.strip().str.lower() == comb.lower()
        if not mask.any():
            # try contains as fallback
            mask = self.df["Combination"].astype(str).str.strip().str.lower().str.contains(comb.lower())
            if not mask.any():
                raise ValueError(f"No row found matching Combination='{comb}'.")
        # return first match
        idx = self.df[mask].index[0]
        return self.df.loc[idx]

    def _build_features_from_row(self, row: pd.Series) -> pd.DataFrame:
        """
        Build the feature vector DataFrame with a single row, matching self.feature_columns.
        Extracts index 0 element for comma-separated columns where relevant.
        """
        # read required raw fields with 'first element' extraction
        combination = _first_element_if_listlike(row.get("Combination", ""))
        individual_times_str = _first_element_if_listlike(row.get("Individual Processing Times", ""))
        scripts_str = _first_element_if_listlike(row.get("Scripts Executed", ""))
        total_processing_time = row.get("Total Processing Time (sec)", np.nan)
        peak_ram = row.get("Peak RAM Usage (MB)", np.nan)
        peak_gpu = row.get("Peak GPU Usage (%)", np.nan)
        peak_gpu_mem = row.get("Peak GPU Memory (MB)", np.nan)

        # If scripts or individual times contain commas, we actually want the full lists to find detect position.
        # So use the full (un-sliced) string from original row:
        scripts_full = row.get("Scripts Executed", "")
        individual_times_full = row.get("Individual Processing Times", "")

        # parse lists (if they are strings)
        if isinstance(scripts_full, str) and scripts_full.strip() != "":
            scripts_list = [s.strip().lower() for s in scripts_full.split(',')]
        else:
            # fallback: infer scripts from combination letters (one letter per op)
            scripts_list = list(combination.strip().lower())

        if isinstance(individual_times_full, str) and individual_times_full.strip() != "":
            try:
                individual_times_list = [float(x.strip()) for x in individual_times_full.split(',')]
            except Exception:
                # if parsing fails, set empty list
                individual_times_list = []
        else:
            individual_times_list = []

        comb_lower = combination.lower()
        # counts
        num_speech = comb_lower.count('s')
        num_detect = comb_lower.count('d')
        num_predict = comb_lower.count('p')
        total_ops = len(combination)

        # find detect position in scripts_list (first occurrence)
        position = 0
        is_first = 0
        is_last = 0
        # default: use the first 'd' position in combination if scripts don't help
        pos_found = None
        # try to find in scripts list
        for i, s in enumerate(scripts_list):
            if s == 'detect' or s == 'd':
                pos_found = i
                break
        if pos_found is None:
            # fallback: find index of 'd' in combination characters
            try:
                pos_found = list(combination.lower()).index('d')
            except ValueError:
                # if not found, set 0
                pos_found = 0

        position = pos_found
        is_first = 1 if position == 0 else 0
        is_last = 1 if (position == (len(scripts_list) - 1)) else 0

        # Build a dict of features expected
        feature_values = {
            'num_speech': num_speech,
            'num_detect': num_detect,
            'num_predict': num_predict,
            'total_ops': total_ops,
            'position': position,
            'is_first': is_first,
            'is_last': is_last,
            'peak_ram': peak_ram,
            'peak_gpu': peak_gpu,
            'peak_gpu_memory': peak_gpu_mem,
            'total_processing_time': total_processing_time
        }

        # create dataframe in the order of feature_columns
        row_features = []
        for col in self.feature_columns:
            if col in feature_values:
                row_features.append(feature_values[col])
            else:
                # if a feature column is missing from our computed values, try to extract from the row directly
                val = row.get(col, np.nan)
                # if value looks like a comma-separated string, take its first element
                row_features.append(_try_parse_number(_first_element_if_listlike(val)))

        X = pd.DataFrame([row_features], columns=self.feature_columns)
        return X


def _try_parse_number(val):
    """Try to convert to float or int, otherwise return val as-is."""
    try:
        if pd.isna(val):
            return np.nan
        # if already numeric
        if isinstance(val, (int, float, np.integer, np.floating)):
            return val
        s = str(val).strip()
        # empty string -> nan
        if s == "":
            return np.nan
        # remove percent sign if present
        s = s.replace('%', '')
        return float(s)
    except Exception:
        return val


    # end helper


# Attach helper to class as static if needed
DetectPredictor._first_element_if_listlike = staticmethod(_first_element_if_listlike)


def _prepare_for_prediction(dp: DetectPredictor, X: pd.DataFrame) -> np.ndarray:
    """
    Apply scaler if present and return the array ready for model.predict
    """
    if dp.scaler is not None:
        # some models in training used scaling; scaler expects numeric ndarray
        X_numeric = X.astype(float)
        X_scaled = dp.scaler.transform(X_numeric)
        return X_scaled
    else:
        # assume model was trained on raw features
        # convert to numeric where possible
        try:
            return X.astype(float).values
        except Exception:
            # fallback to raw values
            return X.values


# Public API functions inside module
def predict_processing_time_for_combination(model_path: str,
                                            csv_path: str,
                                            combination: str) -> float:
    """
    Convenience function: loads model and csv from paths and returns predicted processing time for given combination.
    """
    dp = DetectPredictor()
    return dp.predict_from_combination(combination)  # type: ignore


# Bind method to the class after function definitions to avoid forward-reference
def _predict_from_combination(self, combination: str) -> float:
    """
    Main method: find row by combination, build features, predict and return float seconds.
    """
    row = self._find_row_by_combination(combination)
    X = self._build_features_from_row(row)
    X_ready = _prepare_for_prediction(self, X)

    # predict
    pred = None
    try:
        pred_arr = self.model.predict(X_ready)
        # model.predict may return array-like
        pred = float(np.asarray(pred_arr).ravel()[0])
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    return pred


# attach to class
DetectPredictor.predict_from_combination = _predict_from_combination

