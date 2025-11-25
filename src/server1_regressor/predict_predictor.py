# predict_predictor.py
from pathlib import Path
import os
import pickle
import pandas as pd
import numpy as np


# reuse same helpers
def _first_element_if_listlike(value):
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(',')]
        return parts[0] if parts else ""
    return str(value)


def _try_parse_number(val):
    try:
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return val
        s = str(val).strip()
        if s == "":
            return np.nan
        s = s.replace('%', '')
        return float(s)
    except Exception:
        return val


class PredictPredictor:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        model_path: Path = base_dir.parent.parent / "models" / "server1" / "predict_regressor_model.pkl"
        csv_path: Path = base_dir.parent.parent / "data" / "server1.csv"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(model_path, "rb") as f:
            saved = pickle.load(f)

        self.model = saved.get("model")
        self.scaler = saved.get("scaler", None)
        self.feature_columns = saved.get("feature_columns")
        self.model_name = saved.get("model_name", None)

        if self.model is None or self.feature_columns is None:
            raise ValueError("Pickle must contain 'model' and 'feature_columns'.")

        self.df = pd.read_csv(csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]

    def _find_row_by_combination(self, combination: str) -> pd.Series:
        comb = str(combination).strip()
        mask = self.df["Combination"].astype(str).str.strip().str.lower() == comb.lower()
        if not mask.any():
            mask = self.df["Combination"].astype(str).str.strip().str.lower().str.contains(comb.lower())
            if not mask.any():
                raise ValueError(f"No row found matching Combination='{comb}'.")
        idx = self.df[mask].index[0]
        return self.df.loc[idx]

    def _build_features_from_row(self, row: pd.Series) -> pd.DataFrame:
        combination = _first_element_if_listlike(row.get("Combination", ""))
        individual_times_full = row.get("Individual Processing Times", "")
        scripts_full = row.get("Scripts Executed", "")

        if isinstance(scripts_full, str) and scripts_full.strip() != "":
            scripts_list = [s.strip().lower() for s in scripts_full.split(',')]
        else:
            scripts_list = list(combination.strip().lower())

        if isinstance(individual_times_full, str) and individual_times_full.strip() != "":
            try:
                individual_times_list = [float(x.strip()) for x in individual_times_full.split(',')]
            except Exception:
                individual_times_list = []
        else:
            individual_times_list = []

        comb_lower = combination.lower()
        num_speech = comb_lower.count('s')
        num_detect = comb_lower.count('d')
        num_predict = comb_lower.count('p')
        total_ops = len(combination)

        pos_found = None
        for i, s in enumerate(scripts_list):
            if s == 'predict' or s == 'p':
                pos_found = i
                break
        if pos_found is None:
            try:
                pos_found = list(combination.lower()).index('p')
            except ValueError:
                pos_found = 0

        position = pos_found
        is_first = 1 if position == 0 else 0
        is_last = 1 if (position == (len(scripts_list) - 1)) else 0

        feature_values = {
            'num_speech': num_speech,
            'num_detect': num_detect,
            'num_predict': num_predict,
            'total_ops': total_ops,
            'position': position,
            'is_first': is_first,
            'is_last': is_last,
            'peak_ram': row.get("Peak RAM Usage (MB)", np.nan),
            'peak_gpu': row.get("Peak GPU Usage (%)", np.nan),
            'peak_gpu_memory': row.get("Peak GPU Memory (MB)", np.nan),
            'total_processing_time': row.get("Total Processing Time (sec)", np.nan)
        }

        row_features = []
        for col in self.feature_columns:
            if col in feature_values:
                row_features.append(feature_values[col])
            else:
                val = row.get(col, np.nan)
                row_features.append(_try_parse_number(_first_element_if_listlike(val)))

        X = pd.DataFrame([row_features], columns=self.feature_columns)
        return X

    def predict_from_combination(self, combination: str) -> float:
        row = self._find_row_by_combination(combination)
        X = self._build_features_from_row(row)
        
        if self.scaler is not None:
            # Linear models need scaling - returns numpy array
            X_ready = self.scaler.transform(X.astype(float))
            pred_arr = self.model.predict(X_ready)
        else:
            # Tree-based models: keep as DataFrame with feature names
            X_numeric = X.astype(float)
            # Ensure it's still a DataFrame with proper columns
            if not isinstance(X_numeric, pd.DataFrame):
                X_numeric = pd.DataFrame(X_numeric, columns=self.feature_columns)
            pred_arr = self.model.predict(X_numeric)
        
        try:
            return float(np.asarray(pred_arr).ravel()[0])
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")