import pickle
import pandas as pd
import numpy as np
from pathlib import Path


class DetectPredictor:
    """
    Loads a saved Linear Regression detect-time model and predicts
    detect processing time for a given combination.
    """

    def __init__(self):
        base_dir = Path(__file__).resolve().parent

        self.model_path = (
            base_dir.parent.parent / "models" / "server1" / "detect_regressor_model.pkl"
        )
        self.csv_path = (
            base_dir.parent.parent / "data" / "server1.csv"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        # Load model bundle
        with open(self.model_path, "rb") as f:
            saved = pickle.load(f)

        self.model = saved["model"]
        self.scaler = saved["scaler"]
        self.feature_columns = saved["feature_columns"]
        self.model_name = saved.get("model_name", "Linear Regression")

        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]

        if "Combination" not in self.df.columns:
            raise ValueError("CSV must contain 'Combination' column")

    # ------------------------------------------------------------------ #

    def _find_row(self, combination: str) -> pd.Series:
        comb = combination.strip().lower()
        mask = self.df["Combination"].astype(str).str.lower() == comb

        if not mask.any():
            raise ValueError(f"No row found for combination='{combination}'")

        return self.df.loc[mask].iloc[0]

    # ------------------------------------------------------------------ #

    def _build_features(self, row: pd.Series) -> pd.DataFrame:
        combination = row["Combination"].lower()

        scripts = [s.strip().lower() for s in row["Scripts Executed"].split(",")]
        times = [float(t.strip()) for t in row["Individual Processing Times"].split(",")]

        # Detect position
        detect_idx = scripts.index("detect")

        features = {
            "num_speech": combination.count("s"),
            "num_detect": combination.count("d"),
            "num_predict": combination.count("p"),
            "total_ops": len(combination),
            "position": detect_idx,
            "is_first": 1 if detect_idx == 0 else 0,
            "is_last": 1 if detect_idx == len(scripts) - 1 else 0,
            "peak_ram": row["Peak RAM Usage (MB)"],
            "peak_gpu": row["Peak GPU Usage (%)"],
            "peak_gpu_memory": row["Peak GPU Memory (MB)"],
            "total_processing_time": row["Total Processing Time (sec)"],
        }

        X = pd.DataFrame([[features[c] for c in self.feature_columns]],
                         columns=self.feature_columns)
        return X

    # ------------------------------------------------------------------ #

    def predict_from_combination(self, combination: str) -> float:
        """
        Predict detect processing time (seconds) for a given combination.
        """
        row = self._find_row(combination)
        X = self._build_features(row)

        # Scale features (MANDATORY for Linear Regression)
        X_scaled = self.scaler.transform(X.astype(float))

        prediction = self.model.predict(X_scaled)
        return float(prediction[0])
