import pickle
import pandas as pd
from pathlib import Path


class DetectPredictor:
    """
    Predicts Detect processing time using a Random Forest model.
    """

    def __init__(self):
        base_dir = Path(__file__).resolve().parent

        self.model_path = base_dir.parent.parent / "models" / "server1" / "detect_regressor_model.pkl"
        self.csv_path = base_dir.parent.parent / "dataset" / "server1.csv"

        with open(self.model_path, "rb") as f:
            saved = pickle.load(f)

        self.model = saved["model"]
        self.feature_columns = saved["feature_columns"]

        self.df = pd.read_csv(self.csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]

    def _find_row(self, combination: str) -> pd.Series:
        mask = self.df["Combination"].str.lower() == combination.lower()
        if not mask.any():
            raise ValueError(f"No row found for combination '{combination}'")
        return self.df.loc[mask].iloc[0]

    def _build_features(self, row: pd.Series) -> pd.DataFrame:
        combination = row["Combination"].lower()
        scripts = [s.strip().lower() for s in row["Scripts Executed"].split(",")]
        idx = scripts.index("detect")

        features = {
            "num_tasks": len(scripts),
            "position": idx + 1,
            "total_combination_length": len(combination),
            "has_speech": int("s" in combination),
            "has_predict": int("p" in combination),
            "is_first": int(idx == 0),
            "is_last": int(idx == len(scripts) - 1),
            "num_speech_tasks": combination.count("s"),
            "num_detect_tasks": combination.count("d"),
            "num_predict_tasks": combination.count("p"),
            "peak_ram": row["Peak RAM Usage (MB)"],
            "peak_cpu": row["Peak CPU Usage (%)"],
            "avg_cpu_clock": row["Average CPU Clock (MHz)"],
            "num_files": 500,
        }

        return pd.DataFrame([[features[c] for c in self.feature_columns]],
                            columns=self.feature_columns)

    def predict_from_combination(self, combination: str) -> float:
        row = self._find_row(combination)
        X = self._build_features(row)
        return float(self.model.predict(X)[0])
