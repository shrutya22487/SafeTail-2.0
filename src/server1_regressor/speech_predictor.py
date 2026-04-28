import pickle
import pandas as pd
from pathlib import Path


class SpeechPredictor:
    """
    Predicts Speech processing time using a Linear Regression model.
    """

    def __init__(self):
        base_dir = Path(__file__).resolve().parent

        self.model_path = base_dir.parent.parent / "models" / "server1" / "speech_regressor_model.pkl"
        self.csv_path = base_dir.parent.parent / "dataset" / "server1.csv"

        with open(self.model_path, "rb") as f:
            saved = pickle.load(f)

        self.model = saved["model"]
        self.scaler = saved["scaler"]
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
        speech_idx = scripts.index("speech")

        features = {
            "num_speech": combination.count("s"),
            "num_detect": combination.count("d"),
            "num_predict": combination.count("p"),
            "total_ops": len(combination),
            "position": speech_idx,
            "is_first": int(speech_idx == 0),
            "is_last": int(speech_idx == len(scripts) - 1),
            "peak_ram": row["Peak RAM Usage (MB)"],
            "peak_gpu": row["Peak GPU Usage (%)"],
            "peak_gpu_memory": row["Peak GPU Memory (MB)"],
            "total_processing_time": row["Total Processing Time (sec)"],
        }

        return pd.DataFrame([[features[c] for c in self.feature_columns]],
                            columns=self.feature_columns)

    def predict_from_combination(self, combination: str) -> float:
        row = self._find_row(combination)
        X = self._build_features(row)

        X_scaled = self.scaler.transform(X.astype(float))
        return float(self.model.predict(X_scaled)[0])
