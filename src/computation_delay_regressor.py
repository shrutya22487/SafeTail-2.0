import os
import pickle
import ast
import numpy as np
import pandas as pd
from scipy import stats
from scipy import special
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
warnings.filterwarnings("ignore")

EPS = 1e-10


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

class AdvancedEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, max_features='sqrt', random_state=42),
            'et': ExtraTreesRegressor(n_estimators=300, max_depth=25, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        self.weights = None
        self.scalers = {}

    def fit(self, X, y):
        self.fitted_models = {}
        predictions = {}
        for name, model in self.models.items():
            if name in ['ridge', 'elastic']:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers[name] = scaler
                self.fitted_models[name] = model.fit(X_scaled, y)
                predictions[name] = self.fitted_models[name].predict(X_scaled)
            else:
                self.fitted_models[name] = model.fit(X, y)
                predictions[name] = self.fitted_models[name].predict(X)
        self.weights = np.array([0.3, 0.3, 0.25, 0.1, 0.05])
        return self

    def predict(self, X):
        predictions = {}
        for name, model in self.fitted_models.items():
            if name in ['ridge', 'elastic']:
                X_scaled = self.scalers[name].transform(X)
                predictions[name] = model.predict(X_scaled)
            else:
                predictions[name] = model.predict(X)
        pred_array = np.column_stack([predictions[name] for name in self.models])
        return np.average(pred_array, axis=1, weights=self.weights)
# --- Feature engineering helper functions (mirror training) ---
def parse_image_pixel(df):
    if 'Image Pixel' in df.columns:
        extracted = df['Image Pixel'].astype(str).str.extract(r'(\d+)x(\d+)')
        df['width'] = pd.to_numeric(extracted[0], errors='coerce')
        df['height'] = pd.to_numeric(extracted[1], errors='coerce')
        df.drop(columns=['Image Pixel'], inplace=True, errors='ignore')
    else:
        # ensure width/height exist (NaN) so later code won't break
        if 'width' not in df.columns:
            df['width'] = np.nan
        if 'height' not in df.columns:
            df['height'] = np.nan
    return df


def cpu_stats_from_str(s):
    """Safe parsing of CPU Usage Per Core strings (use ast.literal_eval, not eval)."""
    cols = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'range',
            'active_cores', 'peak_cores', 'idle_cores', 'load_variance', 'load_skew', 'core_efficiency']
    try:
        # handle NaN or empty
        if pd.isna(s):
            return {k: np.nan for k in cols}
        vals = ast.literal_eval(s) if isinstance(s, str) else s
        if isinstance(vals, (list, tuple, np.ndarray)) and len(vals) > 0:
            arr = np.array(vals, dtype=float)
            mean = np.mean(arr)
            mx = np.max(arr)
            load_skew = stats.skew(arr) if len(arr) > 2 else 0.0
            core_eff = mean / mx if mx > 0 else 0.0
            return {
                'mean': mean,
                'std': np.std(arr),
                'min': np.min(arr),
                'max': mx,
                'median': np.median(arr),
                'q25': np.percentile(arr, 25),
                'q75': np.percentile(arr, 75),
                'range': np.ptp(arr),
                'active_cores': np.sum(arr > 1),
                'peak_cores': np.sum(arr > 50),
                'idle_cores': np.sum(arr < 1),
                'load_variance': np.var(arr),
                'load_skew': load_skew,
                'core_efficiency': core_eff
            }
    except Exception:
        pass
    return {k: np.nan for k in cols}


def add_cpu_features(df):
    if 'CPU Usage Per Core' in df.columns:
        cpu_parsed = df['CPU Usage Per Core'].apply(cpu_stats_from_str)
        cpu_df = pd.DataFrame(cpu_parsed.tolist(), index=df.index)
        df = df.drop(columns=['CPU Usage Per Core'], errors='ignore')
        df = pd.concat([df, cpu_df], axis=1)
    else:
        # Ensure columns exist (NaN) if missing
        for k in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75',
                  'range', 'active_cores', 'peak_cores', 'idle_cores',
                  'load_variance', 'load_skew', 'core_efficiency']:
            if k not in df.columns:
                df[k] = np.nan
    return df


def add_basic_engineering(df):
    # safe numeric casts
    for col in ['width', 'height', 'RAM Memory Usage (MB)', 'Total RAM (GB)',
                'GPU Memory Usage (MB)', 'Total GPU Memory (GB)',
                'Number of Cores Used', 'Total CPU Cores',
                'CPU Clock Speed (MHz)', 'GPU Clock Speed (MHz)', 'mean']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # pixel area / aspect ratio
    df['pixel_area'] = df['width'] * df['height']
    # avoid divide by zero
    df['aspect_ratio'] = df['width'] / (df['height'] + EPS)

    # resolution class - fall back to NaN if pixel_area missing
    try:
        df['resolution_class'] = pd.cut(df['pixel_area'], bins=10, labels=False)
    except Exception:
        df['resolution_class'] = np.nan

    # RAM/GPU/core utilization (safe with EPS)
    df['ram_utilization'] = df['RAM Memory Usage (MB)'] / (df.get('Total RAM (GB)', 0) * 1024 + EPS)
    df['gpu_memory_utilization'] = df['GPU Memory Usage (MB)'] / (df.get('Total GPU Memory (GB)', 0) * 1024 + EPS)
    df['core_utilization'] = df['Number of Cores Used'] / (df.get('Total CPU Cores', 0) + EPS)

    # Efficiency features (guard denominators)
    df['processing_efficiency'] = df['pixel_area'] / ((df.get('CPU Clock Speed (MHz)', 0) * (df.get('Number of Cores Used', 0)) + EPS))
    df['memory_efficiency'] = df['pixel_area'] / (df['RAM Memory Usage (MB)'] + EPS)
    df['gpu_efficiency'] = df['pixel_area'] / (df['GPU Memory Usage (MB)'] + 1.0)

    # balances and interactions
    df['cpu_gpu_balance'] = df['mean'] / (df.get('GPU Usage (%)', 0) + 1.0)
    df['memory_compute_balance'] = df['ram_utilization'] / (df['core_utilization'] + 0.01)
    df['resource_pressure'] = df['ram_utilization'] * df['gpu_memory_utilization'] * df['core_utilization']
    df['complexity_score'] = (df['pixel_area'] * df.get('combination_complexity', 0)) / ((df.get('Total CPU Cores', 0) * df.get('Total GPU Memory (GB)', 0)) + EPS)
    df['processing_density'] = df['pixel_area'] / ((df.get('Total RAM (GB)', 0) * 1024) + (df.get('Total GPU Memory (GB)', 0) * 1024) + EPS)

    # logs, roots
    df['log_pixel_area'] = np.log1p(df['pixel_area'].replace(-np.inf, np.nan).fillna(0))
    df['log_ram_usage'] = np.log1p(df['RAM Memory Usage (MB)'].replace(-np.inf, np.nan).fillna(0))
    df['log_gpu_memory'] = np.log1p(df['GPU Memory Usage (MB)'].replace(-np.inf, np.nan).fillna(0))
    df['sqrt_pixel_area'] = np.sqrt(df['pixel_area'].clip(lower=0))
    df['cbrt_pixel_area'] = np.cbrt(df['pixel_area'].clip(lower=0))

    # interactions
    df['pixel_cpu_interaction'] = df['pixel_area'] * df.get('CPU Clock Speed (MHz)', 0)
    df['pixel_gpu_interaction'] = df['pixel_area'] * df.get('GPU Clock Speed (MHz)', 0)
    df['ram_cpu_interaction'] = df['RAM Memory Usage (MB)'] * df['mean']
    df['gpu_cpu_interaction'] = df.get('GPU Usage (%)', 0) * df['mean']
    df['total_compute_power'] = (df.get('CPU Clock Speed (MHz)', 0) * df.get('Total CPU Cores', 0)) + df.get('GPU Clock Speed (MHz)', 0)
    df['total_memory'] = df.get('Total RAM (GB)', 0) + df.get('Total GPU Memory (GB)', 0)
    df['hardware_score'] = df['total_compute_power'] / (df['total_memory'] + EPS)

    return df


def add_combination_features(df):
    if 'Combination' in df.columns:
        df['combination_complexity'] = df['Combination'].astype(str).str.len()
        df['combination_words'] = df['Combination'].astype(str).str.split().str.len()
        df.drop(columns=['Combination'], inplace=True, errors='ignore')
    else:
        df['combination_complexity'] = 0
        df['combination_words'] = 0
    return df


def add_latent_features(df, latent_prefix='latent_dim_'):
    latent_cols = [c for c in df.columns if c.startswith(latent_prefix)]
    if len(latent_cols) == 0:
        # nothing to do
        df['latent_magnitude'] = np.nan
        df['latent_mean'] = np.nan
        df['latent_std'] = np.nan
        df['latent_min'] = np.nan
        df['latent_max'] = np.nan
        df['latent_range'] = np.nan
        df['latent_skewness'] = np.nan
        df['latent_kurtosis'] = np.nan
        df['latent_entropy'] = np.nan
        df['latent_energy'] = np.nan
        df['latent_l1_norm'] = np.nan
        df['latent_max_abs'] = np.nan
        df['latent_positive_dims'] = np.nan
        df['latent_negative_dims'] = np.nan
        return df, latent_cols

    latent_vals = df[latent_cols].values.astype(float)
    df['latent_magnitude'] = np.sqrt(np.sum(latent_vals ** 2, axis=1))
    df['latent_mean'] = np.mean(latent_vals, axis=1)
    df['latent_std'] = np.std(latent_vals, axis=1)
    df['latent_min'] = np.min(latent_vals, axis=1)
    df['latent_max'] = np.max(latent_vals, axis=1)
    df['latent_range'] = df['latent_max'] - df['latent_min']
    # per-row stats using apply (slower but safe)
    df['latent_skewness'] = df[latent_cols].apply(lambda x: stats.skew(x), axis=1)
    df['latent_kurtosis'] = df[latent_cols].apply(lambda x: stats.kurtosis(x), axis=1)
    df['latent_entropy'] = df[latent_cols].apply(lambda x: stats.entropy(np.abs(x) + EPS), axis=1)
    df['latent_energy'] = np.sum(latent_vals ** 2, axis=1)
    df['latent_l1_norm'] = np.sum(np.abs(latent_vals), axis=1)
    df['latent_max_abs'] = np.max(np.abs(latent_vals), axis=1)
    # interactions
    df['latent_pixel_interaction'] = df['latent_magnitude'] * df.get('pixel_area', 0)
    df['latent_cpu_interaction'] = df['latent_mean'] * df.get('mean', 0)
    df['latent_gpu_interaction'] = df['latent_mean'] * df.get('GPU Usage (%)', 0)
    df['latent_memory_interaction'] = df['latent_magnitude'] * df.get('RAM Memory Usage (MB)', 0)
    df['latent_processing_efficiency'] = df['latent_magnitude'] / (df.get('processing_efficiency', EPS) + EPS)
    df['latent_memory_efficiency'] = df['latent_energy'] / (df.get('memory_efficiency', EPS) + EPS)
    df['latent_hardware_score'] = df['latent_magnitude'] * df.get('hardware_score', 0)
    # dim ratios if available
    if len(latent_cols) >= 3:
        df['latent_dim_ratio_01'] = df[latent_cols[0]] / (df[latent_cols[1]] + EPS)
        df['latent_dim_ratio_02'] = df[latent_cols[0]] / (df[latent_cols[2]] + EPS)
    df['latent_positive_dims'] = np.sum(latent_vals > 0, axis=1)
    df['latent_negative_dims'] = np.sum(latent_vals < 0, axis=1)
    return df, latent_cols


def encode_categoricals(df, categorical_cols=['CPU Model', 'GPU Model']):
    # One-hot encode; note we will align columns later to pipeline expectations
    existing = [c for c in categorical_cols if c in df.columns]
    if existing:
        df = pd.get_dummies(df, columns=existing, prefix=existing, dummy_na=False)
    return df


# --- Main preprocessing function for a DataFrame of raw rows ---
def preprocess_dataframe(df_raw):
    df = df_raw.copy().reset_index(drop=True)
    # Drop stray unnamed columns
    unnamed = [c for c in df.columns if str(c).startswith('Unnamed')]
    df = df.drop(columns=unnamed, errors='ignore')

    # Steps mirroring training
    df = parse_image_pixel(df)
    df = add_combination_features(df)
    df = add_cpu_features(df)
    df = add_basic_engineering(df)
    df = encode_categoricals(df)
    df, latent_cols = add_latent_features(df)

    # PCA & KMeans will be applied from pipeline objects (they expect latent columns to exist)
    return df, latent_cols


# --- Prediction pipeline ---
def predict_rows(row_num=None, verbose=True):
    
    model_path = "../models/regressor_model.pkl"
    pipeline_path = "../models/full_pipeline.pkl"
    input_csv = "../data/updated_file_Detect.csv"
    output_csv = "../data/predictions.csv"
    
    # load model & pipeline
    model = load_pickle(model_path)
    pipeline = load_pickle(pipeline_path)

    # load input CSV
    df_input = pd.read_csv(input_csv)

    # --- NEW PART ---
    if row_num is not None:
        if row_num < 0 or row_num >= len(df_input):
            raise IndexError(f"Row number {row_num} is out of range (0–{len(df_input)-1})")
        df_input = df_input.iloc[[row_num]].copy()
        if verbose:
            print(f"Predicting only for row {row_num}")
    # ----------------

    df_features, latent_cols = preprocess_dataframe(df_input)

    # Apply PCA & KMeans from pipeline (if available)
    if 'pca' in pipeline and pipeline['pca'] is not None and len(latent_cols) > 0:
        try:
            pca = pipeline['pca']
            latent_vals = df_features[latent_cols].values.astype(float)
            latent_pca = pca.transform(latent_vals)
            for i in range(latent_pca.shape[1]):
                df_features[f'latent_pca_{i}'] = latent_pca[:, i]
        except Exception as e:
            print("Warning: PCA transform failed:", e)

    if 'kmeans' in pipeline and pipeline['kmeans'] is not None and len(latent_cols) > 0:
        try:
            kmeans = pipeline['kmeans']
            df_features['latent_cluster'] = kmeans.predict(df_features[latent_cols].values.astype(float))
        except Exception as e:
            print("Warning: KMeans predict failed:", e)

    imputer = pipeline.get('imputer', None)
    selector = pipeline.get('selector', None)
    if imputer is not None and hasattr(imputer, 'feature_names_in_'):
        expected_cols = list(imputer.feature_names_in_)
    elif selector is not None and hasattr(selector, 'feature_names_in_'):
        expected_cols = list(selector.feature_names_in_)
    else:
        expected_cols = list(df_features.select_dtypes(include=[np.number]).columns)

    missing = [c for c in expected_cols if c not in df_features.columns]
    for c in missing:
        df_features[c] = 0.0
    X_num = df_features[expected_cols].copy()

    if imputer is not None:
        try:
            X_imputed = pd.DataFrame(imputer.transform(X_num), columns=expected_cols, index=X_num.index)
        except Exception as e:
            print("Warning: imputer.transform failed; doing columnwise median imputation. Error:", e)
            X_imputed = X_num.fillna(X_num.median())
    else:
        X_imputed = X_num.fillna(X_num.median())

    if selector is not None:
        try:
            X_selected = selector.transform(X_imputed)
        except Exception as e:
            try:
                support = selector.get_support(indices=True)
                sel_cols = [X_imputed.columns[i] for i in support]
                X_selected = X_imputed[sel_cols].values
                print("Warning: selector.transform failed; used manual column selection.")
            except Exception:
                raise RuntimeError("Feature selector transform failed and fallback couldn't recover.") from e
    else:
        X_selected = X_imputed.values

    y_pred_transformed = model.predict(X_selected)

    transform_method = pipeline.get('target_transform', None)
    y_pred_original = None
    if transform_method == 'log':
        y_pred_original = np.expm1(y_pred_transformed)
    elif transform_method == 'sqrt':
        y_pred_original = np.power(y_pred_transformed, 2.0)
    elif transform_method == 'boxcox':
        boxcox_lambda = pipeline.get('boxcox_lambda', None)
        if boxcox_lambda is not None:
            y_pred_original = special.inv_boxcox(y_pred_transformed, boxcox_lambda)
        else:
            print("Warning: boxcox lambda not saved in pipeline; returning transformed predictions.")
            y_pred_original = y_pred_transformed
    elif transform_method in ['yeo-johnson', 'yeo_johnson']:
        if 'power_transformer' in pipeline and pipeline['power_transformer'] is not None:
            pt = pipeline['power_transformer']
            y_pred_original = pt.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()
        else:
            print("Warning: Yeo-Johnson transformer not found; returning transformed predictions.")
            y_pred_original = y_pred_transformed
    else:
        y_pred_original = y_pred_transformed

    # Prepare output
    out_df = df_input.copy().reset_index(drop=True)
    out_df['predicted_processing_time'] = y_pred_original

    if row_num is not None:
        pred = float(out_df.loc[0, 'predicted_processing_time'])
        print(f"\nRow {row_num} -> Predicted Processing Time: {pred}")
        return pred

    # otherwise, print all and save CSV
    for idx, row in out_df.iterrows():
        print(f"Row {idx} -> Predicted Processing Time: {row['predicted_processing_time']}")

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to: {output_csv}")
    return out_df

