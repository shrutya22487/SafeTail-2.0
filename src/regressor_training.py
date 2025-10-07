#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# -------------------------
# Helper functions
# -------------------------

def smart_outlier_removal(df, target_col):
    """Smart outlier removal using IsolationForest"""
    from sklearn.ensemble import IsolationForest
    X_features = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    X_features = X_features.fillna(X_features.median())
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso_forest.fit_predict(X_features)
    high_value_mask = df[target_col] > df[target_col].quantile(0.9)
    keep_outliers = outliers == -1
    final_mask = (outliers == 1) | (keep_outliers & high_value_mask)
    return df[final_mask]


def create_balanced_sample(df, target_col, n_samples=100000):
    """Create balanced sample with stratification"""
    if len(df) <= n_samples:
        return df

    high_threshold = df[target_col].quantile(0.95)
    high_value_df = df[df[target_col] >= high_threshold]
    regular_df = df[df[target_col] < high_threshold].copy()

    n_high = len(high_value_df)
    n_regular = min(n_samples - n_high, len(regular_df))

    # Stratified sampling
    regular_df['bins'] = pd.qcut(regular_df[target_col], q=20, labels=False, duplicates='drop')

    regular_sample = regular_df.groupby('bins').apply(
        lambda x: x.sample(n=max(1, int(n_regular * len(x) / len(regular_df))), 
                          random_state=42, replace=False)
    ).reset_index(drop=True)

    final_sample = pd.concat([high_value_df, regular_sample], ignore_index=True)
    return final_sample.drop(columns=['bins'], errors='ignore')


def find_best_transformation(y):
    """Find the best transformation for target variable"""
    transformations = {
        'original': y,
        'log': np.log1p(y),
        'sqrt': np.sqrt(y)
    }

    try:
        transformations['boxcox'] = stats.boxcox(y + 1)[0]
    except Exception:
        pass

    try:
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        transformations['yeo-johnson'] = pt.fit_transform(y.values.reshape(-1, 1)).flatten()
    except Exception:
        pass

    # Choose transformation with lowest skewness
    best_transform = min(transformations, key=lambda k: abs(stats.skew(transformations[k])))
    return transformations[best_transform], best_transform


def inverse_transform(y_pred, y_test, method, aux=None):
    """Inverse transform predictions (basic)"""
    # aux may contain saved transformer or boxcox lambda in future
    if method == 'log':
        return np.expm1(y_pred), np.expm1(y_test)
    elif method == 'sqrt':
        return y_pred**2, y_test**2
    else:
        return y_pred, y_test


def save_model_and_pipeline(model, pipeline,
                            model_path="../models/regressor_model.pkl",
                            pipeline_path="../models/full_pipeline.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)


# -------------------------
# Ensemble definition
# -------------------------
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


# -------------------------
# Main training routine
# -------------------------
def main():
    # Load datasets
    df = pd.read_csv('../data/updated_file_Predict.csv')
    latent_df = pd.read_csv('../data/var250e32b.csv')
    print("dataset loaded")

    # Drop unnecessary column
    if latent_df.columns[0] in ['Unnamed: 0', ''] or latent_df.columns[0].isdigit():
        latent_df = latent_df.drop(columns=latent_df.columns[0])

    # Rename latent columns
    latent_columns = [f'latent_dim_{i}' for i in range(latent_df.shape[1])]
    latent_df.columns = latent_columns

    # Reset index and concatenate
    df = df.reset_index(drop=True)
    latent_df = latent_df.reset_index(drop=True)
    df = pd.concat([df, latent_df], axis=1)

    print("feature selection")
    # Feature selection
    base_columns_to_keep = [
        'Image Pixel', 'Processing Time', 'Combination', 'RAM Memory Usage (MB)', 
        'CPU Usage Per Core', 'GPU Usage (%)', 'GPU Memory Usage (MB)', 
        'Total RAM (GB)', 'Total CPU Cores', 'Total GPU Memory (GB)', 
        'CPU Clock Speed (MHz)', 'GPU Clock Speed (MHz)', 'Number of Cores Used', 
        'CPU Model', 'GPU Model'
    ]
    columns_to_keep = base_columns_to_keep + latent_columns
    df = df[columns_to_keep]

    # Parse image pixel
    df[['width', 'height']] = df['Image Pixel'].str.extract(r'(\d+)x(\d+)').astype(float)
    df.drop(columns=['Image Pixel'], inplace=True)

    # Combination features
    df['combination_complexity'] = df['Combination'].astype(str).str.len()
    df['combination_words'] = df['Combination'].astype(str).str.split().str.len()
    df.drop(columns=['Combination'], inplace=True)

    # CPU Usage Parsing
    def enhanced_cpu_parsing(core_str):
        try:
            values = eval(core_str)
            if isinstance(values, list) and len(values) > 0:
                arr = np.array(values)
                return {
                    'mean': np.mean(arr),
                    'std': np.std(arr),
                    'min': np.min(arr),
                    'max': np.max(arr),
                    'median': np.median(arr),
                    'q25': np.percentile(arr, 25),
                    'q75': np.percentile(arr, 75),
                    'range': np.ptp(arr),
                    'active_cores': np.sum(arr > 1),
                    'peak_cores': np.sum(arr > 50),
                    'idle_cores': np.sum(arr < 1),
                    'load_variance': np.var(arr),
                    'load_skew': stats.skew(arr) if len(arr) > 2 else 0,
                    'core_efficiency': np.mean(arr) / np.max(arr) if np.max(arr) > 0 else 0
                }
        except:
            pass
        return {k: np.nan for k in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 
                                    'range', 'active_cores', 'peak_cores', 'idle_cores', 
                                    'load_variance', 'load_skew', 'core_efficiency']}

    cpu_df = df['CPU Usage Per Core'].apply(enhanced_cpu_parsing)
    cpu_df = pd.DataFrame(cpu_df.tolist())
    df.drop(columns=['CPU Usage Per Core'], inplace=True)
    df = pd.concat([df, cpu_df], axis=1)


    print("feature engineering")
    # Feature Engineering
    df['pixel_area'] = df['width'] * df['height']
    df['aspect_ratio'] = df['width'] / df['height']
    df['resolution_class'] = pd.cut(df['pixel_area'], bins=10, labels=False)
    df['ram_utilization'] = df['RAM Memory Usage (MB)'] / (df['Total RAM (GB)'] * 1024)
    df['gpu_memory_utilization'] = df['GPU Memory Usage (MB)'] / (df['Total GPU Memory (GB)'] * 1024)
    df['core_utilization'] = df['Number of Cores Used'] / df['Total CPU Cores']
    df['processing_efficiency'] = df['pixel_area'] / (df['CPU Clock Speed (MHz)'] * df['Number of Cores Used'])
    df['memory_efficiency'] = df['pixel_area'] / df['RAM Memory Usage (MB)']
    df['gpu_efficiency'] = df['pixel_area'] / (df['GPU Memory Usage (MB)'] + 1)
    df['cpu_gpu_balance'] = df['mean'] / (df['GPU Usage (%)'] + 1)
    df['memory_compute_balance'] = df['ram_utilization'] / (df['core_utilization'] + 0.01)
    df['resource_pressure'] = df['ram_utilization'] * df['gpu_memory_utilization'] * df['core_utilization']
    df['complexity_score'] = (df['pixel_area'] * df['combination_complexity']) / (df['Total CPU Cores'] * df['Total GPU Memory (GB)'])
    df['processing_density'] = df['pixel_area'] / (df['Total RAM (GB)'] * 1024 + df['Total GPU Memory (GB)'] * 1024)
    df['log_pixel_area'] = np.log1p(df['pixel_area'])
    df['log_ram_usage'] = np.log1p(df['RAM Memory Usage (MB)'])
    df['log_gpu_memory'] = np.log1p(df['GPU Memory Usage (MB)'])
    df['sqrt_pixel_area'] = np.sqrt(df['pixel_area'])
    df['cbrt_pixel_area'] = np.cbrt(df['pixel_area'])
    df['pixel_cpu_interaction'] = df['pixel_area'] * df['CPU Clock Speed (MHz)']
    df['pixel_gpu_interaction'] = df['pixel_area'] * df['GPU Clock Speed (MHz)']
    df['ram_cpu_interaction'] = df['RAM Memory Usage (MB)'] * df['mean']
    df['gpu_cpu_interaction'] = df['GPU Usage (%)'] * df['mean']
    df['total_compute_power'] = df['CPU Clock Speed (MHz)'] * df['Total CPU Cores'] + df['GPU Clock Speed (MHz)']
    df['total_memory'] = df['Total RAM (GB)'] + df['Total GPU Memory (GB)']
    df['hardware_score'] = df['total_compute_power'] / df['total_memory']

    print("categorical encoding")
    # Encode categorical
    categorical_cols = ['CPU Model', 'GPU Model']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)

    print("latent repr feature engg")
    # Latent stats
    df_encoded['latent_magnitude'] = np.sqrt(np.sum(df_encoded[latent_columns].values**2, axis=1))
    df_encoded['latent_mean'] = df_encoded[latent_columns].mean(axis=1)
    df_encoded['latent_std'] = df_encoded[latent_columns].std(axis=1)
    df_encoded['latent_min'] = df_encoded[latent_columns].min(axis=1)
    df_encoded['latent_max'] = df_encoded[latent_columns].max(axis=1)
    df_encoded['latent_range'] = df_encoded['latent_max'] - df_encoded['latent_min']
    df_encoded['latent_skewness'] = df_encoded[latent_columns].apply(lambda x: stats.skew(x), axis=1)
    df_encoded['latent_kurtosis'] = df_encoded[latent_columns].apply(lambda x: stats.kurtosis(x), axis=1)
    df_encoded['latent_entropy'] = df_encoded[latent_columns].apply(lambda x: stats.entropy(np.abs(x) + 1e-10), axis=1)
    df_encoded['latent_energy'] = np.sum(df_encoded[latent_columns].values**2, axis=1)
    df_encoded['latent_l1_norm'] = np.sum(np.abs(df_encoded[latent_columns].values), axis=1)
    df_encoded['latent_max_abs'] = np.max(np.abs(df_encoded[latent_columns].values), axis=1)
    df_encoded['latent_pixel_interaction'] = df_encoded['latent_magnitude'] * df_encoded['pixel_area']
    df_encoded['latent_cpu_interaction'] = df_encoded['latent_mean'] * df_encoded['mean']
    df_encoded['latent_gpu_interaction'] = df_encoded['latent_mean'] * df_encoded['GPU Usage (%)']
    df_encoded['latent_memory_interaction'] = df_encoded['latent_magnitude'] * df_encoded['RAM Memory Usage (MB)']
    df_encoded['latent_processing_efficiency'] = df_encoded['latent_magnitude'] / (df_encoded['processing_efficiency'] + 1e-10)
    df_encoded['latent_memory_efficiency'] = df_encoded['latent_energy'] / (df_encoded['memory_efficiency'] + 1e-10)
    df_encoded['latent_hardware_score'] = df_encoded['latent_magnitude'] * df_encoded['hardware_score']
    df_encoded['latent_dim_ratio_01'] = df_encoded['latent_dim_0'] / (df_encoded['latent_dim_1'] + 1e-10)
    df_encoded['latent_dim_ratio_02'] = df_encoded['latent_dim_0'] / (df_encoded['latent_dim_2'] + 1e-10)
    df_encoded['latent_positive_dims'] = np.sum(df_encoded[latent_columns].values > 0, axis=1)
    df_encoded['latent_negative_dims'] = np.sum(df_encoded[latent_columns].values < 0, axis=1)

    print("PCA")
    # PCA
    pca = PCA(n_components=5)
    latent_pca = pca.fit_transform(df_encoded[latent_columns])
    for i in range(5):
        df_encoded[f'latent_pca_{i}'] = latent_pca[:, i]

    # KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df_encoded['latent_cluster'] = kmeans.fit_predict(df_encoded[latent_columns])

    # Drop NaNs and inf
    df_encoded = df_encoded.dropna(subset=['Processing Time'])
    df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan)


    print("outlier removal")
    df_clean = smart_outlier_removal(df_encoded, 'Processing Time')

    df_sampled = create_balanced_sample(df_clean, 'Processing Time')

    print("y transformation")
    y_original = df_sampled['Processing Time'].values
    y_transformed, transform_method = find_best_transformation(df_sampled['Processing Time'])

    X = df_sampled.drop(columns=['Processing Time']).select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    print("select K best")
    selector = SelectKBest(score_func=mutual_info_regression, k=min(250, X_imputed.shape[1]))
    X_selected = selector.fit_transform(X_imputed, y_transformed)

    print("ensemble models")
    # Train/test split
    y_bins = pd.qcut(y_transformed, q=10, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_transformed, test_size=0.5, stratify=y_bins, random_state=42)

    print("training")
    # Train model
    ensemble = AdvancedEnsemble()
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    print("training done")

    # Inverse transform
    y_pred_original, y_test_original = inverse_transform(y_pred, y_test, transform_method)

    # Metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)

    print("\n" + "="*50)
    print("OPTIMIZED RESULTS")
    print("="*50)
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_original, y_pred_original, alpha=0.6, color='blue', s=30)
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    plt.xlabel('Actual Processing Time')
    plt.ylabel('Predicted Processing Time')
    plt.title(f'Optimized: Actual vs Predicted\nR² = {r2:.3f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ======================
    # Save model & pipeline
    # ======================
    full_pipeline = {
        "pca": pca,
        "kmeans": kmeans,
        "imputer": imputer,
        "selector": selector,
        "target_transform": transform_method  # Needed for inverse transform later
    }

    save_model_and_pipeline(ensemble, full_pipeline)
    print("model and pipeline saved")


if __name__ == "__main__":
    main()
