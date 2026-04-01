"""
modeling.py — Phase 2: Model training, evaluation, anomaly detection, persistence.

Functions: split_data, train_all_models, evaluate_models,
           run_isolation_forest, save_models
"""
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from utils import timer, print_section


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Standard train/test split for regression (no stratification).

    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Fraction for test set (default 0.2)
        random_state: Seed for reproducibility (default 42)

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


def evaluate_models(models_dict: dict, X_test, y_test) -> pd.DataFrame:
    """
    Compute RMSE, MAE, and R² for each model. Print comparison table.
    Plot bar chart of RMSE across models (steelblue).

    Args:
        models_dict: Dict mapping model key (str) -> fitted model
        X_test: Test feature DataFrame
        y_test: Test target Series

    Returns:
        results_df: DataFrame indexed by model name with columns RMSE, MAE, R2
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rows = []
    for name, model in models_dict.items():
        preds = model.predict(X_test)
        rows.append({
            'Model': name,
            'RMSE': round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
            'MAE': round(float(mean_absolute_error(y_test, preds)), 3),
            'R2': round(float(r2_score(y_test, preds)), 3),
        })

    results_df = pd.DataFrame(rows).set_index('Model')
    print("\nModel Evaluation Results:")
    print(results_df.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    results_df['RMSE'].plot(kind='bar', color='steelblue', ax=ax)
    ax.set_title('Model Comparison: RMSE (lower is better)')
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Model')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

    return results_df


def save_models(models_dict: dict, output_dir: str) -> None:
    """
    Save each fitted model to output_dir/<key>_model.joblib using joblib.

    Args:
        models_dict: Dict mapping model key (str) -> fitted model
        output_dir: Target directory (created if missing)
    """
    os.makedirs(output_dir, exist_ok=True)
    for key, model in models_dict.items():
        path = os.path.join(output_dir, f'{key}_model.joblib')
        joblib.dump(model, path)
        print(f"Saved {key} → {path}")


@timer
def train_all_models(X_train, y_train) -> dict:
    """
    Train LinearRegression, RandomForest, XGBoost, and LightGBM. Prints per-model time.

    Args:
        X_train: Training feature DataFrame
        y_train: Training target Series

    Returns:
        Dict mapping 'linear', 'rf', 'xgb', 'lgbm' -> fitted model objects
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    print_section("Model Training")
    configs = [
        ('linear', LinearRegression()),
        ('rf',     RandomForestRegressor(n_estimators=100, max_depth=15,
                                         n_jobs=-1, random_state=42)),
        ('xgb',    XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                n_jobs=-1, random_state=42, verbosity=0)),
        ('lgbm',   LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  n_jobs=-1, random_state=42, verbose=-1)),
    ]

    models = {}
    for name, model in configs:
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"  {name:8s} trained in {elapsed:.2f}s")
        models[name] = model

    return models


def run_isolation_forest(X_train, contamination: float = 0.05):
    """
    Fit IsolationForest on training data to detect anomalies.
    Labels: -1 = anomaly, 1 = normal.

    Args:
        X_train: Training feature DataFrame
        contamination: Expected fraction of outliers (default 0.05)

    Returns:
        (anomaly_labels, anomaly_scores) — both as pandas Series indexed like X_train
    """
    from sklearn.ensemble import IsolationForest

    print_section("Anomaly Detection — Isolation Forest")
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    labels = iso.fit_predict(X_train)
    scores = iso.score_samples(X_train)

    labels_s = pd.Series(labels, index=X_train.index, name='anomaly_label')
    scores_s = pd.Series(scores, index=X_train.index, name='anomaly_score')

    n_anomalies = int((labels == -1).sum())
    pct = n_anomalies / len(X_train) * 100
    print(f"Anomalies detected: {n_anomalies} ({pct:.1f}% of training data)")

    return labels_s, scores_s
