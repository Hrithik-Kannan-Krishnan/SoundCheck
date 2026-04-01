import os
import sys
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ── split_data ───────────────────────────────────────────────────────────────

def test_split_data_sizes(engineered):
    from modeling import split_data
    X, y, _ = engineered
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert abs(len(X_test) / len(X) - 0.2) < 0.05


def test_split_data_custom_test_size(engineered):
    from modeling import split_data
    X, y, _ = engineered
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
    assert abs(len(X_test) / len(X) - 0.3) < 0.05


def test_split_data_reproducible(engineered):
    from modeling import split_data
    X, y, _ = engineered
    _, X_test1, _, _ = split_data(X, y, random_state=42)
    _, X_test2, _, _ = split_data(X, y, random_state=42)
    pd.testing.assert_frame_equal(X_test1, X_test2)


# ── evaluate_models ──────────────────────────────────────────────────────────

def test_evaluate_models_returns_dataframe(small_models, engineered):
    from modeling import evaluate_models
    X, y, _ = engineered
    X_test, y_test = X.iloc[48:], y.iloc[48:]
    result = evaluate_models(small_models, X_test, y_test)
    assert isinstance(result, pd.DataFrame)


def test_evaluate_models_has_correct_columns(small_models, engineered):
    from modeling import evaluate_models
    X, y, _ = engineered
    X_test, y_test = X.iloc[48:], y.iloc[48:]
    result = evaluate_models(small_models, X_test, y_test)
    for col in ['RMSE', 'MAE', 'R2']:
        assert col in result.columns


def test_evaluate_models_row_per_model(small_models, engineered):
    from modeling import evaluate_models
    X, y, _ = engineered
    X_test, y_test = X.iloc[48:], y.iloc[48:]
    result = evaluate_models(small_models, X_test, y_test)
    assert len(result) == len(small_models)


# ── save_models ───────────────────────────────────────────────────────────────

def test_save_models_creates_joblib_files(small_models, tmp_path):
    from modeling import save_models
    save_models(small_models, str(tmp_path))
    for key in small_models:
        assert os.path.isfile(os.path.join(tmp_path, f'{key}_model.joblib'))


def test_save_models_files_loadable(small_models, tmp_path):
    import joblib
    from modeling import save_models
    save_models(small_models, str(tmp_path))
    for key in small_models:
        loaded = joblib.load(os.path.join(tmp_path, f'{key}_model.joblib'))
        assert loaded is not None


# ── train_all_models ──────────────────────────────────────────────────────────

def test_train_all_models_returns_dict_with_4_keys(engineered):
    from modeling import train_all_models
    X, y, _ = engineered
    models = train_all_models(X.iloc[:48], y.iloc[:48])
    assert set(models.keys()) == {'linear', 'rf', 'xgb', 'lgbm'}


def test_train_all_models_all_models_can_predict(engineered):
    from modeling import train_all_models
    X, y, _ = engineered
    models = train_all_models(X.iloc[:48], y.iloc[:48])
    X_test = X.iloc[48:]
    for name, model in models.items():
        preds = model.predict(X_test)
        assert len(preds) == len(X_test), f"{name} prediction length mismatch"


# ── run_isolation_forest ──────────────────────────────────────────────────────

def test_run_isolation_forest_returns_two_series(engineered):
    from modeling import run_isolation_forest
    X, y, _ = engineered
    labels, scores = run_isolation_forest(X.iloc[:48])
    assert isinstance(labels, pd.Series)
    assert isinstance(scores, pd.Series)


def test_run_isolation_forest_labels_are_minus1_or_1(engineered):
    from modeling import run_isolation_forest
    X, y, _ = engineered
    labels, _ = run_isolation_forest(X.iloc[:48])
    assert set(labels.unique()).issubset({-1, 1})


def test_run_isolation_forest_length_matches_input(engineered):
    from modeling import run_isolation_forest
    X, y, _ = engineered
    labels, scores = run_isolation_forest(X.iloc[:48])
    assert len(labels) == 48
    assert len(scores) == 48
