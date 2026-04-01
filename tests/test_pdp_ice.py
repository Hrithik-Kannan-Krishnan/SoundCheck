import os
import sys
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ── select_pdp_features ───────────────────────────────────────────────────────

def test_select_pdp_features_returns_6(engineered):
    from pdp_ice import select_pdp_features
    _, _, feature_names = engineered
    features = select_pdp_features(feature_names)
    assert len(features) == 6


def test_select_pdp_features_exact_list(engineered):
    from pdp_ice import select_pdp_features
    _, _, feature_names = engineered
    features = select_pdp_features(feature_names)
    expected = ['energy', 'danceability', 'loudness', 'acousticness', 'valence', 'tempo']
    assert features == expected


def test_select_pdp_features_all_in_feature_names(engineered):
    from pdp_ice import select_pdp_features
    _, _, feature_names = engineered
    features = select_pdp_features(feature_names)
    for f in features:
        assert f in feature_names


# ── pdp_ice_interpretation_table ─────────────────────────────────────────────

def test_interpretation_table_returns_dataframe(small_models, engineered):
    from pdp_ice import select_pdp_features, pdp_ice_interpretation_table
    X, y, feature_names = engineered
    X_test = X.iloc[48:]
    features = select_pdp_features(feature_names)
    rf_model = small_models['rf']
    result = pdp_ice_interpretation_table(rf_model, X_test, features, 'TestModel')
    assert isinstance(result, pd.DataFrame)


def test_interpretation_table_has_correct_columns(small_models, engineered):
    from pdp_ice import select_pdp_features, pdp_ice_interpretation_table
    X, y, feature_names = engineered
    X_test = X.iloc[48:]
    features = select_pdp_features(feature_names)
    rf_model = small_models['rf']
    result = pdp_ice_interpretation_table(rf_model, X_test, features, 'TestModel')
    for col in ['Feature', 'PDP Range', 'PDP Shape', 'ICE Heterogeneity (std)']:
        assert col in result.columns


def test_interpretation_table_one_row_per_feature(small_models, engineered):
    from pdp_ice import select_pdp_features, pdp_ice_interpretation_table
    X, y, feature_names = engineered
    X_test = X.iloc[48:]
    features = select_pdp_features(feature_names)
    rf_model = small_models['rf']
    result = pdp_ice_interpretation_table(rf_model, X_test, features, 'TestModel')
    assert len(result) == len(features)


def test_interpretation_table_shape_values_valid(small_models, engineered):
    from pdp_ice import select_pdp_features, pdp_ice_interpretation_table
    X, y, feature_names = engineered
    X_test = X.iloc[48:]
    features = select_pdp_features(feature_names)
    rf_model = small_models['rf']
    result = pdp_ice_interpretation_table(rf_model, X_test, features, 'TestModel')
    valid_shapes = {'monotonic increasing', 'monotonic decreasing', 'non-monotonic'}
    assert set(result['PDP Shape'].unique()).issubset(valid_shapes)


def test_interpretation_table_pdp_range_non_negative(small_models, engineered):
    from pdp_ice import select_pdp_features, pdp_ice_interpretation_table
    X, y, feature_names = engineered
    X_test = X.iloc[48:]
    features = select_pdp_features(feature_names)
    rf_model = small_models['rf']
    result = pdp_ice_interpretation_table(rf_model, X_test, features, 'TestModel')
    assert (result['PDP Range'] >= 0).all()


# ── plot_pdp_single ───────────────────────────────────────────────────────────

def test_plot_pdp_single_saves_file(small_models, engineered, tmp_path):
    from pdp_ice import plot_pdp_single
    X, y, feature_names = engineered
    X_test = X.iloc[48:]
    plot_pdp_single(small_models['rf'], X_test, 'energy', 'RF', str(tmp_path))
    assert os.path.isfile(os.path.join(tmp_path, 'pdp_RF_energy.png'))


# ── plot_pdp_grid ─────────────────────────────────────────────────────────────

def test_plot_pdp_grid_saves_file(small_models, engineered, tmp_path):
    from pdp_ice import select_pdp_features, plot_pdp_grid
    X, y, feature_names = engineered
    X_test = X.iloc[48:]
    features = select_pdp_features(feature_names)
    plot_pdp_grid(small_models['rf'], X_test, features, 'RF', str(tmp_path))
    assert os.path.isfile(os.path.join(tmp_path, 'pdp_grid_RF.png'))


# ── plot_ice ──────────────────────────────────────────────────────────────────

def test_plot_ice_standard_saves_file(small_models, engineered, tmp_path):
    from pdp_ice import plot_ice
    X, y, feature_names = engineered
    X_test = X.iloc[48:]
    plot_ice(small_models['rf'], X_test, 'energy', 'RF', str(tmp_path), n_samples=5)
    assert os.path.isfile(os.path.join(tmp_path, 'ice_RF_energy_standard.png'))


def test_plot_ice_centered_saves_file(small_models, engineered, tmp_path):
    from pdp_ice import plot_ice
    X, y, feature_names = engineered
    X_test = X.iloc[48:]
    plot_ice(small_models['rf'], X_test, 'energy', 'RF', str(tmp_path),
             n_samples=5, centered=True)
    assert os.path.isfile(os.path.join(tmp_path, 'ice_RF_energy_centered.png'))


# ── plot_ice_grid ─────────────────────────────────────────────────────────────

def test_plot_ice_grid_standard_saves_file(small_models, engineered, tmp_path):
    from pdp_ice import select_pdp_features, plot_ice_grid
    X, y, feature_names = engineered
    X_test = X.iloc[48:]
    features = select_pdp_features(feature_names)
    plot_ice_grid(small_models['rf'], X_test, features, 'RF', str(tmp_path), n_samples=5)
    assert os.path.isfile(os.path.join(tmp_path, 'ice_grid_RF_standard.png'))


def test_plot_ice_grid_centered_saves_file(small_models, engineered, tmp_path):
    from pdp_ice import select_pdp_features, plot_ice_grid
    X, y, feature_names = engineered
    X_test = X.iloc[48:]
    features = select_pdp_features(feature_names)
    plot_ice_grid(small_models['rf'], X_test, features, 'RF', str(tmp_path),
                  n_samples=5, centered=True)
    assert os.path.isfile(os.path.join(tmp_path, 'ice_grid_RF_centered.png'))


# ── run_full_pdp_ice_analysis ─────────────────────────────────────────────────

def test_run_full_pdp_ice_analysis_saves_expected_files(small_models, engineered, tmp_path):
    from pdp_ice import run_full_pdp_ice_analysis
    X, y, feature_names = engineered
    X_test = X.iloc[48:]
    # Only 'rf' is in small_models; xgb is skipped gracefully
    run_full_pdp_ice_analysis(small_models, X_test, feature_names, str(tmp_path))
    # PDP grid for rf
    assert os.path.isfile(os.path.join(tmp_path, 'pdp_grid_RandomForest.png'))
    # ICE grids for rf
    assert os.path.isfile(os.path.join(tmp_path, 'ice_grid_RandomForest_standard.png'))
    assert os.path.isfile(os.path.join(tmp_path, 'ice_grid_RandomForest_centered.png'))
