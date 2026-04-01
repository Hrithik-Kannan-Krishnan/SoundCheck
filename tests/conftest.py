import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_df():
    """60-row synthetic DataFrame mirroring the Spotify dataset structure."""
    np.random.seed(42)
    n = 60
    genres = ['pop', 'rock', 'jazz', 'classical', 'hip-hop']
    return pd.DataFrame({
        'track_id': [f'id_{i}' for i in range(n)],
        'artists': [f'artist_{i % 10}' for i in range(n)],
        'album_name': [f'album_{i % 10}' for i in range(n)],
        'track_name': [f'track_{i}' for i in range(n)],
        'popularity': np.random.randint(0, 100, n).astype(float),
        'duration_ms': np.random.randint(120000, 400000, n),
        'explicit': np.random.choice([True, False], n),
        'danceability': np.random.uniform(0, 1, n),
        'energy': np.random.uniform(0, 1, n),
        'key': np.random.randint(0, 11, n),
        'loudness': np.random.uniform(-20, 0, n),
        'mode': np.random.randint(0, 2, n),
        'speechiness': np.random.uniform(0, 0.5, n),
        'acousticness': np.random.uniform(0, 1, n),
        'instrumentalness': np.random.uniform(0, 1, n),
        'liveness': np.random.uniform(0, 1, n),
        'valence': np.random.uniform(0, 1, n),
        'tempo': np.random.uniform(60, 200, n),
        'time_signature': np.random.randint(3, 5, n),
        'track_genre': np.random.choice(genres, n),
    })


@pytest.fixture
def engineered(sample_df):
    """Returns (X, y, feature_names) from engineer_features applied to sample_df."""
    from preprocessing import engineer_features
    return engineer_features(sample_df)


@pytest.fixture
def small_models(engineered):
    """Returns a dict of tiny fitted models for fast testing."""
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    X, y, _ = engineered
    X_tr, y_tr = X.iloc[:48], y.iloc[:48]
    return {
        'linear': LinearRegression().fit(X_tr, y_tr),
        'rf': RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42).fit(X_tr, y_tr),
    }
