import os
import sys
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ── load_and_clean ──────────────────────────────────────────────────────────

def test_load_and_clean_drops_duplicate_track_ids(tmp_path):
    from preprocessing import load_and_clean
    csv = tmp_path / "test.csv"
    csv.write_text(
        ",track_id,artists,album_name,track_name,popularity,duration_ms,explicit,"
        "danceability,energy,key,loudness,mode,speechiness,acousticness,"
        "instrumentalness,liveness,valence,tempo,time_signature,track_genre\n"
        "0,AAA,A,Al,T1,50,200000,False,0.5,0.5,1,-5.0,1,0.05,0.1,0.0,0.1,0.5,120,4,pop\n"
        "1,AAA,A,Al,T2,60,200000,False,0.6,0.6,2,-6.0,0,0.06,0.2,0.0,0.2,0.6,130,4,pop\n"
        "2,BBB,B,Bl,T3,70,200000,True,0.7,0.7,3,-7.0,1,0.07,0.3,0.0,0.3,0.7,140,4,rock\n"
    )
    df = load_and_clean(str(csv))
    assert df['track_id'].nunique() == len(df)
    assert len(df) == 2


def test_load_and_clean_drops_null_popularity(tmp_path):
    from preprocessing import load_and_clean
    csv = tmp_path / "test.csv"
    csv.write_text(
        ",track_id,artists,album_name,track_name,popularity,duration_ms,explicit,"
        "danceability,energy,key,loudness,mode,speechiness,acousticness,"
        "instrumentalness,liveness,valence,tempo,time_signature,track_genre\n"
        "0,AAA,A,Al,T1,,200000,False,0.5,0.5,1,-5.0,1,0.05,0.1,0.0,0.1,0.5,120,4,pop\n"
        "1,BBB,B,Bl,T2,60,200000,False,0.6,0.6,2,-6.0,0,0.06,0.2,0.0,0.2,0.6,130,4,rock\n"
    )
    df = load_and_clean(str(csv))
    assert df['popularity'].isnull().sum() == 0
    assert len(df) == 1


def test_load_and_clean_converts_explicit_to_int(tmp_path):
    from preprocessing import load_and_clean
    csv = tmp_path / "test.csv"
    csv.write_text(
        ",track_id,artists,album_name,track_name,popularity,duration_ms,explicit,"
        "danceability,energy,key,loudness,mode,speechiness,acousticness,"
        "instrumentalness,liveness,valence,tempo,time_signature,track_genre\n"
        "0,AAA,A,Al,T1,50,200000,False,0.5,0.5,1,-5.0,1,0.05,0.1,0.0,0.1,0.5,120,4,pop\n"
        "1,BBB,B,Bl,T2,60,200000,True,0.6,0.6,2,-6.0,0,0.06,0.2,0.0,0.2,0.6,130,4,rock\n"
    )
    df = load_and_clean(str(csv))
    assert df['explicit'].dtype in [int, 'int64', 'int32']
    assert set(df['explicit'].unique()).issubset({0, 1})


def test_load_and_clean_returns_dataframe(tmp_path):
    from preprocessing import load_and_clean
    csv = tmp_path / "test.csv"
    csv.write_text(
        ",track_id,artists,album_name,track_name,popularity,duration_ms,explicit,"
        "danceability,energy,key,loudness,mode,speechiness,acousticness,"
        "instrumentalness,liveness,valence,tempo,time_signature,track_genre\n"
        "0,AAA,A,Al,T1,50,200000,False,0.5,0.5,1,-5.0,1,0.05,0.1,0.0,0.1,0.5,120,4,pop\n"
    )
    df = load_and_clean(str(csv))
    assert isinstance(df, pd.DataFrame)


# ── engineer_features ───────────────────────────────────────────────────────

def test_engineer_features_drops_identifier_columns(sample_df):
    from preprocessing import engineer_features
    X, y, feature_names = engineer_features(sample_df)
    for col in ['track_id', 'artists', 'album_name', 'track_name']:
        assert col not in X.columns


def test_engineer_features_drops_popularity_from_X(sample_df):
    from preprocessing import engineer_features
    X, y, feature_names = engineer_features(sample_df)
    assert 'popularity' not in X.columns


def test_engineer_features_y_is_popularity(sample_df):
    from preprocessing import engineer_features
    X, y, feature_names = engineer_features(sample_df)
    assert y.name == 'popularity'
    assert len(y) == len(sample_df)


def test_engineer_features_creates_genre_dummies(sample_df):
    from preprocessing import engineer_features
    X, y, feature_names = engineer_features(sample_df)
    genre_cols = [c for c in X.columns if c.startswith('track_genre_')]
    assert len(genre_cols) > 0


def test_engineer_features_no_track_genre_column(sample_df):
    from preprocessing import engineer_features
    X, y, feature_names = engineer_features(sample_df)
    assert 'track_genre' not in X.columns


def test_engineer_features_feature_names_matches_X_columns(sample_df):
    from preprocessing import engineer_features
    X, y, feature_names = engineer_features(sample_df)
    assert feature_names == X.columns.tolist()


def test_engineer_features_row_count_preserved(sample_df):
    from preprocessing import engineer_features
    X, y, feature_names = engineer_features(sample_df)
    assert len(X) == len(sample_df)


# ── distribution_analysis ────────────────────────────────────────────────────

def test_distribution_analysis_saves_histograms(sample_df, tmp_path):
    from preprocessing import distribution_analysis
    distribution_analysis(sample_df, str(tmp_path))
    assert os.path.isfile(os.path.join(tmp_path, 'histograms.png'))


def test_distribution_analysis_saves_boxplots(sample_df, tmp_path):
    from preprocessing import distribution_analysis
    distribution_analysis(sample_df, str(tmp_path))
    assert os.path.isfile(os.path.join(tmp_path, 'boxplots.png'))


def test_distribution_analysis_returns_none(sample_df, tmp_path):
    from preprocessing import distribution_analysis
    result = distribution_analysis(sample_df, str(tmp_path))
    assert result is None


# ── correlation_with_popularity ──────────────────────────────────────────────

def test_correlation_saves_bar_chart(sample_df, tmp_path):
    from preprocessing import correlation_with_popularity
    correlation_with_popularity(sample_df, str(tmp_path))
    assert os.path.isfile(os.path.join(tmp_path, 'correlation_bar.png'))


def test_correlation_saves_heatmap(sample_df, tmp_path):
    from preprocessing import correlation_with_popularity
    correlation_with_popularity(sample_df, str(tmp_path))
    assert os.path.isfile(os.path.join(tmp_path, 'correlation_heatmap.png'))


# ── bias_check ───────────────────────────────────────────────────────────────

def test_bias_check_returns_dict(sample_df, tmp_path):
    from preprocessing import bias_check
    result = bias_check(sample_df, str(tmp_path))
    assert isinstance(result, dict)
    assert 'genre_imbalance_ratio' in result
    assert 'dominant_artists' in result


def test_bias_check_imbalance_ratio_is_positive(sample_df, tmp_path):
    from preprocessing import bias_check
    result = bias_check(sample_df, str(tmp_path))
    assert result['genre_imbalance_ratio'] > 0


def test_bias_check_saves_genre_figure(sample_df, tmp_path):
    from preprocessing import bias_check
    bias_check(sample_df, str(tmp_path))
    assert os.path.isfile(os.path.join(tmp_path, 'genre_distribution.png'))


def test_bias_check_saves_artist_figure(sample_df, tmp_path):
    from preprocessing import bias_check
    bias_check(sample_df, str(tmp_path))
    assert os.path.isfile(os.path.join(tmp_path, 'artist_distribution.png'))
