"""
preprocessing.py — Phase 1: Data loading, EDA, and feature engineering.

Functions: load_and_clean, distribution_analysis, correlation_with_popularity,
           bias_check, engineer_features
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import save_figure, print_section


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load Spotify CSV, deduplicate on track_id, drop null popularity rows,
    convert explicit column to int (0/1).

    Args:
        filepath: Path to dataset.csv (has unnamed index column)

    Returns:
        Cleaned pandas DataFrame
    """
    print_section("Data Loading & Cleaning")
    df = pd.read_csv(filepath, index_col=0)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    before = len(df)
    df = df.drop_duplicates(subset='track_id')
    print(f"Dropped {before - len(df)} duplicate track_ids")

    before = len(df)
    df = df.dropna(subset=['popularity'])
    print(f"Dropped {before - len(df)} null popularity rows")

    df['explicit'] = (
        df['explicit'].astype(str).str.strip().map({'True': 1, 'False': 0}).fillna(0).astype(int)
    )
    print(f"Converted 'explicit' to int. Final shape: {df.shape}")

    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts):
        print(f"\nRemaining nulls:\n{null_counts}")
    else:
        print("No remaining nulls.")

    return df


def engineer_features(df: pd.DataFrame):
    """
    Prepare X and y for modeling: drop identifiers, one-hot encode track_genre,
    separate popularity target.

    Args:
        df: Cleaned DataFrame from load_and_clean

    Returns:
        X (DataFrame of features), y (Series of popularity), feature_names (list of str)
    """
    print_section("Feature Engineering")
    df = df.copy()

    drop_cols = ['track_id', 'artists', 'album_name', 'track_name']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df['popularity'].copy()
    df = df.drop(columns=['popularity'])

    df = pd.get_dummies(df, columns=['track_genre'], drop_first=True)

    feature_names = df.columns.tolist()
    print(f"Feature matrix: {df.shape[0]} rows × {df.shape[1]} features")
    print(f"Genre dummies created. Total features: {len(feature_names)}")

    return df, y, feature_names


def distribution_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot histograms for all numeric features and boxplots for key audio features.
    Print skewness table and flag features with |skew| > 1.

    Args:
        df: Cleaned DataFrame
        output_dir: Directory to save histograms.png and boxplots.png
    """
    print_section("Distribution Analysis")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # Histograms grid (4 columns)
    n_cols = 4
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=40, color='steelblue', edgecolor='white')
        axes[i].set_title(col, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Feature Distributions', fontsize=14, y=1.01)
    save_figure(fig, 'histograms.png', output_dir)

    # Boxplots
    boxplot_cols = [
        'danceability', 'energy', 'loudness', 'tempo', 'acousticness',
        'instrumentalness', 'valence', 'liveness', 'speechiness', 'popularity',
    ]
    boxplot_cols = [c for c in boxplot_cols if c in df.columns]
    n_bp = len(boxplot_cols)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, col in enumerate(boxplot_cols):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor='steelblue', alpha=0.7))
        axes[i].set_title(col)
    for j in range(n_bp, len(axes)):
        axes[j].set_visible(False)
    save_figure(fig, 'boxplots.png', output_dir)

    # Skewness table
    skew = df[numeric_cols].skew().sort_values(ascending=False)
    skew_df = skew.reset_index()
    skew_df.columns = ['Feature', 'Skewness']
    print("\nSkewness Table (sorted descending):")
    print(skew_df.to_string(index=False))
    high_skew = skew_df[skew_df['Skewness'].abs() > 1]['Feature'].tolist()
    print(f"\nHigh skew features (|skew| > 1): {high_skew}")


def correlation_with_popularity(df: pd.DataFrame, output_dir: str) -> None:
    """
    Compute Pearson correlation of all numeric features vs popularity.
    Plot horizontal bar chart (steelblue) and full heatmap (viridis).
    Print top 5 positive and top 5 negative correlations.

    Args:
        df: Cleaned DataFrame
        output_dir: Directory to save correlation_bar.png and correlation_heatmap.png
    """
    print_section("Correlation with Popularity")
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()['popularity'].drop('popularity').sort_values(
        key=abs, ascending=False
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['steelblue' if v >= 0 else 'tomato' for v in corr.values]
    corr.plot(kind='barh', color=colors, ax=ax)
    ax.set_title('Pearson Correlation with Popularity')
    ax.set_xlabel('Correlation Coefficient')
    ax.axvline(0, color='black', linewidth=0.8)
    save_figure(fig, 'correlation_bar.png', output_dir)

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='viridis',
                linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    save_figure(fig, 'correlation_heatmap.png', output_dir)

    print("\nTop 5 positively correlated with popularity:")
    pos = corr[corr > 0].head(5)
    print(pos.to_string() if len(pos) else "  (none)")
    print("\nTop 5 negatively correlated with popularity:")
    neg = corr[corr < 0].tail(5)
    print(neg.to_string() if len(neg) else "  (none)")


def bias_check(df: pd.DataFrame, output_dir: str) -> dict:
    """
    Check genre imbalance and artist dominance in the dataset.
    Plots top 20 genres and top 20 artists. Flags dominant artists (>1% of tracks).

    Args:
        df: Cleaned DataFrame (must have 'track_genre' and 'artists' columns)
        output_dir: Directory to save genre_distribution.png and artist_distribution.png

    Returns:
        bias_report dict with keys:
            'genre_imbalance_ratio': float (max genre count / min genre count)
            'dominant_artists': list of artist names with >1% of total tracks
    """
    print_section("Bias Check")

    # Genre imbalance
    genre_counts = df['track_genre'].value_counts()
    imbalance_ratio = float(genre_counts.max() / genre_counts.min())
    print(f"Genre imbalance ratio (max/min): {imbalance_ratio:.1f}x")

    fig, ax = plt.subplots(figsize=(12, 6))
    genre_counts.head(20).plot(kind='bar', color='steelblue', ax=ax)
    ax.set_title('Top 20 Genres by Track Count')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    save_figure(fig, 'genre_distribution.png', output_dir)

    # Artist dominance
    artist_counts = df['artists'].value_counts()
    total = len(df)
    dominant = artist_counts[artist_counts / total > 0.01].index.tolist()
    if dominant:
        print(f"Artists with >1% of total tracks: {dominant}")
    else:
        print("No single artist dominates >1% of tracks.")

    fig, ax = plt.subplots(figsize=(12, 6))
    artist_counts.head(20).plot(kind='bar', color='steelblue', ax=ax)
    ax.set_title('Top 20 Artists by Track Count')
    ax.set_xlabel('Artist')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    save_figure(fig, 'artist_distribution.png', output_dir)

    return {
        'genre_imbalance_ratio': imbalance_ratio,
        'dominant_artists': dominant,
    }
