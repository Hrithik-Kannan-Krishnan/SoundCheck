# Spotify XAI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular XAI pipeline for Spotify popularity prediction with PDP/ICE explanations across 4 reusable Python modules and a master Jupyter notebook.

**Architecture:** Flat function-based modules under `src/` (utils → preprocessing → modeling → pdp_ice). `utils.py` is a shared dependency imported by the other three modules. The master notebook in `notebooks/` adds `../src` to `sys.path`, calls all functions sequentially, and saves plots to `outputs/figures/` and models to `outputs/models/`.

**Tech Stack:** Python 3.9+, pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, joblib, tabulate, pytest, nbformat

---

## File Map

| File | Role |
|---|---|
| `src/__init__.py` | Empty package marker |
| `src/utils.py` | 5 shared helpers: ensure_dirs, set_plot_style, save_figure, timer, print_section |
| `src/preprocessing.py` | Phase 1: load_and_clean, distribution_analysis, correlation_with_popularity, bias_check, engineer_features |
| `src/modeling.py` | Phase 2: split_data, train_all_models, evaluate_models, run_isolation_forest, save_models |
| `src/pdp_ice.py` | Phase 3: select_pdp_features, plot_pdp_single, plot_pdp_grid, plot_ice, plot_ice_grid, pdp_ice_interpretation_table, run_full_pdp_ice_analysis |
| `notebooks/spotify_xai.ipynb` | Master notebook: imports src/, calls all functions, inline outputs with markdown commentary |
| `outputs/figures/` | All saved plot PNGs |
| `outputs/models/` | Saved model .joblib files |
| `tests/__init__.py` | Empty test package marker |
| `tests/conftest.py` | Shared pytest fixtures (synthetic 60-row DataFrame) |
| `tests/test_utils.py` | Tests for utils.py |
| `tests/test_preprocessing.py` | Tests for preprocessing.py |
| `tests/test_modeling.py` | Tests for modeling.py |
| `tests/test_pdp_ice.py` | Tests for pdp_ice.py |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `outputs/figures/.gitkeep`
- Create: `outputs/models/.gitkeep`
- Create: `notebooks/` directory
- Create: `requirements.txt`

- [ ] **Step 1: Create all directories and empty package files**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
mkdir -p src tests notebooks outputs/figures outputs/models
touch src/__init__.py tests/__init__.py outputs/figures/.gitkeep outputs/models/.gitkeep
```

- [ ] **Step 2: Create requirements.txt**

Write to `requirements.txt`:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
joblib
tabulate
pytest
nbformat
ipykernel
```

- [ ] **Step 3: Create tests/conftest.py with shared fixtures**

Write to `tests/conftest.py`:
```python
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
```

- [ ] **Step 4: Verify directory structure**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
ls -R src/ tests/ outputs/
```
Expected: `src/__init__.py`, `tests/__init__.py`, `tests/conftest.py`, `outputs/figures/.gitkeep`, `outputs/models/.gitkeep`

- [ ] **Step 5: Commit**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
git init
git add src/__init__.py tests/__init__.py tests/conftest.py outputs/ requirements.txt
git commit -m "chore: project scaffolding — dirs, fixtures, requirements"
```

---

### Task 2: src/utils.py

**Files:**
- Create: `src/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write failing tests for utils**

Write to `tests/test_utils.py`:
```python
import os
import sys
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_ensure_dirs_creates_figures_and_models(tmp_path):
    from utils import ensure_dirs
    ensure_dirs(str(tmp_path))
    assert os.path.isdir(os.path.join(tmp_path, 'figures'))
    assert os.path.isdir(os.path.join(tmp_path, 'models'))


def test_ensure_dirs_is_idempotent(tmp_path):
    from utils import ensure_dirs
    ensure_dirs(str(tmp_path))
    ensure_dirs(str(tmp_path))  # should not raise


def test_save_figure_creates_file(tmp_path):
    from utils import save_figure
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    save_figure(fig, 'test_plot.png', str(tmp_path))
    assert os.path.isfile(os.path.join(tmp_path, 'test_plot.png'))


def test_save_figure_closes_figure(tmp_path):
    from utils import save_figure
    fig, ax = plt.subplots()
    ax.plot([1], [1])
    open_before = plt.get_fignums()
    save_figure(fig, 'closed.png', str(tmp_path))
    assert fig.number not in plt.get_fignums()


def test_timer_returns_function_result():
    from utils import timer

    @timer
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_timer_preserves_function_name():
    from utils import timer

    @timer
    def my_func():
        pass

    assert my_func.__name__ == 'my_func'


def test_set_plot_style_does_not_raise():
    from utils import set_plot_style
    set_plot_style()


def test_print_section_does_not_raise(capsys):
    from utils import print_section
    print_section("Test Section")
    captured = capsys.readouterr()
    assert "Test Section" in captured.out
```

- [ ] **Step 2: Run tests — expect ALL to fail**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_utils.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'utils'`

- [ ] **Step 3: Implement src/utils.py**

Write to `src/utils.py`:
```python
"""
utils.py — Shared helpers for the Spotify XAI project.

Provides: ensure_dirs, set_plot_style, save_figure, timer, print_section
"""
import os
import time
from functools import wraps

import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dirs(base_dir: str) -> None:
    """
    Create outputs/figures/ and outputs/models/ under base_dir if they don't exist.

    Args:
        base_dir: Root output directory (e.g. '../outputs' or '/tmp/outputs')
    """
    os.makedirs(os.path.join(base_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)


def set_plot_style() -> None:
    """
    Apply consistent global matplotlib/seaborn style.
    Sets whitegrid theme, dpi=120, figsize=(10, 6).
    """
    sns.set_theme(style='whitegrid')
    plt.rcParams.update({
        'figure.dpi': 120,
        'figure.figsize': (10, 6),
    })


def save_figure(fig, filename: str, output_dir: str) -> None:
    """
    Apply tight_layout, save figure to output_dir/filename at dpi=120, then close.

    Args:
        fig: matplotlib Figure object
        filename: Output filename (e.g. 'histograms.png')
        output_dir: Directory to save into
    """
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=120, bbox_inches='tight')
    plt.close(fig)


def timer(func):
    """
    Decorator that prints elapsed wall-clock time after the wrapped function returns.
    Preserves function name and docstring via functools.wraps.

    Usage:
        @timer
        def expensive_function(...): ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[timer] {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def print_section(title: str) -> None:
    """
    Print a visible divider banner for notebook readability.

    Args:
        title: Section title to display
    """
    width = 60
    print('\n' + '=' * width)
    print(f"  {title}")
    print('=' * width + '\n')
```

- [ ] **Step 4: Run tests — expect ALL to pass**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_utils.py -v
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/utils.py tests/test_utils.py
git commit -m "feat: implement src/utils.py with 5 shared helpers"
```

---

### Task 3: src/preprocessing.py — load_and_clean + engineer_features

**Files:**
- Create: `src/preprocessing.py` (partial — these two functions only)
- Create: `tests/test_preprocessing.py` (partial)

- [ ] **Step 1: Write failing tests for load_and_clean and engineer_features**

Write to `tests/test_preprocessing.py`:
```python
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
```

- [ ] **Step 2: Run tests — expect ALL to fail**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_preprocessing.py -v -k "load_and_clean or engineer_features" 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'preprocessing'`

- [ ] **Step 3: Implement load_and_clean and engineer_features in src/preprocessing.py**

Write to `src/preprocessing.py`:
```python
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
    drop_cols = ['track_id', 'artists', 'album_name', 'track_name']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df['popularity'].copy()
    df = df.drop(columns=['popularity'])

    df = pd.get_dummies(df, columns=['track_genre'], drop_first=True)

    feature_names = df.columns.tolist()
    print(f"Feature matrix: {df.shape[0]} rows × {df.shape[1]} features")
    print(f"Genre dummies created. Total features: {len(feature_names)}")

    return df, y, feature_names
```

- [ ] **Step 4: Run tests — expect ALL to pass**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_preprocessing.py -v -k "load_and_clean or engineer_features"
```
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: implement load_and_clean and engineer_features"
```

---

### Task 4: src/preprocessing.py — distribution_analysis, correlation_with_popularity, bias_check

**Files:**
- Modify: `src/preprocessing.py` (add 3 functions)
- Modify: `tests/test_preprocessing.py` (add tests)

- [ ] **Step 1: Add failing tests for the three analysis functions**

Append to `tests/test_preprocessing.py`:
```python
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
```

- [ ] **Step 2: Run new tests — expect to fail**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_preprocessing.py -v -k "distribution or correlation or bias" 2>&1 | head -20
```
Expected: failures for all 9 new tests

- [ ] **Step 3: Append the three functions to src/preprocessing.py**

Append to `src/preprocessing.py`:
```python

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
```

- [ ] **Step 4: Run all preprocessing tests — expect ALL to pass**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_preprocessing.py -v
```
Expected: `20 passed`

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: implement distribution_analysis, correlation_with_popularity, bias_check"
```

---

### Task 5: src/modeling.py — split_data, evaluate_models, save_models

**Files:**
- Create: `src/modeling.py` (partial — these three functions)
- Create: `tests/test_modeling.py` (partial)

- [ ] **Step 1: Write failing tests**

Write to `tests/test_modeling.py`:
```python
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
```

- [ ] **Step 2: Run tests — expect ALL to fail**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_modeling.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'modeling'`

- [ ] **Step 3: Implement the three functions in src/modeling.py**

Write to `src/modeling.py`:
```python
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
```

- [ ] **Step 4: Run tests — expect ALL to pass**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_modeling.py -v -k "split_data or evaluate_models or save_models"
```
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/modeling.py tests/test_modeling.py
git commit -m "feat: implement split_data, evaluate_models, save_models"
```

---

### Task 6: src/modeling.py — train_all_models, run_isolation_forest

**Files:**
- Modify: `src/modeling.py` (add 2 functions)
- Modify: `tests/test_modeling.py` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_modeling.py`:
```python
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
```

- [ ] **Step 2: Run new tests — expect to fail**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_modeling.py -v -k "train_all or isolation" 2>&1 | head -20
```
Expected: `AttributeError` or `ImportError`

- [ ] **Step 3: Append the two functions to src/modeling.py**

Append to `src/modeling.py`:
```python

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
```

- [ ] **Step 4: Run all modeling tests — expect ALL to pass**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_modeling.py -v
```
Expected: `14 passed`

- [ ] **Step 5: Commit**

```bash
git add src/modeling.py tests/test_modeling.py
git commit -m "feat: implement train_all_models and run_isolation_forest"
```

---

### Task 7: src/pdp_ice.py — select_pdp_features + pdp_ice_interpretation_table

**Files:**
- Create: `src/pdp_ice.py` (partial — these two functions)
- Create: `tests/test_pdp_ice.py` (partial)

- [ ] **Step 1: Write failing tests**

Write to `tests/test_pdp_ice.py`:
```python
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
```

- [ ] **Step 2: Run tests — expect ALL to fail**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_pdp_ice.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'pdp_ice'`

- [ ] **Step 3: Implement the two functions in src/pdp_ice.py**

Write to `src/pdp_ice.py`:
```python
"""
pdp_ice.py — Phase 3: Partial Dependence Plots and Individual Conditional Expectation.

Functions: select_pdp_features, plot_pdp_single, plot_pdp_grid, plot_ice,
           plot_ice_grid, pdp_ice_interpretation_table, run_full_pdp_ice_analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import save_figure, print_section, timer


def select_pdp_features(feature_names: list) -> list:
    """
    Return the 6 most interpretable continuous audio features for PDP/ICE analysis.
    Hardcoded: musically meaningful, continuous, not encoded categoricals.

    Args:
        feature_names: Full list of model feature names (used to validate presence)

    Returns:
        List of 6 feature name strings present in feature_names
    """
    pdp_features = ['energy', 'danceability', 'loudness', 'acousticness', 'valence', 'tempo']
    available = [f for f in pdp_features if f in feature_names]
    missing = set(pdp_features) - set(available)
    if missing:
        print(f"Warning: PDP features not found in feature_names: {missing}")
    return available


def pdp_ice_interpretation_table(model, X_test, features_list: list,
                                   model_name: str) -> pd.DataFrame:
    """
    For each feature compute PDP range (impact), PDP shape, and ICE heterogeneity
    (std of per-instance ranges — high value indicates interaction effects).
    Prints a formatted table.

    Args:
        model: Fitted sklearn-compatible estimator
        X_test: Test feature DataFrame
        features_list: List of feature names to analyze
        model_name: Label for output heading

    Returns:
        DataFrame with columns: Feature, PDP Range, PDP Shape, ICE Heterogeneity (std)
    """
    from sklearn.inspection import partial_dependence

    rows = []
    for feature in features_list:
        feat_idx = list(X_test.columns).index(feature)

        # PDP values (average)
        pd_avg = partial_dependence(model, X_test, [feat_idx], kind='average')
        pdp_vals = pd_avg['average'][0]
        pdp_range = float(np.max(pdp_vals) - np.min(pdp_vals))

        # Shape detection
        diffs = np.diff(pdp_vals)
        if np.all(diffs >= 0):
            shape = 'monotonic increasing'
        elif np.all(diffs <= 0):
            shape = 'monotonic decreasing'
        else:
            shape = 'non-monotonic'

        # ICE heterogeneity (std of per-instance ranges)
        pd_ice = partial_dependence(model, X_test, [feat_idx], kind='individual')
        ice_vals = pd_ice['individual'][0]  # shape: (n_samples, n_grid_points)
        per_instance_ranges = ice_vals.max(axis=1) - ice_vals.min(axis=1)
        heterogeneity = float(np.std(per_instance_ranges))

        rows.append({
            'Feature': feature,
            'PDP Range': round(pdp_range, 3),
            'PDP Shape': shape,
            'ICE Heterogeneity (std)': round(heterogeneity, 3),
        })

    result_df = pd.DataFrame(rows)
    print(f"\nPDP/ICE Interpretation Table — {model_name}")
    print(result_df.to_string(index=False))
    return result_df
```

- [ ] **Step 4: Run tests — expect ALL to pass**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_pdp_ice.py -v -k "select_pdp or interpretation"
```
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pdp_ice.py tests/test_pdp_ice.py
git commit -m "feat: implement select_pdp_features and pdp_ice_interpretation_table"
```

---

### Task 8: src/pdp_ice.py — plot_pdp_single, plot_pdp_grid, plot_ice, plot_ice_grid

**Files:**
- Modify: `src/pdp_ice.py` (add 4 plot functions)
- Modify: `tests/test_pdp_ice.py` (add smoke tests)

- [ ] **Step 1: Add failing smoke tests for plot functions**

Append to `tests/test_pdp_ice.py`:
```python
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
```

- [ ] **Step 2: Run new tests — expect to fail**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_pdp_ice.py -v -k "plot_pdp or plot_ice" 2>&1 | head -20
```
Expected: `ImportError` for plot functions not yet defined

- [ ] **Step 3: Append the 4 plot functions to src/pdp_ice.py**

Append to `src/pdp_ice.py`:
```python

def plot_pdp_single(model, X_test, feature_name: str, model_name: str,
                    output_dir: str):
    """
    Plot a single PDP for one feature with an annotation explaining PDP.

    Args:
        model: Fitted sklearn-compatible estimator
        X_test: Test feature DataFrame
        feature_name: Feature to analyze
        model_name: Label used in title and filename
        output_dir: Directory to save pdp_{model_name}_{feature_name}.png

    Returns:
        PartialDependenceDisplay object
    """
    from sklearn.inspection import PartialDependenceDisplay
    feat_idx = list(X_test.columns).index(feature_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    disp = PartialDependenceDisplay.from_estimator(
        model, X_test, [feat_idx], kind='average', ax=ax
    )
    ax.set_title(f"PDP: Effect of {feature_name} on Popularity ({model_name})")
    ax.text(
        0.02, 0.97,
        "PDP shows the average effect of this feature\non predicted popularity,"
        " marginalizing all others.",
        transform=ax.transAxes, fontsize=8, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
    )
    save_figure(fig, f"pdp_{model_name}_{feature_name}.png", output_dir)
    return disp


def plot_pdp_grid(model, X_test, features_list: list, model_name: str,
                  output_dir: str) -> None:
    """
    Plot PDP for all features in a 2×3 grid.

    Args:
        model: Fitted sklearn-compatible estimator
        X_test: Test feature DataFrame
        features_list: List of feature names (length 6 expected)
        model_name: Label used in title and filename
        output_dir: Directory to save pdp_grid_{model_name}.png
    """
    from sklearn.inspection import PartialDependenceDisplay
    feat_indices = [list(X_test.columns).index(f) for f in features_list]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    PartialDependenceDisplay.from_estimator(
        model, X_test, feat_indices, kind='average',
        ax=axes.flatten()[:len(feat_indices)],
    )
    fig.suptitle(f"PDP Grid - {model_name}", fontsize=14)
    save_figure(fig, f"pdp_grid_{model_name}.png", output_dir)


def plot_ice(model, X_test, feature_name: str, model_name: str, output_dir: str,
             n_samples: int = 200, centered: bool = False) -> None:
    """
    Plot ICE lines + PDP average overlay for a single feature.
    ICE lines: steelblue, alpha=0.05. PDP line: red, linewidth=3.

    Args:
        model: Fitted sklearn-compatible estimator
        X_test: Test feature DataFrame
        feature_name: Feature to analyze
        model_name: Label used in title and filename
        output_dir: Directory to save the figure
        n_samples: Number of ICE lines to draw (default 200)
        centered: If True, center ICE lines at first grid point (default False)
    """
    from sklearn.inspection import PartialDependenceDisplay
    feat_idx = list(X_test.columns).index(feature_name)
    centered_label = 'centered' if centered else 'standard'

    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        model, X_test, [feat_idx], kind='both',
        subsample=n_samples, centered=centered,
        ice_lines_kw={'alpha': 0.05, 'color': 'steelblue'},
        pd_line_kw={'color': 'red', 'linewidth': 3, 'label': 'PDP (average)'},
        ax=ax,
    )
    ax.set_title(
        f"ICE Plot: {feature_name} ({'Centered' if centered else 'Standard'}) | {model_name}"
    )
    ax.text(
        0.02, 0.03,
        "Each line = one song. Red = average (PDP).\nCrossing lines indicate interaction effects.",
        transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
    )
    ax.legend(loc='upper right')
    save_figure(fig, f"ice_{model_name}_{feature_name}_{centered_label}.png", output_dir)


def plot_ice_grid(model, X_test, features_list: list, model_name: str,
                  output_dir: str, n_samples: int = 150,
                  centered: bool = False) -> None:
    """
    Plot ICE lines + PDP overlay for all features in a 2×3 grid.
    ICE lines: steelblue, alpha=0.05. PDP line: red, linewidth=3.

    Args:
        model: Fitted sklearn-compatible estimator
        X_test: Test feature DataFrame
        features_list: List of feature names (length 6 expected)
        model_name: Label used in title and filename
        output_dir: Directory to save the figure
        n_samples: Number of ICE lines per subplot (default 150)
        centered: If True, center ICE lines (default False)
    """
    from sklearn.inspection import PartialDependenceDisplay
    feat_indices = [list(X_test.columns).index(f) for f in features_list]
    centered_label = 'centered' if centered else 'standard'

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    PartialDependenceDisplay.from_estimator(
        model, X_test, feat_indices, kind='both',
        subsample=n_samples, centered=centered,
        ice_lines_kw={'alpha': 0.05, 'color': 'steelblue'},
        pd_line_kw={'color': 'red', 'linewidth': 3},
        ax=axes.flatten()[:len(feat_indices)],
    )
    fig.suptitle(
        f"ICE Grid ({'Centered' if centered else 'Standard'}) - {model_name}",
        fontsize=14,
    )
    save_figure(fig, f"ice_grid_{model_name}_{centered_label}.png", output_dir)
```

- [ ] **Step 4: Run all pdp_ice tests so far — expect ALL to pass**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_pdp_ice.py -v
```
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pdp_ice.py tests/test_pdp_ice.py
git commit -m "feat: implement 4 PDP/ICE plot functions"
```

---

### Task 9: src/pdp_ice.py — run_full_pdp_ice_analysis

**Files:**
- Modify: `src/pdp_ice.py` (add final function)
- Modify: `tests/test_pdp_ice.py` (add smoke test)

- [ ] **Step 1: Add failing smoke test**

Append to `tests/test_pdp_ice.py`:
```python
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
```

- [ ] **Step 2: Run test — expect to fail**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_pdp_ice.py::test_run_full_pdp_ice_analysis_saves_expected_files -v 2>&1 | head -20
```
Expected: `ImportError` — function not yet defined

- [ ] **Step 3: Append run_full_pdp_ice_analysis to src/pdp_ice.py**

Append to `src/pdp_ice.py`:
```python

@timer
def run_full_pdp_ice_analysis(models_dict: dict, X_test, feature_names: list,
                               output_dir: str) -> None:
    """
    Run complete PDP and ICE analysis for RandomForest and XGBoost.
    For each model: PDP grid, standard ICE grid, centered ICE grid,
    interpretation table, and side-by-side detail plots for top 2 features
    by PDP range. Prints a final summary.

    Args:
        models_dict: Dict mapping model keys to fitted models (uses 'rf' and 'xgb')
        X_test: Test feature DataFrame (recommend X_test.sample(5000, random_state=42))
        feature_names: Full list of model feature names
        output_dir: Directory to save all generated figures
    """
    from sklearn.inspection import PartialDependenceDisplay

    features = select_pdp_features(feature_names)
    summary = {}

    for model_key, model_name in [('rf', 'RandomForest'), ('xgb', 'XGBoost')]:
        if model_key not in models_dict:
            print(f"Skipping {model_name}: key '{model_key}' not in models_dict")
            continue

        model = models_dict[model_key]
        print_section(f"PDP/ICE Analysis: {model_name}")

        # a. PDP grid
        plot_pdp_grid(model, X_test, features, model_name, output_dir)

        # b. ICE grid (standard)
        plot_ice_grid(model, X_test, features, model_name, output_dir, centered=False)

        # c. ICE grid (centered)
        plot_ice_grid(model, X_test, features, model_name, output_dir, centered=True)

        # d. Interpretation table
        interp_df = pdp_ice_interpretation_table(model, X_test, features, model_name)
        summary[model_name] = interp_df

        # e. Side-by-side detail for top 2 features by PDP range
        top2 = interp_df.nlargest(2, 'PDP Range')['Feature'].tolist()
        for feat in top2:
            feat_idx = list(X_test.columns).index(feat)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            PartialDependenceDisplay.from_estimator(
                model, X_test, [feat_idx], kind='average', ax=axes[0]
            )
            axes[0].set_title(f"PDP: {feat} | {model_name}")

            PartialDependenceDisplay.from_estimator(
                model, X_test, [feat_idx], kind='both', subsample=200,
                ice_lines_kw={'alpha': 0.05, 'color': 'steelblue'},
                pd_line_kw={'color': 'red', 'linewidth': 3},
                ax=axes[1],
            )
            axes[1].set_title(f"ICE: {feat} | {model_name}")
            axes[1].text(
                0.02, 0.03,
                "Each line = one song. Red = average (PDP).",
                transform=axes[1].transAxes, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            )
            save_figure(fig, f"top_feature_{model_name}_{feat}.png", output_dir)

    # Final summary
    print_section("PDP/ICE Final Summary")
    for model_name, df in summary.items():
        top_impact = df.nlargest(2, 'PDP Range')[['Feature', 'PDP Range']].to_string(index=False)
        top_interact = df.nlargest(2, 'ICE Heterogeneity (std)')[
            ['Feature', 'ICE Heterogeneity (std)']
        ].to_string(index=False)
        print(f"\n{model_name}:")
        print(f"  Highest impact (PDP range):\n{top_impact}")
        print(f"  Strongest interactions (ICE heterogeneity):\n{top_interact}")
```

- [ ] **Step 4: Run all pdp_ice tests — expect ALL to pass**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/test_pdp_ice.py -v
```
Expected: `16 passed`

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/ -v
```
Expected: `39 passed`

- [ ] **Step 6: Commit**

```bash
git add src/pdp_ice.py tests/test_pdp_ice.py
git commit -m "feat: implement run_full_pdp_ice_analysis — complete Phase 3"
```

---

### Task 10: notebooks/spotify_xai.ipynb

**Files:**
- Create: `notebooks/spotify_xai.ipynb`

- [ ] **Step 1: Create notebook generator script and run it**

Write the following to a temporary file `create_notebook.py` in the project root, then run it:

```python
import nbformat as nbf
import json

nb = nbf.v4.new_notebook()
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.9.0'},
}

cells = []

def md(source):
    return nbf.v4.new_markdown_cell(source)

def code(source):
    return nbf.v4.new_code_cell(source)

# ── Section 0: Setup ──────────────────────────────────────────────────────────
cells.append(md("# Spotify XAI Analysis\n\nEnd-to-end pipeline: EDA → Modeling → Explainability (PDP/ICE)"))

cells.append(code("""\
%pip install -q pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib tabulate nbformat ipykernel
"""))

cells.append(md("## Section 0: Setup"))

cells.append(code("""\
import sys
import os

sys.path.insert(0, '../src')

import matplotlib
matplotlib.use('inline')
import matplotlib.pyplot as plt

from utils import ensure_dirs, set_plot_style, print_section
from preprocessing import (
    load_and_clean, distribution_analysis,
    correlation_with_popularity, bias_check, engineer_features,
)
from modeling import (
    split_data, train_all_models, evaluate_models,
    run_isolation_forest, save_models,
)
from pdp_ice import (
    select_pdp_features, run_full_pdp_ice_analysis,
)

ensure_dirs('../outputs')
set_plot_style()
print("Setup complete.")
"""))

# ── Section 1: Phase 1 ────────────────────────────────────────────────────────
cells.append(md("## Section 1: Phase 1 — Data Loading and EDA"))

cells.append(md("### 1.1 Load and Clean"))
cells.append(code("df = load_and_clean('../dataset.csv')"))

cells.append(md("### 1.2 Distribution Analysis\n\nHistograms and boxplots reveal the spread and skewness of each audio feature. Highly skewed features (|skew| > 1) may need transformation for linear models, though tree-based models are robust to skew."))
cells.append(code("distribution_analysis(df, '../outputs/figures')"))

cells.append(md("### 1.3 Correlation with Popularity\n\nPearson correlation shows linear relationships. Loudness and energy tend to have the strongest positive correlations with popularity — louder, energetic tracks are associated with mainstream appeal. Acousticness and instrumentalness often show negative correlation, reflecting a preference for vocal, produced sound in popular music."))
cells.append(code("correlation_with_popularity(df, '../outputs/figures')"))

cells.append(md("### 1.4 Bias Check\n\nUneven genre representation can bias model predictions toward over-represented genres. Artists with many tracks may also skew the learned relationships between audio features and popularity."))
cells.append(code("bias_report = bias_check(df, '../outputs/figures')\nprint(bias_report)"))

# ── Section 2: Phase 2 ────────────────────────────────────────────────────────
cells.append(md("## Section 2: Phase 2 — Feature Engineering and Modeling"))

cells.append(md("### 2.1 Feature Engineering"))
cells.append(code("""\
X, y, feature_names = engineer_features(df)
print(f"Features: {len(feature_names)}")
print(X.dtypes.value_counts())
"""))

cells.append(md("### 2.2 Train/Test Split"))
cells.append(code("X_train, X_test, y_train, y_test = split_data(X, y)"))

cells.append(md("### 2.3 Train All Models\n\nWe train four models: a linear baseline (LinearRegression), two ensemble tree models (RandomForest, XGBoost), and a gradient boosting model (LightGBM). Tree-based models are expected to capture non-linear audio feature interactions."))
cells.append(code("models_dict = train_all_models(X_train, y_train)"))

cells.append(md("### 2.4 Evaluate Models\n\nRMSE measures prediction error in popularity units (0–100). R² indicates how much variance in popularity the model explains. Popularity is notoriously hard to predict from audio features alone — values of R² ~ 0.2–0.4 are typical for audio-only models."))
cells.append(code("results_df = evaluate_models(models_dict, X_test, y_test)\nresults_df"))

cells.append(md("### 2.5 Anomaly Detection — Isolation Forest\n\nIsolation Forest flags tracks whose audio feature combinations are unusual compared to the rest of the dataset. These may be mislabelled tracks, niche experimental recordings, or data entry errors."))
cells.append(code("""\
anomaly_labels, anomaly_scores = run_isolation_forest(X_train)
print(anomaly_scores.describe())
"""))

cells.append(md("### 2.6 Save Models"))
cells.append(code("save_models(models_dict, '../outputs/models')"))

# ── Section 3: Phase 3 ────────────────────────────────────────────────────────
cells.append(md("## Section 3: Phase 3 — PDP and ICE Analysis"))

cells.append(md("""\
Partial Dependence Plots (PDP) show the **average** effect of a single feature on predicted popularity, holding all other features constant. Individual Conditional Expectation (ICE) plots show this effect **per track** — diverging ICE lines reveal interaction effects where the feature's impact depends on other feature values.

We analyse RandomForest (primary) and XGBoost (secondary). Both are non-linear models whose internal mechanisms are opaque — PDP/ICE provides post-hoc transparency.
"""))

cells.append(md("### 3.1 Feature Selection"))
cells.append(code("""\
X_test_sample = X_test.sample(5000, random_state=42)
features = select_pdp_features(feature_names)
print("PDP/ICE features:", features)
"""))

cells.append(md("### 3.2 RandomForest — Full PDP/ICE Analysis\n\nRandomForest is our primary model for explanation because its ensemble averaging produces stable, interpretable PDP curves. The ICE plots reveal whether individual tracks respond differently to feature changes."))
cells.append(code("""\
run_full_pdp_ice_analysis(
    {'rf': models_dict['rf']},
    X_test_sample,
    feature_names,
    '../outputs/figures',
)
"""))

cells.append(md("""\
**Interpretation — RandomForest PDP/ICE:**

- **Loudness** typically shows a monotonic positive relationship with predicted popularity — the model has learned that louder (more heavily mastered) tracks tend to chart better, consistent with the loudness war in commercial music production.
- **Energy** also tends to show a positive association — high-energy tracks dominate streaming charts in pop, hip-hop, and electronic genres.
- **Acousticness** commonly shows a negative association — highly acoustic tracks (folk, classical) receive lower predicted popularity scores on average, reflecting the dataset's genre distribution.
- **Danceability** may show non-monotonic behaviour — moderate danceability scores align with the widest range of popular genres, while extremes (very low or very high) narrow genre appeal.
- **ICE heterogeneity** for loudness and energy is often high, indicating that the effect of these features depends on other characteristics — a loud acoustic ballad behaves differently from a loud electronic track.
"""))

cells.append(md("### 3.3 XGBoost — Full PDP/ICE Analysis\n\nXGBoost is our secondary model. Its gradient-boosted trees can capture sharper feature thresholds. Comparing RF and XGBoost PDPs reveals whether the learned relationships are consistent across model architectures."))
cells.append(code("""\
run_full_pdp_ice_analysis(
    {'xgb': models_dict['xgb']},
    X_test_sample,
    feature_names,
    '../outputs/figures',
)
"""))

cells.append(md("""\
**Interpretation — XGBoost PDP/ICE:**

- XGBoost PDPs often show **sharper transitions** than RandomForest — gradient boosting captures hard decision boundaries that manifest as step-like PDP curves, particularly for loudness and tempo.
- Where RF and XGBoost agree on the direction of an effect (e.g. both show loudness positively correlated with popularity), that finding is more robust — it is not an artefact of a single model's inductive bias.
- Discrepancies between the two models in the shape of the tempo PDP may reflect that tempo's effect on popularity is highly context-dependent (genre-mediated), making it sensitive to the model's feature interaction capacity.
- High ICE heterogeneity in XGBoost valence plots suggests the model has captured that the emotional tone of a track (valence) affects different subsets of songs differently — happy-sounding tracks benefit in pop contexts but not in hip-hop.
"""))

nb.cells = cells

with open('notebooks/spotify_xai.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/spotify_xai.ipynb")
```

Run it:
```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python create_notebook.py
```
Expected: `Notebook written to notebooks/spotify_xai.ipynb`

- [ ] **Step 2: Verify the notebook is valid JSON and has expected sections**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -c "
import json
with open('notebooks/spotify_xai.ipynb') as f:
    nb = json.load(f)
print(f'Cell count: {len(nb[\"cells\"])}')
md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f'Markdown cells: {len(md_cells)}')
print(f'Code cells: {len(code_cells)}')
"
```
Expected: Cell count >= 22, markdown cells >= 12, code cells >= 10

- [ ] **Step 3: Remove the generator script**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
rm create_notebook.py
```

- [ ] **Step 4: Run the full test suite one final time**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
python -m pytest tests/ -v
```
Expected: `39 passed`, 0 failures

- [ ] **Step 5: Final commit**

```bash
cd "/Users/hrithikkannankrishnan/Desktop/DBA5102 - Blockchain/XAI"
git add notebooks/spotify_xai.ipynb
git commit -m "feat: add master Jupyter notebook spotify_xai.ipynb"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ `load_and_clean` — Task 3
- ✅ `distribution_analysis` — Task 4
- ✅ `correlation_with_popularity` — Task 4
- ✅ `bias_check` — Task 4
- ✅ `engineer_features` — Task 3
- ✅ `split_data` — Task 5
- ✅ `train_all_models` (4 models, per-model timing, @timer) — Task 6
- ✅ `evaluate_models` (RMSE/MAE/R², bar chart) — Task 5
- ✅ `run_isolation_forest` — Task 6
- ✅ `save_models` (joblib) — Task 5
- ✅ `select_pdp_features` (hardcoded 6 features) — Task 7
- ✅ `plot_pdp_single` (annotation box) — Task 8
- ✅ `plot_pdp_grid` (2×3) — Task 8
- ✅ `plot_ice` (standard + centered, steelblue/red, text note) — Task 8
- ✅ `plot_ice_grid` (2×3, standard + centered) — Task 8
- ✅ `pdp_ice_interpretation_table` (range, shape, heterogeneity) — Task 7
- ✅ `run_full_pdp_ice_analysis` (RF+XGB, top-2 side-by-side, summary) — Task 9
- ✅ Master notebook (4 sections, markdown interpretation, X_test_sample) — Task 10
- ✅ Global constraints: dpi=120, steelblue/viridis, random_state=42, @timer, docstrings

**Type consistency:**
- `engineer_features` returns `(X, y, feature_names)` — used as `X, y, feature_names = engineer_features(df)` throughout ✅
- `train_all_models` returns dict with keys `'linear', 'rf', 'xgb', 'lgbm'` — `run_full_pdp_ice_analysis` accesses `'rf'` and `'xgb'` ✅
- `run_isolation_forest` returns `(anomaly_labels, anomaly_scores)` — used as tuple unpack ✅
- `select_pdp_features(feature_names)` — called with `feature_names` list from `engineer_features` ✅
- `run_full_pdp_ice_analysis(models_dict, X_test_sample, feature_names, output_dir)` — matches notebook call ✅
