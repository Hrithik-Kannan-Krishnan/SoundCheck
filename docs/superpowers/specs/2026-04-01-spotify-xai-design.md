# Spotify XAI Project — Design Spec
**Date:** 2026-04-01
**Status:** Approved

---

## Overview

Build a complete Explainable AI (XAI) pipeline for a Spotify dataset (114,000 rows, 21 columns). The goal is to predict track popularity and explain model predictions using Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots. The project is structured as reusable Python modules under `src/` with a master Jupyter notebook as the top layer.

---

## Dataset

- **File:** `dataset.csv` (project root)
- **Rows:** 114,000 (114,001 including header)
- **Target:** `popularity` (continuous, 0–100)
- **Key columns:** `track_id`, `artists`, `album_name`, `track_name`, `popularity`, `duration_ms`, `explicit`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`, `track_genre`

---

## File Architecture

```
XAI/
├── dataset.csv
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── preprocessing.py
│   ├── modeling.py
│   └── pdp_ice.py
├── notebooks/
│   └── spotify_xai.ipynb
└── outputs/
    ├── figures/
    └── models/
```

---

## Module Design

### `src/utils.py` — Shared Helpers

| Function | Signature | Purpose |
|---|---|---|
| `ensure_dirs` | `(base_dir: str)` | Creates `outputs/figures/` and `outputs/models/` if missing |
| `set_plot_style` | `()` | Sets seaborn theme, `dpi=120`, `figsize=(10,6)` globally |
| `save_figure` | `(fig, filename, output_dir)` | `tight_layout` → `savefig` → `plt.close(fig)` |
| `timer` | decorator | Prints elapsed time; applied to expensive training/analysis functions |
| `print_section` | `(title: str)` | Prints a divider banner for notebook readability |

---

### `src/preprocessing.py` — Phase 1

#### `load_and_clean(filepath)`
- Load CSV
- Drop duplicates on `track_id`
- Drop rows where `popularity` is null
- Convert `explicit` (bool string) to int
- Print summary: shape before/after, null counts
- **Returns:** cleaned `DataFrame`

#### `distribution_analysis(df, output_dir)`
- Histograms for all numeric features in a 4-column grid
- Boxplots for: `danceability`, `energy`, `loudness`, `tempo`, `acousticness`, `instrumentalness`, `valence`, `liveness`, `speechiness`, `popularity`
- Print skewness table sorted descending; flag `|skew| > 1` as high skew
- Save figures to `output_dir`
- **Returns:** None

#### `correlation_with_popularity(df, output_dir)`
- Pearson correlation of all numeric features vs `popularity`
- Horizontal bar chart sorted by absolute correlation (steelblue)
- Full heatmap of numeric features (viridis)
- Print top 5 positively and top 5 negatively correlated features
- Save figures to `output_dir`
- **Returns:** None

#### `bias_check(df, output_dir)`
- Genre imbalance: `value_counts` for `track_genre`, bar chart of top 20 genres, print imbalance ratio (max count / min count)
- Artist dominance: count tracks per artist, show top 20, flag any artist with > 1% of total tracks
- Save figures to `output_dir`
- **Returns:** `bias_report` dict with keys `genre_imbalance_ratio`, `dominant_artists`

#### `engineer_features(df)`
- Drop: `track_id`, `artists`, `album_name`, `track_name`
- One-hot encode `track_genre` with `pd.get_dummies(drop_first=True)`
- Keep `key`, `mode`, `time_signature` as integer categories
- **Returns:** `X` (DataFrame), `y` (Series), `feature_names` (list of str)

---

### `src/modeling.py` — Phase 2

#### `split_data(X, y, test_size=0.2, random_state=42)`
- Standard train/test split (no stratification — regression target)
- **Returns:** `X_train, X_test, y_train, y_test`

#### `train_all_models(X_train, y_train)`
Decorated with `@timer`. Trains 4 models:

| Key | Model | Hyperparameters |
|---|---|---|
| `linear` | `LinearRegression` | default |
| `rf` | `RandomForestRegressor` | `n_estimators=100, max_depth=15, n_jobs=-1, random_state=42` |
| `xgb` | `XGBRegressor` | `n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42` |
| `lgbm` | `LGBMRegressor` | `n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42` |

- Print training time per model
- **Returns:** `dict` mapping key → fitted model

#### `evaluate_models(models_dict, X_test, y_test)`
- Metrics per model: RMSE, MAE, R²
- Print clean comparison table (pandas or tabulate)
- Bar chart comparing RMSE across models (steelblue)
- **Returns:** `results_df` (DataFrame)

#### `run_isolation_forest(X_train, contamination=0.05)`
- Fit `IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)`
- Print count and % of anomalies detected
- **Returns:** `(anomaly_labels, anomaly_scores)` as Series

#### `save_models(models_dict, output_dir)`
- Save each model via `joblib.dump` to `outputs/models/<key>_model.joblib`
- **Returns:** None

---

### `src/pdp_ice.py` — Phase 3

#### `select_pdp_features(feature_names)`
- Returns hardcoded list: `['energy', 'danceability', 'loudness', 'acousticness', 'valence', 'tempo']`
- These are continuous, musically meaningful, non-encoded features

#### `plot_pdp_single(model, X_test, feature_name, model_name, output_dir)`
- `PartialDependenceDisplay.from_estimator(kind='average')`
- Title: `f"PDP: Effect of {feature_name} on Popularity ({model_name})"`
- Annotation box explaining what PDP shows
- Save to `outputs/figures/pdp_{model_name}_{feature_name}.png`
- **Returns:** display object

#### `plot_pdp_grid(model, X_test, features_list, model_name, output_dir)`
- All 6 features in a 2×3 grid, `kind='average'`
- Title: `f"PDP Grid - {model_name}"`
- Save to `outputs/figures/pdp_grid_{model_name}.png`

#### `plot_ice(model, X_test, feature_name, model_name, output_dir, n_samples=200, centered=False)`
- `PartialDependenceDisplay.from_estimator(kind='both', subsample=n_samples)`
- ICE lines: `alpha=0.05, color='steelblue'`
- PDP line: `color='red', linewidth=3, label='PDP (average)'`
- If `centered=True`: pass `centered=True` to display
- Title: `f"ICE Plot: {feature_name} ({'Centered' if centered else 'Standard'}) | {model_name}"`
- Text note: `"Each line = one song. Red = average (PDP). Crossing lines indicate interaction effects."`
- Save to `outputs/figures/ice_{model_name}_{feature_name}_{'centered' if centered else 'standard'}.png`

#### `plot_ice_grid(model, X_test, features_list, model_name, output_dir, n_samples=150, centered=False)`
- All 6 features in a 2×3 grid, `kind='both'`, same styling as `plot_ice`
- Save to `outputs/figures/ice_grid_{model_name}_{'centered' if centered else 'standard'}.png`

#### `pdp_ice_interpretation_table(model, X_test, features_list, model_name)`
For each feature computes:
- **PDP range:** `max(pdp_values) - min(pdp_values)` — overall impact magnitude
- **PDP shape:** `'monotonic increasing'`, `'monotonic decreasing'`, or `'non-monotonic'` (via sorted value comparison)
- **ICE heterogeneity:** std of per-instance PDP ranges — high std = interaction effects
- Prints as formatted table
- **Returns:** DataFrame of results

#### `run_full_pdp_ice_analysis(models_dict, X_test, feature_names, output_dir)`
Decorated with `@timer`. Runs for RF and XGBoost only:

For each model:
1. `plot_pdp_grid`
2. `plot_ice_grid` (standard)
3. `plot_ice_grid` (centered)
4. `pdp_ice_interpretation_table` (printed)
5. For top 2 features by PDP range: side-by-side PDP + ICE single figure

Prints final summary: features with most impact, features with interaction effects.

---

## Master Notebook (`notebooks/spotify_xai.ipynb`)

### Section 0: Setup
- `pip install` block for all dependencies
- `sys.path.insert` to add `../src` to path
- All imports from `src/`
- `ensure_dirs('../outputs')`, `set_plot_style()`

### Section 1: Phase 1 — Data Loading and EDA
```
load_and_clean('../dataset.csv')
distribution_analysis(df, '../outputs/figures')
correlation_with_popularity(df, '../outputs/figures')
bias_check(df, '../outputs/figures')
```

### Section 2: Phase 2 — Feature Engineering and Modeling
```
engineer_features(df)
split_data(X, y)
train_all_models(X_train, y_train)
evaluate_models(models_dict, X_test, y_test)
run_isolation_forest(X_train)
save_models(models_dict, '../outputs/models')
```

### Section 3: Phase 3 — PDP and ICE Analysis
```python
X_test_sample = X_test.sample(5000, random_state=42)
features = select_pdp_features(feature_names)
run_full_pdp_ice_analysis(models_dict, X_test_sample, feature_names, '../outputs/figures')
```
- Markdown cells between each output block with music-domain interpretation
- RF is primary (deeper commentary), XGBoost is secondary
- Example: *"Loudness shows a monotonic positive relationship with popularity, suggesting the model has learned that louder mastered tracks tend to chart better."*

---

## Global Implementation Constraints

| Constraint | Value |
|---|---|
| Plot DPI | 120 |
| Layout | `tight_layout=True` always |
| Bar chart color | `steelblue` |
| Heatmap colormap | `viridis` |
| ICE lines color | `steelblue`, `alpha=0.05` |
| PDP average line | `red`, `linewidth=3` |
| Random state | `42` everywhere applicable |
| Model serialization | `joblib` |
| Timing | `@timer` decorator on expensive functions |
| Docstrings | Every function: inputs, outputs, description |
| Notebook path to data | `../dataset.csv` |
| X_test for PDP/ICE | `X_test.sample(5000, random_state=42)` — sampled in notebook |

---

## Dependencies

```
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, joblib, tabulate
```
