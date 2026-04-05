"""
pfi.py — Phase 3: Permutation Feature Importance (PFI)

Supports:
1. Standard feature-wise PFI
2. Grouped PFI (e.g. all track_genre_* columns shuffled together)

Functions:
    compute_pfi
    plot_pfi
    compute_grouped_pfi
    plot_grouped_pfi
    run_pfi_analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer

from utils import save_figure, print_section, timer


def _get_model_score(model, X, y, scoring: str) -> float:
    """
    Compute model score using sklearn scorer API.

    Args:
        model: fitted sklearn-compatible estimator
        X: feature DataFrame
        y: target Series
        scoring: sklearn scoring string

    Returns:
        Score as float
    """
    scorer = get_scorer(scoring)
    return float(scorer(model, X, y))


def _infer_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """
    Infer grouped feature blocks from column names.

    Current grouping rule:
    - all columns starting with 'track_genre_' -> one group called 'track_genre'

    All remaining columns are treated as their own individual groups.

    Args:
        columns: list of feature names

    Returns:
        Dict mapping group name -> list of columns
    """
    groups = {}
    genre_cols = [c for c in columns if c.startswith("track_genre_")]

    if genre_cols:
        groups["track_genre"] = genre_cols

    used = set(sum(groups.values(), []))
    for col in columns:
        if col not in used:
            groups[col] = [col]

    return groups


@timer
def compute_pfi(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    scoring: str = "r2",
    n_repeats: int = 10,
    random_state: int = 42,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Standard sklearn permutation feature importance (column-by-column).

    Args:
        model: fitted sklearn-compatible estimator
        X_test: test feature DataFrame
        y_test: test target Series
        model_name: display label
        scoring: sklearn scoring metric
        n_repeats: number of shuffles per feature
        random_state: reproducibility seed
        top_n: number of top features to print

    Returns:
        DataFrame with feature importances sorted descending
    """
    print_section(f"Permutation Feature Importance — {model_name}")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    pfi_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance Mean": result.importances_mean,
        "Importance Std": result.importances_std,
    }).sort_values("Importance Mean", ascending=False).reset_index(drop=True)

    print(f"\nTop {top_n} features by standard PFI ({scoring}) — {model_name}")
    print(pfi_df.head(top_n).to_string(index=False))

    return pfi_df


def plot_pfi(
    pfi_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
    top_n: int = 15,
) -> None:
    """
    Plot top-N standard PFI results.

    Args:
        pfi_df: DataFrame returned by compute_pfi
        model_name: display label
        output_dir: figure output directory
        top_n: number of rows to plot
    """
    plot_df = pfi_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        plot_df["Feature"],
        plot_df["Importance Mean"],
        xerr=plot_df["Importance Std"],
        color="steelblue",
        alpha=0.9,
    )
    ax.set_title(f"Permutation Feature Importance — {model_name}")
    ax.set_xlabel("Decrease in model performance after permutation")
    ax.set_ylabel("Feature")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    save_figure(fig, f"pfi_{model_name}.png", output_dir)


@timer
def compute_grouped_pfi(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    scoring: str = "r2",
    n_repeats: int = 10,
    random_state: int = 42,
    feature_groups: dict[str, list[str]] | None = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Grouped permutation feature importance.

    Entire feature groups are shuffled together in each repeat.
    This is especially useful for one-hot encoded categorical variables,
    such as track_genre_* columns.

    Args:
        model: fitted sklearn-compatible estimator
        X_test: test feature DataFrame
        y_test: test target Series
        model_name: display label
        scoring: sklearn scoring metric
        n_repeats: number of group shuffles
        random_state: reproducibility seed
        feature_groups: dict mapping group name -> list of columns.
                        If None, inferred automatically.
        top_n: number of top groups to print

    Returns:
        DataFrame with grouped importances sorted descending
    """
    print_section(f"Grouped Permutation Feature Importance — {model_name}")

    if feature_groups is None:
        feature_groups = _infer_feature_groups(list(X_test.columns))

    rng = np.random.RandomState(random_state)
    baseline_score = _get_model_score(model, X_test, y_test, scoring)

    rows = []

    for group_name, cols in feature_groups.items():
        missing = [c for c in cols if c not in X_test.columns]
        if missing:
            print(f"Skipping group '{group_name}' due to missing columns: {missing}")
            continue

        score_drops = []

        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm.loc[:, cols] = X_perm[cols].iloc[perm_idx].to_numpy()

            perm_score = _get_model_score(model, X_perm, y_test, scoring)
            drop = baseline_score - perm_score
            score_drops.append(drop)

        rows.append({
            "Feature Group": group_name,
            "Importance Mean": float(np.mean(score_drops)),
            "Importance Std": float(np.std(score_drops)),
            "Num Columns": len(cols),
        })

    grouped_df = pd.DataFrame(rows).sort_values(
        "Importance Mean", ascending=False
    ).reset_index(drop=True)

    print(f"\nTop {top_n} feature groups by grouped PFI ({scoring}) — {model_name}")
    print(grouped_df.head(top_n).to_string(index=False))

    return grouped_df


def plot_grouped_pfi(
    grouped_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
    top_n: int = 15,
) -> None:
    """
    Plot top-N grouped PFI results.

    Args:
        grouped_df: DataFrame returned by compute_grouped_pfi
        model_name: display label
        output_dir: figure output directory
        top_n: number of rows to plot
    """
    plot_df = grouped_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        plot_df["Feature Group"],
        plot_df["Importance Mean"],
        xerr=plot_df["Importance Std"],
        color="darkorange",
        alpha=0.9,
    )
    ax.set_title(f"Grouped Permutation Feature Importance — {model_name}")
    ax.set_xlabel("Decrease in model performance after group permutation")
    ax.set_ylabel("Feature Group")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    save_figure(fig, f"grouped_pfi_{model_name}.png", output_dir)


@timer
def run_pfi_analysis(
    models_dict: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
    scoring: str = "r2",
    n_repeats: int = 10,
    top_n: int = 15,
    model_keys: list[str] | None = None,
    run_standard: bool = True,
    run_grouped: bool = True,
    feature_groups: dict[str, list[str]] | None = None,
) -> dict:
    """
    Run standard and/or grouped PFI for selected models.

    Args:
        models_dict: dict of fitted models
        X_test: test feature DataFrame
        y_test: test target Series
        output_dir: figure output directory
        scoring: sklearn scoring metric
        n_repeats: number of shuffles
        top_n: number of rows to display/plot
        model_keys: subset of model keys to analyze
        run_standard: whether to run standard PFI
        run_grouped: whether to run grouped PFI
        feature_groups: optional custom groups

    Returns:
        Dict with structure:
        {
            model_key: {
                "standard": DataFrame or None,
                "grouped": DataFrame or None
            }
        }
    """
    if model_keys is None:
        model_keys = list(models_dict.keys())

    label_map = {
        "linear": "LinearRegression",
        "rf": "RandomForest",
        "xgb": "XGBoost",
        "lgbm": "LightGBM",
    }

    results = {}

    for key in model_keys:
        if key not in models_dict:
            print(f"Skipping missing model key: {key}")
            continue

        model = models_dict[key]
        model_name = label_map.get(key, key)

        results[key] = {"standard": None, "grouped": None}

        if run_standard:
            pfi_df = compute_pfi(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                scoring=scoring,
                n_repeats=n_repeats,
                random_state=42,
                top_n=top_n,
            )
            plot_pfi(pfi_df, model_name, output_dir, top_n=top_n)
            results[key]["standard"] = pfi_df

        if run_grouped:
            grouped_df = compute_grouped_pfi(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                scoring=scoring,
                n_repeats=n_repeats,
                random_state=42,
                feature_groups=feature_groups,
                top_n=top_n,
            )
            plot_grouped_pfi(grouped_df, model_name, output_dir, top_n=top_n)
            results[key]["grouped"] = grouped_df

    return results