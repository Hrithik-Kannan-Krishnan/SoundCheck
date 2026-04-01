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
    Plot PDP for all features in a 2x3 grid.

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
    Plot ICE lines + PDP overlay for all features in a 2x3 grid.
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
