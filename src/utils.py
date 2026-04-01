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
