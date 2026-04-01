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
