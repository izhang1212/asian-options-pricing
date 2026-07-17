import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Palette (light mode, colorblind-safe categorical order)
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"


def _new_axes(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED_INK)
    ax.tick_params(colors=SECONDARY_INK)
    return fig, ax


def _finish(fig, ax, title, xlabel, ylabel, save_path):
    ax.set_title(title, color=PRIMARY_INK, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=SECONDARY_INK)
    ax.set_ylabel(ylabel, color=SECONDARY_INK)
    ax.legend(frameon=False, labelcolor=SECONDARY_INK)
    fig.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_sample_paths(paths, T, save_path, n_show=25):
    n_steps = paths.shape[1] - 1
    t = np.linspace(0, T, n_steps + 1)
    n_show = min(n_show, paths.shape[0])
    idx = np.random.choice(paths.shape[0], n_show, replace=False)

    fig, ax = _new_axes()
    for i in idx:
        ax.plot(t, paths[i], color=BLUE, alpha=0.35, linewidth=1, zorder=2)
    ax.plot(t, paths.mean(axis=0), color=PRIMARY_INK, linewidth=2, zorder=3, label="Mean path")

    _finish(fig, ax, "Simulated GBM Stock Price Paths", "Time (years)", "Stock price", save_path)


def plot_terminal_distribution(paths, S0, mu, sigma, T, save_path):
    terminal = paths[:, -1]

    fig, ax = _new_axes()
    ax.hist(terminal, bins=50, density=True, color=BLUE, alpha=0.55,
            edgecolor=SURFACE, linewidth=0.5, zorder=2, label="Simulated terminal price")

    s = sigma * np.sqrt(T)
    scale = S0 * np.exp((mu - 0.5 * sigma ** 2) * T)
    x = np.linspace(terminal.min(), terminal.max(), 500)
    ax.plot(x, lognorm.pdf(x, s=s, scale=scale), color=RED, linewidth=2, zorder=3,
            label="Theoretical lognormal pdf")

    _finish(fig, ax, "Terminal Stock Price Distribution", "Stock price", "Density", save_path)


def plot_convergence(M_values, means, stds, save_path, true_price=None):
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = _new_axes()
    ax.fill_between(M_values, means - 1.96 * stds, means + 1.96 * stds,
                     color=BLUE, alpha=0.15, zorder=1, label="95% CI (across repeats)")
    ax.plot(M_values, means, color=BLUE, marker="o", markersize=5, linewidth=2,
            zorder=3, label="MC price estimate")
    if true_price is not None:
        ax.axhline(true_price, color=RED, linestyle="--", linewidth=1.5, zorder=2, label="Reference price")

    ax.set_xscale("log")
    _finish(fig, ax, "Monte Carlo Convergence — Arithmetic Asian Call",
            "Number of simulated paths (M)", "Estimated option price", save_path)
