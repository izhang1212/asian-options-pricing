import numpy as np
import matplotlib.pyplot as plt

# Palette (light mode, colorblind-safe categorical order)
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BLUE = "#2a78d6"
GREEN = "#008300"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED_INK)
    ax.tick_params(colors=SECONDARY_INK)


def _new_axes(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    return fig, ax


def _finish(fig, ax, title, xlabel, ylabel, save_path, legend=True):
    ax.set_title(title, color=PRIMARY_INK, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=SECONDARY_INK)
    ax.set_ylabel(ylabel, color=SECONDARY_INK)
    if legend:
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


def plot_payoff_correlation(arith_payoffs, geo_payoffs, save_path):
    arith_payoffs, geo_payoffs = np.array(arith_payoffs), np.array(geo_payoffs)
    corr = np.corrcoef(arith_payoffs, geo_payoffs)[0, 1]

    fig, ax = _new_axes()
    ax.scatter(arith_payoffs, geo_payoffs, s=10, color=ORANGE, alpha=0.3, zorder=2, linewidths=0)

    upper = max(arith_payoffs.max(), geo_payoffs.max()) * 1.05
    ax.plot([0, upper], [0, upper], color=MUTED_INK, linestyle="--", linewidth=1, zorder=1)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal")

    ax.text(0.05, 0.95, f"ρ = {corr:.4f}", transform=ax.transAxes, ha="left", va="top",
            color=SECONDARY_INK, fontsize=11)

    _finish(fig, ax, "Arithmetic vs. Geometric Payoff (per path)",
            "Arithmetic payoff ($)", "Geometric payoff ($)", save_path, legend=False)


def plot_convergence_comparison(M_values, arith_means, arith_stds, geo_means, geo_stds, save_path):
    arith_means, arith_stds = np.array(arith_means), np.array(arith_stds)
    geo_means, geo_stds = np.array(geo_means), np.array(geo_stds)

    fig, ax = _new_axes()

    ax.fill_between(M_values, arith_means - 1.96 * arith_stds, arith_means + 1.96 * arith_stds,
                     color=BLUE, alpha=0.15, zorder=1)
    ax.plot(M_values, arith_means, color=BLUE, alpha=0.7, marker="o", markersize=5,
            linewidth=2, zorder=3, label="Arithmetic (MC)")

    ax.fill_between(M_values, geo_means - 1.96 * geo_stds, geo_means + 1.96 * geo_stds,
                     color=GREEN, alpha=0.15, zorder=1)
    ax.plot(M_values, geo_means, color=GREEN, alpha=0.7, marker="o", markersize=5,
            linewidth=2, zorder=3, label="Geometric (MC)")

    ax.set_xscale("log")
    _finish(fig, ax, "Monte Carlo Convergence — Arithmetic vs. Geometric Asian Call",
            "Number of simulated paths (M)", "Estimated option price", save_path)


def plot_variance_reduction_combined(M_values, naive_means, naive_stds, agg_means, agg_stds,
                                      path_means, path_stds, anti_means, anti_stds,
                                      S0, K, r, sigma, T, option_type, save_path):
    naive_means, naive_stds = np.array(naive_means), np.array(naive_stds)
    agg_means, agg_stds = np.array(agg_means), np.array(agg_stds)
    path_means, path_stds = np.array(path_means), np.array(path_stds)
    anti_means, anti_stds = np.array(anti_means), np.array(anti_stds)

    # Pathwise has the tightest CI by construction — use its converged (largest-M) estimate as the reference price
    reference_price = path_means[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax1)
    _style_axes(ax2)

    # Left panel: convergence of each estimator to the reference price
    for means, stds, color, label in (
        (naive_means, naive_stds, BLUE, "Naive"),
        (agg_means, agg_stds, GREEN, "Control variate (aggregate)"),
        (path_means, path_stds, ORANGE, "Control variate (pathwise)"),
        (anti_means, anti_stds, VIOLET, "Antithetic variate"),
    ):
        ax1.fill_between(M_values, means - 1.96 * stds, means + 1.96 * stds, color=color, alpha=0.15, zorder=1)
        ax1.plot(M_values, means, color=color, alpha=0.8, marker="o", markersize=5,
                 linewidth=2, zorder=3, label=label)
    ax1.axhline(reference_price, color=MUTED_INK, linestyle="--", linewidth=1.5, zorder=2,
                label=f"Reference ≈ ${reference_price:.4f}")
    ax1.set_xscale("log")
    ax1.set_title("Convergence to Reference Price", color=PRIMARY_INK, fontsize=12, pad=10)
    ax1.set_xlabel("Number of simulated paths (M)", color=SECONDARY_INK)
    ax1.set_ylabel("Estimated price ($)", color=SECONDARY_INK)
    ax1.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=8.5)

    # Right panel: std across repeats at the largest M, naive as baseline
    labels = ["Naive", "Aggregate CV", "Pathwise CV", "Antithetic"]
    stds_at_max = [naive_stds[-1], agg_stds[-1], path_stds[-1], anti_stds[-1]]
    colors = [BLUE, GREEN, ORANGE, VIOLET]
    x = np.arange(len(labels))
    bars = ax2.bar(x, stds_at_max, color=colors, width=0.6, zorder=3, edgecolor=SURFACE, linewidth=1)

    for rect, std, label in zip(bars, stds_at_max, labels):
        if label == "Naive":
            tag = "baseline"
        else:
            pct = (1 - (std / stds_at_max[0]) ** 2) * 100
            tag = f"-{pct:.1f}% var"
        ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                  tag, ha="center", va="bottom", color=SECONDARY_INK, fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color=SECONDARY_INK)
    ax2.set_ylim(0, stds_at_max[0] * 1.25)
    ax2.set_title(f"Std Error at {M_values[-1]:,} Paths", color=PRIMARY_INK, fontsize=12, pad=10)
    ax2.set_ylabel("Std error ($)", color=SECONDARY_INK)

    fig.suptitle(
        f"Variance Reduction Comparison   "
        f"(S0={S0}, K={K}, r={r:.0%}, σ={sigma:.0%}, T={T} yr, {option_type})",
        color=PRIMARY_INK, fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_variance_reduction_process_comparison(M_values, agg_means, agg_stds, path_means, path_stds, save_path):
    agg_means, agg_stds = np.array(agg_means), np.array(agg_stds)
    path_means, path_stds = np.array(path_means), np.array(path_stds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, means, stds, title in (
        (ax1, agg_means, agg_stds, "Process 1: Aggregate"),
        (ax2, path_means, path_stds, "Process 2: Pathwise"),
    ):
        _style_axes(ax)
        ax.fill_between(M_values, means - 1.96 * stds, means + 1.96 * stds,
                         color=ORANGE, alpha=0.15, zorder=1)
        ax.plot(M_values, means, color=ORANGE, alpha=0.8, marker="o", markersize=5,
                linewidth=2, zorder=3)
        ax.set_xscale("log")
        ax.set_title(title, color=PRIMARY_INK, fontsize=12, pad=10)
        ax.set_xlabel("Number of simulated paths (M)", color=SECONDARY_INK)

    ax1.set_ylabel("Estimated option price", color=SECONDARY_INK)
    fig.suptitle("Control Variate Estimator — Aggregate vs. Pathwise", color=PRIMARY_INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
