import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Palette (light mode, colorblind-safe categorical order)
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BLUE = "#2a78d6"
GREEN = "#008300"
MAGENTA = "#e87ba4"
AQUA = "#1baf7a"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
RED = "#e34948"


def _lighten(hex_color, amount):
    # amount in [0, 1]: 0 = original color, 1 = white. Used to push noisier (wide-band)
    # series toward a pale shade so tighter, more precise series stay visible on top.
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


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


def plot_payoff_correlation_combined(datasets, save_path):
    # datasets: list of (x_values, x_label, y_values, y_label, title, color, same_scale)
    # Sort by measured correlation ascending -- least correlated top-left, most correlated bottom,
    # matching the classic Glasserman-style layout (2 panels on top, the strongest candidate below)
    scored = []
    for x_values, x_label, y_values, y_label, title, color, same_scale in datasets:
        x_values, y_values = np.array(x_values), np.array(y_values)
        corr = np.corrcoef(x_values, y_values)[0, 1]
        scored.append((corr, x_values, x_label, y_values, y_label, title, color, same_scale))
    scored.sort(key=lambda s: s[0])

    fig = plt.figure(figsize=(11, 10), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 4)
    positions = [gs[0, 0:2], gs[0, 2:4], gs[1, 1:3]]

    for pos, (corr, x_values, x_label, y_values, y_label, title, color, same_scale) in zip(positions, scored):
        ax = fig.add_subplot(pos)
        _style_axes(ax)
        ax.scatter(x_values, y_values, s=10, color=color, alpha=0.3, zorder=2, linewidths=0)

        # The y=x reference diagonal and equal aspect only make sense when both axes are
        # commensurate payoffs (e.g. arithmetic vs. geometric/European) -- not when comparing
        # a payoff against a raw price on a different scale (e.g. terminal stock price)
        if same_scale:
            upper = max(x_values.max(), y_values.max()) * 1.05
            ax.plot([0, upper], [0, upper], color=MUTED_INK, linestyle="--", linewidth=1, zorder=1)
            ax.set_xlim(0, upper)
            ax.set_ylim(0, upper)
            ax.set_aspect("equal")

        ax.text(0.05, 0.95, f"ρ = {corr:.4f}", transform=ax.transAxes, ha="left", va="top",
                color=SECONDARY_INK, fontsize=11)
        ax.set_title(title, color=PRIMARY_INK, fontsize=11, pad=10)
        ax.set_xlabel(x_label, color=SECONDARY_INK)
        ax.set_ylabel(y_label, color=SECONDARY_INK)

    fig.suptitle("Control Variate Candidates — Payoff Correlation (least → most correlated)",
                 color=PRIMARY_INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_convergence_comparison(M_values, arith_means, arith_stds, geo_means, geo_stds, save_path):
    arith_means, arith_stds = np.array(arith_means), np.array(arith_stds)
    geo_means, geo_stds = np.array(geo_means), np.array(geo_stds)

    fig, ax = _new_axes()

    for means, stds, color, label in (
        (arith_means, arith_stds, BLUE, "Arithmetic (MC)"),
        (geo_means, geo_stds, GREEN, "Geometric (MC)"),
    ):
        ax.fill_between(M_values, means - 1.96 * stds, means + 1.96 * stds, color=color, alpha=0.15, zorder=1)
        ax.plot(M_values, means, color=color, alpha=0.7, marker="o", markersize=5,
                linewidth=2, zorder=3, label=label)
        ax.annotate(f"${means[-1]:.4f}", xy=(M_values[-1], means[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", ha="left", color=color,
                    fontsize=9, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(right=M_values[-1] * 2.5)  # room for the direct value labels
    _finish(fig, ax, "Monte Carlo Convergence — Arithmetic vs. Geometric Asian Call",
            "Number of simulated paths (M)", "Estimated option price", save_path)


def plot_control_variate_overlay(M_values, stock_means, stock_stds, euro_means, euro_stds,
                                  agg_means, agg_stds, path_means, path_stds, save_path):
    series = [
        ("Aggregate geometric", np.array(agg_means), np.array(agg_stds), GREEN),
        ("European payoff", np.array(euro_means), np.array(euro_stds), MAGENTA),
        ("Terminal stock S(T)", np.array(stock_means), np.array(stock_stds), AQUA),
        ("Pathwise geometric", np.array(path_means), np.array(path_stds), ORANGE),
    ]

    # Rank by overall variance: the noisiest (widest-band) series draws first/lightest,
    # the most precise (tightest-band) series draws last/darkest and on top -- so a wide
    # pale band never buries a narrow, precise line underneath it
    order = sorted(range(len(series)), key=lambda i: np.mean(series[i][2] ** 2), reverse=True)
    lighten_fracs = np.linspace(0.55, 0.0, len(series))

    fig, ax = _new_axes()
    for rank, idx in enumerate(order):
        label, means, stds, base_color = series[idx]
        color = _lighten(base_color, lighten_fracs[rank])
        ax.fill_between(M_values, means - 1.96 * stds, means + 1.96 * stds, color=color, alpha=0.25, zorder=1 + rank)
        ax.plot(M_values, means, color=color, alpha=0.9, marker="o", markersize=5,
                linewidth=2, zorder=10 + rank, label=label)

    ax.set_xscale("log")
    _finish(fig, ax, "Control Variate Methods — Convergence Comparison",
            "Number of simulated paths (M)", "Estimated option price", save_path)


def plot_variance_reduction_bars(labels, stds, colors, save_path, S0, K, r, sigma, T, option_type):
    stds = np.array(stds)

    # Sort bars by value (descending) -- color stays attached to its entity, not its rank
    order = np.argsort(stds)[::-1]
    labels = [labels[i] for i in order]
    colors = [colors[i] for i in order]
    stds = stds[order]

    # Naive is always the reference, regardless of where it lands after sorting -- a method
    # that comes out noisier than naive (possible at low repeat counts) is shown as an
    # increase rather than mislabeling whichever bar happens to be tallest as "baseline"
    naive_idx = labels.index("Naive")
    naive_std = stds[naive_idx]

    fig, ax = _new_axes()
    x = np.arange(len(labels))
    bars = ax.bar(x, stds, color=colors, width=0.6, zorder=3, edgecolor=SURFACE, linewidth=1)

    for i, (rect, std) in enumerate(zip(bars, stds)):
        if i == naive_idx:
            tag = "baseline"
        else:
            pct = (1 - (std / naive_std) ** 2) * 100
            tag = f"-{pct:.1f}% var" if pct >= 0 else f"+{-pct:.1f}% var"
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                tag, ha="center", va="bottom", color=SECONDARY_INK, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=SECONDARY_INK, rotation=20, ha="right")
    ax.set_ylim(0, stds[0] * 1.25)

    _finish(fig, ax,
            f"Std Error Comparison — All Methods (S0={S0}, K={K}, r={r:.0%}, σ={sigma:.0%}, T={T} yr, {option_type})",
            "", "Std error ($)", save_path, legend=False)


def plot_variance_reduction_all_methods(M_values, naive_means, naive_stds, agg_means, agg_stds,
                                         path_means, path_stds, stock_means, stock_stds,
                                         euro_means, euro_stds, anti_means, anti_stds,
                                         S0, K, r, sigma, T, option_type, save_path):
    # Pathwise has the tightest CI by construction — use its converged (largest-M) estimate as the reference price
    reference_price = np.array(path_means)[-1]

    series = [
        ("Naive", np.array(naive_means), np.array(naive_stds), BLUE),
        ("Aggregate geometric", np.array(agg_means), np.array(agg_stds), GREEN),
        ("European payoff", np.array(euro_means), np.array(euro_stds), MAGENTA),
        ("Terminal stock S(T)", np.array(stock_means), np.array(stock_stds), AQUA),
        ("Pathwise geometric", np.array(path_means), np.array(path_stds), ORANGE),
        ("Antithetic", np.array(anti_means), np.array(anti_stds), VIOLET),
    ]

    # Same rationale as the control-variate overlay: draw the noisiest (widest-band) series
    # first/lightest, the most precise (tightest-band) series last/darkest and on top
    order = sorted(range(len(series)), key=lambda i: np.mean(series[i][2] ** 2), reverse=True)
    lighten_fracs = np.linspace(0.6, 0.0, len(series))

    fig, ax = _new_axes()
    for rank, idx in enumerate(order):
        label, means, stds, base_color = series[idx]
        color = _lighten(base_color, lighten_fracs[rank])
        ax.fill_between(M_values, means - 1.96 * stds, means + 1.96 * stds, color=color, alpha=0.22, zorder=1 + rank)
        ax.plot(M_values, means, color=color, alpha=0.9, marker="o", markersize=5,
                linewidth=2, zorder=10 + rank, label=label)

    ax.axhline(reference_price, color=MUTED_INK, linestyle="--", linewidth=1.5, zorder=20,
               label=f"Reference ≈ ${reference_price:.4f}")
    ax.set_xscale("log")
    ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=8)

    _finish(fig, ax,
            f"Variance Reduction — All Methods (S0={S0}, K={K}, r={r:.0%}, σ={sigma:.0%}, T={T} yr, {option_type})",
            "Number of simulated paths (M)", "Estimated price ($)", save_path, legend=False)


def plot_effective_sample_size(n_ref, reference_price, M_grid, naive_means, naive_stds, panels, save_path):
    # panels: list of (title, color, control_means, control_stds, rho, m_theory). Naive is fixed
    # at n_ref replications; theory says each control should reach the same precision at only
    # m_theory = n_ref*(1-rho^2) replications. Both naive and every control are simulated across
    # the SAME M_grid (with common random numbers upstream), so m_theory is checked against real
    # data, not an extrapolation. Sorted by rho ascending and laid out like the correlation plot:
    # least correlated top-left, most correlated bottom-left, with a shared legend bottom-right.
    from matplotlib.lines import Line2D

    M_grid = np.array(M_grid)
    naive_means = np.array(naive_means)
    naive_stds = np.array(naive_stds)
    scored = sorted(panels, key=lambda p: p[4])

    fig = plt.figure(figsize=(11, 10.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.4)
    positions = [gs[0, 0:2], gs[0, 2:4], gs[1, 0:2]]

    # Zoom the top of the y-axis to where the methods actually settle down (M >= 100) -- at
    # M=5..50 even the controls are barely better than naive, so including those points would
    # zoom back out to nearly the full naive range. The bottom is fixed at 0 (see below), so
    # only the top needs to adapt to the data.
    zoom_mask = M_grid >= 100
    control_half_ranges = [
        np.max(np.abs(np.array(cm)[zoom_mask] - reference_price) + 1.96 * np.array(cs)[zoom_mask])
        for _, _, cm, cs, _, _ in scored
    ]
    y_hi = reference_price + max(control_half_ranges) * 1.6
    y_lo = 0

    x_formatter = mticker.FuncFormatter(lambda x, _: f"{int(round(x)):,}")

    axes = []
    for pos, (title, color, control_means, control_stds, rho, m_theory) in zip(positions, scored):
        ax = fig.add_subplot(pos)
        axes.append(ax)
        _style_axes(ax)
        control_means = np.array(control_means)
        control_stds = np.array(control_stds)

        ax.fill_between(M_grid, naive_means - 1.96 * naive_stds, naive_means + 1.96 * naive_stds,
                         color=BLUE, alpha=0.15, zorder=1)
        ax.plot(M_grid, naive_means, color=BLUE, alpha=0.8, marker="o", markersize=4, linewidth=2, zorder=3)

        ax.fill_between(M_grid, control_means - 1.96 * control_stds, control_means + 1.96 * control_stds,
                         color=color, alpha=0.2, zorder=2)
        ax.plot(M_grid, control_means, color=color, alpha=0.9, marker="o", markersize=4, linewidth=2, zorder=4)

        ax.axvline(n_ref, color=MUTED_INK, linestyle="-", linewidth=1.3, zorder=5)
        ax.axvline(m_theory, color=RED, linestyle=":", linewidth=2, zorder=6)

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(x_formatter)
        ax.set_xlim(M_grid.min(), M_grid.max())  # no left-margin gap -- the first point sits on the y-axis
        ax.set_ylim(y_lo, y_hi)

        ax.set_title(title, color=PRIMARY_INK, fontsize=11, pad=10)
        ax.set_xlabel("Number of simulated paths (M)", color=SECONDARY_INK, fontsize=9)

        m_theory_label = f"{m_theory:.1f}" if m_theory < 1000 else f"{m_theory:,.0f}"
        ax.text(0.5, -0.24,
                f"ρ = {rho:.4f}    1/(1-ρ²) ≈ {1 / (1 - rho**2):,.1f}×    theory target M ≈ {m_theory_label}",
                transform=ax.transAxes, ha="center", va="top", color=SECONDARY_INK, fontsize=8.5, clip_on=False)

    for ax in axes:
        ax.set_ylabel("Estimated option price ($)", color=SECONDARY_INK)

    # One shared legend, bottom-right, naming every color and line used across all three panels
    legend_ax = fig.add_subplot(gs[1, 2:4])
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", linewidth=2, label="Naive"),
        Line2D([0], [0], color=AQUA, marker="o", linewidth=2, label="Control: Terminal Stock S(T)"),
        Line2D([0], [0], color=MAGENTA, marker="o", linewidth=2, label="Control: European Payoff"),
        Line2D([0], [0], color=ORANGE, marker="o", linewidth=2, label="Control: Pathwise Geometric"),
        Line2D([0], [0], color=MUTED_INK, linewidth=1.5, label=f"Naive fixed at N = {n_ref:,}"),
        Line2D([0], [0], color=RED, linestyle=":", linewidth=2, label="Theoretical convergence point for CV"),
    ]
    legend_ax.legend(handles=handles, loc="center", frameon=False, labelcolor=SECONDARY_INK, fontsize=10)

    fig.suptitle(
        f"Effective Sample Size — Naive Fixed at N={n_ref:,} vs. Theoretical Control Target M=N·(1-ρ²)",
        color=PRIMARY_INK, fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
        ax.annotate(f"${means[-1]:.4f}", xy=(M_values[-1], means[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", ha="left", color=ORANGE,
                    fontsize=9, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlim(right=M_values[-1] * 2.5)  # room for the direct value label
        ax.set_title(title, color=PRIMARY_INK, fontsize=12, pad=10)
        ax.set_xlabel("Number of simulated paths (M)", color=SECONDARY_INK)

    ax1.set_ylabel("Estimated option price", color=SECONDARY_INK)
    fig.suptitle("Control Variate Estimator — Aggregate vs. Pathwise", color=PRIMARY_INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
