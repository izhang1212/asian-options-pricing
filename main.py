import os
import shutil

import numpy as np
from scipy.stats import norm

from config import S0, K, r, SIGMA, T
from simulation.gbm import gbm_sim
from simulation.arithmatic_asian import arithmatic_payoff
from simulation.geometric_asian import geometric_payoff_sim, geometric_payoff_closed_form
from simulation.var_reduction_control import (
    control_variate_arithmatic_payoff_pathwise,
    control_variate_arithmatic_payoff_aggregate,
    control_variate_stock,
    control_variate_european,
)
from simulation.var_reduction_antithetic import variance_reduction_arithmatic_payoff_antithetic
from visualization import plots

OUTPUT_DIR = "output"

N_STEPS = 126           # ~daily monitoring over a 6-month expiry
M_DISPLAY = 30          # paths drawn on the sample-path plot
M_CORRELATION = 2000    # paths used for the payoff-correlation scatter plots
M_PRICE = 50000         # paths used for the headline price estimates
M_CONVERGENCE = [100, 500, 1000, 5000, 10000, 50000]
CONVERGENCE_REPEATS = 5
M_PRICE_REPEATS = 20    # higher-repeat, low-noise estimate at M_PRICE, used only by the bar chart

# Effective sample size study: fix naive at N_REF replications, then check whether each control
# variate's ACTUAL variance at the THEORETICAL target M = N_REF*(1-rho^2) matches naive's.
# N_REF is intentionally separate from M_PRICE/M_CONVERGENCE -- it only scopes this one plot.
N_REF = 100000
M_EFFECTIVE_GRID = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
EFFECTIVE_REPEATS = 30


def _paired_prices(S0, mu, sigma, T, N, K, M):
    # Naive and all three controls computed from the SAME simulated batch (common random
    # numbers) so the curves in the effective-sample-size plot are a fair, paired comparison --
    # not an artifact of each method drawing its own independent random paths at small M.
    discount = np.exp(-mu * T)
    paths = gbm_sim(S0, mu, sigma, T, N, M)

    ari_avg = 1 / N * np.sum(paths[:, 1:], axis=1)
    geo_avg = np.prod(paths[:, 1:], axis=1) ** (1 / N)
    terminal = paths[:, -1]

    arith_payoffs = np.maximum(ari_avg - K, 0)
    geo_payoffs = np.maximum(geo_avg - K, 0)
    euro_payoffs = np.maximum(terminal - K, 0)

    naive_price = discount * np.mean(arith_payoffs)

    # At very small M it's possible for every sampled path to land out-of-the-money, giving a
    # control payoff with zero sample variance -- beta is undefined there, so fall back to no
    # correction (beta=0) for that draw rather than dividing by zero into a NaN
    def _beta(cov):
        return -cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0

    theo_stock = S0 * np.exp(mu * T)
    beta_s = _beta(np.cov(arith_payoffs, terminal))
    stock_price = discount * np.mean(arith_payoffs + beta_s * (terminal - theo_stock))

    d1 = (np.log(S0 / K) + (mu + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theo_euro = discount * (S0 * np.exp(mu * T) * norm.cdf(d1) - K * norm.cdf(d2))
    beta_e = _beta(np.cov(arith_payoffs, euro_payoffs))
    euro_price = np.mean(discount * arith_payoffs + beta_e * (discount * euro_payoffs - theo_euro))

    geo_closed_form = geometric_payoff_closed_form(S0, mu, sigma, T, N, K, "call")
    beta_g = _beta(np.cov(arith_payoffs, geo_payoffs))
    path_price = np.mean(discount * arith_payoffs + beta_g * (discount * geo_payoffs - geo_closed_form))

    return naive_price, stock_price, euro_price, path_price


def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print("=== Asian Option Pricing via Monte Carlo ===")
    print(f"S0={S0}, K={K}, r={r}, sigma={SIGMA}, T={T}, N_STEPS={N_STEPS}")

    # Sanity-check the GBM engine visually
    display_paths = gbm_sim(S0, r, SIGMA, T, N_STEPS, M_DISPLAY)
    plots.plot_sample_paths(display_paths, T, os.path.join(OUTPUT_DIR, "1_sample_paths.png"))

    # Payoff correlation: why each control variate correlates with the arithmetic Asian payoff
    corr_paths = gbm_sim(S0, r, SIGMA, T, N_STEPS, M_CORRELATION)
    corr_arith_avg = 1 / N_STEPS * np.sum(corr_paths[:, 1:], axis=1)
    corr_geo_avg = np.prod(corr_paths[:, 1:], axis=1) ** (1 / N_STEPS)
    corr_terminal_prices = corr_paths[:, -1]
    corr_arith_payoffs = np.maximum(corr_arith_avg - K, 0)
    corr_geo_payoffs = np.maximum(corr_geo_avg - K, 0)
    corr_european_payoffs = np.maximum(corr_terminal_prices - K, 0)

    plots.plot_payoff_correlation_combined(
        [
            (corr_arith_payoffs, "Arithmetic payoff ($)", corr_geo_payoffs, "Geometric payoff ($)",
             "Arithmetic vs. Geometric", plots.ORANGE, True),
            (corr_arith_payoffs, "Arithmetic payoff ($)", corr_terminal_prices, "Terminal stock price S(T) ($)",
             "Arithmetic vs. Terminal Stock Price", plots.AQUA, False),
            (corr_arith_payoffs, "Arithmetic payoff ($)", corr_european_payoffs, "European payoff ($)",
             "Arithmetic vs. European", plots.MAGENTA, True),
        ],
        os.path.join(OUTPUT_DIR, "2_correlation.png"),
    )

    # rho feeds the effective-sample-size demonstration plot below: 1/(1-rho^2) is how many
    # times fewer simulations each control variate needs versus naive MC
    rho_geo = np.corrcoef(corr_arith_payoffs, corr_geo_payoffs)[0, 1]
    rho_stock = np.corrcoef(corr_arith_payoffs, corr_terminal_prices)[0, 1]
    rho_euro = np.corrcoef(corr_arith_payoffs, corr_european_payoffs)[0, 1]

    # Headline option price estimates across every method
    arith_price = arithmatic_payoff(S0, r, SIGMA, T, N_STEPS, K, M_PRICE, option_type="call")
    geo_price = geometric_payoff_sim(S0, r, SIGMA, T, N_STEPS, K, M_PRICE, option_type="call")
    cv_price = control_variate_arithmatic_payoff_pathwise(S0, r, SIGMA, T, N_STEPS, K, M_PRICE, option_type="call")
    stock_cv_price = control_variate_stock(S0, r, SIGMA, T, N_STEPS, K, M_PRICE, option_type="call")
    euro_cv_price = control_variate_european(S0, r, SIGMA, T, N_STEPS, K, M_PRICE, option_type="call")
    antithetic_price = variance_reduction_arithmatic_payoff_antithetic(S0, r, SIGMA, T, N_STEPS, K, M_PRICE, option_type="call")
    print(f"\nArithmetic Asian call price     (M={M_PRICE}): {arith_price:.4f}")
    print(f"Geometric Asian call price      (M={M_PRICE}): {geo_price:.4f}")
    print(f"Control variate (geometric)      (M={M_PRICE}): {cv_price:.4f}")
    print(f"Control variate (stock S(T))     (M={M_PRICE}): {stock_cv_price:.4f}")
    print(f"Control variate (European)       (M={M_PRICE}): {euro_cv_price:.4f}")
    print(f"Antithetic variate               (M={M_PRICE}): {antithetic_price:.4f}")

    # Convergence study: repeat at each M to see Monte Carlo noise shrink
    arith_means, arith_stds = [], []
    geo_means, geo_stds = [], []
    for M in M_CONVERGENCE:
        arith_estimates = [
            arithmatic_payoff(S0, r, SIGMA, T, N_STEPS, K, M, option_type="call")
            for _ in range(CONVERGENCE_REPEATS)
        ]
        geo_estimates = [
            geometric_payoff_sim(S0, r, SIGMA, T, N_STEPS, K, M, option_type="call")
            for _ in range(CONVERGENCE_REPEATS)
        ]
        arith_means.append(np.mean(arith_estimates))
        arith_stds.append(np.std(arith_estimates))
        geo_means.append(np.mean(geo_estimates))
        geo_stds.append(np.std(geo_estimates))
        print(f"M={M:>7}: arithmetic={arith_means[-1]:.4f} (std {arith_stds[-1]:.4f})  "
              f"geometric={geo_means[-1]:.4f} (std {geo_stds[-1]:.4f})")

    plots.plot_convergence_comparison(M_CONVERGENCE, arith_means, arith_stds, geo_means, geo_stds,
                                       os.path.join(OUTPUT_DIR, "3_convergence.png"))

    # Variance reduction study: naive arithmetic vs. control variate (pathwise and aggregate) at the same M values
    path_means, path_stds = [], []
    agg_means, agg_stds = [], []
    for i, M in enumerate(M_CONVERGENCE):
        path_estimates = [
            control_variate_arithmatic_payoff_pathwise(S0, r, SIGMA, T, N_STEPS, K, M, option_type="call")
            for _ in range(CONVERGENCE_REPEATS)
        ]
        agg_estimates = [
            control_variate_arithmatic_payoff_aggregate(S0, r, SIGMA, T, N_STEPS, K, M, option_type="call")
            for _ in range(CONVERGENCE_REPEATS)
        ]
        path_means.append(np.mean(path_estimates))
        path_stds.append(np.std(path_estimates))
        agg_means.append(np.mean(agg_estimates))
        agg_stds.append(np.std(agg_estimates))
        factor = (arith_stds[i] / path_stds[-1]) ** 2
        print(f"M={M:>7}: pathwise={path_means[-1]:.4f} (std {path_stds[-1]:.4f})  "
              f"aggregate={agg_means[-1]:.4f} (std {agg_stds[-1]:.4f})  "
              f"variance reduction factor={factor:,.0f}x")

    # Stock-price control variate study, at the same M values
    stock_means, stock_stds = [], []
    for i, M in enumerate(M_CONVERGENCE):
        stock_estimates = [
            control_variate_stock(S0, r, SIGMA, T, N_STEPS, K, M, option_type="call")
            for _ in range(CONVERGENCE_REPEATS)
        ]
        stock_means.append(np.mean(stock_estimates))
        stock_stds.append(np.std(stock_estimates))
        factor = (arith_stds[i] / stock_stds[-1]) ** 2
        print(f"M={M:>7}: stock CV={stock_means[-1]:.4f} (std {stock_stds[-1]:.4f})  "
              f"variance reduction factor={factor:,.2f}x")

    # European-payoff control variate study, at the same M values
    euro_means, euro_stds = [], []
    for i, M in enumerate(M_CONVERGENCE):
        euro_estimates = [
            control_variate_european(S0, r, SIGMA, T, N_STEPS, K, M, option_type="call")
            for _ in range(CONVERGENCE_REPEATS)
        ]
        euro_means.append(np.mean(euro_estimates))
        euro_stds.append(np.std(euro_estimates))
        factor = (arith_stds[i] / euro_stds[-1]) ** 2
        print(f"M={M:>7}: european CV={euro_means[-1]:.4f} (std {euro_stds[-1]:.4f})  "
              f"variance reduction factor={factor:,.2f}x")

    # Antithetic variate study, at the same M values
    anti_means, anti_stds = [], []
    for i, M in enumerate(M_CONVERGENCE):
        anti_estimates = [
            variance_reduction_arithmatic_payoff_antithetic(S0, r, SIGMA, T, N_STEPS, K, M, option_type="call")
            for _ in range(CONVERGENCE_REPEATS)
        ]
        anti_means.append(np.mean(anti_estimates))
        anti_stds.append(np.std(anti_estimates))
        factor = (arith_stds[i] / anti_stds[-1]) ** 2
        print(f"M={M:>7}: antithetic={anti_means[-1]:.4f} (std {anti_stds[-1]:.4f})  "
              f"variance reduction factor={factor:,.2f}x")

    plots.plot_variance_reduction_process_comparison(M_CONVERGENCE, agg_means, agg_stds, path_means, path_stds,
                                                       os.path.join(OUTPUT_DIR, "4_variance_reduction_process_comparison.png"))
    plots.plot_control_variate_overlay(M_CONVERGENCE, stock_means, stock_stds, euro_means, euro_stds,
                                        agg_means, agg_stds, path_means, path_stds,
                                        os.path.join(OUTPUT_DIR, "5_control_variate_overlay.png"))

    # Effective sample size: fix naive at N_REF replications. Theory says each control variate
    # should need only N_REF*(1-rho^2) replications to match naive's precision at N_REF -- a
    # much smaller number than N_REF itself, so (unlike an M_PRICE-anchored version) it's cheap
    # to actually simulate every method across the same M_EFFECTIVE_GRID and check directly
    # whether each control's real variance has already caught up by its theoretical M.
    m_theory_stock = N_REF * (1 - rho_stock**2)
    m_theory_euro = N_REF * (1 - rho_euro**2)
    m_theory_geo = N_REF * (1 - rho_geo**2)

    naive_eff_means, naive_eff_stds = [], []
    stock_eff_means, stock_eff_stds = [], []
    euro_eff_means, euro_eff_stds = [], []
    path_eff_means, path_eff_stds = [], []
    for M in M_EFFECTIVE_GRID:
        naive_reps, stock_reps, euro_reps, path_reps = [], [], [], []
        for _ in range(EFFECTIVE_REPEATS):
            n, s, e, p = _paired_prices(S0, r, SIGMA, T, N_STEPS, K, M)
            naive_reps.append(n); stock_reps.append(s); euro_reps.append(e); path_reps.append(p)
        naive_eff_means.append(np.mean(naive_reps)); naive_eff_stds.append(np.std(naive_reps))
        stock_eff_means.append(np.mean(stock_reps)); stock_eff_stds.append(np.std(stock_reps))
        euro_eff_means.append(np.mean(euro_reps)); euro_eff_stds.append(np.std(euro_reps))
        path_eff_means.append(np.mean(path_reps)); path_eff_stds.append(np.std(path_reps))

    naive_ref_std = naive_eff_stds[M_EFFECTIVE_GRID.index(N_REF)]
    print(f"\nEffective sample size: naive at N_REF={N_REF:,} has std={naive_ref_std:.5f} "
          f"(~1/{1 / naive_ref_std:,.0f})")
    print(f"  stock:      rho={rho_stock:.4f}  theory: CV needs M~{m_theory_stock:,.1f}")
    print(f"  european:   rho={rho_euro:.4f}  theory: CV needs M~{m_theory_euro:,.1f}")
    print(f"  geometric:  rho={rho_geo:.4f}  theory: CV needs M~{m_theory_geo:,.1f}")

    # arith_price (the M_PRICE=50,000 headline estimate) is the most reliable single price
    # estimate available, used as the reference line every panel should converge toward
    plots.plot_effective_sample_size(
        N_REF, arith_price, M_EFFECTIVE_GRID, naive_eff_means, naive_eff_stds,
        [
            ("Terminal Stock S(T)", plots.AQUA, stock_eff_means, stock_eff_stds, rho_stock, m_theory_stock),
            ("European Payoff", plots.MAGENTA, euro_eff_means, euro_eff_stds, rho_euro, m_theory_euro),
            ("Pathwise Geometric", plots.ORANGE, path_eff_means, path_eff_stds, rho_geo, m_theory_geo),
        ],
        os.path.join(OUTPUT_DIR, "6_effective_sample_size.png"),
    )

    # Bar chart needs a low-noise comparison at M_PRICE -- CONVERGENCE_REPEATS=5 is fine for
    # showing the shape of convergence across M_CONVERGENCE, but too few repeats to reliably
    # rank methods whose real difference is modest (a single unlucky repeat can make a genuinely
    # variance-reducing method look worse than naive purely by chance)
    def _price_and_std(fn, M):
        estimates = [fn(S0, r, SIGMA, T, N_STEPS, K, M, option_type="call") for _ in range(M_PRICE_REPEATS)]
        return np.mean(estimates), np.std(estimates)

    _, naive_bar_std = _price_and_std(arithmatic_payoff, M_PRICE)
    _, agg_bar_std = _price_and_std(control_variate_arithmatic_payoff_aggregate, M_PRICE)
    _, path_bar_std = _price_and_std(control_variate_arithmatic_payoff_pathwise, M_PRICE)
    _, stock_bar_std = _price_and_std(control_variate_stock, M_PRICE)
    _, euro_bar_std = _price_and_std(control_variate_european, M_PRICE)
    _, anti_bar_std = _price_and_std(variance_reduction_arithmatic_payoff_antithetic, M_PRICE)
    print(f"\nBar chart reference (M={M_PRICE}, {M_PRICE_REPEATS} repeats): "
          f"naive={naive_bar_std:.5f} agg={agg_bar_std:.5f} path={path_bar_std:.5f} "
          f"stock={stock_bar_std:.5f} euro={euro_bar_std:.5f} antithetic={anti_bar_std:.5f}")

    labels = ["Naive", "Aggregate CV", "Pathwise CV", "Stock CV", "European CV", "Antithetic"]
    stds_at_max = [naive_bar_std, agg_bar_std, path_bar_std, stock_bar_std, euro_bar_std, anti_bar_std]
    colors = [plots.BLUE, plots.GREEN, plots.ORANGE, plots.AQUA, plots.MAGENTA, plots.VIOLET]
    plots.plot_variance_reduction_bars(labels, stds_at_max, colors,
                                        os.path.join(OUTPUT_DIR, "7_variance_reduction_bars.png"),
                                        S0, K, r, SIGMA, T, "call")

    plots.plot_variance_reduction_all_methods(M_CONVERGENCE, arith_means, arith_stds, agg_means, agg_stds,
                                               path_means, path_stds, stock_means, stock_stds,
                                               euro_means, euro_stds, anti_means, anti_stds,
                                               S0, K, r, SIGMA, T, "call",
                                               os.path.join(OUTPUT_DIR, "8_variance_reduction_all_methods.png"))

    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
