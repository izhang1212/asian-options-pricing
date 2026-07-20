import os
import shutil

import numpy as np

from config import S0, K, r, SIGMA, T
from simulation.gbm import gbm_sim
from simulation.arithmatic_asian import arithmatic_payoff
from simulation.geometric_asian import geometric_payoff_sim
from simulation.variance_reduction import (
    variance_reduction_arithmatic_payoff_pathwise,
    variance_reduction_arithmatic_payoff_aggregate,
    variance_reduction_arithmatic_payoff_antithetic
)
from visualization import plots

OUTPUT_DIR = "output"

N_STEPS = 126           # ~daily monitoring over a 6-month expiry
M_DISPLAY = 30          # paths drawn on the sample-path plot
M_CORRELATION = 2000    # paths used for the payoff-correlation scatter plot
M_PRICE = 50000         # paths used for the headline price estimates
M_CONVERGENCE = [100, 500, 1000, 5000, 10000, 50000]
CONVERGENCE_REPEATS = 5


def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print("=== Asian Option Pricing via Monte Carlo ===")
    print(f"S0={S0}, K={K}, r={r}, sigma={SIGMA}, T={T}, N_STEPS={N_STEPS}")

    # Sanity-check the GBM engine visually
    display_paths = gbm_sim(S0, r, SIGMA, T, N_STEPS, M_DISPLAY)
    plots.plot_sample_paths(display_paths, T, os.path.join(OUTPUT_DIR, "sample_paths.png"))

    # Payoff correlation: why the geometric average makes a good control variate for the arithmetic one
    corr_paths = gbm_sim(S0, r, SIGMA, T, N_STEPS, M_CORRELATION)
    corr_arith_avg = 1 / N_STEPS * np.sum(corr_paths[:, 1:], axis=1)
    corr_geo_avg = np.prod(corr_paths[:, 1:], axis=1) ** (1 / N_STEPS)
    corr_arith_payoffs = np.maximum(corr_arith_avg - K, 0)
    corr_geo_payoffs = np.maximum(corr_geo_avg - K, 0)
    plots.plot_payoff_correlation(corr_arith_payoffs, corr_geo_payoffs,
                                   os.path.join(OUTPUT_DIR, "payoff_correlation.png"))

    # Headline arithmetic and geometric Asian call prices
    arith_price = arithmatic_payoff(S0, r, SIGMA, T, N_STEPS, K, M_PRICE, option_type="call")
    geo_price = geometric_payoff_sim(S0, r, SIGMA, T, N_STEPS, K, M_PRICE, option_type="call")
    cv_price = variance_reduction_arithmatic_payoff_pathwise(S0, r, SIGMA, T, N_STEPS, K, M_PRICE, option_type="call")
    antithetic_price = variance_reduction_arithmatic_payoff_antithetic(S0, r, SIGMA, T, N_STEPS, K, M_PRICE, option_type="call")
    print(f"\nArithmetic Asian call price (M={M_PRICE}): {arith_price:.4f}")
    print(f"Geometric Asian call price  (M={M_PRICE}): {geo_price:.4f}")
    print(f"Control variate call price  (M={M_PRICE}): {cv_price:.4f}")

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
                                       os.path.join(OUTPUT_DIR, "convergence.png"))

    # Variance reduction study: naive arithmetic vs. control variate (pathwise and aggregate) at the same M values
    path_means, path_stds = [], []
    agg_means, agg_stds = [], []
    for i, M in enumerate(M_CONVERGENCE):
        path_estimates = [
            variance_reduction_arithmatic_payoff_pathwise(S0, r, SIGMA, T, N_STEPS, K, M, option_type="call")
            for _ in range(CONVERGENCE_REPEATS)
        ]
        agg_estimates = [
            variance_reduction_arithmatic_payoff_aggregate(S0, r, SIGMA, T, N_STEPS, K, M, option_type="call")
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
                                                       os.path.join(OUTPUT_DIR, "variance_reduction_process_comparison.png"))
    plots.plot_variance_reduction_combined(M_CONVERGENCE, arith_means, arith_stds, agg_means, agg_stds,
                                            path_means, path_stds, anti_means, anti_stds,
                                            S0, K, r, SIGMA, T, "call",
                                            os.path.join(OUTPUT_DIR, "variance_reduction_combined.png"))

    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
