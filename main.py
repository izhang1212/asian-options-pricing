import os

import numpy as np

from config import S0, K, r, SIGMA, T
from simulation.gbm import gbm_sim
from simulation.arithmatic_asian import arithmatic_payoff
from visualization import plots

OUTPUT_DIR = "output"

N_STEPS = 126           # ~daily monitoring over a 6-month expiry
M_DISPLAY = 30          # paths drawn on the sample-path plot
M_DIST = 20000          # paths used to sanity-check the terminal distribution
M_PRICE = 50000         # paths used for the headline price estimate
M_CONVERGENCE = [100, 500, 1000, 5000, 10000, 50000, 100000]
CONVERGENCE_REPEATS = 5


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Asian Option Pricing via Monte Carlo ===")
    print(f"S0={S0}, K={K}, r={r}, sigma={SIGMA}, T={T}, N_STEPS={N_STEPS}")

    # Sanity-check the GBM engine visually
    display_paths = gbm_sim(S0, r, SIGMA, T, N_STEPS, M_DISPLAY)
    plots.plot_sample_paths(display_paths, T, os.path.join(OUTPUT_DIR, "sample_paths.png"))

    # Terminal distribution vs. theoretical lognormal
    dist_paths = gbm_sim(S0, r, SIGMA, T, N_STEPS, M_DIST)
    plots.plot_terminal_distribution(dist_paths, S0, r, SIGMA, T,
                                      os.path.join(OUTPUT_DIR, "terminal_distribution.png"))

    # Headline arithmetic Asian call price
    price = arithmatic_payoff(S0, r, SIGMA, T, N_STEPS, K, M_PRICE, option_type="call")
    print(f"\nArithmetic Asian call price (M={M_PRICE}): {price:.4f}")

    # Convergence study: repeat at each M to see Monte Carlo noise shrink
    means, stds = [], []
    for M in M_CONVERGENCE:
        estimates = [
            arithmatic_payoff(S0, r, SIGMA, T, N_STEPS, K, M, option_type="call")
            for _ in range(CONVERGENCE_REPEATS)
        ]
        means.append(np.mean(estimates))
        stds.append(np.std(estimates))
        print(f"M={M:>7}: price={means[-1]:.4f}  (std across {CONVERGENCE_REPEATS} repeats: {stds[-1]:.4f})")

    plots.plot_convergence(M_CONVERGENCE, means, stds, os.path.join(OUTPUT_DIR, "convergence.png"))

    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
