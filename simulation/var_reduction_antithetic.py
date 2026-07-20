import numpy as np
from simulation.gbm import gbm_sim
from simulation.geometric_asian import geometric_payoff_closed_form

def variance_reduction_arithmatic_payoff_antithetic(S0, mu, sigma, T, N, K, M, option_type='call'):
    dt = T / N
    discount = np.exp(-mu * T)

    # One shared set of standard normal draws for the whole batch: shape (M, N)
    Z = np.random.standard_normal((M, N))
    dW = np.sqrt(dt) * Z
    drift = (mu - 0.5 * sigma**2) * dt

    # Same GBM recursion as sim_one_path, applied to the whole batch at once, for +Z and -Z
    paths = S0 * np.exp(np.cumsum(drift + sigma * dW, axis=1))
    antithetic_paths = S0 * np.exp(np.cumsum(drift - sigma * dW, axis=1))

    # Calculate the arithmetic average of each path
    ari_avg_prices = np.mean(paths, axis=1)
    antithetic_ari_avg_prices = np.mean(antithetic_paths, axis=1)

    # Calculate the payoff for each path
    if(option_type == 'call'):
        arith_sim_payoffs = np.maximum(ari_avg_prices - K, 0)
        antithetic_arith_sim_payoffs = np.maximum(antithetic_ari_avg_prices - K, 0)
    else:
        arith_sim_payoffs = np.maximum(K - ari_avg_prices, 0)
        antithetic_arith_sim_payoffs = np.maximum(K - antithetic_ari_avg_prices, 0)

    # Average each pair of paths (original, antithetic)
    combined_payoffs = (arith_sim_payoffs + antithetic_arith_sim_payoffs) / 2

    # Average combined payoffs across all paths and discount
    return discount * np.mean(combined_payoffs)