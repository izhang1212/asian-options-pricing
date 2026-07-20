import numpy as np
from simulation.gbm import gbm_sim
from scipy.stats import norm

# Solve for asian option price via Monte-Carlo using Stock's geometric average
    # Formula: For a stock path, stock's average price = prod(S_i)^(1/N
def geometric_payoff_sim(S0, mu, sigma, T, N, K, M, option_type='call'):
    dt = T / N  
    discount = np.exp(-mu * T) 

    # Array of all simulated paths
    paths = gbm_sim(S0, mu, sigma, T, N, M)

    # Calculate the geometric average of each path (excluding the initial price)
        # geometric average = prod(S_i)^(1/N) for i=1 to N
    geo_avg_prices = np.prod(paths[:, 1:], axis=1) ** (1/N)

    # Calculate the payoff for each path
    if(option_type == 'call'):
        payoffs = np.maximum(geo_avg_prices - K, 0)
    else:
        payoffs = np.maximum(K - geo_avg_prices, 0)

    # Return the average payoff across all paths
    return discount * np.mean(payoffs)

# geometric Asian option closed-form solution (for comparison)
    # Kenma-Vorst (1993) formula: S0 * exp((mu - 0.5 * sigma^2) * T) * N(d1) - K * exp(-rT) * N(d2), where we adjust mu and sigma for the geometric average
def geometric_payoff_closed_form(S0, mu, sigma, T, N, K, option_type='call'):
    dt = T / N  
    discount = np.exp(-mu * T) 

    # Adjusted parameters for geometric Asian option
    mu_g = (mu - 0.5 * sigma**2) * (N + 1) / (2 * N)
    sigma_g = sigma * np.sqrt((N + 1) * (2 * N + 1) / (6 * N**2))

    d1 = (np.log(S0 / K) + (mu_g + 0.5 * sigma_g**2) * T) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)

    if option_type == 'call':
        price = discount * (S0 * np.exp((mu_g + 0.5 * sigma_g**2) * T) * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = discount * (K * norm.cdf(-d2) - S0 * np.exp((mu_g + 0.5 * sigma_g**2) * T) * norm.cdf(-d1))

    return price