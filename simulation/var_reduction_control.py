import numpy as np
from simulation.gbm import gbm_sim
from simulation.geometric_asian import geometric_payoff_closed_form
from scipy.stats import norm

# Control using terminal stock price S(T)

def control_variate_stock(S0, mu, sigma, T, N, K, M, option_type='call'):
    discount = np.exp(-mu * T)

    # Array of all simulated paths
    paths = gbm_sim(S0, mu, sigma, T, N, M)

    # Calculate the arithmetic average of each path (excluding the initial price)
        # arithmetic average = 1/N * sum(S_i) for i=1 to N
    ari_avg_prices = 1/N * np.sum(paths[:, 1:], axis=1)

    # Calculate the terminal stock price S(T) for each path
    terminal_prices = paths[:, -1]

    # Calculate theoretical stock price at time T using formula: S(T) = S0 * exp(rT)
    theoretical_stock_price = S0 * np.exp(mu * T)

    # Calculate the payoff for each path
    if(option_type == 'call'):
        arith_sim_payoffs = np.maximum(ari_avg_prices - K, 0)
    else:
        arith_sim_payoffs = np.maximum(K - ari_avg_prices, 0)

    # Covariance Matrix
    cov_matrix = np.cov(arith_sim_payoffs, terminal_prices)
    cov_xy = cov_matrix[0][1]
    var_y = cov_matrix[1][1]

    # Optimal coefficient for control variate
    beta = -cov_xy / var_y

    # Control variate estimate for each path
    control_variate_estimates = arith_sim_payoffs + beta * (terminal_prices - theoretical_stock_price)

    # Return discoutned average of control variate estimates
    return discount * np.mean(control_variate_estimates)

# Control with using European payoff (S(T) - K)+ 
def control_variate_european(S0, mu, sigma, T, N, K, M, option_type='call'):
    discount = np.exp(-mu * T)

    # Array of all simulated paths
    paths = gbm_sim(S0, mu, sigma, T, N, M)

    # Calculate the arithmetic average of each path (excluding the initial price)
        # arithmetic average = 1/N * sum(S_i) for i=1 to N
    ari_avg_prices = 1/N * np.sum(paths[:, 1:], axis=1)

    # Calculate the terminal stock price S(T) for each path
    terminal_prices = paths[:, -1]

    # Calculate the payoff for each path
    if(option_type == 'call'):
        arith_sim_payoffs = np.maximum(ari_avg_prices - K, 0)
        european_payoffs = np.maximum(terminal_prices - K, 0)
    else:
        arith_sim_payoffs = np.maximum(K - ari_avg_prices, 0)
        european_payoffs = np.maximum(K - terminal_prices, 0)

    # Calculate theoretical European option price using Black-Scholes formula
    # S0 must be grown by exp(mu*T) before discounting, else the formula collapses
    # to e^{-rT}(S0*N(d1) - K*N(d2)), which double-discounts the S0 term
    d1 = (np.log(S0 / K) + (mu + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        theoretical_european_price = discount * (S0 * np.exp(mu * T) * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        theoretical_european_price = discount * (K * norm.cdf(-d2) - S0 * np.exp(mu * T) * norm.cdf(-d1))

    # Covariance Matrix
    cov_matrix = np.cov(arith_sim_payoffs, european_payoffs)
    cov_xy = cov_matrix[0][1]
    var_y = cov_matrix[1][1]

    # Optimal coefficient for control variate
    beta = -cov_xy / var_y

    # Control variate estimate for each path (discount before combining -- theoretical_european_price
    # is already a discounted price, so it must be combined with discounted payoffs, not raw ones)
    control_variate_estimates = discount * arith_sim_payoffs + beta * (discount * european_payoffs - theoretical_european_price)

    # Return the average adjusted payoff across all paths
    return np.mean(control_variate_estimates)

# Control using geometric Asian payoff (G(T) - K)+

# Process 1: aggregate control variate 
    # 1. Average all arithmatic and geometric payoffs across all paths
    # 2. Apply the control variate estimator once to these averages to get final price estimate
def control_variate_arithmatic_payoff_aggregate(S0, mu, sigma, T, N, K, M, option_type='call'):
    discount = np.exp(-mu * T)

    # Array of all simulated paths
    paths = gbm_sim(S0, mu, sigma, T, N, M)

    # Calculate the arithmetic average of each path (excluding the initial price)
        # arithmetic average = 1/N * sum(S_i) for i=1 to N
    ari_avg_prices = 1/N * np.sum(paths[:, 1:], axis=1)

    # Calculate the geometric average of each path (excluding the initial price)
        # geometric average = prod(S_i)^(1/N) for i=1 to N
    geo_avg_prices = np.prod(paths[:, 1:], axis=1) ** (1/N)

    # Caclulate the closed-form solution for geometric Asian option (for control variate)
    geo_closed_form_price = geometric_payoff_closed_form(S0, mu, sigma, T, N, K, option_type)

    # Calculate the payoff for each path
    if(option_type == 'call'):
        arith_sim_payoffs = np.maximum(ari_avg_prices - K, 0)
        geo_sim_payoffs = np.maximum(geo_avg_prices - K, 0)
    else:
        arith_sim_payoffs = np.maximum(K - ari_avg_prices, 0)
        geo_sim_payoffs = np.maximum(K - geo_avg_prices, 0)

    # Optimal control variate coefficient, estimated from the per-path payoffs
    cov_matrix = np.cov(arith_sim_payoffs, geo_sim_payoffs)
    beta = -cov_matrix[0][1] / cov_matrix[1][1]

    # Average each payoff series separately first — these are the two standalone
    # (naive) discounted MC prices for the arithmetic and geometric Asian option
    arith_mc_price = discount * np.mean(arith_sim_payoffs)
    geo_mc_price = discount * np.mean(geo_sim_payoffs)

    # Then apply the control variate correction once, at the price level
    return arith_mc_price + beta * (geo_mc_price - geo_closed_form_price)


# Process 2: pathwise control variate 
    # 1. For each stock path i, calculate the arithmetic and geometric average prices and plug into control variate formula to get an adjusted payoff for that path
    # 2. Average the M adjusted payoffs to get the final price estimate
def control_variate_arithmatic_payoff_pathwise(S0, mu, sigma, T, N, K, M, option_type='call'):
    dt = T / N  
    discount = np.exp(-mu * T) 

    # Array of all simulated paths
    paths = gbm_sim(S0, mu, sigma, T, N, M)

    # Calculate the arithmetic average of each path (excluding the initial price)
        # arithmetic average = 1/N * sum(S_i) for i=1 to N
    ari_avg_prices = 1/N * np.sum(paths[:, 1:], axis=1)  
    
    # Calculate the geometric average of each path (excluding the initial price)
        # geometric average = prod(S_i)^(1/N) for i=1 to N
    geo_avg_prices = np.prod(paths[:, 1:], axis=1) ** (1/N)

    # Caclulate the closed-form solution for geometric Asian option (for control variate)
    geo_closed_form_price = geometric_payoff_closed_form(S0, mu, sigma, T, N, K, option_type)

    # Calculate the payoff for each path
    if(option_type == 'call'):
        arith_sim_payoffs = np.maximum(ari_avg_prices - K, 0)
        geo_sim_payoffs = np.maximum(geo_avg_prices - K, 0)
    else:
        arith_sim_payoffs = np.maximum(K - ari_avg_prices, 0)
        geo_sim_payoffs = np.maximum(K - geo_avg_prices, 0)

    # Use closed-form solution as control variate for geometric Asian option
    cov_matrix = np.cov(arith_sim_payoffs, geo_sim_payoffs)
    cov_xy = cov_matrix[0][1]
    var_y = cov_matrix[1][1]
    
    # Optimal coefficient for control variate
    beta = -cov_xy / var_y

    # Adjusted payoffs using control variate (discount to present value first — geo_closed_form_price
    # is already a discounted price, so it must be combined with discounted payoffs, not raw ones)
    adjusted_payoffs = discount * arith_sim_payoffs + beta * (discount * geo_sim_payoffs - geo_closed_form_price)

    # Return the average adjusted payoff across all paths
    return np.mean(adjusted_payoffs)