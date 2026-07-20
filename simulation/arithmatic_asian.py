import numpy as np
from simulation.gbm import gbm_sim

# Solve for asian option price via Monte-Carlo using Stock's arithmatic average 
    # Formula: For a stock path, stock's average price = 1/N * sum(S_i) for i=1 to N
def arithmatic_payoff(S0, mu, sigma, T, N, K, M, option_type='call'):
    dt = T / N  
    discount = np.exp(-mu * T) 

    # Array of all simulated paths
    paths = gbm_sim(S0, mu, sigma, T, N, M)

    # Calculate the arithmetic average of each path (excluding the initial price)
        # arithmetic average = 1/N * sum(S_i) for i=1 to N
    avg_prices = 1/N * np.sum(paths[:, 1:], axis=1)  
    
    # Calculate the payoff for each path
    if(option_type == 'call'):
        payoffs = np.maximum(avg_prices - K, 0)
    else:
        payoffs = np.maximum(K - avg_prices, 0)

    # Return the average payoff across all paths
    return discount * np.mean(payoffs)
