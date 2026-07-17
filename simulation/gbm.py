import numpy as np

def sim_one_path(S0, mu, sigma, T, N):
    dt = T / N  
    t = np.linspace(0, T, N + 1)  
    # Array to hold simulated stock prices
    S = np.zeros(N + 1)  
    # Set initial stock price
    S[0] = S0  

    for i in range(1, N + 1):
        # Brownian motion increment dw = sqrt(dt) * Z, where Z ~ N(0,1)
        dW = np.random.normal(0, np.sqrt(dt))  
        # Increment the stock price using the GBM formula
        S[i] = S[i - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)  

    # return the simulated stock path
    return S

def gbm_sim(S0, mu, sigma, T, N, M):
    # Array to hold all simulated paths
    paths = np.zeros((M, N + 1))  
    # Simulate M single paths and add to paths array
    for j in range(M):
        paths[j] = sim_one_path(S0, mu, sigma, T, N)  

    # return matrix 
        # each row is a simulated path, each column is stock price at a given time step
    return paths