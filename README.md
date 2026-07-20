# asian-options-pricing

Python implementation of Asian option pricing using the Monte Carlo simulation, with variance reduction via control variate

---

## Overview

### What does this do?

This project prices Asian call and put options using two types of averaging: arithmatic averaging and geometric averaging. This program uses Monte-Carlo to simulate many paths, apply these averaging methods, and returns the averaged payoff of an Asian option. We further incorperate variance reduction via control variates to reduce error.

### Purpose

Goal: To use Monte-Carlo to price Asian options. In partiular, since finding Asian option payoff via arithmatic averaging has no closed-form solution, it requires simulation. Since geometric averaging does provide a closed form solution (Kenma-Vorst (1993)), we can test against it to see how accurate our arithmatic pricing simulation is, and even incorperte it for variance reduction.

**Parameters used throughout:**

| Parameter | Value | Description |
|---|---|---|
| S₀ | $50 | Initial stock price |
| K | $50 | Strike price (at-the-money) |
| r | 5% | Risk-free rate (annual) |
| σ | 25% | Volatility (annual) |
| T | 0.5 yr | Time to expiry |

---

### Methodology/Process 

## Arithmatic Averaging

Simulates M GBM stock price paths (`gbm.py`). For each path, the arithmetic average of the N monitored prices is taken, then the call/put payoff is computed and discounted. There is no closed-form solution for the arithmetic average, so the price can only be estimated via Monte Carlo (`arithmatic_asian.py`).

$$
A = \frac{1}{N} \sum_{i=1}^{N} S_i
$$

## Geometric Averaging

Same simulation, but each path is averaged geometrically instead:

$$
G = \left( \prod_{i=1}^{N} S_i \right)^{1/N}
$$

### Simulation

The Monte Carlo geometric estimate (`geometric_payoff_sim`) averages each path geometrically instead of arithmetically, then computes and discounts the payoff exactly like the arithmetic case. Used to sanity-check the closed form below.

### Closed Form

Since the geometric average of a GBM path is itself lognormal, this case has a closed-form solution (Kemna-Vorst, 1993), implemented in `geometric_payoff_closed_form`. This closed form doubles as the control variate target for variance reduction below.

### Variance Reduction

## Control Variates
Arithmetic and geometric averages of the *same* path are highly correlated, so the geometric average's known closed-form price is used as a control variate to reduce the variance of the arithmetic Monte Carlo estimate (`variance_reduction.py`). 

**Terminal Stock Price**: use respective terminal stock price as control variate. Use respective simualted terminal stock prices and find average S(T). Then, solve theoretical stock price E[S(T)] = S0 * exp(rT). Finally, compute control variate estimator.
**European Option**: use respective European option price as control variate. Simulate various European option payouts and find average V(T). Then, solve theoretical price via Black-Scholes and get E[V(T)]. Finally, compute control variate estimator.
- **Geometric Asian (Pathwise)**: combine each path's arithmetic and geometric payoff with the control variate correction first, then average across paths.
- **Geometric Asian (Aggregate)**: average the arithmetic and geometric payoffs separately first, then apply the correction once.

## Antithetic Variate

For each standard normal draw `Z` used to simulate a path, a mirror path is built from `-Z` instead. Since `Z` and `-Z` are perfectly negatively correlated, the two paths' payoffs tend to err in opposite directions, so averaging each `(path, mirror path)` pair before averaging across pairs cancels out some of the sampling noise.

Issue: `gbm_sim` can't produce this pair; it draws its own random numbers internally and never exposes them, so calling it twice just gives two independent path sets, not a path and its mirror image. 
Fix: to fix, we build both paths from one shared array of draws, applying `+Z` and `-Z` to the same GBM recursion. This gives a modest, real variance reduction (although, it is weaker than our control variate since it still relys on underlying randomness, and does not rely on near-perfect coorelation with known closed-form) 

---

## Output

Running `main.py` regenerates the `output/` directory from scratch with the following plots:

| File | What it shows |
|---|---|
| `1_sample_paths.png` | A sample of simulated GBM stock price paths plus the mean path |
| `2_correlation.png` | Graphs that map the coorelation between arithmatic payout to controls (Terminal stock price, European option payout, Geometric Asian payout); higher coorelation makes a better control variate |
| `3_convergence.png` | Arithmetic vs. geometric Monte Carlo price estimates (mean ± 95% CI) as the number of simulated paths grows |
| `4_variance_reduction_process_comparison.png` | The pathwise and aggregate control variate estimators plotted side by side, showing they are theoretically the same (converge to the same price) |
| `5_control_variate_overlay.png` | Overlapping convergence of each control variate method as number of simualated paths grows; better controls converge faster |
| `6_effective_sample_size.png` | Given fixed number of naive simulations, indicates when control variate monte-carlo should theoretically converge based on coorelation with control variate |
| `7_variance_reduction_bars.png` | Compare standard error between all monte-carlo simulation methods, using naive as a base line and testing variance reduction of remaning methods against it |
| `8_variance_reduction_all_methods.png` | Overlap all monte-carlo simulations to see how they all converge compared to each other as number of simulations grows |
