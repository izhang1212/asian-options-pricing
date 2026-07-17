# asian-options-pricing

Python implementation of Asian option pricing using the Monte Carlo simulation, with variance reduction via control variate

---

## Overview

### What does this do?

This project prices Asian call and put options using two types of averaging: arithmatic averaging and geometric averaging. This program uses Monte-Carlo to simulate many paths, apply these averaging methods, and returns the averaged payoff of an Asian option. We further incorperate variance reduction via control variates to reduce error.

### Purpose

Goal: To use Monte-Carlo to price Asian options. In partiular, since finding Asian option payoff via arithmatic averaging has no closed-form solution, it requires simulation. Since geometric averaging does provide a closed form solution (Kenma-Vorst (1993)), we can test and see how accurate the arithmatic pricing simulation is.

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

```
A = (1/N) * Σ Sᵢ,   i = 1 ... N
```

## Geometric Averaging

Same simulation, but each path is averaged geometrically instead:

```
G = (Π Sᵢ)^(1/N),   i = 1 ... N
```

### Simulation

The Monte Carlo geometric estimate (`geometric_payoff_sim`) averages each path geometrically instead of arithmetically, then computes and discounts the payoff exactly like the arithmetic case. Used to sanity-check the closed form below.

### Closed Form

Since the geometric average of a GBM path is itself lognormal, this case has a closed-form solution (Kemna-Vorst, 1993), implemented in `geometric_payoff_closed_form`. This closed form doubles as the control variate target for variance reduction below.

### Variance Reduction

Arithmetic and geometric averages of the *same* path are highly correlated, so the geometric average's known closed-form price is used as a control variate to reduce the variance of the arithmetic Monte Carlo estimate (`variance_reduction.py`). Two mathematically equivalent implementations are included:

- **Pathwise**: combine each path's arithmetic and geometric payoff with the control variate correction first, then average across paths.
- **Aggregate**: average the arithmetic and geometric payoffs separately first, then apply the correction once.

Both converge to the same price with roughly a 100x–1000x+ reduction in variance versus the naive arithmetic estimator at the same path count.

---

## Output

Running `main.py` regenerates the `output/` directory from scratch with the following plots:

| File | What it shows |
|---|---|
| `sample_paths.png` | A sample of simulated GBM stock price paths plus the mean path — visual sanity check that the simulator behaves like GBM. |
| `payoff_correlation.png` | Per-path arithmetic payoff vs. geometric payoff — the tight linear correlation is why the geometric average makes an effective control variate. |
| `convergence.png` | Arithmetic vs. geometric Monte Carlo price estimates (mean ± 95% CI) as the number of simulated paths grows. |
| `variance_reduction_process_comparison.png` | The pathwise and aggregate control variate estimators plotted side by side, showing they converge to the same price. |
| `variance_reduction_combined.png` | Left: naive vs. control variate convergence to a reference price. Right: standard error at the largest path count, showing the variance reduction achieved. |
