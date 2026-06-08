# THE CRUCIBLE PROTOCOL (v1.0)

> **"If it survives The Crucible, it might just survive reality."**

This document establishes the absolute, unbreakable laws for the simulation engine and backtesting environment. It adheres strictly to QUANT ENGINE v4.1 rules (B3, B4, B11) to discover the true Out-of-Sample Alpha ceiling under extreme hardware constraints (ARM 4C 24GB).

## 1. Zero-Tolerance Bias Prevention
### 1.1 Lookahead Bias
*   **Mandatory Shifting**: All signals must be explicitly `shift(1)` against the reference price axis. Any unshifted data alignment will trigger an immediate system halt.
*   **WFO State Isolation**: Walk-Forward Optimization (WFO) folds must enforce an absolute `warm_up_window` cutoff. No state can leak across folds. The first prediction day of a test set must independently infer from `[T - warm_up_window, T-1]`.

### 1.2 Survivorship Bias
*   **Universe Dynamic Membership**: The asset universe must reconstruct historical membership at time $T$. Point-in-time definitions must include delisted, bankrupt, and merged entities.
*   **The "08-09-15" Test**: Any strategy failing to process the Lehman bankruptcy event gracefully (e.g., hanging on missing data) is instantly rejected.

## 2. Microstructure & Regime Friction
### 2.1 Dynamic Slippage Modeling
*   Slippage is not a static constant. It is modeled as a function of instantaneous volatility (e.g., ATR) and trading volume.
*   **Crisis Multipliers**: Slippage multipliers are applied automatically during detected VIX spikes (e.g., 2008H2, 2011H2, 2020-03).
*   **Liquidity Capacity Cap**: Daily execution volume for any single asset cannot exceed 10% of its real daily traded volume. Any excess order volume is brutally truncated.

### 2.2 Execution Parity
*   **T+n Delay**: The Holding Clock must accurately enforce market settlement constraints (e.g., T+1 for A-shares).
*   **Mock Execution Parity**: Before finalizing any Alpha, high-frequency tick/minute simulations must be run over a 1-month subset to align the lower-frequency Vectorized model with realistic execution latency.

## 3. Statistical Rigor & Overfitting Deflation
### 3.1 Deflated Sharpe Ratio (DSR) & Statistical Tests
*   All reported IC, Sharpe, and performance metrics must include the number of trials ($n$) and $p$-values. If $p > 0.05$, the results are flagged as statistically insignificant.
*   **Trial Budgeting**: Maximum of 30 hyperparameter iterations per strategy formulation. Exceeding this budget indicates data mining; the strategy structure must be rejected or redesigned (Level 2).

### 3.2 Regime-Agnostic Validation
*   Performance must be broken down by regime. The strategy must be run through:
    *   **High Volatility Regimes**: 2008H2, 2020-03.
    *   **Low Volatility Regimes**: 2017 (VIX < 10).
*   A failure in extreme regimes marks the strategy as fragile.

## 4. Hardware Optimization (ARM 4C 24GB Constraint)
### 4.1 Memory-Bounded Execution
*   **Chunked Processing**: Multi-decade backtests cannot be loaded entirely into RAM. The engine must stream data in non-overlapping historical blocks (e.g., 5-year chunks).
*   **Garbage Collection**: Explicit `del` and `gc.collect()` hooks must be fired after massive vectorized operations.
*   **Type Downcasting**: `float64` is forbidden for high-density feature matrices unless numerical precision mandates it. Aggressive downcasting to `float32` and `int32`/`int16` is enforced upon data load.

### 4.2 Computational Hierarchy
*   **Vectorization First**: `NumPy`/`Pandas` vectorization is mandatory. `apply()` is banned for core metric loops.
*   **Path-Dependent Iteration**: Where absolute precision in path dependency is required (e.g., capital allocation across a constrained portfolio), explicit state machines (Numba JIT compiled) will be used instead of flawed Pandas masks.
