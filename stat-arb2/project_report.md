# Multivariate Statistical Arbitrage Pipeline: S&P 500
An event-driven backtesting engine for quantitative statistical arbitrage strategies, developed to extract outperformance across the S&P 500 equity universe.

## Pipeline Architecture
The pipeline is modeled directly on the canonical factor-based residual extraction frameworks (e.g., Avellaneda & Lee, 2010), combining dimensionality reduction and stochastic process modeling.

### 1. Data Processing Module
The engine constructs the equity universe by dynamically downloading and aggressively filtering the S&P 500 constituents using `yfinance`. 
Data cleaning systematically strips any dead/halted tickers (dropping securities with internal variance below `1e-12`), flags anomalous daily gap returns (e.g. >50% log-returns), and drops stocks with excessive missing data thresholds.

### 2. PCA Factor Extraction
Price data is mapped into a log-return space and standardized to a correlation matrix. To isolate systematic risks from idiosyncratic opportunities, the engine performs Principal Component Analysis (`svd_solver="full"`). 
By default, the strategy pulls the top `N_FACTORS = 11` principal components representing broad market indices and sector effects, fitting an OLS regression via Ridge penalization against the 250-trading-day lookback window.

### 3. Ornstein-Uhlenbeck (OU) Mean-Reversion Modeling
The isolated regression residuals fundamentally represent specific, potentially tradable cross-sectional spreads. They are mathematically modeled using an Ornstein-Uhlenbeck stochastic process, defined as `dx = θ(μ - x) dt + σ dW`.
We compute:
*   `θ` (Reversion speed): Extracted from the AR(1) autoregressive scalar.
*   `Half-life`: Derived as `ln(2) / θ`, indicating how long typical pricing dislocations take to revert.
*   `Z-Score / S-Score`: The dimensionless deviation signal `(spread - μ) / σ_eq`.

### 4. Mean-Reversion Diagnostics
Before deployment into the backtester, all extracted spreads pass through a battery of rigorous time-series validations:
*   **Hurst Exponent (R/S)**: Confirms the series has long memory reverting properties ($H < 0.5$).
*   **Lo-MacKinlay Variance Ratio**: Validates the time-variance scales sub-linearly (VR < 1).
*   **Augmented Dickey-Fuller (ADF)**: Rejects the null hypothesis of a geometric random walk / unit root (p-value < 0.05).

---

## Event-Driven Backtest & Execution Engine
The signal processing is completely vectorized (Phase 1 logic), identifying exits and stop-losses prior to entering an event loop mapping cross-sectional entries and time-stops.

### Signal Configuration Config
The state-machine operates around distinct continuous S-score boundaries:
*   **Entry** (`S_ENTER = +1.25σ`): Triggers a long or short leg on the systematic spread when specific dislocations breach 1.25 standard deviations.
*   **Target Exit** (`S_EXIT = 0.50σ`): Closes the leg gracefully.
*   **Stop-Loss** (`S_STOP = 4.00σ`): Cuts losses if structural breakdown occurs.
*   **Time-Stop** (`MAX_HOLD_DAYS = 20`): Forces closure on lingering open positions.

### Parallel Computing and Optimization
The engine utilizes a long-lived `ProcessPoolExecutor` passed continuously between recalibration periods (configured to a default of 20 days or roughly monthly periods). The executor eliminates per-cycle process-spawning OS overhead and dynamically threads the mathematical validation matrix (Hurst/ADF/VR) across CPU cores.

### Reality Constraints and Frictions
Models reflect realistic latency and execution logic:
*   **Execution Costs (`COST_BPS = 7.0`)**: Imposed symmetrically representing bid-ask widening, impact, and standard commission. Short borrow fees are largely abstracted given S&P 500 large-cap liquidity.
*   **Kelly Sizing Parameterization**: Supports dynamic continuous Kelly fractional asset scaling to optimally grow geometric equity, restricted by custom safety caps to limit per-stock leverage blowouts.

---

## Results and Performance Analytics
At the conclusion of the backtest loop, the system natively dumps out exhaustive multi-layer trade reporting.
*   **Total Returns & Sharpe Ratio**: Aggregated on log-return units. 
*   **Equal-Weight Benchmark Comparisons**: Directly comparable Alpha and geometric baseline against an equal weight (1/N) benchmark spanning the S&P500 subset.
*   **Equity Curve Modeling**: Two-panel automated chart execution visualizing benchmark-overlaid gross returns and max drawdown depth plotting.
*   **Exit Breakdown Sub-Layer**: Matrix classification quantifying the explicit `Count`, `Total P&L`, and `% of P&L` specifically attributed to clean Mean-Reversion (Long/Short), Stop-Loss events, and general Time-Stops.
