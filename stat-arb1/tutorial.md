# Tutorial: Running the Multivariate Statistical Arbitrage Pipeline

This tutorial walks through every stage of the pipeline — from environment setup to interpreting the final dashboard. By the end, you will know how to configure a sector universe, run cointegration analysis, backtest a mean-reversion strategy, and validate it with walk-forward analysis.

---

## 1. Environment Setup

### Prerequisites

- **Python 3.12+**
- **Git**

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Quant-Finance-Projects

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux

# Install all dependencies
pip install -r requirements.txt
```

### Register the Jupyter Kernel

This step is required so the notebook can find your virtual environment:

```bash
python -m ipykernel install --user --name=quant_env --display-name "Python (Quant Env)"
```

### Launch the Notebook

```bash
jupyter notebook notebooks/main_pipeline.ipynb
```

Make sure you select **"Python (Quant Env)"** as the kernel in the top-right corner of the notebook.

---

## 2. Project Layout

```
Quant-Finance-Projects/
├── notebooks/
│   └── main_pipeline.ipynb      # The 9-cell end-to-end pipeline
├── src/
│   ├── config.py                # Ticker universes & all 33 parameters
│   ├── data_loader.py           # yfinance fetching + parquet caching
│   ├── features.py              # PCA, spreads, z-scores, mean-reversion tests
│   ├── models.py                # Johansen cointegration & signal generation
│   ├── backtest.py              # Single-pass + walk-forward backtester
│   ├── utils.py                 # 28 performance/risk metrics & plotting
│   └── dashboard.py             # Essential (7+6) & Full (28+13) dashboards
├── data/
│   ├── raw/                     # Auto-cached price parquet files
│   └── processed/
├── tests/
└── requirements.txt
```

---

## 3. Pipeline Walkthrough (Cell by Cell)

Open `notebooks/main_pipeline.ipynb` and run each cell in order.

### Cell 1 — Configuration

```python
SECTOR = config.DEFAULT_SECTOR        # 'tech' by default
TICKERS = None                         # None = use sector default
START_DATE = config.DEFAULT_START_DATE  # '2020-01-01'
END_DATE = config.DEFAULT_END_DATE     # '2024-12-31'
```

**What to change:**
- Set `SECTOR` to any key in `config.TICKER_UNIVERSES` — available sectors are: `tech`, `energy`, `semiconductors`, `precious_metals`, `sp500_value`, `sp500_growth`, `lng`.
- Or override `TICKERS` with your own list, e.g. `['AAPL', 'MSFT', 'GOOG', 'AMZN']`.
- Adjust `START_DATE` / `END_DATE` to change the analysis window.

### Cell 2 — Data Loading

Downloads adjusted close prices via `yfinance`. Data is **automatically cached** as a `.parquet` file in `data/raw/`, so subsequent runs are instant.

**Output:** A DataFrame of daily prices with shape `(dates × tickers)` and a missing-value check.

### Cell 3 — PCA Analysis

Computes log returns, then runs **Principal Component Analysis** to extract eigenportfolios.

**Key outputs:**
- **Explained Variance Ratio** — how much of the sector's return variance each principal component captures.
- **Eigenportfolio Weights** — the loading of each stock on each factor. Weights < 5% are zeroed (sparse PCA) and the vector is re-normalized.

**Why it matters:** PC1 is usually the "market factor." The residual components (PC2, PC3) often capture sector rotations or pair-like relationships that the strategy can exploit.

### Cell 4 — Johansen Cointegration

Runs the **Johansen Trace Test** to find how many independent mean-reverting linear combinations exist among the price series.

**Key outputs:**
- **Cointegration Rank** — the number of statistically significant cointegrating relationships (at 95% confidence).
- **Trace Test Table** — test statistics vs. critical values at 90%, 95%, and 99%.
- **Weight Vectors** — the coefficients that define each mean-reverting spread (e.g. `+0.4 AAPL − 0.3 MSFT + 0.2 GOOG ...`).

> **If Rank = 0**, the pipeline stops here. Try a different sector, more tickers, or a longer date range.

### Cell 5 — Spread Construction & Diagnostics

Constructs multi-leg spreads from the Johansen weight vectors and computes rolling z-scores.

**Mean-reversion diagnostics (for spread 1):**

| Metric | Interpretation |
|--------|---------------|
| **Half-Life** | Days for the spread to revert halfway to its mean. Shorter = better. |
| **Hurst Exponent** | < 0.5 = mean-reverting, ≈ 0.5 = random walk, > 0.5 = trending. |
| **Zero-Crossing Rate** | Fraction of days the spread crosses its mean. Higher = more frequent reversion. |
| **Variance Ratios** | VR < 1 at lags 2, 5, 10, 20 implies mean reversion at those horizons. |

**Plots:** The spread level & z-score over time, plus empirical percentile bands (5th/95th) that replace fixed ±2σ thresholds for fat-tailed distributions.

### Cell 6 — Single-Pass Backtest

Generates trading signals using z-score bands and runs a full backtest simulation.

**Signal logic:**
- **Enter Long** when z-score drops below the lower band (spread is cheap).
- **Enter Short** when z-score rises above the upper band (spread is rich).
- **Exit** when z-score crosses back toward zero.

**Exit conditions (risk management):**
1. **Profit take** — z-score crosses the exit threshold.
2. **Stop-loss** — position loss ≥ 5% of position value.
3. **Time stop** — holding > 2× the spread's half-life.
4. **Drawdown halt** — portfolio equity drawdown ≥ 15% of peak.

**Costs modeled:**
- Transaction costs: 12 bps per side.
- Short borrow rate: 1% annualized, applied daily.

**Output:** Final equity, total return, trade count, and an equity curve plot.

### Cell 7 — Walk-Forward Analysis

The most important validation step. Re-estimates the Johansen vectors on a rolling **2-year training window** and trades the subsequent **3-month test window** with *fixed* parameters — no look-ahead bias.

**Process per fold:**
1. **Train:** Johansen + PCA on the training window.
2. **Warm-up:** Z-scores are initialized using `ZSCORE_WINDOW` (60 days) of trailing data from the end of the training period.
3. **Trade:** Signals are generated on the test window using the frozen training parameters.
4. **Collect:** Out-of-sample PnL is recorded.

**Output:** Number of folds, valid vs. skipped folds (rank 0), and the **stitched out-of-sample equity curve**. A fold summary table shows per-fold Sharpe, Max Drawdown, and Final Equity.

### Cell 8 — Essential Dashboard

Displays the 7 most important scalar metrics plus 6 time-varying plots:

**Scalar metrics:** Hurst Exponent, Sharpe Ratio, Sortino Ratio, Max Drawdown, Half-Life, Information Ratio, Johansen Rank.

**Plots:** Equity curve, Spread, Rolling Half-Life, Rolling IC, Turnover, Mahalanobis Distance.

### Cell 9 — Full Dashboard

Extends the essential view with all **28 metrics** and **13 time-varying diagnostic plots**, including:
- PCA persistence (eigenvector stability over time)
- Covariance matrix condition number (numerical stability)
- Copula concordance (tail dependence structure)
- Z-score decay rate (signal persistence via autocorrelation)
- Mean crossing frequency
- Crowding sensitivity
- Eigenportfolio weights heatmap

---

## 4. Key Configuration Reference

All parameters live in `src/config.py`. Here are the most impactful ones:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ENTRY_THRESHOLD` | 1.0σ | Higher = fewer but higher-conviction trades |
| `EXIT_THRESHOLD` | 0.25σ | Lower = hold trades longer |
| `STOP_LOSS` | 5% | Tighter = less drawdown, more whipsaws |
| `DRAWDOWN_LIMIT` | 15% | Portfolio-level circuit breaker |
| `POSITION_SIZE` | 10% | Capital allocated per spread trade |
| `WF_TRAIN_WINDOW` | 504 days (~2 yr) | Longer = more stable estimates, less data for testing |
| `WF_TEST_WINDOW` | 63 days (~3 mo) | Shorter = more folds, more granular OOS evaluation |
| `TRANSACTION_COST_BPS` | 12 bps | Increase for less liquid names |
| `N_COMPONENTS` | 3 | Max PCA / Johansen vectors to retain |

---

## 5. Tips & Troubleshooting

- **"No cointegration found" (Rank 0):** The selected assets have no stationary linear combination in the given window. Try a different sector, add more tickers, or extend the date range.
- **Very long half-life (> 60 days):** The spread reverts too slowly to be practical. Consider tightening `MAX_HALF_LIFE_HOLD` or switching sectors.
- **High condition number:** The covariance matrix is near-singular, meaning the weight estimates are unstable. Remove highly correlated tickers from the universe.
- **Walk-forward folds all skipped:** Cointegration is regime-dependent. If most folds show rank 0, the relationship may not be persistent enough to trade.
- **Data download fails:** Verify ticker symbols in `config.TICKER_UNIVERSES` and check your internet connection. Cached data in `data/raw/` can be deleted to force a re-download.
