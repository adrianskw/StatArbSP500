"""
config.py — Default tickers, date windows, and strategy parameters.

All parameters are documented with units and rationale.
Organized into logical groups for clarity.

Two execution modes are available:
  - "ideal"    : Zero friction — no transaction costs, no borrow fees,
                 no stop-loss, no drawdown circuit breaker.  Useful for
                 measuring raw signal quality.
  - "realistic": Production-grade costs and risk limits.
"""

# ═══════════════════════════════════════════════════════════════
# Execution Mode  —  "ideal" or "realistic"
# ═══════════════════════════════════════════════════════════════

EXECUTION_MODE = "realistic"       # ← change to "ideal" for frictionless run

_FRICTION_PROFILES = {
    "ideal": {
        "TRANSACTION_COST_BPS": 0.0,      # no transaction costs
        "SHORT_BORROW_RATE":    0.0,      # no borrow fees
        "STOP_LOSS":            1.0,      # effectively disabled (100%)
        "DRAWDOWN_LIMIT":       1.0,      # effectively disabled (100%)
        "POSITION_SIZE":        0.10,     # same sizing
        "MAX_EXPOSURE":         1.0,      # same exposure cap
        "MAX_HALF_LIFE_HOLD":   float("inf"),  # no time stop
        "PARTICIPATION_RATE":   1.0,      # no volume constraint
    },
    "realistic": {
        "TRANSACTION_COST_BPS": 7.0,     # basis points per side
        "SHORT_BORROW_RATE":    0.005,     # annualized rate (1%/yr)
        "STOP_LOSS":            0.05,     # 5% of position value
        "DRAWDOWN_LIMIT":       0.15,     # 15% of peak equity
        "POSITION_SIZE":        0.10,     # 10% of capital per trade
        "MAX_EXPOSURE":         1.0,      # 100% of capital
        "MAX_HALF_LIFE_HOLD":   2.0,      # multiples of half-life
        "PARTICIPATION_RATE":   0.03,     # 3% of daily volume
    },
}

_active = _FRICTION_PROFILES[EXECUTION_MODE]


def apply_mode(mode: str) -> None:
    """
    Switch friction profile at runtime.

    Call from the notebook *after* importing config to override
    all mode-dependent parameters without restarting the kernel.

    Usage::

        config.apply_mode("ideal")   # frictionless
        config.apply_mode("realistic")  # production costs
    """
    if mode not in _FRICTION_PROFILES:
        raise ValueError(f"Unknown mode '{mode}'. Options: {list(_FRICTION_PROFILES.keys())}")

    profile = _FRICTION_PROFILES[mode]
    
    # Use globals() to update module-level variables directly
    # This avoids issues with sys.modules[__name__] reflection
    g = globals()
    g["EXECUTION_MODE"] = mode
    g["TRANSACTION_COST_BPS"] = profile["TRANSACTION_COST_BPS"]
    g["SHORT_BORROW_RATE"] = profile["SHORT_BORROW_RATE"]
    g["STOP_LOSS"] = profile["STOP_LOSS"]
    g["DRAWDOWN_LIMIT"] = profile["DRAWDOWN_LIMIT"]
    g["POSITION_SIZE"] = profile["POSITION_SIZE"]
    g["MAX_EXPOSURE"] = profile["MAX_EXPOSURE"]
    g["MAX_HALF_LIFE_HOLD"] = profile["MAX_HALF_LIFE_HOLD"]
    g["PARTICIPATION_RATE"] = profile["PARTICIPATION_RATE"]

# ═══════════════════════════════════════════════════════════════
# Sector-Based Ticker Universes
# ═══════════════════════════════════════════════════════════════

TICKER_UNIVERSES = {
    "tech": [
        "AAPL", "MSFT", "GOOG", "AMZN", "META",
        "NVDA", "TSM", "AVGO", "ORCL", "CRM",
        "CSCO", "ADBE", 
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG",
        "MPC", "VLO", "OXY", "KMI", "WMB",
    ],
    "precious_metals": [
        "GLD", "SLV", "GDX", "NEM", "AEM", 
        "WPM", "RGLD", "PAAS", "SSRM", "CDE",
    ],
    "semiconductors": [
        "NVDA", "AMD", "INTC", "TSM", "AVGO",
        "QCOM", "MU", "AMAT", "LRCX", "KLAC",
        "TXN", "ADI", 
    ],
    "sp500_value": [
        "BRK-B", "JNJ", "JPM", "PG", "UNH",
        "HD", "KO", "PEP", "MRK", "ABBV",
        "MCD", "WMT", 
    ],
    "lng": [
        "LNG", "SHEL", "TTE", "EQNR", "KMI",
        "WMB", "TRP", "OKE", "GLNG",
    ],
    "sp500_growth": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META",
        "GOOGL", "LLY", "V", "MA", "TSLA",
        "AVGO", "COST",
    ],
}

# ═══════════════════════════════════════════════════════════════
# Data & Universe Selection
# ═══════════════════════════════════════════════════════════════

DEFAULT_SECTOR = "tech"
DEFAULT_START_DATE = "2013-01-01"
DEFAULT_END_DATE = "2018-12-31"  # Adjusted end date to keep a reasonable range, or keep as is? The plan said 2025. I'll stick to the existing end date if possible, or just change start. Actually, the user didn't explicitly say change end date, but usually daily backtests go further back. I will use 2020-01-01 as start.
DATA_INTERVAL = "1d"               # "1d", "1h", "15m" etc. Note: Windows are in *bars*.
ANNUALIZATION_FACTOR = 252         # 252 days * 1 bar/day = 252 bars/year

# ═══════════════════════════════════════════════════════════════
# Johansen Cointegration
# ═══════════════════════════════════════════════════════════════

JOHANSEN_SIGNIFICANCE = 0.05    # p-value — 95% confidence level
JOHANSEN_DET_ORDER = 0          # enum (-1, 0, 1) — restricted constant
JOHANSEN_K_AR_DIFF = 5          # lag count — lagged differences in VECM

# ═══════════════════════════════════════════════════════════════
# PCA & Factor Decomposition
# ═══════════════════════════════════════════════════════════════

N_COMPONENTS = 5                # count — top 5 eigenportfolios
PCA_PERSISTENCE_WINDOW = 126    # 126 bars (~6 months)
PCA_MIN_WEIGHT = 0.05           # fraction (0–1) — sparse PCA threshold

# ═══════════════════════════════════════════════════════════════
# Plot Configuration
# ═══════════════════════════════════════════════════════════════

SAVE_PLOTS = True
PLOT_DIR = "plots"

# ═══════════════════════════════════════════════════════════════
# Signal Generation (Z-Score)
# ═══════════════════════════════════════════════════════════════

ZSCORE_WINDOW = 60              # 60 bars (~3 months)
ENTRY_THRESHOLD = 2.0           # standard deviations
EXIT_THRESHOLD = 0.5           # standard deviations
ZSCORE_PERCENTILES = [10, 90]    # percentile (0–100) — empirical bands

# ═══════════════════════════════════════════════════════════════
# Mean Reversion Dynamics
# ═══════════════════════════════════════════════════════════════

LOOKBACK_WINDOW = 252           # 252 bars (~1 year)
ROLLING_HL_WINDOW = 126         # 126 bars (~6 months)
CROSSING_WINDOW = 60            # 60 bars (~3 months)
VR_LAGS = [1, 5, 10, 20]        # bars — approx 1d, 1w, 2w, 1m equivalents

# ═══════════════════════════════════════════════════════════════
# Backtest Execution  (mode-dependent values from _active)
# ═══════════════════════════════════════════════════════════════

INITIAL_CAPITAL = 1_000_000                     # USD
POSITION_SIZE = _active["POSITION_SIZE"]        # fraction of capital (0–1)
STOP_LOSS = _active["STOP_LOSS"]                # fraction of position value (0–1)
MAX_EXPOSURE = _active["MAX_EXPOSURE"]          # fraction of capital (0–1)
DRAWDOWN_LIMIT = _active["DRAWDOWN_LIMIT"]      # fraction of peak equity (0–1)
WF_TRAIN_WINDOW = 504           # 504 bars (~2 years)
WF_TEST_WINDOW = 63             # 63 bars (~3 months)

# ═══════════════════════════════════════════════════════════════
# Performance & Risk Analytics  (mode-dependent values from _active)
# ═══════════════════════════════════════════════════════════════

N_TRIALS = 1                                            # count — for deflated Sharpe ratio
ALPHA_HORIZONS = [1, 2, 5, 10, 20]                      # bars (1d, 2d, 1w, 2w, 1m)
CROWDING_WINDOW = 60                                    # 60 bars (~3 months)
PARTICIPATION_RATE = _active["PARTICIPATION_RATE"]       # fraction of daily volume (0–1)
MAX_HALF_LIFE_HOLD = _active["MAX_HALF_LIFE_HOLD"]      # multiples of half-life
TRANSACTION_COST_BPS = _active["TRANSACTION_COST_BPS"]   # basis points per side
SHORT_BORROW_RATE = _active["SHORT_BORROW_RATE"]         # annualized rate

# ═══════════════════════════════════════════════════════════════
# Expensive Feature Flags (default OFF)
# ═══════════════════════════════════════════════════════════════

COMPUTE_ROLLING_JOHANSEN = True   # re-estimate cointegration daily
COMPUTE_FEATURE_IMPORTANCE = True  # permutation importance
