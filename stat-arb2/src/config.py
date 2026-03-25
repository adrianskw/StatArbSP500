"""Multivariate Statistical Arbitrage — Configuration."""

# ── Date Range ───────────────────────────────────────────────────────────────────────────────────────
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
OOS_TEST_START = "2025-01-01"

# ── Data Loading ─────────────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data"                 # directory for cached CSV files
MISSING_THRESHOLD = 0.05          # drop tickers with > 5 % missing rows
EXTREME_RETURN_THRESHOLD = 0.50   # flag any single-day |log-return| > 50 %

# ── Modeling ─────────────────────────────────────────────────────────────────────────────────────────
N_FACTORS = 11                    # number of PCA components to extract
LOOKBACK_WINDOW = 250             # trading days used for PCA + OU calibration (~1 year)
RECALIB_DAYS = 20                 # recalibrate OU model every N trading days (~monthly)
USE_RIDGE = True                  # if True, Ridge regression strips factors; if False, use PCA back-projection

# ── Diagnostics ──────────────────────────────────────────────────────────────────────────────────────
HURST_MAX_LAG = 60               # max lag for R/S Hurst estimator
HURST_MIN_LAG = 10                # min lag — skip small-sample-biased R/S blocks
VARIANCE_RATIO_LAG = 5            # lag for Lo-MacKinlay variance ratio
VARIANCE_RATIO_CUTOFF = 1.0       # threshold for variance ratio test (VR < 1 = mean-reverting)
ADF_ALPHA = 0.05                  # ADF p-value threshold (below = reject unit root → stationary)

# ── Strategy / Backtest ──────────────────────────────────────────────────────────────────────────────
S_ENTER = 2.0                    # open position when |s| exceeds this
S_EXIT = 0.5                      # close position when |s| falls below this
S_STOP = 4.0                      # stop-loss: force close when |s| exceeds this
COST_BPS = 6.0                    # one-way transaction cost in basis points

# ── Risk Management ──────────────────────────────────────────────────────────────────────────────────
MAX_HOLD_DAYS = 30                # maximum holding period in trading days (time-stop)
USE_KELLY = True                 # if False, use Unit Sizing (size=1.0)
KELLY_FRACTION = 0.1              # scale down raw Kelly to manage risk
MAX_UNIT_SIZE = 10.0              # safety cap on per-stock leverage
