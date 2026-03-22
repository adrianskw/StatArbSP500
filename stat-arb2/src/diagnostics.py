"""
Standalone diagnostic functions for mean-reversion analysis.

Provides Hurst exponent, variance-ratio, and ADF utilities that can operate
on arbitrary series without a full model calibration.
"""

import logging
import concurrent.futures
from typing import Optional

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)


# ── Consensus spread diagnostics ──────────────────────────────


def _process_ticker(args) -> dict:
    """
    Evaluate mean-reversion diagnostics for a single spread series.

    Tests are applied in cheapest-to-most-expensive order (Lazy Evaluation):
        1. Variance Ratio  — O(T), pure NumPy; fastest gate.
        2. ADF             — O(T·lags), statsmodels OLS; skipped if VR fails.
        3. Hurst (R/S)     — O(T·lags²), multi-lag loop; skipped if ADF fails.

    IMPORTANT: The backtest filters on ``tests_passed == 3``.  Because we
    short-circuit early, a stock can only reach ``tests_passed == 3`` if it
    passes ALL THREE tests in the order above.  The threshold logic in
    ``backtest.py`` remains correct as long as this ordering is maintained.
    """
    ticker, s, h_max, h_min, v_lag, v_cut, a_alpha = args
    
    # Base dictionary structure
    res = {
        "ticker": ticker, "hurst": np.nan, "vr": np.nan, "adf_p": np.nan,
        "mr_hurst": False, "mr_vr": False, "mr_adf": False, "tests_passed": 0
    }

    # 1. Variance Ratio (Fastest Math)
    vr = variance_ratio(s, lag=v_lag)
    res["vr"] = vr
    res["mr_vr"] = (not np.isnan(vr)) and vr < v_cut
    if not res["mr_vr"]:
        return res
    res["tests_passed"] = 1

    # 2. Augmented Dickey-Fuller (Moderate, runs OLS internally)
    _, adf_p = adf_test(s)
    res["adf_p"] = adf_p
    res["mr_adf"] = (not np.isnan(adf_p)) and adf_p < a_alpha
    if not res["mr_adf"]:
        return res
    res["tests_passed"] = 2

    # 3. Hurst Exponent (Slowest, multi-lag loops)
    h = hurst_exponent(s, max_lag=h_max, min_lag=h_min)
    res["hurst"] = h
    res["mr_hurst"] = (not np.isnan(h)) and h < 0.5
    if res["mr_hurst"]:
        res["tests_passed"] = 3

    return res


def compute_spread_diagnostics(
    spread_df: pd.DataFrame,
    ou_diagnostics: Optional[pd.DataFrame] = None,
    hurst_max_lag: int = config.HURST_MAX_LAG,
    hurst_min_lag: int = config.HURST_MIN_LAG,
    vr_lag: int = config.VARIANCE_RATIO_LAG,
    vr_cutoff: float = config.VARIANCE_RATIO_CUTOFF,
    adf_alpha: float = config.ADF_ALPHA,
    executor: Optional[concurrent.futures.ProcessPoolExecutor] = None,
) -> pd.DataFrame:
    """
    Run three mean-reversion tests on every column of a spread DataFrame
    and return a tidy summary table, optionally enriched with OU half-lives.

    Tests applied per ticker
    ------------------------
    * **Hurst exponent** (R/S):  H < 0.5  → mean-reverting
    * **Variance ratio** (Lo-MacKinlay):  VR(q) < ``vr_cutoff``  → mean-reverting
    * **Augmented Dickey-Fuller**:  p < ``adf_alpha``  → reject unit root

    Parameters
    ----------
    spread_df : pd.DataFrame
        DataFrame of cumulated idiosyncratic spread series (T × N).
    ou_diagnostics : pd.DataFrame, optional
        Output of ``StatisticalFactorModel.get_diagnostics()``.  When provided,
        the ``half_life_days`` column is merged in on ``ticker``.
    hurst_max_lag : int
        Maximum lag for the R/S Hurst estimator.
    hurst_min_lag : int
        Minimum lag (skips small-sample-biased R/S blocks).
    vr_lag : int
        Holding-period q for the Lo-MacKinlay variance ratio.
    vr_cutoff : float
        Variance ratio threshold; series below this pass the VR test.
    adf_alpha : float
        ADF significance level; p-values below this reject the unit root.
    executor : ProcessPoolExecutor, optional
        A long-lived executor to reuse across multiple calls (avoids the
        fork+IPC overhead of spawning a new pool on every recalibration).
        If None, a fresh pool is created and shut down within this call.

    Returns
    -------
    pd.DataFrame
        One row per ticker, sorted by tests_passed (desc) then hurst (asc).

        Columns: ticker, hurst, vr, adf_p,
                 mr_hurst, mr_vr, mr_adf, tests_passed
                 [, half_life_days]  — if ``ou_diagnostics`` is provided.
    """
    tasks = [
        (ticker, spread_df[ticker], hurst_max_lag, hurst_min_lag, vr_lag, vr_cutoff, adf_alpha)
        for ticker in spread_df.columns
    ]

    if executor is not None:
        # Reuse the long-lived pool — no spawn overhead
        records = list(executor.map(_process_ticker, tasks))
    else:
        # Standalone call: create and immediately shut down a local pool
        with concurrent.futures.ProcessPoolExecutor() as local_executor:
            records = list(local_executor.map(_process_ticker, tasks))

    result = (
        pd.DataFrame(records)
        .sort_values(["tests_passed", "hurst"], ascending=[False, True])
        .reset_index(drop=True)
    )

    # Optionally enrich with OU half-lives from the calibrated model
    if ou_diagnostics is not None:
        result = result.merge(
            ou_diagnostics[["ticker", "half_life_days"]],
            on="ticker",
            how="left",
        )

    return result


# ── Hurst exponent (rescaled-range) ──────────────────────────


def hurst_exponent(
    series: pd.Series | np.ndarray,
    max_lag: int = 150,
    min_lag: int = 10,
) -> float:
    """
    Estimate the Hurst exponent using the rescaled-range (R/S) method.

    The R/S statistic is computed on the **first differences** (increments)
    of the input series, not on the raw levels.

    H < 0.5 → mean-reverting
    H ≈ 0.5 → random walk
    H > 0.5 → trending

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Time series of prices or spreads (levels; differences are taken internally).
    max_lag : int
        Maximum block size for the R/S analysis.
    min_lag : int
        Minimum block size.  Lags below this are skipped to avoid the
        well-known positive small-sample bias of the R/S estimator.

    Returns
    -------
    float
        Estimated Hurst exponent.
    """
    ts = np.asarray(series, dtype=float)
    ts = ts[~np.isnan(ts)]

    # R/S operates on increments (first differences), not levels
    increments = np.diff(ts)
    n = len(increments)

    if n < max_lag + 2:
        log.warning("Series too short for Hurst (len=%d, max_lag=%d).", n, max_lag)
        return np.nan

    lags = range(min_lag, max_lag + 1)
    rs_values = []

    for lag in lags:
        # Split into non-overlapping blocks
        n_blocks = n // lag
        if n_blocks < 1:
            continue

        rs_list = []
        for block_idx in range(n_blocks):
            block = increments[block_idx * lag : (block_idx + 1) * lag]
            mean_block = block.mean()
            cumdev = np.cumsum(block - mean_block)
            R = cumdev.max() - cumdev.min()
            S = block.std(ddof=1)
            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append((lag, np.mean(rs_list)))

    if len(rs_values) < 2:
        return np.nan

    log_lags = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])

    # H = slope of log(R/S) vs log(lag)
    hurst = float(np.polyfit(log_lags, log_rs, 1)[0])
    return hurst


# ── Variance ratio ────────────────────────────────────────────


def variance_ratio(series: pd.Series | np.ndarray, lag: int = 5) -> float:
    """
    Lo-MacKinlay variance ratio test.

    VR(q) = Var(r_t(q)) / (q * Var(r_t))

    VR < 1 → mean-reverting
    VR ≈ 1 → random walk
    VR > 1 → trending

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Price or spread series.
    lag : int
        Holding period q.

    Returns
    -------
    float
        Variance ratio.
    """
    ts = np.asarray(series, dtype=float)
    returns_1 = np.diff(ts)
    returns_q = ts[lag:] - ts[:-lag]

    var_1 = np.var(returns_1, ddof=1)
    var_q = np.var(returns_q, ddof=1)

    if var_1 == 0:
        return np.nan
    return var_q / (lag * var_1)


# ── Augmented Dickey-Fuller test ──────────────────────────────


def adf_test(series: pd.Series | np.ndarray) -> tuple[float, float]:
    """
    Augmented Dickey-Fuller (ADF) test for stationarity.

    Tests H₀: the series has a unit root (non-stationary / random walk).
    Rejecting H₀ (low p-value) is evidence of mean-reversion.

    The ADF regression estimated internally is:
        ΔXₜ = α + β·Xₜ₋₁ + Σⱼ γⱼ·ΔXₜ₋ⱼ + εₜ

    H₀: β = 0  (unit root)
    H₁: β < 0  (stationary, mean-reverting)

    The lag order is chosen automatically by minimising AIC, which
    corrects for serial correlation in the residuals without requiring
    the caller to specify a lag length.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Spread or price series (levels, not differences).

    Returns
    -------
    tuple[float, float]
        (adf_statistic, p_value).
        p_value < 0.05 → reject H₀ at 5% → stationary at 95% confidence.
        p_value < 0.10 → reject H₀ at 10% → stationary at 90% confidence.
    """
    from statsmodels.tsa.stattools import adfuller

    ts = np.asarray(series, dtype=float)
    ts = ts[~np.isnan(ts)]

    if len(ts) < 20:
        log.warning("Series too short for ADF test (len=%d).", len(ts))
        return np.nan, np.nan

    result = adfuller(ts, autolag="AIC", regression="c")
    return float(result[0]), float(result[1])

