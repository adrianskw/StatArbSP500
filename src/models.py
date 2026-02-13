"""
models.py — Johansen cointegration testing and signal generation.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src import config


def johansen_test(
    prices: pd.DataFrame,
    det_order: int = config.JOHANSEN_DET_ORDER,
    k_ar_diff: int = config.JOHANSEN_K_AR_DIFF,
) -> object:
    """
    Run the Johansen cointegration test across all assets.

    Parameters
    ----------
    prices : pd.DataFrame
        Price levels (N dates × M assets).
    det_order : int
        Deterministic order (-1, 0, 1).
    k_ar_diff : int
        Number of lagged differences in the VECM.

    Returns
    -------
    coint_johansen result object with:
        - .evec : eigenvectors (cointegrating vectors)
        - .eig  : eigenvalues
        - .lr1  : trace test statistics
        - .lr2  : max-eigenvalue test statistics
        - .cvt  : critical values for trace test (90%, 95%, 99%)
        - .cvm  : critical values for max-eigenvalue test
    """
    return coint_johansen(prices.dropna(), det_order, k_ar_diff)


def johansen_rank(
    johansen_result: object,
    significance: float = config.JOHANSEN_SIGNIFICANCE,
) -> int:
    """
    Determine the cointegration rank.

    Uses the trace test at the specified significance level.
    Maps significance to critical value column:
        0.10 -> col 0 (90%), 0.05 -> col 1 (95%), 0.01 -> col 2 (99%)

    Returns
    -------
    int : rank (0 = no cointegration, r = r cointegrating relationships)
    """
    sig_map = {0.10: 0, 0.05: 1, 0.01: 2}
    col = sig_map.get(significance, 1)

    trace_stats = johansen_result.lr1
    critical_values = johansen_result.cvt[:, col]

    rank = 0
    for i in range(len(trace_stats)):
        if trace_stats[i] > critical_values[i]:
            rank += 1
        else:
            break

    return rank


def johansen_trace_statistic(johansen_result: object) -> pd.DataFrame:
    """
    Extract per-rank trace test statistics vs. critical values.

    Returns a DataFrame with columns: trace_stat, cv_90, cv_95, cv_99, significant_95.
    """
    n_ranks = len(johansen_result.lr1)

    df = pd.DataFrame({
        "trace_stat": johansen_result.lr1,
        "cv_90": johansen_result.cvt[:, 0],
        "cv_95": johansen_result.cvt[:, 1],
        "cv_99": johansen_result.cvt[:, 2],
    }, index=[f"r<={i}" for i in range(n_ranks)])

    df["significant_95"] = df["trace_stat"] > df["cv_95"]

    return df


def select_vectors(
    johansen_result: object,
    significance: float = config.JOHANSEN_SIGNIFICANCE,
    n_max: int = config.N_COMPONENTS,
) -> np.ndarray:
    """
    Select cointegrating vectors that pass the trace test.

    Parameters
    ----------
    johansen_result : coint_johansen result
    significance : float
    n_max : int
        Maximum number of vectors to return.

    Returns
    -------
    np.ndarray of shape (K, M) where K = min(rank, n_max).
    """
    rank = johansen_rank(johansen_result, significance)
    n_select = min(rank, n_max)

    if n_select == 0:
        return np.array([])

    return johansen_result.evec[:, :n_select].T


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = config.ENTRY_THRESHOLD,
    exit_threshold: float = config.EXIT_THRESHOLD,
    zscore_percentiles: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Generate trading signals from z-scores.

    Logic:
        - If ``zscore_percentiles`` is provided, uses empirical bands.
        - Otherwise, uses fixed ``entry_threshold`` / ``exit_threshold``.

    Signal values:
        +1 = Long spread (z-score below lower band / -entry)
        -1 = Short spread (z-score above upper band / +entry)
         0 = Flat (z-score within exit zone)

    Signals persist until an exit condition is met.
    """
    signals = pd.Series(0, index=zscore.index, name="signal")
    position = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]

        if np.isnan(z):
            signals.iloc[i] = position
            continue

        if zscore_percentiles is not None and not zscore_percentiles.empty:
            lower = zscore_percentiles["lower"].iloc[i]
            upper = zscore_percentiles["upper"].iloc[i]

            if np.isnan(lower) or np.isnan(upper):
                signals.iloc[i] = position
                continue

            if position == 0:
                if z <= lower:
                    position = 1   # Long spread
                elif z >= upper:
                    position = -1  # Short spread
            elif position == 1:
                if z >= 0:
                    position = 0   # Exit long
            elif position == -1:
                if z <= 0:
                    position = 0   # Exit short
        else:
            # Fixed thresholds
            if position == 0:
                if z <= -entry_threshold:
                    position = 1   # Long spread
                elif z >= entry_threshold:
                    position = -1  # Short spread
            elif position == 1:
                if z >= -exit_threshold:
                    position = 0   # Exit long
            elif position == -1:
                if z <= exit_threshold:
                    position = 0   # Exit short

        signals.iloc[i] = position

    return signals
