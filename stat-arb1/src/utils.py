"""
utils.py — Plotting helpers and all 28 performance/risk metrics.

Metrics are organized into:
  1. PnL & Performance (Scalar) — 7 metrics
  2. Model Health & Signal Quality (Scalar) — 8 metrics
  3. Time-Varying Diagnostics (Series) — 13 metrics
"""

import numpy as np
import pandas as pd
from scipy import stats

from src import config, features


# ═══════════════════════════════════════════════════════════════
# 1. PnL & Performance Metrics (Scalar)
# ═══════════════════════════════════════════════════════════════

def net_sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio (after costs)."""
    excess = returns - rf / 252
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(252)


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sortino ratio (penalizes downside only)."""
    excess = returns - rf / 252
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.inf
    return (excess.mean() / downside.std()) * np.sqrt(252)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-valley drawdown (fraction)."""
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    return dd.min()


def cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    n_years = len(equity_curve) / 252
    if n_years <= 0 or equity_curve.iloc[0] <= 0:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = config.N_TRIALS,
) -> float:
    """
    Sharpe adjusted for multiple testing, skew, and kurtosis.

    Uses the Bailey-López de Prado correction.
    """
    sr = net_sharpe_ratio(returns)
    n = len(returns)
    skew = returns.skew()
    kurt = returns.kurtosis()

    # Expected max SR under null
    e_max_sr = np.sqrt(2 * np.log(n_trials)) * (
        1 - np.euler_gamma / (2 * np.log(n_trials + 1e-12))
    )

    # Variance of SR estimator
    sr_var = (1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / n

    if sr_var <= 0:
        return sr

    # Probability that observed SR is genuine
    z = (sr - e_max_sr) / np.sqrt(sr_var + 1e-12)
    return stats.norm.cdf(z)


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series = None) -> float:
    """Active return / Tracking error. If no benchmark, uses 0."""
    if benchmark_returns is None:
        benchmark_returns = pd.Series(0, index=returns.index)

    active = returns - benchmark_returns
    tracking_error = active.std()

    if tracking_error == 0:
        return 0.0
    return (active.mean() / tracking_error) * np.sqrt(252)


def capacity_estimate(
    returns: pd.Series,
    turnover: pd.Series,
    avg_daily_volume_usd: float,
    participation_rate: float = config.PARTICIPATION_RATE,
    cost_bps: float = config.TRANSACTION_COST_BPS,
) -> float:
    """
    Estimate maximum AUM before costs erode alpha to zero.

    capacity = (alpha * avg_volume * participation) / (turnover * cost)
    """
    alpha = returns.mean() * 252
    avg_turnover = turnover.mean()

    if avg_turnover <= 0 or alpha <= 0:
        return 0.0

    cost_per_dollar = cost_bps / 10_000
    max_volume = avg_daily_volume_usd * participation_rate
    capacity = (alpha * max_volume) / (avg_turnover * cost_per_dollar)

    return capacity


# ═══════════════════════════════════════════════════════════════
# 2. Model Health & Signal Quality (Scalar)
# ═══════════════════════════════════════════════════════════════
# Note: hurst_exponent, ou_half_life, variance_ratio_test, zero_crossing_rate,
# avg_excursion_duration are implemented in features.py.
# The following are computed here for organizational clarity.

def ic_information_ratio(ic_series: pd.Series) -> float:
    """Mean(IC) / Std(IC) — stability of predictive skill."""
    if ic_series.std() == 0:
        return 0.0
    return ic_series.mean() / ic_series.std()


def alpha_decay_profile(
    zscore: pd.Series,
    returns: pd.Series,
    horizons: list = None,
) -> dict:
    """
    IC / Sharpe at forward horizons to find optimal holding period.

    Returns dict mapping horizon -> IC.
    """
    if horizons is None:
        horizons = config.ALPHA_HORIZONS

    profile = {}
    for h in horizons:
        fwd_returns = returns.shift(-h)
        valid = pd.concat([zscore, fwd_returns], axis=1).dropna()
        if len(valid) > 10:
            corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
            profile[h] = corr
        else:
            profile[h] = np.nan

    return profile


# ═══════════════════════════════════════════════════════════════
# 3. Time-Varying Diagnostics (Series) — 13 Plots
# ═══════════════════════════════════════════════════════════════
# Note: rolling_half_life, mean_crossing_count, zscore_percentile_bands,
# zscore_decay_rate, mahalanobis_distance, covariance_condition_number,
# pca_persistence, copula_concordance are in features.py.
# The following are computed here.

def information_coefficient(
    zscore: pd.Series,
    returns: pd.Series,
    window: int = config.ZSCORE_WINDOW,
) -> pd.Series:
    """Rolling IC: correlation between yesterday's z-score and today's return."""
    lagged_z = zscore.shift(1)
    ic = lagged_z.rolling(window).corr(returns)
    return ic.rename("information_coefficient")


def turnover_ratio(
    weights: pd.DataFrame,
) -> pd.Series:
    """Daily turnover: sum of absolute weight changes."""
    turnover = weights.diff().abs().sum(axis=1)
    return turnover.rename("turnover_ratio")


def crowding_sensitivity(
    returns: pd.Series,
    factor_returns: pd.DataFrame = None,
    window: int = config.CROWDING_WINDOW,
) -> pd.Series:
    """
    Rolling correlation of strategy returns to common factors.

    If no factor_returns provided, returns NaN series.
    """
    if factor_returns is None or factor_returns.empty:
        return pd.Series(np.nan, index=returns.index, name="crowding_sensitivity")

    avg_corr = pd.Series(0.0, index=returns.index)
    n_factors = 0

    for col in factor_returns.columns:
        corr = returns.rolling(window).corr(factor_returns[col])
        avg_corr += corr.fillna(0)
        n_factors += 1

    if n_factors > 0:
        avg_corr /= n_factors

    return avg_corr.rename("crowding_sensitivity")


def brinson_fachler_attribution(
    portfolio_returns: pd.Series,
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame = None,
    asset_returns: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Brinson-Fachler return attribution: Allocation vs Selection.

    Returns DataFrame with 'allocation' and 'selection' columns.
    """
    if asset_returns is None or benchmark_weights is None:
        return pd.DataFrame({
            "allocation": np.nan,
            "selection": np.nan,
        }, index=portfolio_returns.index)

    # Allocation effect: (w_p - w_b) * (R_b_sector - R_b_total)
    # Selection effect: w_p * (R_p_sector - R_b_sector)
    # Simplified: decompose at portfolio level
    active_weights = portfolio_weights - benchmark_weights
    allocation = (active_weights * asset_returns).sum(axis=1)
    selection = portfolio_returns - allocation

    return pd.DataFrame({
        "allocation": allocation,
        "selection": selection,
    }, index=portfolio_returns.index)


def tail_dependence_series(
    returns: pd.DataFrame,
    window: int = config.LOOKBACK_WINDOW,
    quantile: float = 0.05,
) -> pd.Series:
    """Rolling average tail dependence across all pairs."""
    results = []
    for i in range(window, len(returns)):
        chunk = returns.iloc[i - window : i]
        td = features.tail_dependence_coefficient(chunk, quantile=quantile)
        if td:
            avg_lower = np.mean([v["lambda_lower"] for v in td.values() if not np.isnan(v["lambda_lower"])])
            results.append(avg_lower)
        else:
            results.append(np.nan)

    return pd.Series(results, index=returns.index[window:], name="tail_dependence")


# ═══════════════════════════════════════════════════════════════
# 4. Compute All Metrics
# ═══════════════════════════════════════════════════════════════

def compute_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    spread: pd.Series = None,
    zscore: pd.Series = None,
    asset_returns: pd.DataFrame = None,
    weights: pd.DataFrame = None,
    johansen_rank_val: int = None,
) -> dict:
    """Compute all 28 metrics and return as a dict."""
    result = {}

    # PnL Scalars
    result["net_sharpe_ratio"] = net_sharpe_ratio(returns)
    result["sortino_ratio"] = sortino_ratio(returns)
    result["max_drawdown"] = max_drawdown(equity_curve)
    result["cagr"] = cagr(equity_curve)
    result["deflated_sharpe_ratio"] = deflated_sharpe_ratio(returns)
    result["information_ratio"] = information_ratio(returns)

    # Model Scalars
    if spread is not None:
        result["hurst_exponent"] = features.hurst_exponent(spread)
        result["ou_half_life"] = features.ou_half_life(spread)
        result["zero_crossing_rate"] = features.zero_crossing_rate(spread)
        result["avg_excursion_duration"] = features.avg_excursion_duration(spread)
        result["variance_ratio"] = features.variance_ratio_test(spread)

    if johansen_rank_val is not None:
        result["johansen_rank"] = johansen_rank_val

    # Alpha decay
    if zscore is not None and returns is not None:
        result["alpha_decay_profile"] = alpha_decay_profile(zscore, returns)

    # IC
    if zscore is not None:
        ic_series = information_coefficient(zscore, returns)
        result["ic_information_ratio"] = ic_information_ratio(ic_series.dropna())

    # Time-varying
    if spread is not None:
        result["rolling_half_life"] = features.rolling_half_life(spread)

    if zscore is not None:
        result["rolling_ic"] = information_coefficient(zscore, returns)

    if weights is not None:
        result["turnover_ratio"] = turnover_ratio(weights)

    if asset_returns is not None:
        result["mahalanobis_distance"] = features.mahalanobis_distance(asset_returns)
        result["condition_number"] = features.covariance_condition_number(asset_returns)

    return result


# ═══════════════════════════════════════════════════════════════
# 5. Plotting Helpers
# ═══════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")


def _setup_ax(ax, title: str, ylabel: str):
    """Standard axis formatting."""
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)


def plot_equity_curve(equity: pd.Series, ax=None):
    """Plot equity curve with peak and drawdown shading."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    peak = equity.expanding().max()
    ax.plot(equity.index, equity, label="Equity", linewidth=1.5)
    ax.fill_between(equity.index, equity, peak, alpha=0.2, color="red", label="Drawdown")
    _setup_ax(ax, "Equity Curve", "USD")
    ax.legend()
    return ax


def plot_spread(spread: pd.Series, zscore: pd.Series = None, ax=None):
    """Plot spread level and optional z-score overlay."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(spread.index, spread, label="Spread", linewidth=1)
    ax.axhline(spread.mean(), color="gray", linestyle="--", alpha=0.5)

    if zscore is not None:
        ax2 = ax.twinx()
        ax2.plot(zscore.index, zscore, color="orange", alpha=0.6, label="Z-Score")
        ax2.set_ylabel("Z-Score")
        ax2.legend(loc="upper left")

    _setup_ax(ax, "Spread", "Level")
    ax.legend()
    return ax


def plot_signals(spread: pd.Series, signals: pd.Series, ax=None):
    """Plot spread with signal markers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(spread.index, spread, linewidth=1, alpha=0.7)

    long_mask = signals == 1
    short_mask = signals == -1

    ax.scatter(spread.index[long_mask], spread[long_mask],
               marker="^", color="green", s=30, label="Long", zorder=5)
    ax.scatter(spread.index[short_mask], spread[short_mask],
               marker="v", color="red", s=30, label="Short", zorder=5)

    _setup_ax(ax, "Signal Overlay", "Spread Level")
    ax.legend()
    return ax


def plot_rolling_half_life(hl_series: pd.Series, ax=None):
    """Plot rolling half-life with reference lines."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(hl_series.index, hl_series, linewidth=1)
    ax.axhline(np.median(hl_series.dropna()), color="gray", linestyle="--", alpha=0.5,
               label=f"Median: {np.median(hl_series.dropna()):.1f}d")
    _setup_ax(ax, "Rolling Half-Life", "Trading Days")
    ax.legend()
    return ax


def plot_rolling_ic(ic_series: pd.Series, ax=None):
    """Plot rolling information coefficient."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(ic_series.index, ic_series, linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(ic_series.index, ic_series, 0,
                    where=ic_series > 0, alpha=0.2, color="green")
    ax.fill_between(ic_series.index, ic_series, 0,
                    where=ic_series < 0, alpha=0.2, color="red")
    _setup_ax(ax, "Rolling IC (Information Coefficient)", "Correlation")
    return ax


def plot_turnover_series(turnover: pd.Series, ax=None):
    """Plot daily turnover ratio."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(turnover.index, turnover, width=1, alpha=0.6, color="steelblue")
    _setup_ax(ax, "Daily Turnover", "Fraction")
    return ax


def plot_rolling_mahalanobis(maha: pd.Series, ax=None):
    """Plot rolling Mahalanobis distance with alert threshold."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(maha.index, maha, linewidth=1)
    threshold = maha.quantile(0.95)
    ax.axhline(threshold, color="red", linestyle="--", alpha=0.7,
               label=f"95th pctl: {threshold:.1f}")
    _setup_ax(ax, "Mahalanobis Distance", "Distance")
    ax.legend()
    return ax


def plot_zscore_bands(spread: pd.Series, bands: pd.DataFrame, ax=None):
    """Plot spread with empirical percentile bands."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(spread.index, spread, linewidth=1, label="Spread")
    ax.plot(bands.index, bands["lower"], "--", color="green", alpha=0.7, label="5th pctl")
    ax.plot(bands.index, bands["upper"], "--", color="red", alpha=0.7, label="95th pctl")
    ax.fill_between(bands.index, bands["lower"], bands["upper"], alpha=0.1, color="gray")
    _setup_ax(ax, "Z-Score Percentile Bands", "Level")
    ax.legend()
    return ax


def plot_zscore_decay(acf_series: pd.Series, ax=None):
    """Plot z-score autocorrelation decay."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(acf_series.index, acf_series, color="steelblue", alpha=0.7)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("Z-Score Autocorrelation Decay", fontsize=12, fontweight="bold")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    return ax


def plot_crossing_frequency(crossing_series: pd.Series, ax=None):
    """Plot rolling mean-crossing frequency."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(crossing_series.index, crossing_series, linewidth=1)
    _setup_ax(ax, "Mean-Crossing Frequency", "Count")
    return ax


def plot_crowding(crowding: pd.Series, ax=None):
    """Plot crowding sensitivity."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(crowding.index, crowding, linewidth=1)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Crowding threshold")
    _setup_ax(ax, "Crowding Sensitivity", "Avg Correlation")
    ax.legend()
    return ax


def plot_attribution(attribution: pd.DataFrame, ax=None):
    """Plot Brinson-Fachler attribution decomposition."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(attribution.index, attribution["allocation"].cumsum(), label="Allocation")
    ax.plot(attribution.index, attribution["selection"].cumsum(), label="Selection")
    _setup_ax(ax, "Brinson-Fachler Attribution", "Cumulative Return")
    ax.legend()
    return ax


def plot_pca_persistence(persistence: pd.DataFrame, ax=None):
    """Plot PCA eigenvector persistence."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    for col in persistence.columns:
        ax.plot(persistence.index, persistence[col], label=col, linewidth=1)

    ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="Stability threshold")
    _setup_ax(ax, "PCA Persistence", "Cosine Similarity")
    ax.legend()
    return ax


def plot_condition_number(cn_series: pd.Series, ax=None):
    """Plot rolling condition number with instability threshold."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(cn_series.index, cn_series, linewidth=1)
    ax.axhline(1000, color="red", linestyle="--", label="Instability (1000)")
    ax.set_yscale("log")
    _setup_ax(ax, "Covariance Condition Number", "Condition #")
    ax.legend()
    return ax


def plot_copula_concordance(concordance: pd.Series, ax=None):
    """Plot rolling Kendall's tau."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(concordance.index, concordance, linewidth=1)
    _setup_ax(ax, "Copula Concordance (Kendall τ)", "Tau")
    return ax


def plot_tail_dependence(td_series: pd.Series, ax=None):
    """Plot rolling tail dependence."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(td_series.index, td_series, linewidth=1, color="darkred")
    _setup_ax(ax, "Tail Dependence (λ_lower)", "Probability")
    return ax


def plot_mean_reversion_diagnostics(spread: pd.Series, ax=None):
    """Composite: half-life + Hurst + VR in one summary panel."""
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        axes = [ax]

    if len(axes) >= 3:
        hl = features.ou_half_life(spread)
        h = features.hurst_exponent(spread)
        vr = features.variance_ratio_test(spread)

        axes[0].text(0.5, 0.5, f"Half-Life\n{hl:.1f} days",
                     ha="center", va="center", fontsize=16, transform=axes[0].transAxes)
        axes[0].set_title("OU Half-Life")

        axes[1].text(0.5, 0.5, f"Hurst\n{h:.3f}",
                     ha="center", va="center", fontsize=16, transform=axes[1].transAxes)
        axes[1].set_title("Hurst Exponent")

        vr_text = "\n".join([f"Lag {k}: {v:.3f}" for k, v in vr.items()])
        axes[2].text(0.5, 0.5, f"VR\n{vr_text}",
                     ha="center", va="center", fontsize=10, transform=axes[2].transAxes)
        axes[2].set_title("Variance Ratio")

    plt.tight_layout()
    return axes
