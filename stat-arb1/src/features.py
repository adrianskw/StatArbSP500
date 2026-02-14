"""
features.py — Feature engineering for multivariate statistical arbitrage.

Functions are organized into logical groups:
  1. Data Transforms
  2. Factor Decomposition (PCA)
  3. Spread Construction & Z-Score
  4. Mean Reversion Speed
  5. Crossing & Duration Analysis
  6. Multivariate Dependence Structure (Copulas)
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA

from src import config


# ═══════════════════════════════════════════════════════════════
# 1. Data Transforms
# ═══════════════════════════════════════════════════════════════

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price levels."""
    return np.log(prices / prices.shift(1)).dropna()


# ═══════════════════════════════════════════════════════════════
# 2. Factor Decomposition (PCA)
# ═══════════════════════════════════════════════════════════════

def run_pca(
    returns: pd.DataFrame,
    n_components: int = config.N_COMPONENTS,
    min_weight: float = config.PCA_MIN_WEIGHT,
) -> dict:
    """
    PCA decomposition into eigenportfolios.

    Sparse filtering: weights with |w| < ``min_weight`` are zeroed,
    then re-normalized so each vector sums to 1.0.

    Returns
    -------
    dict with keys:
        - 'components': np.ndarray (n_components, n_assets)
        - 'explained_variance_ratio': np.ndarray
        - 'eigenvalues': np.ndarray
        - 'residuals': pd.DataFrame
    """
    pca = PCA(n_components=n_components)
    pca.fit(returns.values)

    components = pca.components_.copy()

    # Sparse PCA: zero weights below threshold, re-normalize
    if min_weight > 0:
        for i in range(components.shape[0]):
            mask = np.abs(components[i]) < min_weight
            components[i, mask] = 0.0
            total = np.sum(np.abs(components[i]))
            if total > 0:
                components[i] /= total

    residuals = returns - returns.values @ components.T @ components

    return {
        "components": components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "eigenvalues": pca.explained_variance_,
        "residuals": pd.DataFrame(residuals, index=returns.index, columns=returns.columns),
    }


def pca_persistence(
    returns: pd.DataFrame,
    n_components: int = config.N_COMPONENTS,
    window: int = config.PCA_PERSISTENCE_WINDOW,
) -> pd.DataFrame:
    """
    Rolling eigenvector stability (cosine similarity between consecutive windows).

    Returns a DataFrame of persistence scores per component, indexed by date.
    """
    dates = returns.index[window:]
    records = []

    prev_components = None
    for i in range(len(dates)):
        chunk = returns.iloc[i : i + window]
        pca = PCA(n_components=n_components)
        pca.fit(chunk.values)
        curr = pca.components_

        if prev_components is not None:
            sims = []
            for j in range(n_components):
                cos_sim = np.abs(np.dot(prev_components[j], curr[j])) / (
                    np.linalg.norm(prev_components[j]) * np.linalg.norm(curr[j]) + 1e-12
                )
                sims.append(cos_sim)
            records.append(sims)
        else:
            records.append([np.nan] * n_components)

        prev_components = curr

    cols = [f"PC{j+1}_persistence" for j in range(n_components)]
    return pd.DataFrame(records, index=dates, columns=cols)


def check_unit_root(series: pd.Series, significance: float = 0.05) -> dict:
    """
    Augmented Dickey-Fuller (ADF) test for stationarity.

    Returns dict with 'statistic', 'p_value', 'is_stationary'.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < significance,
        "critical_values": result[4],
    }


def check_normality(series: pd.Series) -> dict:
    """
    Jarque-Bera test for normality.

    Returns dict with 'statistic', 'p_value', 'is_normal'.
    """
    jb_stat, jb_p = stats.jarque_bera(series.dropna())
    return {
        "statistic": jb_stat,
        "p_value": jb_p,
        "is_normal": jb_p > 0.05,
    }


# ═══════════════════════════════════════════════════════════════
# 3. Spread Construction & Z-Score
# ═══════════════════════════════════════════════════════════════

def construct_spreads(prices: pd.DataFrame, weight_vectors: np.ndarray) -> pd.DataFrame:
    """
    Construct multi-leg spreads from cointegrating weight vectors.

    Parameters
    ----------
    prices : pd.DataFrame
        Price levels (N dates × M assets).
    weight_vectors : np.ndarray
        Shape (K, M) — K cointegrating vectors.

    Returns
    -------
    pd.DataFrame with K spread columns.
    """
    spreads = prices.values @ weight_vectors.T
    cols = [f"spread_{i+1}" for i in range(weight_vectors.shape[0])]
    return pd.DataFrame(spreads, index=prices.index, columns=cols)


def compute_zscore(
    spread: pd.Series,
    window: int = config.ZSCORE_WINDOW,
) -> pd.Series:
    """Rolling z-score: (spread - rolling_mean) / rolling_std."""
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    return (spread - rolling_mean) / (rolling_std + 1e-12)


def zscore_percentile_bands(
    spread: pd.Series,
    window: int = config.ZSCORE_WINDOW,
    percentiles: list = None,
) -> pd.DataFrame:
    """
    Empirical percentile bands (no normality assumption).

    Returns DataFrame with 'lower' and 'upper' columns.
    """
    if percentiles is None:
        percentiles = config.ZSCORE_PERCENTILES

    lower = spread.rolling(window).quantile(percentiles[0] / 100.0)
    upper = spread.rolling(window).quantile(percentiles[1] / 100.0)
    return pd.DataFrame({"lower": lower, "upper": upper}, index=spread.index)


def zscore_decay_rate(zscore_series: pd.Series, max_lag: int = 20) -> pd.Series:
    """
    Autocorrelation of z-score at lags 1..max_lag.

    High autocorrelation = slow decay (signal persists).
    """
    acf_values = [zscore_series.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
    return pd.Series(acf_values, index=range(1, max_lag + 1), name="zscore_acf")


# ═══════════════════════════════════════════════════════════════
# 4. Mean Reversion Speed
# ═══════════════════════════════════════════════════════════════

def ou_half_life(spread: pd.Series) -> float:
    """
    Ornstein-Uhlenbeck AR(1) fit: half-life = -ln(2) / beta.

    Returns half-life in **bars**. Negative beta = mean-reverting.
    """
    spread_clean = spread.dropna()
    y = spread_clean.diff().dropna().values
    x = spread_clean.iloc[:-1].values.reshape(-1, 1)

    # OLS: Δy = β * y_{t-1}
    beta = np.linalg.lstsq(x, y, rcond=None)[0][0]

    if beta >= 0:
        return np.inf  # Not mean-reverting

    return -np.log(2) / beta


def rolling_half_life(
    spread: pd.Series,
    window: int = config.ROLLING_HL_WINDOW,
) -> pd.Series:
    """Rolling half-life to detect regime changes."""
    results = []
    for i in range(window, len(spread)):
        chunk = spread.iloc[i - window : i]
        hl = ou_half_life(chunk)
        results.append(hl)

    return pd.Series(results, index=spread.index[window:], name="rolling_half_life")


def hurst_exponent(spread: pd.Series) -> float:
    """
    R/S (Rescaled Range) analysis for Hurst exponent.

    H < 0.5 = mean-reverting, H ≈ 0.5 = random walk, H > 0.5 = trending.
    """
    ts = spread.dropna().values
    n = len(ts)

    if n < 20:
        return np.nan

    max_k = min(n // 2, 100)
    lags = range(2, max_k)
    rs_values = []

    for lag in lags:
        chunks = [ts[i : i + lag] for i in range(0, n - lag, lag)]
        rs_chunk = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_c)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_values.append((lag, np.mean(rs_chunk)))

    if len(rs_values) < 2:
        return np.nan

    log_lags = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])

    slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
    return slope


def variance_ratio_test(
    spread: pd.Series,
    lags: list = None,
) -> dict:
    """
    Lo-MacKinlay Variance Ratio test.

    VR < 1 implies mean reversion at the given lag.

    Returns dict mapping lag -> VR value.
    """
    if lags is None:
        lags = config.VR_LAGS

    ts = spread.dropna().values
    results = {}

    for q in lags:
        if len(ts) < q + 1:
            results[q] = np.nan
            continue

        # Variance of q-period returns vs 1-period returns
        ret_1 = np.diff(ts)
        ret_q = ts[q:] - ts[:-q]

        var_1 = np.var(ret_1, ddof=1)
        var_q = np.var(ret_q, ddof=1)

        vr = var_q / (q * var_1 + 1e-12)
        results[q] = vr

    return results


# ═══════════════════════════════════════════════════════════════
# 5. Crossing & Duration Analysis
# ═══════════════════════════════════════════════════════════════

def zero_crossing_rate(spread: pd.Series) -> float:
    """Fraction of periods where the spread crosses its own mean."""
    centered = spread - spread.mean()
    signs = np.sign(centered.values)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return crossings / (len(signs) - 1)


def mean_crossing_count(
    spread: pd.Series,
    window: int = config.CROSSING_WINDOW,
) -> pd.Series:
    """Rolling count of mean crossings within a window."""
    centered = spread - spread.rolling(window).mean()
    signs = np.sign(centered)
    crossings = signs.diff().abs().rolling(window).sum() / 2
    return crossings.rename("mean_crossing_count")


def avg_excursion_duration(spread: pd.Series) -> float:
    """Average number of days the spread stays above/below its mean before reverting."""
    centered = spread - spread.mean()
    signs = np.sign(centered.values)

    durations = []
    current_duration = 1

    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1] and signs[i] != 0:
            current_duration += 1
        else:
            durations.append(current_duration)
            current_duration = 1

    durations.append(current_duration)

    return np.mean(durations) if durations else 0.0


# ═══════════════════════════════════════════════════════════════
# 6. Multivariate Dependence Structure
# ═══════════════════════════════════════════════════════════════

def mahalanobis_distance(
    returns: pd.DataFrame,
    window: int = config.LOOKBACK_WINDOW,
) -> pd.Series:
    """
    Covariance-adjusted distance from historical mean.

    High values flag regime breaks.
    """
    results = []
    for i in range(window, len(returns)):
        chunk = returns.iloc[i - window : i]
        mu = chunk.mean().values
        cov = chunk.cov().values
        current = returns.iloc[i].values

        try:
            cov_inv = np.linalg.inv(cov)
            diff = current - mu
            d = np.sqrt(diff @ cov_inv @ diff)
        except np.linalg.LinAlgError:
            d = np.nan

        results.append(d)

    return pd.Series(results, index=returns.index[window:], name="mahalanobis_distance")


def covariance_condition_number(
    returns: pd.DataFrame,
    window: int = config.LOOKBACK_WINDOW,
) -> pd.Series:
    """
    Rolling condition number of the covariance matrix.

    Condition number > 1000 implies mathematical instability.
    """
    results = []
    for i in range(window, len(returns)):
        chunk = returns.iloc[i - window : i]
        cov = chunk.cov().values
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]

        if len(eigenvalues) > 0:
            cn = eigenvalues[-1] / (eigenvalues[0] + 1e-12)
        else:
            cn = np.nan

        results.append(cn)

    return pd.Series(results, index=returns.index[window:], name="condition_number")


def fit_copula(returns: pd.DataFrame, family: str = "gaussian") -> object:
    """
    Fit a parametric copula (Gaussian, Student-t, Clayton, Gumbel).

    Returns fitted copula object from the ``copulas`` library.
    """
    from copulas.multivariate import GaussianMultivariate

    # Note: copulas library primarily supports Gaussian multivariate.
    # For Clayton/Gumbel, bivariate implementations may be needed.
    copula = GaussianMultivariate()
    copula.fit(returns)
    return copula


def tail_dependence_coefficient(
    returns: pd.DataFrame,
    family: str = "gaussian",
    quantile: float = 0.05,
) -> dict:
    """
    Estimate upper/lower tail dependence coefficients.

    Uses empirical quantile-based estimation.
    """
    n = len(returns)
    results = {}

    for i, col_i in enumerate(returns.columns):
        for j, col_j in enumerate(returns.columns):
            if j <= i:
                continue

            u = returns[col_i].rank() / (n + 1)
            v = returns[col_j].rank() / (n + 1)

            # Lower tail: P(V <= q | U <= q)
            lower_mask = u <= quantile
            if lower_mask.sum() > 0:
                lambda_l = (v[lower_mask] <= quantile).mean()
            else:
                lambda_l = np.nan

            # Upper tail: P(V >= 1-q | U >= 1-q)
            upper_mask = u >= (1 - quantile)
            if upper_mask.sum() > 0:
                lambda_u = (v[upper_mask] >= (1 - quantile)).mean()
            else:
                lambda_u = np.nan

            results[f"{col_i}_{col_j}"] = {
                "lambda_lower": lambda_l,
                "lambda_upper": lambda_u,
            }

    return results


def copula_concordance(
    returns: pd.DataFrame,
    window: int = config.LOOKBACK_WINDOW,
) -> pd.Series:
    """
    Rolling Kendall's tau (concordance measure).

    Captures non-linear dependence shifts over time.
    Uses the average pairwise tau across all asset pairs.
    """
    from scipy.stats import kendalltau

    results = []
    for i in range(window, len(returns)):
        chunk = returns.iloc[i - window : i]
        taus = []
        cols = chunk.columns
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                tau, _ = kendalltau(chunk[cols[a]], chunk[cols[b]])
                if not np.isnan(tau):
                    taus.append(tau)
        results.append(np.mean(taus) if taus else np.nan)

    return pd.Series(results, index=returns.index[window:], name="copula_concordance")
