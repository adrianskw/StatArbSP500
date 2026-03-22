"""
Statistical factor model for multivariate statistical arbitrage.

Pipeline:
    1. PCA on log-returns → extract K systematic factors
    2. Ridge regression → strip factors, isolate idiosyncratic residuals
    3. Diagonal AR(1) on cumulated residual spreads → estimate per-stock coefficients
    4. Continuous-time OU calibration (θ, μ, Σ) from the AR(1) fit
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src import config

log = logging.getLogger(__name__)


class StatisticalFactorModel:
    """
    Multivariate statistical factor model for statistical arbitrage.

    Pipeline (call ``process_pipeline`` to execute all steps):
        1. ``compute_logreturns``  — log-price differences, derives Δt
        2. ``extract_factors``     — PCA → K systematic factor series
        3. ``extract_logreturns_residuals_OLS`` — OLS strips factors → idiosyncratic residuals
        4. ``fit_ou_dAR1``  — continuous-time diagonal OU calibration (θ, μ, Σ)
    """

    def __init__(self, n_factors: int = 10):
        """
        Initialize the factor model.

        Parameters
        ----------
        n_factors : int
            Number of principal components to extract.
            Typically 5-15 factors explain ~50% of S&P 500 variance.
        """
        self.n_factors = n_factors
        self.pca = PCA(n_components=n_factors, svd_solver="full")
        self.linear_model_logreturns = Ridge(alpha=1.0, fit_intercept=True)

        # State variables — dimensions use: T=days, N=stocks, K=n_factors
        self.tickers: pd.Index | None = None                    # (N,)   ticker labels
        self.logreturns: np.ndarray | None = None               # (T, N)  T = trading days after dropna
        self.logreturns_scaler: StandardScaler | None = None    # fitted StandardScaler from extract_factors
        self.factors: np.ndarray | None = None                  # (T, K)
        self.betas: np.ndarray | None = None                    # (N, K)
        self.logreturns_residuals: np.ndarray | None = None     # (T, N)
        self.spread: np.ndarray | None = None                   # (T, N)  cumsum of residuals

        # OU process parameters
        self.theta: np.ndarray | None = None                    # (N, N)
        self.mu: np.ndarray | None = None                       # (N,)
        self.sigma: np.ndarray | None = None                    # (N, N)
        self.delta_t: float | None = None                       # derived from data

    def compute_logreturns(self, clean_prices_df: pd.DataFrame) -> np.ndarray:
        """
        Transform raw prices into stationary log-returns.

        Also derives ``delta_t`` (in trading-year fractions) from the
        median gap in the datetime index.

        Parameters
        ----------
        clean_prices_df : pd.DataFrame
            DataFrame of adjusted close prices (days × stocks).

        Returns
        -------
        np.ndarray
            Log returns of shape (T-1, N).
        """
        logreturns_df = np.log(clean_prices_df / clean_prices_df.shift(1)).dropna()

        # Store ticker labels for downstream use, then convert to ndarray
        self.tickers = logreturns_df.columns
        self.logreturns = logreturns_df.values  # (T-1, N)

        # Derive delta_t from the data's own datetime index (trading days)
        time_deltas = pd.Series(logreturns_df.index).diff().dropna()
        median_gap_days = time_deltas.dt.total_seconds().median() / 86_400
        self.delta_t = median_gap_days / 252  # fraction of a trading year
        log.info("Derived Δt = %.6f (≈ %.1f calendar day(s) / 252).",
                 self.delta_t, median_gap_days)

        return self.logreturns

    def extract_factors(self) -> np.ndarray:
        """
        Run PCA on the log-returns to find market drivers.

        PCA is performed on the **correlation matrix** (i.e. standardised
        returns) rather than the raw covariance matrix.  Without this step,
        high-volatility stocks (e.g. NVDA, TSLA) dominate the first few
        principal components simply because their variance is larger — not
        because they are more representative market drivers.  Standardising
        to unit variance first gives each stock equal weight in the
        decomposition, producing factors that represent genuine cross-stock
        co-movement patterns.

        This is the approach used in Avellaneda & Lee (2010).

        The downstream OLS step regresses the *raw* log-returns (not
        standardised) on these factor scores, so the betas absorb the
        scale difference and the residuals remain in raw return units.

        Returns
        -------
        np.ndarray
            Factor time-series (T days x K factors).

        Raises
        ------
        ValueError
            If log returns have not been computed yet.
        """
        if self.logreturns is None:
            raise ValueError("Must compute log returns before extracting factors.")

        log.info("Running PCA to extract top %d standardized factors...", self.n_factors)

        # Standardise to unit variance: equivalent to PCA on the correlation matrix.
        # Each stock contributes equally regardless of its volatility level.
        self.logreturns_scaler = StandardScaler(with_mean=True, with_std=True)
        logreturns_std = self.logreturns_scaler.fit_transform(self.logreturns)

        # Project standardised returns onto top-K principal components
        self.factors = self.pca.fit_transform(logreturns_std)

        # Calculate how much market variance these factors explain
        explained_variance = np.sum(self.pca.explained_variance_ratio_)

        log.info("Factor Variance Explanation:")
        for i, var in enumerate(self.pca.explained_variance_ratio_):
            cumulative = np.sum(self.pca.explained_variance_ratio_[: i + 1])
            log.info("\tFactor %d:\t%.2f%%\t(%.2f%% cumulative)", i + 1, var * 100, cumulative * 100)

        log.info("Total: %d factors explain %.2f%% of market variance.", self.n_factors, explained_variance * 100)

        return self.factors

    def extract_logreturns_residuals_OLS(self) -> np.ndarray:
        """
        Strip systematic factors from stocks to isolate idiosyncratic residuals.

        Two methods are available, controlled by ``config.USE_RIDGE``:

        * ``USE_RIDGE = True`` (default): fits a Ridge regression of raw log-returns
          on PCA factor scores.  The Ridge penalty (α = 1.0) shrinks the betas,
          preventing the model from over-hedging noise when deployed OOS.

        * ``USE_RIDGE = False``: reconstructs systematic returns via direct PCA
          back-projection (``factors @ components_``) then rescales to raw return
          units using the saved ``StandardScaler`` statistics.  No regression is run
          and no L₂ penalty is applied.

        Returns
        -------
        np.ndarray
            Residual returns (idiosyncratic component only), shape (T, N).

        Raises
        ------
        ValueError
            If factors have not been extracted yet.
        """
        if self.factors is None:
            raise ValueError("Must extract factors before calculating residuals.")

        if config.USE_RIDGE:
            log.info("Stripping factors via Ridge regression (alpha=1.0)...")

            # Fit Ridge: X = factor scores (T x K), Y = raw log-returns (T x N)
            self.linear_model_logreturns.fit(self.factors, self.logreturns)

            # Beta matrix (N x K): how much each stock loads on each factor
            # Ridge penalty prevents massive betas on noise factors
            self.betas = self.linear_model_logreturns.coef_

            systematic_returns = self.linear_model_logreturns.predict(self.factors)

        else:
            log.info("Stripping factors via PCA back-projection (no Ridge)...")

            # Reconstruct K-factor approximation in standardized space:
            # factors (T x K) @ components (K x N) → systematic_std (T x N)
            systematic_std = self.factors @ self.pca.components_

            # Rescale to raw return units: multiply by per-stock σ only.
            # We do NOT add back the mean — the scaler was fit on demeaned returns,
            # so the factor scores already carry zero-mean information.  The stock
            # drift (mean) is idiosyncratic and must remain in the residuals.
            scale = self.logreturns_scaler.scale_   # per-stock σ (N,)
            systematic_returns = systematic_std * scale

            # Store PCA components as betas for reference (N x K)
            self.betas = self.pca.components_.T

        # Residual = Actual raw return − Systematic return
        self.logreturns_residuals = self.logreturns - systematic_returns

        return self.logreturns_residuals


    # ── OU parameter estimation ───────────────────────────────

    def _compute_sigma_eq(self) -> np.ndarray:
        """
        Calculate the equilibrium standard deviation (σ_eq) per stock.

        Since ``self.sigma`` is a diagonal matrix (produced by ``fit_ou_dAR1``),
        we extract per-stock variances via ``np.sum(self.sigma**2, axis=0)`` —
        an O(N) operation — rather than the O(N²) full matrix product Σ = UᵀU.
        """
        theta_diag = np.diag(self.theta)
        # Diagonal of (sigma.T @ sigma) equals column-wise sum of squares for
        # a diagonal matrix; avoids an unnecessary full N×N matrix multiply.
        sigma_sq = np.sum(self.sigma ** 2, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            sigma_eq = np.where(
                theta_diag > 0,
                np.sqrt(sigma_sq / (2.0 * theta_diag)),
                np.inf,
            )
        return sigma_eq


    def fit_ou_dAR1(self) -> None:
        """
        Diagonal (scalar) OU calibration: treats each stock independently.

        Instead of calling ``np.linalg.lstsq`` 500 times in a Python loop
        (which performs a full SVD per call), all N regressions are solved
        simultaneously using the closed-form scalar OLS formula:

            a_i = Cov(X_t, X_{t+1}) / Var(X_t)
            b_i = ȳ − a_i · x̄

        where X_t = spread[:-1, i] and X_{t+1} = spread[1:, i].
        This is O(T·N) as a single vectorized NumPy pass, versus O(T²·N)
        for 500 sequential SVD-backed lstsq calls.

        OU mapping (continuous time at resolution Δt):
            θ_i  = −ln(a_i) / Δt          (mean-reversion speed, annualised)
            μ_i  = b_i / (1 − a_i)        (equilibrium level)
            σ_i  = std(ε_i) / √(Δt · (1 − a_i²))  (diffusion)
        """
        N = self.spread.shape[1]
        
        # Vectorized scalar OLS across all N spread columns simultaneously
        # Formula: a_i = Cov(X, Y) / Var(X), where X = spread[:-1], Y = spread[1:]
        X_mat = self.spread[:-1, :]    # (T-1, N)
        Y_mat = self.spread[1:, :]     # (T-1, N)
        
        x_bar = np.mean(X_mat, axis=0) # (N,)
        y_bar = np.mean(Y_mat, axis=0) # (N,)
        
        x_ctr = X_mat - x_bar
        y_ctr = Y_mat - y_bar
        
        cov_xy = np.sum(x_ctr * y_ctr, axis=0) # (N,)
        var_x  = np.sum(x_ctr ** 2, axis=0)    # (N,)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            a_arr = np.where(var_x > 0, cov_xy / var_x, 0.0)
            
        b_arr = y_bar - a_arr * x_bar
        
        # Clamp root into (0, 1) to guarantee a positive, finite half-life
        a_arr = np.clip(a_arr, 1e-6, 1.0 - 1e-6)
        
        resid_mat = Y_mat - (a_arr * X_mat + b_arr)
        # Standard deviation of residuals per stock (ddof=0 matches numpy default)
        sigma_arr = np.std(resid_mat, axis=0)

        self.theta = np.diag(-np.log(a_arr) / self.delta_t)           # (N, N)
        self.mu    = b_arr / (1.0 - a_arr)                            # (N,)

        # Stationary-variance identity: Var(X_∞) = σ²_ε / (1 − a²)
        sigma_eq   = sigma_arr / np.sqrt(1.0 - a_arr ** 2)            # (N,)
        self.sigma = np.diag(sigma_eq / np.sqrt(self.delta_t))        # (N, N)
        log.info("Diagonal OU calibration complete (N=%d).", N)

    def process_pipeline(
        self,
        clean_prices_df: pd.DataFrame,
        lookback: int = config.LOOKBACK_WINDOW,
    ) -> None:
        """
        Execute the full factor extraction pipeline.

        Parameters
        ----------
        clean_prices_df : pd.DataFrame
            DataFrame of adjusted close prices (days × stocks).
        lookback : int
            Number of most-recent trading days to use for fitting.
            All steps — PCA, OLS, OU calibration — operate on this window
            so the parameters reflect the current market regime.

            Rule of thumb:
                250  → 1-year window  (default, regime-aware)
                500  → 2-year window  (smoother, slower to adapt)
                125  → 6-month window (reactive, noisier estimates)
        """
        available = len(clean_prices_df)
        if available < lookback:
            log.warning(
                "Requested lookback=%d but only %d days available. "
                "Using full history.",
                lookback,
                available,
            )
        else:
            log.info(
                "Applying %d-day lookback window (%s → %s).",
                lookback,
                clean_prices_df.index[-lookback].date(),
                clean_prices_df.index[-1].date(),
            )
            clean_prices_df = clean_prices_df.iloc[-lookback:]

        self.compute_logreturns(clean_prices_df)
        self.extract_factors()
        self.extract_logreturns_residuals_OLS()
        self.spread = self.logreturns_residuals.cumsum(axis=0)

        log.info("OU method: diagonal AR(1).")
        self.fit_ou_dAR1()

    # ── Diagnostics ────────────────────────────────────────────

    def get_diagnostics(self) -> pd.DataFrame:
        """
        Build a per-stock summary of calibrated OU parameters.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, theta, mu, sigma, half_life_days, s_score.

        Raises
        ------
        ValueError
            If the pipeline has not been run yet.
        """
        if self.theta is None or self.mu is None or self.sigma is None:
            raise ValueError("Pipeline must be run before calling get_diagnostics().")

        theta_diag = np.diag(self.theta)

        # self.sigma is diagonal (produced by fit_ou_dAR1):
        #   diag(σ_eq / √Δt).  Per-stock variance = column-wise sum of squares.
        sigma_sq  = np.sum(self.sigma ** 2, axis=0)            # per-stock variance
        sigma_val = np.sqrt(np.maximum(sigma_sq, 0.0))         # std dev

        # Half-life: ln(2) / θ_ii  (in trading days, not annualised)
        with np.errstate(divide="ignore", invalid="ignore"):
            half_life = np.where(theta_diag > 0, np.log(2) / theta_diag, np.inf)
        half_life_days = half_life * 252  # convert from trading-year fraction

        # S-score: (current_spread − μ) / σ_eq
        sigma_eq = self._compute_sigma_eq()
        current_spread = self.spread[-1, :]  # last observation
        with np.errstate(divide="ignore", invalid="ignore"):
            s_scores = np.where(sigma_eq > 0, (current_spread - self.mu) / sigma_eq, 0.0)

        diagnostics = pd.DataFrame(
            {
                "ticker": self.tickers,
                "theta": theta_diag,
                "mu": self.mu,
                "sigma": sigma_val,
                "half_life_days": half_life_days,
                "s_score": s_scores,
            }
        )
        diagnostics = diagnostics.sort_values("half_life_days").reset_index(drop=True)
        return diagnostics