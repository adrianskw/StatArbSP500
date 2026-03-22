"""
Data loader for S&P 500 price data.

NOTE — Survivorship bias:
    The ticker list is scraped from the *current* S&P 500 Wikipedia page.
    Companies that were removed from the index before today are excluded,
    which introduces survivorship bias into any back-test that uses this
    data.  For research-grade work, use a point-in-time constituent list.
"""

from io import StringIO
import logging
import os

import pandas as pd
import requests
import yfinance as yf

from src import config

log = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self,
        start_date: str = config.START_DATE,
        end_date: str = config.END_DATE,
        data_dir: str = config.DATA_DIR,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        self.tickers: list[str] = []
        self.raw_prices = pd.DataFrame()
        self.clean_prices = pd.DataFrame()

        os.makedirs(self.data_dir, exist_ok=True)

    # ── Ticker scraping ───────────────────────────────────────

    def _fetch_sp500_tickers(self) -> list[str]:
        """Scrapes the current S&P 500 tickers from Wikipedia."""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = tables[0]

        # Clean tickers for yfinance (e.g., BRK.B -> BRK-B)
        all_tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info("Fetched %d S&P 500 tickers from Wikipedia.", len(all_tickers))

        self.tickers = all_tickers
        return self.tickers

    # ── Caching helpers ───────────────────────────────────────

    def _get_cache_path(self) -> str:
        """
        Construct a date-stamped cache file path.

        Using start/end dates in the filename prevents a collision where
        a run with different date ranges would silently load stale data.
        """
        filename = f"sp500_adjprices_{self.start_date}_{self.end_date}.csv"
        return os.path.join(self.data_dir, filename)


    def get_data(self) -> pd.DataFrame:
        """
        Downloads adjusted-close prices from yfinance, or loads from cache.
        """
        cache_path = self._get_cache_path()

        if os.path.exists(cache_path):
            log.info("Loading data from cache: %s", cache_path)
            self.raw_prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            self.tickers = self.raw_prices.columns.tolist()
            return self.raw_prices

        if not self.tickers:
            self._fetch_sp500_tickers()

        log.info(
            "Downloading data for %d tickers from %s to %s …",
            len(self.tickers),
            self.start_date,
            self.end_date,
        )

        try:
            data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception:
            log.exception("Failed to download data from yfinance.")
            raise

        # Extract 'Adj Close' to guarantee price adjustments are included
        self.raw_prices = data["Adj Close"]

        # Save to cache
        log.info("Saving data to cache: %s", cache_path)
        self.raw_prices.to_csv(cache_path)

        return self.raw_prices

    # ── Missing-data audit ────────────────────────────────────

    def check_and_clean_missing_data(
        self, missing_threshold: float = config.MISSING_THRESHOLD
    ) -> pd.DataFrame:
        """
        Audits the dataframe for NaNs, drops highly sparse columns,
        forward-fills the rest, then runs sanity checks.

        Parameters
        ----------
        missing_threshold : float
            Fraction of rows a ticker may be missing before it is dropped.
        """
        df = self.raw_prices.copy()

        total_missing = df.isna().sum().sum()
        total_points = df.shape[0] * df.shape[1]
        log.info("--- Data Audit (%s to %s) ---", self.start_date, self.end_date)
        log.info("Missing data points: %d / %d", total_missing, total_points)

        # Drop stocks exceeding the threshold
        min_count = int(len(df) * (1 - missing_threshold))
        df_cleaned = df.dropna(thresh=min_count, axis=1)

        dropped_tickers = sorted(set(df.columns) - set(df_cleaned.columns))

        # Per-ticker breakdown
        missing_pct = df.isna().mean()
        tickers_with_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)

        if not tickers_with_missing.empty:
            log.info("Tickers with missing data (>0%):")
            for ticker, pct in tickers_with_missing.items():
                status = "[DROPPED]" if ticker in dropped_tickers else "[KEPT]"
                log.info("\t%s:\t%.2f%%\t%s", ticker, pct * 100, status)
        else:
            log.info("No tickers have missing data.")

        log.info(
            "Dropped %d tickers exceeding %.0f%% missing threshold.",
            len(dropped_tickers),
            missing_threshold * 100,
        )

        # Forward-fill remaining gaps (trading halts, single-day gaps)
        df_cleaned = df_cleaned.ffill()

        # Drop any leading rows that are still NaN (e.g. first row)
        df_cleaned = df_cleaned.dropna(axis=0)

        self.clean_prices = df_cleaned
        log.info(
            "Clean price matrix: %d days × %d stocks.",
            self.clean_prices.shape[0],
            self.clean_prices.shape[1],
        )

        # Run sanity checks on the cleaned data
        self.clean_prices = self.run_sanity_checks(self.clean_prices)
        return self.clean_prices

    # ── Sanity checks ─────────────────────────────────────────

    def run_sanity_checks(
        self,
        df: pd.DataFrame,
        extreme_threshold: float = config.EXTREME_RETURN_THRESHOLD,
        min_variance: float = 1e-12, # minimum variance to avoid flat data or "divide by zero" situations
    ) -> pd.DataFrame:
        """
        Post-cleaning data quality checks.

        1. Flags tickers with extreme single-day log-returns (data errors).
        2. Drops zero-variance (dead) stocks.
        3. Warns about duplicate column names.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned price matrix (days × stocks).
        extreme_threshold : float
            Flag any |log-return| above this value (default 0.50 = 50 %).
        min_variance : float
            Drop tickers whose price variance is below this.

        Returns
        -------
        pd.DataFrame
            Price matrix after dropping problematic columns.
        """
        import numpy as np

        log.info("--- Sanity Checks ---")
        n_before = df.shape[1]

        # 1. Duplicate columns
        dupes = df.columns[df.columns.duplicated()].tolist()
        if dupes:
            log.warning("Duplicate tickers found (keeping first): %s", dupes)
            df = df.loc[:, ~df.columns.duplicated(keep="first")]

        # 2. Zero-variance / near-constant stocks
        variances = df.var()
        dead = variances[variances < min_variance].index.tolist()
        if dead:
            log.warning("Dropping %d zero-variance tickers: %s", len(dead), dead)
            df = df.drop(columns=dead)

        # 3. Extreme single-day log-returns (likely data errors)
        logreturns = np.log(df / df.shift(1)).iloc[1:]
        extreme_mask = logreturns.abs() > extreme_threshold
        flagged = extreme_mask.any()
        flagged_tickers = flagged[flagged].index.tolist()

        if flagged_tickers:
            log.warning(
                "%d tickers have single-day |log-return| > %.0f%% — dropping:",
                len(flagged_tickers),
                extreme_threshold * 100,
            )
            for ticker in flagged_tickers:
                worst = logreturns[ticker].abs().max()
                log.warning("\t%s: max |log-return| = %.2f%%", ticker, worst * 100)
            df = df.drop(columns=flagged_tickers)
        else:
            log.info("No extreme returns detected (threshold: %.0f%%).", extreme_threshold * 100)

        n_after = df.shape[1]
        if n_after < n_before:
            log.info(
                "Sanity checks removed %d tickers (%d → %d).",
                n_before - n_after,
                n_before,
                n_after,
            )
        else:
            log.info("All %d tickers passed sanity checks.", n_after)

        return df