"""
data_loader.py — Fetch and cache market data via yfinance.
"""

import os
import hashlib

import pandas as pd
import yfinance as yf

from src import config

# ─── Constants ───────────────────────────────────────────────
_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _cache_path(tickers: list[str], start: str, end: str) -> str:
    """Generate a deterministic cache filename from query parameters."""
    key = f"{'_'.join(sorted(tickers))}_{start}_{end}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(_RAW_DIR, f"prices_{h}.parquet")


def load_prices(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    *,
    sector: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted close prices via yfinance, with local caching.

    Parameters
    ----------
    tickers : list[str], optional
        Explicit list of ticker symbols. Overrides ``sector``.
    start : str, optional
        Start date (YYYY-MM-DD). Defaults to ``config.DEFAULT_START_DATE``.
    end : str, optional
        End date (YYYY-MM-DD). Defaults to ``config.DEFAULT_END_DATE``.
    sector : str, optional
        Key into ``config.TICKER_UNIVERSES``. Ignored if ``tickers`` is provided.
    use_cache : bool
        If True, load from parquet cache when available.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted close prices, indexed by date, columns = tickers.
    """
    # Resolve defaults
    if tickers is None:
        sector = sector or config.DEFAULT_SECTOR
        tickers = config.TICKER_UNIVERSES[sector]
    start = start or config.DEFAULT_START_DATE
    end = end or config.DEFAULT_END_DATE

    # Check cache
    cache_file = _cache_path(tickers, start, end)
    if use_cache and os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    # Download
    data = yf.download(tickers, start=start, end=end, interval=config.DATA_INTERVAL, auto_adjust=True, progress=False)

    # Handle single vs multi-ticker yfinance output
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all")

    # Cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    prices.to_parquet(cache_file)

    return prices
