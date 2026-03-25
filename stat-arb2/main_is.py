"""
In-Sample backtest pipeline for S&P 500 statistical arbitrage.

Runs the backtest on the IS period only (START_DATE → OOS_TEST_START).
Use this file freely during strategy development and parameter tuning.

Split boundary is controlled by ``config.OOS_TEST_START``.
"""

import logging

import pandas as pd

from src import config
from src.loader import DataLoader
from src.backtest import run_backtest, print_backtest_summary, print_ticker_pnl_summary, plot_equity_curve


def main():
    """Execute the in-sample statistical arbitrage backtest."""
    # ── Logging setup ─────────────────────────────────────────────
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── 1. Data Loading & Cleaning ────────────────────────────────
    print("=" * 60)
    print("  S&P 500 Stat-Arb  —  IN-SAMPLE BACKTEST")
    print("=" * 60)
    print()

    loader = DataLoader()
    loader.get_data()
    clean_price_matrix = loader.check_and_clean_missing_data()

    # ── 2. IS Slice ───────────────────────────────────────────────
    oos_boundary = pd.Timestamp(config.OOS_TEST_START)
    is_prices    = clean_price_matrix[clean_price_matrix.index < oos_boundary]

    print(f"IS  period : {is_prices.index[0].date()} → {is_prices.index[-1].date()} "
          f"({len(is_prices)} days)")
    print(f"OOS holdout: {oos_boundary.date()} → {clean_price_matrix.index[-1].date()} "
          f"(LOCKED — do not run main_oos.py until parameters are finalised)")
    print()

    # ── 3. In-Sample Backtest ─────────────────────────────────────
    print("--- Running In-Sample Backtest ---")
    is_result = run_backtest(is_prices)
    print_backtest_summary(is_result, benchmark_prices=is_prices, label="IN-SAMPLE")
    print_ticker_pnl_summary(is_result, label="IN-SAMPLE")

    # ── 4. Equity Curve ───────────────────────────────────────────
    plot_equity_curve(is_result, benchmark_prices=is_prices, save_path="equity_curve_is.png")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
