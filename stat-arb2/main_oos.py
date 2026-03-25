"""
Out-of-Sample backtest pipeline for S&P 500 statistical arbitrage.

⚠️  RUN THIS FILE ONLY ONCE — when you have completely finalised your strategy.
    Every run consumes your single unbiased performance estimate.
    Do NOT tweak parameters based on these results.

Split boundary is controlled by ``config.OOS_TEST_START``.
"""

import logging

import pandas as pd

from src import config
from src.loader import DataLoader
from src.backtest import run_backtest, print_backtest_summary, print_ticker_pnl_summary, plot_equity_curve


def main():
    """Execute the out-of-sample statistical arbitrage backtest."""
    # ── Logging setup ─────────────────────────────────────────────
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── 1. Data Loading & Cleaning ────────────────────────────────
    print("=" * 60)
    print("  S&P 500 Stat-Arb  —  OUT-OF-SAMPLE BACKTEST")
    print("=" * 60)
    print()

    loader = DataLoader()
    loader.get_data()
    clean_price_matrix = loader.check_and_clean_missing_data()

    # ── 2. IS & OOS Slices ────────────────────────────────────────
    # The OOS buffer prepends exactly LOOKBACK_WINDOW IS trading days so the
    # model can calibrate on day 1 of the true OOS period.  We use an exact
    # index lookup (not BDay) to avoid a holiday-induced offset.
    oos_boundary = pd.Timestamp(config.OOS_TEST_START)
    is_prices    = clean_price_matrix[clean_price_matrix.index <  oos_boundary]
    oos_prices   = clean_price_matrix[clean_price_matrix.index >= oos_boundary]

    buffer_start = is_prices.index[-config.LOOKBACK_WINDOW]
    oos_buffer   = clean_price_matrix[clean_price_matrix.index >= buffer_start]

    print(f"IS  period : {is_prices.index[0].date()} → {is_prices.index[-1].date()}")
    print(f"OOS period : {oos_boundary.date()} → {oos_prices.index[-1].date()} "
          f"({len(oos_prices)} days)")
    print()

    # ── 3. IS Backtest (reference only) ──────────────────────────
    print("--- Running IS Backtest (reference) ---")
    is_result = run_backtest(is_prices)
    print_backtest_summary(is_result, benchmark_prices=is_prices, label="IN-SAMPLE")
    print_ticker_pnl_summary(is_result, label="IN-SAMPLE")

    # ── 4. OOS Backtest ───────────────────────────────────────────
    print("--- Running Out-of-Sample Backtest ---")
    oos_result = run_backtest(oos_buffer)

    # Trim reported results to the true OOS window (strip the warm-up buffer)
    oos_result.daily_pnl      = oos_result.daily_pnl[oos_result.daily_pnl.index >= oos_boundary]
    oos_result.weekly_pnl     = oos_result.weekly_pnl[oos_result.weekly_pnl.index >= oos_boundary]
    oos_result.monthly_pnl    = oos_result.monthly_pnl[oos_result.monthly_pnl.index >= oos_boundary]
    oos_result.cumulative_pnl = oos_result.daily_pnl.cumsum().rename("cumulative_pnl")

    print_backtest_summary(oos_result, benchmark_prices=oos_prices, label="OUT-OF-SAMPLE")
    print_ticker_pnl_summary(oos_result, label="OUT-OF-SAMPLE")


    # ── 5. Combined Equity Curve ──────────────────────────────────
    plot_equity_curve(
        is_result,
        oos_result=oos_result,
        benchmark_prices=clean_price_matrix,
        save_path="equity_curve_oos.png",
    )

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
