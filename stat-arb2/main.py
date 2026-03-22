"""
Main pipeline for S&P 500 statistical arbitrage.

Loads historical price data, cleans it, runs the rolling S-score backtest,
and saves an equity curve chart.
"""

import logging

from src.loader import DataLoader
from src.backtest import run_backtest, print_backtest_summary, plot_equity_curve


def main():
    """Execute the statistical arbitrage data pipeline."""
    # ── Logging setup ─────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # ── 1. Data Loading & Cleaning ────────────────────────────────
    print("=" * 60)
    print("  S&P 500 Multivariate Statistical Arbitrage Pipeline")
    print("=" * 60)
    print()

    loader = DataLoader()
    loader.get_data()
    clean_price_matrix = loader.check_and_clean_missing_data()

    print(f"\nClean matrix: {clean_price_matrix.shape[0]} days x {clean_price_matrix.shape[1]} stocks")
    print(f"Date range: {clean_price_matrix.index[0].date()} to {clean_price_matrix.index[-1].date()}")
    print()

    # # ── 2. Modeling Pipeline ──────────────────────────────────────
    # factor_model = StatisticalFactorModel(n_factors=config.N_FACTORS)
    # factor_model.process_pipeline(clean_price_matrix)
    # print()

    # # ── 3. OU Diagnostics ─────────────────────────────────────────
    # diag = factor_model.get_diagnostics()


    # # ── 4. Full-Universe Spread Diagnostics ───────────────────────
    # print("--- Full-Universe Spread Diagnostics (all tickers) ---")
    # spread_df = pd.DataFrame(factor_model.spread, columns=factor_model.tickers)

    # print(f"  Running Hurst / VR / ADF on {spread_df.shape[1]} spreads...\n")
    # diag_tests = compute_spread_diagnostics(
    #     spread_df,
    #     ou_diagnostics=diag,
    # )

    # # Summary counts across full universe
    # n_all3  = int((diag_tests["tests_passed"] == 3).sum())
    # n_two   = int((diag_tests["tests_passed"] == 2).sum())
    # n_one   = int((diag_tests["tests_passed"] == 1).sum())
    # n_none  = int((diag_tests["tests_passed"] == 0).sum())
    # print(f"  Universe breakdown ({spread_df.shape[1]} stocks):")
    # print(f"    ALL 3 tests pass : {n_all3:>4d}")
    # print(f"    2/3  tests pass  : {n_two:>4d}")
    # print(f"    1/3  tests pass  : {n_one:>4d}")
    # print(f"    0/3  tests pass  : {n_none:>4d}")
    # print()

    # # Show only tickers that pass ALL 3 mean-reversion tests
    # all_pass = diag_tests[diag_tests["tests_passed"] == 3]
    # print(f"  Tickers passing ALL 3 tests ({len(all_pass)}/{spread_df.shape[1]}):\n")
    # print(f"  {'Ticker':<8s} {'Hurst':>7s} {'VR(5)':>7s} {'ADF p':>7s} {'Half-Life':>12s}")
    # print(f"  {'─' * 8} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 12}")
    # for _, row in all_pass.iterrows():
    #     adf_str = f"{row['adf_p']:.3f}" if not np.isnan(row['adf_p']) else "  n/a"
    #     hl_str  = f"{row['half_life_days']:.1f} days" if np.isfinite(row['half_life_days']) else "∞"
    #     print(
    #         f"  {row['ticker']:<8s} {row['hurst']:>7.4f} {row['vr']:>7.4f} "
    #         f"{adf_str:>7s} {hl_str:>12s}"
    #     )

    # print()

    # # ── 5. Summary Statistics ─────────────────────────────────────
    # theta_diag = np.diag(factor_model.theta)
    # n_mean_reverting = int(np.sum(theta_diag > 0))
    # print("--- Summary ---")
    # print(f"  Mean-reverting stocks (θ > 0): {n_mean_reverting} / {len(theta_diag)}")
    # print(f"  Median half-life: {np.median(diag['half_life_days'][np.isfinite(diag['half_life_days'])]):.1f} days")
    # print(f"  S-score range: [{diag['s_score'].min():.2f}, {diag['s_score'].max():.2f}]")
    # print(f"  Extreme s-scores (|s| > 2): {int((diag['s_score'].abs() > 2).sum())} stocks")
    # print()

    # ── 6. S-Score Backtest ───────────────────────────────────────
    print("--- Running S-Score Backtest ---")
    bt_result = run_backtest(clean_price_matrix)
    print_backtest_summary(bt_result, benchmark_prices=clean_price_matrix)

    # ── 7. Equity Curve Chart ─────────────────────────────────────
    plot_equity_curve(bt_result, benchmark_prices=clean_price_matrix,
                      save_path="equity_curve.png")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()