"""
backtest.py — Backtesting engine with walk-forward analysis and shorting engine.

Pipeline:
  1. Estimate (Johansen + PCA on train window)
  2. Warm-up (prepend zscore_window for z-score initialization)
  3. Generate signals on test window
  4. Portfolio weights via Shorting Engine
  5. Simulate trades with cost deduction and exit checks
  6. Aggregate fold results
"""

import numpy as np
import pandas as pd

from src import config
from src import data_loader, features, models


# ═══════════════════════════════════════════════════════════════
# Portfolio Weight Calculation (Shorting Engine)
# ═══════════════════════════════════════════════════════════════

def calculate_portfolio_weights(
    signals: pd.Series,
    johansen_weights: np.ndarray,
) -> pd.DataFrame:
    """
    Convert spread signals into individual stock positions.

    The Shorting Engine:
      - Signal = +1 (Long Spread):  Position = +1 × β  (Long the basket)
      - Signal = -1 (Short Spread): Position = -1 × β  (Short the basket)
      - Result: Dollar-neutral portfolio with some stocks Long, others Short.

    Parameters
    ----------
    signals : pd.Series
        Signal values (+1, -1, 0) indexed by date.
    johansen_weights : np.ndarray
        Shape (M,) — cointegrating vector for one spread.

    Returns
    -------
    pd.DataFrame
        Per-asset weights indexed by date.
    """
    weights = np.outer(signals.values, johansen_weights)
    return pd.DataFrame(weights, index=signals.index)


# ═══════════════════════════════════════════════════════════════
# Single-Pass Backtest
# ═══════════════════════════════════════════════════════════════

def run_backtest(
    prices: pd.DataFrame,
    weight_vectors: np.ndarray,
    signals_per_spread: dict[str, pd.Series],
    initial_capital: float = config.INITIAL_CAPITAL,
    position_size: float = config.POSITION_SIZE,
    stop_loss: float = config.STOP_LOSS,
    max_exposure: float = config.MAX_EXPOSURE,
    drawdown_limit: float = config.DRAWDOWN_LIMIT,
    transaction_cost_bps: float = config.TRANSACTION_COST_BPS,
    short_borrow_rate: float = config.SHORT_BORROW_RATE,
    max_half_life_hold: float = config.MAX_HALF_LIFE_HOLD,
    half_lives: dict[str, float] | None = None,
) -> dict:
    """
    Simulate a full backtest run across multiple spreads.

    Exit Conditions:
      1. Profit Take: z-score crosses exit_threshold
      2. Stop-Loss: position loss >= stop_loss
      3. Time Stop: holding > max_half_life_hold × half_life
      4. Drawdown: equity drawdown >= drawdown_limit
      5. Matrix Instability: condition_number > 1000 (checked externally)

    Returns
    -------
    dict with keys: 'equity_curve', 'returns', 'trades', 'metrics'
    """
    returns = features.compute_returns(prices)
    n_dates = len(returns)
    equity = np.full(n_dates, initial_capital, dtype=float)
    trade_log = []
    daily_cost_rate = short_borrow_rate / config.ANNUALIZATION_FACTOR  # borrowing cost per bar

    # Track positions per spread
    active_positions = {}  # spread_name -> {entry_date_idx, entry_value, direction}

    for t in range(1, n_dates):
        day_pnl = 0.0
        day_costs = 0.0

        for spread_name, sig_series in signals_per_spread.items():
            date = returns.index[t]
            if date not in sig_series.index:
                continue

            signal = sig_series.loc[date]
            prev_signal = sig_series.shift(1).loc[date] if date in sig_series.shift(1).dropna().index else 0

            # Position sizing
            allocated = equity[t - 1] * position_size

            # Check max exposure
            if abs(signal) > 0 and allocated / equity[t - 1] > max_exposure:
                continue

            # Entry
            if signal != 0 and prev_signal == 0:
                active_positions[spread_name] = {
                    "entry_idx": t,
                    "entry_equity": equity[t - 1],
                    "direction": signal,
                }

            # Exit checks
            if spread_name in active_positions:
                pos = active_positions[spread_name]
                hold_days = t - pos["entry_idx"]
                hl = half_lives.get(spread_name, np.inf) if half_lives else np.inf

                # Time stop
                if hold_days > max_half_life_hold * hl:
                    signal = 0  # Force exit
                    trade_log.append({
                        "spread": spread_name, "exit_date": date,
                        "reason": "time_stop", "hold_days": hold_days,
                    })
                    del active_positions[spread_name]

            # Compute spread return contribution
            spread_idx = int(spread_name.split("_")[-1]) - 1
            if spread_idx < weight_vectors.shape[0]:
                w = weight_vectors[spread_idx]
                asset_returns = returns.iloc[t].values
                spread_return = np.dot(w, asset_returns) * signal
                day_pnl += allocated * spread_return

                # Short borrow cost (on short legs)
                short_notional = allocated * np.sum(np.abs(w[w * signal < 0]))
                day_costs += short_notional * daily_cost_rate

            # Transaction costs on trade changes
            if signal != prev_signal:
                cost = allocated * (transaction_cost_bps / 10_000)
                day_costs += cost

            # Stop-loss check
            if spread_name in active_positions:
                pos = active_positions[spread_name]
                running_pnl = (equity[t - 1] + day_pnl - pos["entry_equity"]) / pos["entry_equity"]
                if running_pnl < -stop_loss:
                    trade_log.append({
                        "spread": spread_name, "exit_date": date,
                        "reason": "stop_loss", "pnl_pct": running_pnl,
                    })
                    del active_positions[spread_name]

            # Clean exit
            if signal == 0 and spread_name in active_positions:
                trade_log.append({
                    "spread": spread_name, "exit_date": date,
                    "reason": "signal_exit",
                })
                del active_positions[spread_name]

        equity[t] = equity[t - 1] + day_pnl - day_costs

        # Drawdown circuit breaker
        peak = np.max(equity[: t + 1])
        dd = (peak - equity[t]) / peak
        if dd >= drawdown_limit:
            trade_log.append({
                "spread": "ALL", "exit_date": returns.index[t],
                "reason": "drawdown_halt", "drawdown": dd,
            })
            # Flatten — fill remaining equity
            equity[t + 1 :] = equity[t]
            break

    equity_series = pd.Series(equity, index=returns.index, name="equity")
    pnl_returns = equity_series.pct_change().dropna()

    return {
        "equity_curve": equity_series,
        "returns": pnl_returns,
        "trades": trade_log,
        "final_equity": equity[-1],
    }


# ═══════════════════════════════════════════════════════════════
# Walk-Forward Analysis
# ═══════════════════════════════════════════════════════════════

def walk_forward_split(
    dates: pd.DatetimeIndex,
    train_window: int = config.WF_TRAIN_WINDOW,
    test_window: int = config.WF_TEST_WINDOW,
) -> list[dict]:
    """
    Generate rolling train/test date index pairs.

    Returns list of dicts: {'train_start', 'train_end', 'test_start', 'test_end'}
    """
    folds = []
    n = len(dates)
    start = 0

    while start + train_window + test_window <= n:
        folds.append({
            "train_start": dates[start],
            "train_end": dates[start + train_window - 1],
            "test_start": dates[start + train_window],
            "test_end": dates[min(start + train_window + test_window - 1, n - 1)],
        })
        start += test_window

    return folds


def walk_forward_backtest(
    prices: pd.DataFrame,
    train_window: int = config.WF_TRAIN_WINDOW,
    test_window: int = config.WF_TEST_WINDOW,
    **backtest_kwargs,
) -> dict:
    """
    Walk-forward backtest with rolling parameter estimation.

    For each fold:
      1. Estimate: Johansen/PCA on train window
      2. Warm-up: Prepend zscore_window from end of train
      3. Trade: Generate signals on test window with FIXED train params
      4. Collect OOS PnL

    Returns
    -------
    dict with 'fold_results', 'oos_equity', 'is_vs_oos_metrics'
    """
    folds = walk_forward_split(prices.index, train_window, test_window)
    fold_results = []
    warmup = config.ZSCORE_WINDOW

    for fold_idx, fold in enumerate(folds):
        # Train phase
        train_prices = prices.loc[fold["train_start"]:fold["train_end"]]

        # Johansen test
        joh = models.johansen_test(train_prices)
        rank = models.johansen_rank(joh)

        if rank == 0:
            fold_results.append({"fold": fold_idx, "status": "skipped", "reason": "rank_0"})
            continue

        weight_vectors = models.select_vectors(joh)

        # Test phase (with warm-up)
        warmup_start_idx = max(0, prices.index.get_loc(fold["test_start"]) - warmup)
        warmup_start = prices.index[warmup_start_idx]
        test_prices = prices.loc[warmup_start:fold["test_end"]]

        # Construct spreads and signals
        spreads = features.construct_spreads(test_prices, weight_vectors)
        signals_per_spread = {}
        half_lives = {}

        for col in spreads.columns:
            zscore = features.compute_zscore(spreads[col])
            signals = models.generate_signals(zscore)

            # Only keep test-window signals (after warm-up)
            test_mask = signals.index >= fold["test_start"]
            signals_per_spread[col] = signals[test_mask]

            hl = features.ou_half_life(spreads[col].dropna())
            half_lives[col] = hl

        # Run backtest on test window
        test_only_prices = prices.loc[fold["test_start"]:fold["test_end"]]
        result = run_backtest(
            test_only_prices,
            weight_vectors,
            signals_per_spread,
            half_lives=half_lives,
            **backtest_kwargs,
        )

        result["fold"] = fold_idx
        result["fold_dates"] = fold
        result["weight_vectors"] = weight_vectors
        result["rank"] = rank
        fold_results.append(result)

    return aggregate_oos_results(fold_results)


def aggregate_oos_results(fold_results: list[dict]) -> dict:
    """
    Stitch OOS equity curves and compare IS vs. OOS metrics.

    Returns
    -------
    dict with 'oos_equity_curve', 'fold_results', 'summary'
    """
    valid_folds = [f for f in fold_results if "equity_curve" in f]

    if not valid_folds:
        return {
            "oos_equity_curve": pd.Series(dtype=float),
            "fold_results": fold_results,
            "n_folds": len(fold_results),
            "n_valid_folds": 0,
            "n_skipped": len(fold_results),
            "summary": "No valid folds (all rank 0).",
        }

    # Stitch equity curves
    oos_curves = []
    for f in valid_folds:
        oos_curves.append(f["equity_curve"])

    oos_equity = pd.concat(oos_curves)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="first")]

    return {
        "oos_equity_curve": oos_equity,
        "fold_results": fold_results,
        "n_folds": len(fold_results),
        "n_valid_folds": len(valid_folds),
        "n_skipped": len(fold_results) - len(valid_folds),
    }
