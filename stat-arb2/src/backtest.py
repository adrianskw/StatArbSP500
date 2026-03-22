"""
S-score threshold strategy with rolling recalibration and P&L tracking.

Strategy rules (Avellaneda & Lee, 2010):
    - LONG  when s_i < -S_ENTER   (spread is cheap relative to equilibrium)
    - SHORT when s_i > +S_ENTER   (spread is rich)
    - CLOSE when |s_i| < S_EXIT   (spread has mean-reverted)
    - STOP  when |s_i| > S_STOP   (spread has blown out — cut the loss)

The backtest walks forward through a clean price matrix:
    1.  At each recalibration date, fit PCA + diagonal OU on the trailing
        LOOKBACK_WINDOW days.
    2.  Run Hurst / VR / ADF on the resulting spreads; only trade stocks
        that pass all three tests.
    3.  Compute daily s-scores and apply the threshold rules above.
    4.  P&L is tracked as position × Δspread (log-return space), minus
        a flat transaction cost on each entry/exit.
"""

import logging
import time
import concurrent.futures
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.diagnostics import compute_spread_diagnostics
from src.modeling import StatisticalFactorModel

log = logging.getLogger(__name__)


# ── Configuration dataclass ───────────────────────────────────


@dataclass
class StrategyConfig:
    """All tuneable knobs for the backtest, with defaults from config.py."""

    s_enter: float = config.S_ENTER
    s_exit: float = config.S_EXIT
    s_stop: float = config.S_STOP
    execution_delay: int = 1
    max_hold_days: int = config.MAX_HOLD_DAYS
    recalib_days: int = config.RECALIB_DAYS
    lookback: int = config.LOOKBACK_WINDOW
    n_factors: int = config.N_FACTORS
    cost_bps: float = config.COST_BPS
    use_kelly: bool = config.USE_KELLY
    kelly_fraction: float = config.KELLY_FRACTION
    max_unit_size: float = config.MAX_UNIT_SIZE


# ── Backtest engine ───────────────────────────────────────────


@dataclass
class BacktestResult:
    """Container returned by ``run_backtest``."""

    daily_pnl: pd.Series             # date-indexed daily portfolio P&L
    weekly_pnl: pd.Series            # date-indexed weekly portfolio P&L
    monthly_pnl: pd.Series           # date-indexed monthly portfolio P&L
    cumulative_pnl: pd.Series        # running sum
    positions: pd.DataFrame          # (T_oos × N) position matrix (+1/0/−1)
    s_scores: pd.DataFrame           # (T_oos × N) daily s-scores
    daily_holdings: list[list[str]] = field(default_factory=list)  # list of tickers held each day
    daily_exposure: np.ndarray = field(default_factory=lambda: np.array([]))  # total gross exposure (abs units)
    trade_log: list[dict] = field(default_factory=list)
    n_recalibs: int = 0
    execution_time_secs: float = 0.0


def run_backtest(
    clean_prices: pd.DataFrame,
    cfg: StrategyConfig | None = None,
) -> BacktestResult:
    """
    Run the S-score statistical arbitrage backtest.

    Parameters
    ----------
    clean_prices : pd.DataFrame
        Cleaned price matrix (T × N), forward-filled, no NaNs.
    cfg : StrategyConfig, optional
        Strategy parameters.  Defaults to values from ``config.py``.

    Returns
    -------
    BacktestResult
    """
    if cfg is None:
        cfg = StrategyConfig()

    start_time = time.time()
    dates = clean_prices.index
    tickers = clean_prices.columns
    N = len(tickers)
    T = len(dates)

    if T <= cfg.lookback:
        raise ValueError(
            f"Need > {cfg.lookback} rows for at least 1 OOS day, got {T}."
        )

    # ── Allocate output arrays ────────────────────────────────
    oos_start = cfg.lookback          # first out-of-sample day index
    T_oos = T - oos_start
    oos_dates = dates[oos_start:]

    positions = np.zeros((T_oos, N), dtype=np.float64)   # +1 / 0 / −1
    s_scores_arr = np.full((T_oos, N), np.nan)
    daily_pnl = np.zeros(T_oos)
    daily_exposure = np.zeros(T_oos)
    daily_holdings: list[list[str]] = []
    trade_log: list[dict] = []

    # Carry forward live positions between days
    pos = np.zeros(N, dtype=np.float64)
    target_pos = np.zeros(N, dtype=np.float64)  # pending orders
    days_held = np.zeros(N, dtype=int)
    accum_pnl = np.zeros(N, dtype=np.float64)  # P&L accumulated during current trade lifecycle

    # ── Model & parameter state ───────────────────────────────
    model: StatisticalFactorModel | None = None
    mu: np.ndarray | None = None
    sigma_eq: np.ndarray | None = None
    tradeable_mask = np.zeros(N, dtype=bool)   # stocks passing all 3 tests
    days_since_calib = cfg.recalib_days         # force calibration on first day

    cost_frac = cfg.cost_bps / 1e4              # one-way cost as a fraction
    n_recalibs = 0
    spread_prev: np.ndarray = np.zeros(N)       # initialized to zero; overwritten on first loop tick

    # A single process pool is kept alive for the entire backtest run.
    # This eliminates the fork+IPC spawn overhead (~0.1–0.3s) that would
    # otherwise occur on each of the ~N_RECALIBS recalibration steps.
    _executor = concurrent.futures.ProcessPoolExecutor()
    try:
        for t_oos in range(T_oos):
            t_abs = oos_start + t_oos               # absolute index into clean_prices
            today = dates[t_abs]

            # ── Recalibrate? ──────────────────────────────────────
            if days_since_calib >= cfg.recalib_days:
                calib_slice = clean_prices.iloc[t_abs - cfg.lookback : t_abs]

                # Suppress verbose per-recalibration logging from sub-modules
                _model_logger = logging.getLogger("src.modeling")
                _diag_logger  = logging.getLogger("src.diagnostics")
                _prev_model_level = _model_logger.level
                _prev_diag_level  = _diag_logger.level
                _model_logger.setLevel(logging.WARNING)
                _diag_logger.setLevel(logging.WARNING)

                model = StatisticalFactorModel(n_factors=cfg.n_factors)
                model.process_pipeline(calib_slice, lookback=cfg.lookback)

                # OU parameters
                mu = model.mu                                # (N,)
                sigma_eq = model._compute_sigma_eq()         # (N,)

                # Universe filter: only trade stocks passing all 3 tests
                spread_df = pd.DataFrame(model.spread, columns=tickers)
                diag_tests = compute_spread_diagnostics(spread_df, executor=_executor)
                pass_set = set(
                    diag_tests.loc[diag_tests["tests_passed"] == 3, "ticker"]
                )
                tradeable_mask = np.array(
                    [t in pass_set for t in tickers], dtype=bool
                )

                days_since_calib = 0
                n_recalibs += 1

                # Restore log levels
                _model_logger.setLevel(_prev_model_level)
                _diag_logger.setLevel(_prev_diag_level)

            days_since_calib += 1

            # ── Compute today's spread & s-scores ─────────────────
            # Build today's spread value using the *current* model's betas
            # spread_i = cumsum of (log-return - factor_exposure × factor_returns)
            # Faster: just use the model's OLS to predict today's residual and
            #         cumulate from the last known spread level.
            if model is None or mu is None or sigma_eq is None:
                continue  # safety — shouldn't happen

            # Today's log return
            today_prices = clean_prices.iloc[t_abs].values       # (N,)
            yest_prices  = clean_prices.iloc[t_abs - 1].values   # (N,)
            today_logret = np.log(today_prices / yest_prices)     # (N,)

            # Project today's return through the calibration-period model to
            # strip out the systematic factor component.
            today_logret_scaled = model.logreturns_scaler.transform(
                today_logret.reshape(1, -1)
            )
            today_factor = model.pca.transform(today_logret_scaled)  # (1, K)

            if config.USE_RIDGE:
                today_predicted = model.linear_model_logreturns.predict(
                    today_factor
                ).flatten()                                           # (N,)
            else:
                # PCA back-projection: undo standardization (scale only, no mean)
                today_predicted = (today_factor @ model.pca.components_).flatten() * model.logreturns_scaler.scale_

            today_resid = today_logret - today_predicted             # (N,)

            # Update spread: S_t = S_{t-1} + ε_t
            if t_oos == 0:
                spread_now = model.spread[-1, :] + today_resid
            else:
                spread_now = spread_prev + today_resid

            # S-score
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.where(
                    sigma_eq > 0,
                    (spread_now - mu) / sigma_eq,
                    0.0,
                )
            s_scores_arr[t_oos] = s

            # ── P&L from carrying yesterday's position ────────────
            # Δspread = spread_now − spread_prev
            if t_oos > 0:
                delta_spread = spread_now - spread_prev
                pnl_contrib = pos * delta_spread
                daily_pnl[t_oos] = np.nansum(pnl_contrib)
                accum_pnl += pnl_contrib                      # Update per-ticker lifecycle P&L

            # ── 1-Day Execution Delay ─────────────────────────────
            # On day T, we execute the pending orders from day T-1 by syncing
            # target_pos -> pos. This must happen BEFORE force-close and days_held
            # so that all downstream logic sees the final live position for the day.
            if cfg.execution_delay > 0 and t_oos > 0:
                newly_entered = (target_pos != 0) & (pos == 0)  # stocks just promoted
                pos = target_pos.copy()
                # Prime accum_pnl with the entry cost for newly entered positions
                # (entry cost was deducted from daily_pnl on entry day, so we record
                # that cost in accum_pnl so trade-log P&L attribution is accurate.)
                for idx in np.where(newly_entered)[0]:
                    entry_cost = abs(pos[idx]) * cost_frac
                    accum_pnl[idx] = -entry_cost
                # days_held must reflect that newly promoted positions have lived 1 day
                days_held[newly_entered] = 1

            # Increment holding days for active positions
            days_held[pos != 0] += 1

            # ── Signal logic ──────────────────────────────────────
            # Phase 1 (vectorized): resolve clean exits and stop-losses for all
            # stocks simultaneously using NumPy boolean masks, bypassing the
            # Python interpreter for the majority of active positions.
            new_pos = pos.copy()

            is_long  = pos > 0
            is_short = pos < 0
            is_held  = pos != 0
            valid_s  = ~np.isnan(s)

            # Clean mean-reversion exits
            close_long  = is_long  & valid_s & (s > -cfg.s_exit)
            close_short = is_short & valid_s & (s <  cfg.s_exit)
            # Stop-loss exits
            stop_out    = is_held  & valid_s & (np.abs(s) > cfg.s_stop)
            # Time-stop exits (handled per-stock below for accurate trade_log)
            time_stop   = is_held & (days_held >= cfg.max_hold_days)

            # Exits to process this tick (time-stop takes priority over stop/close)
            to_close = (close_long | close_short | stop_out) & ~time_stop

            for i in np.where(to_close)[0]:
                size_to_close = pos[i]
                cost = abs(size_to_close) * cost_frac
                daily_pnl[t_oos] -= cost
                si = s[i]
                if stop_out[i]:
                    action = "STOP"
                    reason = f"|s|={abs(si):.2f} > {cfg.s_stop}"
                elif close_long[i]:
                    action = "CLOSE_LONG"
                    reason = f"s={si:.2f} > -{cfg.s_exit}"
                else:
                    action = "CLOSE_SHORT"
                    reason = f"s={si:.2f} < {cfg.s_exit}"
                trade_log.append({
                    "date": today, "ticker": tickers[i],
                    "action": action, "s_score": si,
                    "reason": reason,
                    "pnl": accum_pnl[i] - cost,
                    "size": size_to_close,
                })
                accum_pnl[i] = 0.0
                new_pos[i] = 0.0

            # Phase 2: per-stock loop — handles time-stops and entries only.
            # Exits and stop-losses are already resolved in Phase 1 above.
            for i in range(N):
                # If not tradeable, we can only process exits (if gracefully exiting is permitted)
                if not tradeable_mask[i] and pos[i] == 0:
                    continue

                si = s[i]
                if np.isnan(si):
                    continue

                currently_flat  = pos[i] == 0
                currently_long  = pos[i] > 0
                currently_short = pos[i] < 0

                # Time-Stop (Maximum Holding Period Exceeded)
                if not currently_flat and days_held[i] >= cfg.max_hold_days:
                    size_to_close = pos[i]
                    new_pos[i] = 0.0
                    cost = abs(size_to_close) * cost_frac
                    daily_pnl[t_oos] -= cost   # exit cost
                    trade_log.append({
                        "date": today, "ticker": tickers[i],
                        "action": "TIME_STOP", "s_score": si,
                        "reason": f"held {days_held[i]} days > {cfg.max_hold_days}",
                        "pnl": accum_pnl[i] - cost,
                        "size": size_to_close,
                    })
                    accum_pnl[i] = 0.0
                    continue

                # Entry
                if currently_flat and tradeable_mask[i]:
                    if cfg.use_kelly:
                        # Continuous Kelly: w = -s / (2 * sigma_eq)
                        kelly_weight = abs(si / (2.0 * sigma_eq[i])) * cfg.kelly_fraction
                        size = np.clip(kelly_weight, 0, cfg.max_unit_size)
                    else:
                        size = 1.0  # Default Unit sizing

                    if si < -cfg.s_enter:
                        new_pos[i] = size
                        cost = size * cost_frac
                        daily_pnl[t_oos] -= cost   # entry cost
                        accum_pnl[i] -= cost       # Initial cost
                        trade_log.append({
                            "date": today, "ticker": tickers[i],
                            "action": "LONG", "s_score": si,
                            "size": size,
                            "reason": f"s={si:.2f} < -{cfg.s_enter}, size={size:.2f}",
                        })
                    elif si > cfg.s_enter:
                        new_pos[i] = -size
                        cost = size * cost_frac
                        daily_pnl[t_oos] -= cost   # entry cost
                        accum_pnl[i] -= cost       # Initial cost
                        trade_log.append({
                            "date": today, "ticker": tickers[i],
                            "action": "SHORT", "s_score": si,
                            "size": -size,
                            "reason": f"s={si:.2f} > +{cfg.s_enter}, size={-size:.2f}",
                        })

            if cfg.execution_delay > 0:
                target_pos = new_pos
            else:
                pos = new_pos

            days_held[pos == 0] = 0

            # Track holdings and exposure
            current_daily_pos = np.where(pos != 0)[0]
            daily_holdings.append([tickers[idx] for idx in current_daily_pos])
            daily_exposure[t_oos] = np.sum(np.abs(pos))

            positions[t_oos] = pos
            spread_prev = spread_now

    finally:
        # Always shut down the executor cleanly, even if an exception occurs
        _executor.shutdown(wait=True)

    # ── Package results ───────────────────────────────────────
    daily_pnl_series   = pd.Series(daily_pnl, index=oos_dates, name="daily_pnl")
    weekly_pnl_series  = daily_pnl_series.resample("W-FRI").sum().rename("weekly_pnl")
    monthly_pnl_series = daily_pnl_series.resample("ME").sum().rename("monthly_pnl")
    cum_pnl            = daily_pnl_series.cumsum().rename("cumulative_pnl")

    positions_df = pd.DataFrame(positions, index=oos_dates, columns=tickers)
    s_scores_df  = pd.DataFrame(s_scores_arr, index=oos_dates, columns=tickers)

    execution_time = time.time() - start_time

    return BacktestResult(
        daily_pnl=daily_pnl_series,
        weekly_pnl=weekly_pnl_series,
        monthly_pnl=monthly_pnl_series,
        cumulative_pnl=cum_pnl,
        positions=positions_df,
        s_scores=s_scores_df,
        daily_holdings=daily_holdings,
        daily_exposure=daily_exposure,
        trade_log=trade_log,
        n_recalibs=n_recalibs,
        execution_time_secs=execution_time,
    )


# ── Reporting ─────────────────────────────────────────────────


def print_backtest_summary(
    result: BacktestResult,
    cfg: StrategyConfig | None = None,
    benchmark_prices: pd.DataFrame | None = None,
) -> None:
    """
    Print a concise human-readable summary of backtest results.

    Parameters
    ----------
    result : BacktestResult
    cfg : StrategyConfig, optional
    benchmark_prices : pd.DataFrame, optional
        Full price matrix (same tickers as the backtest) used to compute an
        equal-weight buy-and-hold benchmark over the OOS period.  When provided,
        the benchmark Sharpe and cumulative return are printed alongside the
        strategy figures.
    """
    if cfg is None:
        cfg = StrategyConfig()

    pnl = result.daily_pnl
    cum = result.cumulative_pnl
    n_days = len(pnl)

    total_return = cum.iloc[-1]
    sharpe = (pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0

    # Max drawdown
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = drawdown.min()

    # Win rate and daily trade statistics
    trades = pd.DataFrame(result.trade_log)
    if len(trades) > 0:
        trades_per_day = trades.groupby("date").size().reindex(pnl.index).fillna(0)
        avg_daily_trades = trades_per_day.mean()
        med_daily_trades = trades_per_day.median()
        closed = trades[trades["pnl"].notna()]
    else:
        avg_daily_trades = 0.0
        med_daily_trades = 0.0
        closed = pd.DataFrame()

    win_rate = (closed["pnl"] > 0).mean() if len(closed) > 0 else float("nan")
    avg_pnl  = closed["pnl"].mean() if len(closed) > 0 else float("nan")
    std_pnl  = closed["pnl"].std() if len(closed) > 1 else float("nan")

    pos_days = (pnl > 0).mean()
    pos_weeks = (result.weekly_pnl > 0).mean()
    pos_months = (result.monthly_pnl > 0).mean()

    print("\n" + "=" * 60)
    print("  BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Period          : {pnl.index[0].date()} → {pnl.index[-1].date()} ({n_days} days)")
    print(f"  Recalibrations  : {result.n_recalibs}")
    print(f"  Execution Time  : {result.execution_time_secs:.1f}s")
    print("-" * 60)
    print(f"  Total Return    : {total_return * 100:+.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:+.3f}  (annualised)")
    print(f"  Max Drawdown    : {max_dd * 100:.2f}%")
    print(f"  Trades          : {len(closed)}")
    print("-" * 60)
    print("  PERIODIC P&L BREAKDOWN (%)")
    print(f"  {'Period':<15s} {'% Positive':>10s} {'Avg (%)':>11s} {'StDev (%)':>10s}")
    print(f"  {'─' * 15} {'─' * 10} {'─' * 11} {'─' * 10}")

    if not np.isnan(avg_pnl):
        print(f"  {'Trade (Closed)':<15s} {win_rate:>10.1%} {avg_pnl * 100:>+11.3f} {std_pnl * 100:>10.3f}")
    else:
        print(f"  {'Trade (Closed)':<15s} {'N/A':>10s} {'N/A':>11s} {'N/A':>10s}")

    d_mean, d_std = pnl.mean(), pnl.std()
    print(f"  {'Daily':<15s} {pos_days:>10.1%} {d_mean * 100:>+11.3f} {d_std * 100:>10.3f}")

    w_mean, w_std = result.weekly_pnl.mean(), result.weekly_pnl.std()
    print(f"  {'Weekly':<15s} {pos_weeks:>10.1%} {w_mean * 100:>+11.3f} {w_std * 100:>10.3f}")

    m_mean, m_std = result.monthly_pnl.mean(), result.monthly_pnl.std()
    print(f"  {'Monthly':<15s} {pos_months:>10.1%} {m_mean * 100:>+11.3f} {m_std * 100:>10.3f}")

    # ── Exit mechanism breakdown ──────────────────────────────
    if len(closed) > 0:
        EXIT_LABELS = {
            "CLOSE_LONG":  "Mean-Rev (Long) ",
            "CLOSE_SHORT": "Mean-Rev (Short)",
            "STOP":        "Stop-Loss       ",
            "TIME_STOP":   "Time-Stop       ",
        }
        print("-" * 60)
        print("  EXIT MECHANISM BREAKDOWN (%)")
        print(f"  {'Exit Type':<20s} {'Count':>6s} {'Total (%)':>12s} {'% of P&L':>10s}")
        print(f"  {'─' * 20} {'─' * 6} {'─' * 12} {'─' * 10}")
        all_closed_pnl = closed["pnl"].sum()
        for action, label in EXIT_LABELS.items():
            grp = closed[closed["action"] == action]
            if len(grp) == 0:
                continue
            grp_pnl  = grp["pnl"].sum()
            pct      = (grp_pnl / all_closed_pnl * 100) if all_closed_pnl != 0 else 0.0
            print(f"  {label:<20s} {len(grp):>6d} {grp_pnl * 100:>+11.2f}% {pct:>+9.1f}%")

    # ── Benchmark comparison ──────────────────────────────────
    if benchmark_prices is not None:
        # Align benchmark to the OOS period
        bm = benchmark_prices.loc[pnl.index]
        bm_ret = np.log(bm / bm.shift(1)).dropna()
        bm_daily = bm_ret.mean(axis=1)   # equal-weight daily log-return
        bm_cum   = bm_daily.cumsum()
        bm_sharpe = (bm_daily.mean() / bm_daily.std() * np.sqrt(252)) if bm_daily.std() > 0 else 0.0
        bm_total  = bm_cum.iloc[-1]

        print("-" * 60)
        print("  BENCHMARK  (Equal-Weight S&P 500 Buy & Hold)")
        print(f"  Total Return    : {bm_total * 100:+.2f}%")
        print(f"  Sharpe Ratio    : {bm_sharpe:+.3f}  (annualised)")
        print(f"  Alpha (vs BM)   : {(total_return - bm_total) * 100:+.2f}%")

    print("=" * 60 + "\n")


def plot_equity_curve(
    result: BacktestResult,
    benchmark_prices: pd.DataFrame | None = None,
    save_path: str = "equity_curve.png",
) -> None:
    """
    Plot cumulative P&L and drawdown.  Saves to ``save_path``.

    Parameters
    ----------
    result : BacktestResult
    benchmark_prices : pd.DataFrame, optional
        Full price matrix aligned to the full date range.  When provided, an
        equal-weight buy-and-hold curve is overlaid on the equity chart.
    save_path : str
        File path for the saved PNG.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    pnl = result.daily_pnl
    cum = result.cumulative_pnl
    drawdown = cum - cum.cummax()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle("Statistical Arbitrage — Equity Curve", fontsize=14, fontweight="bold")

    # ── Equity curve ──────────────────────────────────────────
    ax1.plot(cum.index, cum.values, color="#2196F3", linewidth=1.5, label="Strategy")

    if benchmark_prices is not None:
        bm = benchmark_prices.loc[pnl.index]
        bm_ret = np.log(bm / bm.shift(1)).dropna()
        bm_cum = bm_ret.mean(axis=1).cumsum()
        ax1.plot(bm_cum.index, bm_cum.values, color="#FF9800", linewidth=1.2,
                 linestyle="--", label="Equal-Weight S&P 500")

    ax1.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax1.set_ylabel("Cumulative P&L (log-return units)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Drawdown ──────────────────────────────────────────────
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color="#F44336", alpha=0.5, label="Drawdown")
    ax2.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Equity curve saved to: {save_path}")
