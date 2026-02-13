"""
dashboard.py — Metrics dashboards (Essential + Full).

Essential Dashboard: 7 scalar metrics + 6 time-varying plots
Full Dashboard: all 28 metrics + 13 time-varying plots
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src import utils, features, config


# ═══════════════════════════════════════════════════════════════
# Scalar Summary Table
# ═══════════════════════════════════════════════════════════════

def render_scalar_table(metrics: dict, ax=None) -> None:
    """Render scalar metrics as a formatted table in a matplotlib axes."""
    scalar_keys = [
        k for k, v in metrics.items()
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)
    ]

    if not scalar_keys:
        return

    rows = []
    for k in sorted(scalar_keys):
        val = metrics[k]
        if isinstance(val, float):
            if abs(val) < 0.01 and val != 0:
                fmt = f"{val:.6f}"
            elif abs(val) > 1000:
                fmt = f"{val:,.0f}"
            else:
                fmt = f"{val:.4f}"
        else:
            fmt = str(val)
        rows.append([k.replace("_", " ").title(), fmt])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, max(2, len(rows) * 0.4)))

    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(2):
        cell = table[0, j]
        cell.set_facecolor("#2d3436")
        cell.set_text_props(color="white", fontweight="bold")

    return ax


# ═══════════════════════════════════════════════════════════════
# Essential Dashboard
# ═══════════════════════════════════════════════════════════════

def build_essential_dashboard(results: dict) -> None:
    """
    Key metrics at a glance.

    Scalar panel (7): hurst_exponent, information_ratio, ou_half_life,
        max_drawdown, net_sharpe_ratio, johansen_rank, sortino_ratio

    Plots (6): equity_curve, spread, rolling_half_life, rolling_ic,
        turnover_series, rolling_mahalanobis
    """
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("Essential Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # Grid: 4 rows × 2 cols = 8 panels
    # Row 0: Scalar table (span 2 cols)
    # Row 1: Equity curve | Spread
    # Row 2: Rolling HL | Rolling IC
    # Row 3: Turnover | Mahalanobis

    ax_table = fig.add_subplot(4, 2, (1, 2))
    essential_scalars = {
        k: results[k]
        for k in ["hurst_exponent", "information_ratio", "ou_half_life",
                   "max_drawdown", "net_sharpe_ratio", "johansen_rank", "sortino_ratio"]
        if k in results
    }
    render_scalar_table(essential_scalars, ax=ax_table)
    ax_table.set_title("Key Metrics", fontsize=13, fontweight="bold", pad=20)

    plot_data = [
        ("equity_curve", utils.plot_equity_curve),
        ("spread", lambda s, ax: utils.plot_spread(s, ax=ax)),
        ("rolling_half_life", utils.plot_rolling_half_life),
        ("rolling_ic", utils.plot_rolling_ic),
        ("turnover_ratio", utils.plot_turnover_series),
        ("mahalanobis_distance", utils.plot_rolling_mahalanobis),
    ]

    for idx, (key, plot_fn) in enumerate(plot_data):
        if key in results and results[key] is not None:
            ax = fig.add_subplot(4, 2, idx + 3)
            try:
                plot_fn(results[key], ax=ax)
            except Exception:
                ax.text(0.5, 0.5, f"No data: {key}", ha="center", va="center",
                        transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ═══════════════════════════════════════════════════════════════
# Full Dashboard
# ═══════════════════════════════════════════════════════════════

def build_full_dashboard(results: dict) -> None:
    """
    All 28 metrics + all 13 time-varying plots.

    Scalar panel: all PnL scalars + all Model scalars
    Time-varying plots (13):
        rolling_half_life, rolling_ic, mahalanobis_distance,
        zscore_bands, zscore_decay, crossing_frequency,
        turnover_series, crowding, attribution,
        pca_persistence, condition_number,
        copula_concordance, tail_dependence
    """
    # ── Scalar Summary ──
    fig_scalar, ax_scalar = plt.subplots(figsize=(8, 12))
    render_scalar_table(results, ax=ax_scalar)
    ax_scalar.set_title("All Scalar Metrics", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()

    # ── Time-Varying Plots ──
    plot_configs = [
        ("rolling_half_life", utils.plot_rolling_half_life),
        ("rolling_ic", utils.plot_rolling_ic),
        ("mahalanobis_distance", utils.plot_rolling_mahalanobis),
        ("zscore_bands", lambda d, ax: utils.plot_zscore_bands(
            results.get("spread", pd.Series()), d, ax=ax)),
        ("zscore_decay", utils.plot_zscore_decay),
        ("mean_crossing_count", utils.plot_crossing_frequency),
        ("turnover_ratio", utils.plot_turnover_series),
        ("crowding_sensitivity", utils.plot_crowding),
        ("attribution", utils.plot_attribution),
        ("pca_persistence", utils.plot_pca_persistence),
        ("condition_number", utils.plot_condition_number),
        ("copula_concordance", utils.plot_copula_concordance),
        ("tail_dependence", utils.plot_tail_dependence),
    ]

    # Filter to available data
    available = [(k, fn) for k, fn in plot_configs if k in results and results[k] is not None]

    if not available:
        print("No time-varying data available for plotting.")
        return

    n_plots = len(available)
    n_cols = 2
    n_rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle("Time-Varying Diagnostics", fontsize=16, fontweight="bold", y=1.01)

    axes_flat = axes.flatten() if n_plots > 2 else [axes] if n_plots == 1 else axes

    for idx, (key, plot_fn) in enumerate(available):
        try:
            plot_fn(results[key], ax=axes_flat[idx])
        except Exception as e:
            axes_flat[idx].text(0.5, 0.5, f"Error: {key}\n{e}",
                                ha="center", va="center", transform=axes_flat[idx].transAxes)

    # Hide unused axes
    for idx in range(len(available), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
