
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import config, data_loader, features, models, backtest, utils

def main():
    # ── Execution Mode ──
    # "ideal"     → zero friction (no costs, no borrow fees, no stop-loss, no drawdown limit)
    # "realistic" → production-grade costs and risk limits
    EXECUTION_MODE = 'ideal'
    config.apply_mode(EXECUTION_MODE)

    # ── Override defaults here ──
    SECTOR = 'tech' #config.DEFAULT_SECTOR        # e.g. 'tech', 'energy', 'semiconductors'
    TICKERS = None                         # None = use sector default, or ['AAPL', 'MSFT', ...]
    START_DATE = config.DEFAULT_START_DATE  # "2013-01-01"
    END_DATE = config.DEFAULT_END_DATE     # "2018-12-31"

    print(f'=== Execution Mode: {config.EXECUTION_MODE.upper()} ===')
    print(f'  Transaction Cost: {config.TRANSACTION_COST_BPS} bps')
    print(f'  Short Borrow Rate: {config.SHORT_BORROW_RATE}')
    print(f'  Stop Loss: {config.STOP_LOSS}')
    print(f'  Drawdown Limit: {config.DRAWDOWN_LIMIT}')
    print()
    print(f'Sector: {SECTOR}')
    print(f'Tickers: {TICKERS or config.TICKER_UNIVERSES[SECTOR]}')
    print(f'Window: {START_DATE} -> {END_DATE}')

    # ── 2. Data Loading ──
    print('\n--- 2. Data Loading ---')
    prices = data_loader.load_prices(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        sector=SECTOR,
    )

    print(f'Shape: {prices.shape}')
    if not prices.empty:
        print(f'Date range: {prices.index[0].date()} -> {prices.index[-1].date()}')
    print(f'Missing values:\n{prices.isnull().sum()}')
    
    if prices.empty:
        print("No price data loaded. Exiting.")
        return

    # ── 3. PCA Analysis ──
    print('\n--- 3. PCA Analysis ---')
    returns = features.compute_returns(prices)

    # PCA decomposition
    pca_result = features.run_pca(returns, n_components=config.N_COMPONENTS)

    print('Explained Variance Ratio:')
    for i, ev in enumerate(pca_result['explained_variance_ratio']):
        print(f'  PC{i+1}: {ev:.4f} ({ev*100:.1f}%)')
    print(f'  Total: {sum(pca_result["explained_variance_ratio"]):.4f}')

    # Eigenportfolio weights
    weight_df = pd.DataFrame(
        pca_result['components'],
        columns=prices.columns,
        index=[f'PC{i+1}' for i in range(config.N_COMPONENTS)],
    )
    print('\nEigenportfolio Weights (sparse):')
    print(weight_df.round(4))

    # ── 4. Johansen Cointegration ──
    print('\n--- 4. Johansen Cointegration ---')
    # Johansen test
    joh = models.johansen_test(prices)
    rank = models.johansen_rank(joh)

    print(f'Cointegration Rank: {rank}')
    print(f'  -> {rank} independent mean-reverting relationships found\n')

    # Trace statistics
    trace_df = models.johansen_trace_statistic(joh)
    print('Trace Test Results:')
    print(trace_df)

    # Select significant vectors
    weight_vectors = models.select_vectors(joh)
    print(f'\nSelected {weight_vectors.shape[0]} weight vector(s)')

    if weight_vectors.size > 0:
        vec_df = pd.DataFrame(weight_vectors, columns=prices.columns,
                              index=[f'Vector {i+1}' for i in range(weight_vectors.shape[0])])
        print(vec_df.round(4))
    
    # ── 5. Spread Construction & Z-Scores ──
    print('\n--- 5. Spread Construction & Z-Scores ---')
    if weight_vectors.size > 0:
        spreads = features.construct_spreads(prices, weight_vectors)
        print(f'Constructed {spreads.shape[1]} spread(s)')
        
        # Diagnostics for first spread
        spread_1 = spreads.iloc[:, 0]
        zscore_1 = features.compute_zscore(spread_1)
        bands_1 = features.zscore_percentile_bands(spread_1)
        
        # Mean reversion diagnostics
        hl = features.ou_half_life(spread_1)
        h = features.hurst_exponent(spread_1)
        vr = features.variance_ratio_test(spread_1)
        zcr = features.zero_crossing_rate(spread_1)
        
        print(f'Spread 1 Diagnostics:')
        print(f'  Half-Life:        {hl:.1f} days')
        print(f'  Hurst Exponent:   {h:.4f} ({"Mean-Reverting" if h < 0.5 else "Trending"})')
        print(f'  Zero-Crossing:    {zcr:.4f}')
        print(f'  Variance Ratios:  {vr}')
        
        # Plot
        # In a script, plt.show() blocks execution. We might want to save or just show if interactive.
        # For now, we will execute plt.show() as requested but warn it might block.
        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            utils.plot_spread(spread_1, zscore_1, ax=axes[0])
            utils.plot_zscore_bands(spread_1, bands_1, ax=axes[1])
            plt.tight_layout()
            plt.show() # Note: This will block if a display is available until closed
        except Exception as e:
            print(f"Skipping plot display: {e}")

    else:
        print('No cointegration found. Cannot construct spreads.')

    # ── 6. Signal Generation & Backtest ──
    print('\n--- 6. Signal Generation & Backtest ---')
    if weight_vectors.size > 0:
        # Generate signals for each spread
        signals_per_spread = {}
        half_lives = {}
        
        for col in spreads.columns:
            z = features.compute_zscore(spreads[col])
            bands = features.zscore_percentile_bands(spreads[col])
            sig = models.generate_signals(z, zscore_percentiles=bands)
            signals_per_spread[col] = sig
            half_lives[col] = features.ou_half_life(spreads[col])
        
        # Run backtest
        bt_result = backtest.run_backtest(
            prices,
            weight_vectors,
            signals_per_spread,
            initial_capital=config.INITIAL_CAPITAL,
            position_size=config.POSITION_SIZE,
            stop_loss=config.STOP_LOSS,
            max_exposure=config.MAX_EXPOSURE,
            drawdown_limit=config.DRAWDOWN_LIMIT,
            transaction_cost_bps=config.TRANSACTION_COST_BPS,
            short_borrow_rate=config.SHORT_BORROW_RATE,
            max_half_life_hold=config.MAX_HALF_LIFE_HOLD,
            half_lives=half_lives,
        )
        
        print(f'Final Equity: ${bt_result["final_equity"]:,.2f}')
        print(f'Return: {(bt_result["final_equity"] / config.INITIAL_CAPITAL - 1) * 100:.2f}%')
        print(f'Trades: {len(bt_result["trades"])}')
        print(f'Sharpe Ratio: {utils.net_sharpe_ratio(bt_result["returns"]):.2f}')
        print(f'Max Drawdown: {utils.max_drawdown(bt_result["equity_curve"]):.2%}')
        
        # Equity curve
        try:
            fig, ax = plt.subplots(figsize=(14, 5))
            utils.plot_equity_curve(bt_result['equity_curve'], ax=ax)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Skipping plot display: {e}")
    else:
        print('No cointegration -- skipping backtest.')

    # ── 7. Walk-Forward Analysis ──
    print('\n--- 7. Walk-Forward Analysis ---')
    # Walk-forward backtest
    wf_result = backtest.walk_forward_backtest(
        prices,
        train_window=config.WF_TRAIN_WINDOW,
        test_window=config.WF_TEST_WINDOW,
    )

    print(f'Walk-Forward Results:')
    print(f'  Total Folds: {wf_result["n_folds"]}')
    print(f'  Valid Folds: {wf_result["n_valid_folds"]}')
    print(f'  Skipped (Rank 0): {wf_result["n_skipped"]}')

    # Plot OOS equity curve
    if not wf_result['oos_equity_curve'].empty:
        try:
            fig, ax = plt.subplots(figsize=(14, 5))
            utils.plot_equity_curve(wf_result['oos_equity_curve'], ax=ax)
            ax.set_title('Walk-Forward OOS Equity Curve', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Skipping plot display: {e}")

    # Fold summary
    fold_summary = []
    for f in wf_result['fold_results']:
        if 'equity_curve' in f:
            ret = f['returns']
            fold_summary.append({
                'Fold': f['fold'],
                'Rank': f.get('rank', 0),
                'Sharpe': utils.net_sharpe_ratio(ret),
                'Max DD': utils.max_drawdown(f['equity_curve']),
                'Final $': f['final_equity'],
            })
        else:
            fold_summary.append({
                'Fold': f['fold'],
                'Rank': 0,
                'Sharpe': np.nan,
                'Max DD': np.nan,
                'Final $': np.nan,
            })

    fold_df = pd.DataFrame(fold_summary)
    print(fold_df)

if __name__ == '__main__':
    main()
