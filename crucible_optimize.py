"""
CRUCIBLE Level 1 Parameter Optimization + B&H Benchmark
========================================================
Sweeps epochs and learning rate on the LOCKED Baseline feature set.
Also computes Buy & Hold benchmark for factor contribution assessment.

Trial Budget: 12 configs (4 epochs × 3 lr) out of 30 max.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os, sys, gc, json, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ.setdefault('PUSH_EMPTY_RESULT', 'true')
os.environ.setdefault('DATA_CACHE_MODE', 'offline')

from ml_engine import PyTorchDLModel, apply_liquidity_gate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# ── Load locked genes ────────────────────────────────────────────────
with open('.quantbot_data/default_genes.json') as f:
    GENES = json.load(f)
FEATURE_COLS = GENES['features']
log.info(f"Locked features: {FEATURE_COLS}")

# ── Load & preprocess data (single pass, reuse across all trials) ──
def load_panel():
    parquet_path = ".quantbot_data/ashare_daily.parquet"
    macro_path = ".quantbot_data/macro_daily.parquet"

    log.info("Loading A-share data...")
    panel = pd.read_parquet(parquet_path)
    panel = panel.rename(columns={'volume': 'vol'})
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel[panel['date'] >= pd.to_datetime("2020-01-01")].copy()
    panel['code'] = panel['code'].str.replace('sh.', '', regex=False).str.replace('sz.', '', regex=False).str.replace('bj.', '', regex=False)

    for col in ['open', 'high', 'low', 'close', 'vol']:
        panel[col] = pd.to_numeric(panel[col], errors='coerce')

    panel['prev_close'] = panel.groupby('code')['close'].shift(1)
    panel['pct_chg'] = (panel['close'] / panel['prev_close'] - 1) * 100
    panel['is_limit'] = (panel['pct_chg'].abs() >= 9.5) & (panel['high'] == panel['low'])

    panel['next_open'] = panel.groupby('code')['open'].shift(-1)
    panel['next_close'] = panel.groupby('code')['close'].shift(-1)
    panel['fwd_ret_real'] = panel['next_close'] / (panel['next_open'] + 1e-5) - 1
    panel = panel.dropna(subset=['fwd_ret_real'])

    # Factors
    log.info("Computing factors...")
    amihud_raw = panel['pct_chg'].abs() / (panel['vol'] * panel['close'] + 1e-5) * 1e6
    panel['amihud'] = np.where(panel.get('is_limit', False), 99999.0, amihud_raw)
    panel['amihud_20'] = panel.groupby('code')['amihud'].transform(lambda x: x.rolling(20).mean())
    panel['clv'] = (panel['close'] - panel['low']) / (panel['high'] - panel['low'] + 1e-8)
    panel['volatility_5d'] = panel.groupby('code')['pct_chg'].transform(lambda x: x.rolling(5).std())
    panel['alpha_reversal_5d'] = -(panel['close'] / panel.groupby('code')['close'].shift(5) - 1)
    panel['alpha_024_approx'] = panel.groupby('code')['close'].transform(lambda x: x.rolling(20).mean()) / (panel['close'] + 1e-5) - 1

    market_daily = panel.groupby('date')['pct_chg'].mean().reset_index()
    market_daily.rename(columns={'pct_chg': 'market_ret'}, inplace=True)
    market_daily['market_ret_20d'] = market_daily['market_ret'].rolling(20, min_periods=5).mean()
    market_daily['market_ret_60d'] = market_daily['market_ret'].rolling(60, min_periods=20).mean()
    market_daily['market_vol_20d'] = market_daily['market_ret'].rolling(20, min_periods=5).std()
    panel = pd.merge(panel, market_daily[['date', 'market_ret_20d', 'market_ret_60d', 'market_vol_20d']], on='date', how='left')

    # macro_staleness_days: days since last macro update (placeholder = 0 if not available)
    if 'macro_staleness_days' not in panel.columns:
        macro_path_check = '.quantbot_data/macro_daily.parquet'
        if os.path.exists(macro_path_check):
            macro_df = pd.read_parquet(macro_path_check)
            if 'macro_staleness_days' in macro_df.columns:
                panel = pd.merge(panel, macro_df[['macro_staleness_days']], left_on='date', right_index=True, how='left')
                panel['macro_staleness_days'] = panel['macro_staleness_days'].ffill().fillna(0)
            else:
                panel['macro_staleness_days'] = 0
        else:
            panel['macro_staleness_days'] = 0

    # Liquidity gate + limit filter
    panel = apply_liquidity_gate(panel, amihud_col='amihud_20', threshold_pct=0.90)
    panel = panel[~panel['is_limit']].copy()

    # Downcast to float32
    float_cols = panel.select_dtypes(include=['float64']).columns
    panel[float_cols] = panel[float_cols].astype(np.float32)

    ml_df = panel.dropna(subset=FEATURE_COLS + ['fwd_ret_real', 'date']).copy()
    log.info(f"Final ML dataset: {len(ml_df)} rows, {ml_df['date'].nunique()} unique dates")
    return ml_df, panel


def run_wfo(ml_df, epochs, lr, batch_size=2048):
    """Single WFO pass with given hyperparameters."""
    dates = sorted(ml_df['date'].unique())
    train_window = 500
    step = 125

    all_test_preds = []
    for idx in range(train_window, len(dates), step):
        train_dates = dates[max(0, idx - train_window):idx]
        test_dates = dates[idx:min(len(dates), idx + step)]
        if len(test_dates) == 0:
            break

        train_df = ml_df[ml_df['date'].isin(train_dates)].copy()
        test_df = ml_df[ml_df['date'].isin(test_dates)].copy()

        model = PyTorchDLModel(input_dim=len(FEATURE_COLS))
        # Override lr
        import torch.optim as optim
        model.optimizer = optim.Adam(model.model.parameters(), lr=lr)

        model.train(train_df, FEATURE_COLS, target_col='fwd_ret_real', group_col='date',
                    epochs=epochs, batch_size=batch_size)

        test_df['score'] = model.predict(test_df, FEATURE_COLS)
        all_test_preds.append(test_df[['date', 'code', 'score', 'fwd_ret_real']])

        del model
        gc.collect()

    if not all_test_preds:
        return None

    oos_df = pd.concat(all_test_preds, ignore_index=True)

    # Quantile scoring
    def _qcut(g):
        if len(g) < 5:
            return pd.Series(index=g.index, dtype=float)
        try:
            return pd.qcut(g['score'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except:
            return pd.Series(index=g.index, dtype=float)

    oos_df['quantile'] = oos_df.groupby('date', group_keys=False).apply(_qcut)
    oos_df = oos_df.dropna(subset=['quantile'])

    group_returns = oos_df.groupby('quantile', observed=False)['fwd_ret_real'].mean() * 10000
    spread = group_returns.get(5, 0) - group_returns.get(1, 0)

    # Sharpe
    daily_ls = oos_df.groupby('date', group_keys=False).apply(
        lambda x: x[x['quantile'] == 5]['fwd_ret_real'].mean() - x[x['quantile'] == 1]['fwd_ret_real'].mean()
    ).fillna(0)

    sharpe = (daily_ls.mean() / daily_ls.std()) * np.sqrt(252) if daily_ls.std() > 0 else 0.0

    # MaxDD of Q5 portfolio
    q5_daily = oos_df[oos_df['quantile'] == 5].groupby('date')['fwd_ret_real'].mean()
    q5_cum = (1 + q5_daily).cumprod()
    q5_peak = q5_cum.cummax()
    q5_dd = (q5_cum / q5_peak - 1).min()

    return {
        'spread_bps': float(spread),
        'sharpe': float(sharpe),
        'max_dd': float(q5_dd),
        'q5_mean_bps': float(group_returns.get(5, 0)),
        'q1_mean_bps': float(group_returns.get(1, 0)),
        'n_oos_days': int(oos_df['date'].nunique()),
    }


def calc_bh_benchmark(panel):
    """Equal-weighted Buy & Hold benchmark over the same OOS period."""
    log.info("Computing B&H benchmark...")
    # Use the same date range as WFO OOS (after first 500 trading days)
    dates = sorted(panel['date'].unique())
    if len(dates) < 500:
        return None
    oos_start = dates[500]

    oos_panel = panel[panel['date'] >= oos_start].copy()
    daily_ret = oos_panel.groupby('date')['fwd_ret_real'].mean()

    cum_ret = (1 + daily_ret).cumprod()
    peak = cum_ret.cummax()
    max_dd = (cum_ret / peak - 1).min()

    total_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
    n_years = len(daily_ret) / 252
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

    return {
        'total_return': float(total_ret),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'n_days': int(len(daily_ret)),
    }


if __name__ == '__main__':
    t0 = time.time()
    ml_df, panel = load_panel()

    # ── B&H Benchmark ──
    bh = calc_bh_benchmark(panel)
    log.info(f"B&H Benchmark: CAGR={bh['cagr']:.2%}, Sharpe={bh['sharpe']:.2f}, MaxDD={bh['max_dd']:.2%}")

    # ── Parameter Sweep ──
    EPOCH_GRID = [5, 10, 15]
    LR_GRID = [0.0003, 0.001, 0.003]

    results = {}
    trial_count = 0

    for epochs in EPOCH_GRID:
        for lr in LR_GRID:
            trial_count += 1
            config_name = f"ep{epochs}_lr{lr}"
            log.info(f"===== Trial {trial_count}/9: {config_name} =====")

            res = run_wfo(ml_df, epochs=epochs, lr=lr)
            if res:
                results[config_name] = res
                log.info(f"[{config_name}] Spread={res['spread_bps']:.2f} bps | Sharpe={res['sharpe']:.2f} | MaxDD={res['max_dd']:.2%}")

                # B8 auto-rollback: if Sharpe drops >10% vs baseline
                if res['sharpe'] < GENES['baseline_sharpe'] * 0.9:
                    log.warning(f"[{config_name}] Sharpe {res['sharpe']:.2f} < 90% of baseline {GENES['baseline_sharpe']}. Flagged.")
            else:
                log.error(f"[{config_name}] WFO produced no results.")

            gc.collect()

    # ── Summary ──
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"OPTIMIZATION COMPLETE ({elapsed/60:.1f} min, {trial_count} trials)")
    log.info("=" * 60)
    log.info(f"B&H Benchmark: CAGR={bh['cagr']:.2%} | Sharpe={bh['sharpe']:.2f} | MaxDD={bh['max_dd']:.2%}")
    log.info("-" * 60)

    # Sort by Sharpe
    ranked = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    for rank, (name, r) in enumerate(ranked, 1):
        marker = " ★" if r['sharpe'] > GENES['baseline_sharpe'] else ""
        log.info(f"  #{rank} {name}: Spread={r['spread_bps']:.2f} bps | Sharpe={r['sharpe']:.2f} | MaxDD={r['max_dd']:.2%}{marker}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'benchmark_bh': bh,
        'baseline_reference': {'sharpe': GENES['baseline_sharpe'], 'spread_bps': GENES['baseline_spread_bps']},
        'sweep_results': results,
        'best_config': ranked[0][0] if ranked else None,
        'best_sharpe': ranked[0][1]['sharpe'] if ranked else None,
        'trials_used': trial_count,
    }
    with open('.quantbot_data/optimization_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Results saved to .quantbot_data/optimization_results.json")
