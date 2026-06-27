"""
CRUCIBLE Level 3: Architecture Breakthrough
============================================
Trials 22-30 budget.

Trial 22: Pure LSTM (seq_len=10)
Trial 23: Pure LSTM (seq_len=5, lighter)
Trial 24: Ensemble MLP+LSTM (60/40)
Trial 25: Ensemble MLP+LSTM (50/50)
Trial 26: Ensemble MLP+LSTM (70/30)
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os, sys, gc, json, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ.setdefault('PUSH_EMPTY_RESULT', 'true')
os.environ.setdefault('DATA_CACHE_MODE', 'offline')

from ml_engine import PyTorchDLModel, LSTMModel, EnsembleModel, apply_liquidity_gate
import torch.optim as optim
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

with open('.quantbot_data/default_genes.json') as f:
    GENES = json.load(f)
FEATURES = GENES['features']
BASELINE_SHARPE = GENES['baseline_sharpe']  # 1.83

# ── Data Loading ─────────────────────────────────────────────────────
def load_panel():
    log.info("Loading A-share data...")
    panel = pd.read_parquet(".quantbot_data/ashare_daily.parquet")
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
    panel['macro_staleness_days'] = 0

    panel = apply_liquidity_gate(panel, amihud_col='amihud_20', threshold_pct=0.90)
    panel = panel[~panel['is_limit']].copy()

    float_cols = panel.select_dtypes(include=['float64']).columns
    panel[float_cols] = panel[float_cols].astype(np.float32)

    return panel


def run_wfo_with_model(ml_df, feature_cols, model_factory, epochs=5, batch_size=2048):
    """WFO pass with a generic model factory function."""
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

        model = model_factory()
        model.train(train_df, feature_cols, target_col='fwd_ret_real', group_col='date',
                    epochs=epochs, batch_size=batch_size)

        test_df['score'] = model.predict(test_df, feature_cols)
        all_test_preds.append(test_df[['date', 'code', 'score', 'fwd_ret_real']])
        del model; gc.collect()

    if not all_test_preds:
        return None

    oos_df = pd.concat(all_test_preds, ignore_index=True)

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

    daily_ls = oos_df.groupby('date', group_keys=False).apply(
        lambda x: x[x['quantile'] == 5]['fwd_ret_real'].mean() - x[x['quantile'] == 1]['fwd_ret_real'].mean()
    ).fillna(0)
    sharpe = (daily_ls.mean() / daily_ls.std()) * np.sqrt(252) if daily_ls.std() > 0 else 0.0

    q5_daily = oos_df[oos_df['quantile'] == 5].groupby('date')['fwd_ret_real'].mean()
    q5_cum = (1 + q5_daily).cumprod()
    q5_dd = (q5_cum / q5_cum.cummax() - 1).min()

    # Win rate
    win_rate = (daily_ls > 0).mean()

    return {
        'spread_bps': float(spread),
        'sharpe': float(sharpe),
        'max_dd': float(q5_dd),
        'win_rate': float(win_rate),
        'n_oos_days': int(oos_df['date'].nunique()),
    }


if __name__ == '__main__':
    t0 = time.time()
    panel = load_panel()
    ml_df = panel.dropna(subset=FEATURES + ['fwd_ret_real', 'date']).copy()
    n_feat = len(FEATURES)

    configs = [
        ("LSTM_seq10", lambda: LSTMModel(n_feat, seq_len=10), 5),
        ("LSTM_seq5",  lambda: LSTMModel(n_feat, seq_len=5),  5),
        ("Ensemble_60_40", lambda: EnsembleModel(n_feat, seq_len=10, weights=[0.6, 0.4]), 5),
        ("Ensemble_50_50", lambda: EnsembleModel(n_feat, seq_len=10, weights=[0.5, 0.5]), 5),
        ("Ensemble_70_30", lambda: EnsembleModel(n_feat, seq_len=10, weights=[0.7, 0.3]), 5),
    ]

    results = {}
    trial_offset = 21

    for i, (name, factory, epochs) in enumerate(configs):
        trial_num = trial_offset + i + 1
        if trial_num > 30:
            log.error(f"[CRUCIBLE] Trial budget exhausted. Stopping.")
            break

        log.info(f"===== Trial {trial_num}/30: {name} =====")
        res = run_wfo_with_model(ml_df, FEATURES, factory, epochs=epochs)
        if res:
            results[name] = res
            delta = res['sharpe'] - BASELINE_SHARPE
            marker = " ★ NEW CHAMPION!" if res['sharpe'] > BASELINE_SHARPE else ""
            log.info(f"[{name}] Spread={res['spread_bps']:.2f} | Sharpe={res['sharpe']:.2f} (Δ={delta:+.2f}) | MaxDD={res['max_dd']:.2%} | WinRate={res['win_rate']:.1%}{marker}")
        else:
            log.error(f"[{name}] WFO failed.")
        gc.collect()

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info(f"LEVEL 3 ARCHITECTURE EXPLORATION COMPLETE ({elapsed/60:.1f} min)")
    log.info(f"MLP Baseline (ep5_lr0.0003): Sharpe={BASELINE_SHARPE}")
    log.info("=" * 70)

    ranked = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    for rank, (name, r) in enumerate(ranked, 1):
        delta = r['sharpe'] - BASELINE_SHARPE
        log.info(f"  #{rank} {name}: Spread={r['spread_bps']:.2f} | Sharpe={r['sharpe']:.2f} (Δ={delta:+.2f}) | MaxDD={r['max_dd']:.2%} | WinRate={r['win_rate']:.1%}")

    output = {
        'timestamp': datetime.now().isoformat(),
        'baseline_sharpe': BASELINE_SHARPE,
        'results': results,
        'trials_used': trial_offset + len(results),
    }
    with open('.quantbot_data/level3_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Results saved to .quantbot_data/level3_results.json")
