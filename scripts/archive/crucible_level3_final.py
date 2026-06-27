"""
CRUCIBLE Level 3 (Final Attempt): Deep Micro-Structural Features
================================================================
Trial 27: Introduce Beta, VoV, Skewness, and Cross-sectional Ranks.

If this fails to surpass Sharpe 1.83, we declare the ceiling reached.
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
import torch.optim as optim
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

with open('.quantbot_data/default_genes.json') as f:
    GENES = json.load(f)
BASELINE_FEATURES = GENES['features']
BASELINE_SHARPE = GENES['baseline_sharpe']  # 1.83

def load_panel():
    log.info("Loading A-share data...")
    panel = pd.read_parquet(".quantbot_data/ashare_daily.parquet")
    panel = panel.rename(columns={'volume': 'vol'})
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel[panel['date'] >= pd.to_datetime("2020-01-01")].copy()
    panel['code'] = panel['code'].str.replace('sh.', '', regex=False)\
                        .str.replace('sz.', '', regex=False)\
                        .str.replace('bj.', '', regex=False)

    for col in ['open', 'high', 'low', 'close', 'vol']:
        panel[col] = pd.to_numeric(panel[col], errors='coerce')

    panel['prev_close'] = panel.groupby('code')['close'].shift(1)
    panel['pct_chg'] = (panel['close'] / panel['prev_close'] - 1) * 100
    panel['is_limit'] = (panel['pct_chg'].abs() >= 9.5) & (panel['high'] == panel['low'])
    panel['next_open'] = panel.groupby('code')['open'].shift(-1)
    panel['next_close'] = panel.groupby('code')['close'].shift(-1)
    panel['fwd_ret_real'] = panel['next_close'] / (panel['next_open'] + 1e-5) - 1
    panel = panel.dropna(subset=['fwd_ret_real'])

    # --- Baseline Features ---
    log.info("Computing Baseline Factors...")
    amihud_raw = panel['pct_chg'].abs() / (panel['vol'] * panel['close'] + 1e-5) * 1e6
    panel['amihud'] = np.where(panel.get('is_limit', False), 99999.0, amihud_raw)
    panel['amihud_20'] = panel.groupby('code')['amihud'].transform(lambda x: x.rolling(20).mean())
    panel['clv'] = (panel['close'] - panel['low']) / (panel['high'] - panel['low'] + 1e-8)
    panel['volatility_5d'] = panel.groupby('code')['pct_chg'].transform(lambda x: x.rolling(5).std())
    panel['alpha_reversal_5d'] = -(panel['close'] / panel.groupby('code')['close'].shift(5) - 1)
    panel['alpha_024_approx'] = panel.groupby('code')['close'].transform(
        lambda x: x.rolling(20).mean()) / (panel['close'] + 1e-5) - 1

    market_daily = panel.groupby('date')['pct_chg'].mean().reset_index()
    market_daily.rename(columns={'pct_chg': 'market_ret'}, inplace=True)
    market_daily['market_ret_20d'] = market_daily['market_ret'].rolling(20, min_periods=5).mean()
    market_daily['market_ret_60d'] = market_daily['market_ret'].rolling(60, min_periods=20).mean()
    market_daily['market_vol_20d'] = market_daily['market_ret'].rolling(20, min_periods=5).std()
    
    # Market var for beta
    market_daily['market_var_20d'] = market_daily['market_ret'].rolling(20, min_periods=5).var()
    
    panel = pd.merge(panel, market_daily[['date', 'market_ret', 'market_ret_20d', 'market_ret_60d', 'market_vol_20d', 'market_var_20d']],
                     on='date', how='left')
    panel['macro_staleness_days'] = 0

    # --- NEW High-Order Structural Features ---
    log.info("Computing High-Order Structural Features...")
    
    # 1. Cross-sectional Ranks (using pure pandas grouping)
    panel['amihud_20_rank'] = panel.groupby('date')['amihud_20'].rank(pct=True)
    panel['clv_rank'] = panel.groupby('date')['clv'].rank(pct=True)
    
    # 2. Return Skewness (20d)
    panel['skew_20d'] = panel.groupby('code')['pct_chg'].transform(lambda x: x.rolling(20).skew())
    
    # 3. Volatility of Volatility (VoV)
    panel['vov_20d'] = panel.groupby('code')['volatility_5d'].transform(lambda x: x.rolling(20).std())
    
    # 4. Beta to Market (Covariance / Market Variance)
    # rolling covariance of stock pct_chg and market_ret
    # Using simple formula: cov(X,Y) = E(XY) - E(X)E(Y)
    panel['xy'] = panel['pct_chg'] * panel['market_ret']
    roll_xy = panel.groupby('code')['xy'].transform(lambda x: x.rolling(20).mean())
    roll_x = panel.groupby('code')['pct_chg'].transform(lambda x: x.rolling(20).mean())
    roll_y = panel['market_ret_20d']
    panel['cov_20d'] = roll_xy - roll_x * roll_y
    # Note: this is population covariance, sample covariance needs * (N/(N-1)), ignoring for Beta approx.
    panel['beta_20d'] = panel['cov_20d'] / (panel['market_var_20d'] + 1e-8)

    # --- Clean up ---
    panel = apply_liquidity_gate(panel, amihud_col='amihud_20', threshold_pct=0.90)
    panel = panel[~panel['is_limit']].copy()

    float_cols = panel.select_dtypes(include=['float64']).columns
    panel[float_cols] = panel[float_cols].astype(np.float32)
    return panel


def run_wfo(ml_df, feature_cols, epochs=5, lr=0.0003, batch_size=2048):
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

        model = PyTorchDLModel(input_dim=len(feature_cols))
        model.optimizer = optim.Adam(model.model.parameters(), lr=lr)
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

    return {
        'spread_bps': float(spread),
        'sharpe': float(sharpe),
        'max_dd': float(q5_dd)
    }

if __name__ == '__main__':
    t0 = time.time()
    panel = load_panel()
    
    NEW_FEATURES = ['amihud_20_rank', 'clv_rank', 'skew_20d', 'vov_20d', 'beta_20d']
    
    # We will test:
    # Trial 27: Baseline + All 5 New Structural Features
    # Trial 28: Pure Rank Replace (replace raw amihud/clv with ranked)
    
    feat_t27 = BASELINE_FEATURES + NEW_FEATURES
    
    feat_t28 = [f for f in BASELINE_FEATURES if f not in ['amihud_20', 'clv']] + ['amihud_20_rank', 'clv_rank', 'skew_20d', 'vov_20d', 'beta_20d']

    configs = [
        ("Baseline+5_Structural", feat_t27, 5, 0.0003),
        ("Structural_RankReplace", feat_t28, 5, 0.0003)
    ]

    results = {}
    trial_offset = 26

    for i, (name, features, epochs, lr) in enumerate(configs):
        trial_num = trial_offset + i + 1
        log.info(f"===== Trial {trial_num}/30: {name} =====")
        
        ml_df = panel.dropna(subset=features + ['fwd_ret_real', 'date']).copy()
        res = run_wfo(ml_df, features, epochs=epochs, lr=lr)
        
        if res:
            results[name] = res
            delta = res['sharpe'] - BASELINE_SHARPE
            marker = " ★ NEW BEST" if res['sharpe'] > BASELINE_SHARPE else ""
            log.info(f"[{name}] Spread={res['spread_bps']:.2f} bps | Sharpe={res['sharpe']:.2f} (Δ={delta:+.2f}) | MaxDD={res['max_dd']:.2%}{marker}")
        else:
            log.error(f"[{name}] WFO failed.")
        gc.collect()

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info(f"FINAL LEVEL 3 EXPLORATION COMPLETE ({elapsed/60:.1f} min)")
    log.info(f"Baseline reference: Sharpe={BASELINE_SHARPE}")
    log.info("=" * 70)

    for rank, (name, r) in enumerate(sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True), 1):
        delta = r['sharpe'] - BASELINE_SHARPE
        log.info(f"  #{rank} {name}: Spread={r['spread_bps']:.2f} | Sharpe={r['sharpe']:.2f} (Δ={delta:+.2f}) | MaxDD={r['max_dd']:.2%}")
