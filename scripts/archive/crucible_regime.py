"""
CRUCIBLE Level 3 (Revised): Regime-Conditional Training
=======================================================
Split data by market volatility regime, train separate MLP models per regime.

Scientific basis: A-share 2020-2026 spans multiple distinct regimes
(2020H2 bull, 2022 bear, 2024 recovery). A single model is forced to
average across structurally different signal environments.

Trials 24-26 out of 30.
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
FEATURES = GENES['features']
BASELINE_SHARPE = GENES['baseline_sharpe']  # 1.83

# ── Data Loading ─────────────────────────────────────────────────────
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

    log.info("Computing factors...")
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
    panel = pd.merge(panel, market_daily[['date', 'market_ret_20d', 'market_ret_60d', 'market_vol_20d']],
                     on='date', how='left')
    panel['macro_staleness_days'] = 0

    panel = apply_liquidity_gate(panel, amihud_col='amihud_20', threshold_pct=0.90)
    panel = panel[~panel['is_limit']].copy()

    float_cols = panel.select_dtypes(include=['float64']).columns
    panel[float_cols] = panel[float_cols].astype(np.float32)
    return panel


def classify_regime(dates_df, vol_col='market_vol_20d', method='median'):
    """
    Classify each date into a regime based on market volatility.
    Uses ONLY historical data (expanding window) to avoid lookahead.
    
    Returns: Series mapping date → regime label
    """
    daily = dates_df.drop_duplicates('date').set_index('date')[[vol_col]].sort_index()

    if method == 'median':
        # Expanding median: each day's threshold = median of ALL prior days
        expanding_median = daily[vol_col].expanding(min_periods=20).median()
        regime = (daily[vol_col] > expanding_median).map({True: 'high_vol', False: 'low_vol'})
    elif method == 'tercile':
        # Expanding terciles
        q33 = daily[vol_col].expanding(min_periods=60).quantile(0.33)
        q66 = daily[vol_col].expanding(min_periods=60).quantile(0.66)
        regime = pd.Series('mid_vol', index=daily.index)
        regime[daily[vol_col] <= q33] = 'low_vol'
        regime[daily[vol_col] > q66] = 'high_vol'
    else:
        raise ValueError(f"Unknown method: {method}")

    return regime


def run_regime_wfo(ml_df, feature_cols, regime_method='median', epochs=5, lr=0.0003):
    """
    WFO with regime-conditional models.
    In each WFO window:
      1. Classify train dates into regimes (using expanding window, no lookahead)
      2. Train a separate MLP per regime
      3. At test time, classify each test date and route to the matching model
    """
    dates = sorted(ml_df['date'].unique())
    train_window = 500
    step = 125
    n_features = len(feature_cols)

    # Pre-compute regime labels for ALL dates using expanding window (no lookahead)
    regime_labels = classify_regime(ml_df, method=regime_method)
    regime_counts = regime_labels.value_counts()
    log.info(f"Regime distribution ({regime_method}): {regime_counts.to_dict()}")

    all_test_preds = []
    for idx in range(train_window, len(dates), step):
        train_dates = dates[max(0, idx - train_window):idx]
        test_dates = dates[idx:min(len(dates), idx + step)]
        if len(test_dates) == 0:
            break

        train_df = ml_df[ml_df['date'].isin(train_dates)].copy()
        test_df = ml_df[ml_df['date'].isin(test_dates)].copy()

        # Assign regime labels
        train_df['regime'] = train_df['date'].map(regime_labels)
        test_df['regime'] = test_df['date'].map(regime_labels)

        train_df = train_df.dropna(subset=['regime'])
        test_df = test_df.dropna(subset=['regime'])

        regimes_in_train = train_df['regime'].unique()
        models = {}

        for regime in regimes_in_train:
            regime_train = train_df[train_df['regime'] == regime]
            if len(regime_train) < 1000:
                log.warning(f"  Regime '{regime}' only {len(regime_train)} samples, skipping")
                continue

            model = PyTorchDLModel(input_dim=n_features)
            model.optimizer = optim.Adam(model.model.parameters(), lr=lr)
            log.info(f"  Training regime='{regime}' on {len(regime_train)} samples...")
            model.train(regime_train, feature_cols, target_col='fwd_ret_real',
                        group_col='date', epochs=epochs, batch_size=2048)
            models[regime] = model

        # Predict: route each test day to matching regime model
        test_scores = pd.Series(np.nan, index=test_df.index)
        for regime, model in models.items():
            mask = test_df['regime'] == regime
            if mask.any():
                regime_test = test_df[mask]
                test_scores[mask] = model.predict(regime_test, feature_cols)

        # Fallback: if a test regime has no model, use the model with most data
        unmapped = test_scores.isna()
        if unmapped.any() and models:
            fallback_model = list(models.values())[0]
            test_scores[unmapped] = fallback_model.predict(test_df[unmapped], feature_cols)

        test_df['score'] = test_scores.values
        all_test_preds.append(test_df[['date', 'code', 'score', 'fwd_ret_real', 'regime']])

        for m in models.values():
            del m
        gc.collect()

    if not all_test_preds:
        return None

    oos_df = pd.concat(all_test_preds, ignore_index=True)
    oos_df = oos_df.dropna(subset=['score'])

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

    daily_ls = oos_df.groupby('date', group_keys=False).apply(
        lambda x: x[x['quantile'] == 5]['fwd_ret_real'].mean() - x[x['quantile'] == 1]['fwd_ret_real'].mean()
    ).fillna(0)
    sharpe = (daily_ls.mean() / daily_ls.std()) * np.sqrt(252) if daily_ls.std() > 0 else 0.0

    q5_daily = oos_df[oos_df['quantile'] == 5].groupby('date')['fwd_ret_real'].mean()
    q5_cum = (1 + q5_daily).cumprod()
    q5_dd = (q5_cum / q5_cum.cummax() - 1).min()
    win_rate = (daily_ls > 0).mean()

    # Per-regime breakdown
    regime_breakdown = {}
    for regime in oos_df['regime'].unique():
        r_df = oos_df[oos_df['regime'] == regime]
        r_daily = r_df.groupby('date', group_keys=False).apply(
            lambda x: x[x['quantile'] == 5]['fwd_ret_real'].mean() - x[x['quantile'] == 1]['fwd_ret_real'].mean()
        ).fillna(0)
        r_sharpe = (r_daily.mean() / r_daily.std()) * np.sqrt(252) if r_daily.std() > 0 and len(r_daily) > 5 else 0.0
        regime_breakdown[regime] = {'sharpe': float(r_sharpe), 'n_days': int(len(r_daily))}

    return {
        'spread_bps': float(spread),
        'sharpe': float(sharpe),
        'max_dd': float(q5_dd),
        'win_rate': float(win_rate),
        'regime_breakdown': regime_breakdown,
    }


if __name__ == '__main__':
    t0 = time.time()
    panel = load_panel()
    ml_df = panel.dropna(subset=FEATURES + ['fwd_ret_real', 'date']).copy()

    configs = [
        ("Regime_Median_2way", 'median', 5, 0.0003),
        ("Regime_Tercile_3way", 'tercile', 5, 0.0003),
        ("Regime_Median_ep3", 'median', 3, 0.0003),  # less epochs per regime (less data per model)
    ]

    results = {}
    trial_offset = 23

    for i, (name, method, epochs, lr) in enumerate(configs):
        trial_num = trial_offset + i + 1
        log.info(f"{'='*70}")
        log.info(f"Trial {trial_num}/30: {name} (method={method}, ep={epochs}, lr={lr})")
        log.info(f"{'='*70}")

        res = run_regime_wfo(ml_df, FEATURES, regime_method=method, epochs=epochs, lr=lr)
        if res:
            results[name] = res
            delta = res['sharpe'] - BASELINE_SHARPE
            marker = " ★ NEW CHAMPION!" if res['sharpe'] > BASELINE_SHARPE else ""
            log.info(f"[{name}] Spread={res['spread_bps']:.2f} | Sharpe={res['sharpe']:.2f} (Δ={delta:+.2f}) | MaxDD={res['max_dd']:.2%} | WinRate={res['win_rate']:.1%}{marker}")
            log.info(f"  Regime breakdown: {res['regime_breakdown']}")
        gc.collect()

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info(f"REGIME-CONDITIONAL EXPLORATION COMPLETE ({elapsed/60:.1f} min)")
    log.info(f"Single-model Baseline: Sharpe={BASELINE_SHARPE}")
    log.info("=" * 70)

    ranked = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    for rank, (name, r) in enumerate(ranked, 1):
        delta = r['sharpe'] - BASELINE_SHARPE
        log.info(f"  #{rank} {name}: Spread={r['spread_bps']:.2f} | Sharpe={r['sharpe']:.2f} (Δ={delta:+.2f}) | MaxDD={r['max_dd']:.2%}")
        for regime, rd in r['regime_breakdown'].items():
            log.info(f"       └─ {regime}: Sharpe={rd['sharpe']:.2f} ({rd['n_days']} days)")

    output = {
        'timestamp': datetime.now().isoformat(),
        'baseline_sharpe': BASELINE_SHARPE,
        'results': results,
        'trials_used': trial_offset + len(results),
    }
    with open('.quantbot_data/regime_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Results saved to .quantbot_data/regime_results.json")
