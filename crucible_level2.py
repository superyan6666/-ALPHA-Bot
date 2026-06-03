"""
CRUCIBLE Level 2: New Factor Discovery + Ultra-Low LR Exploration
=================================================================
Trials 13-20 out of 30 budget.

Phase A: Push lr even lower (0.0001) with ep=3 and ep=5
Phase B: Introduce new technical factors and ablate their contribution
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
BASELINE_SHARPE = GENES['baseline_sharpe']
log.info(f"Baseline Sharpe: {BASELINE_SHARPE}, Features: {BASELINE_FEATURES}")

# ── Data Loading (single pass) ───────────────────────────────────────
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

    # ── Existing Baseline Factors ──
    log.info("Computing baseline factors...")
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
    panel['macro_staleness_days'] = 0  # placeholder

    # ── NEW Level 2 Factors ──
    log.info("Computing NEW Level 2 factors...")

    # 1. RSI(14) — Relative Strength Index
    def _rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    panel['rsi_14'] = panel.groupby('code')['close'].transform(lambda x: _rsi(x, 14))

    # 2. Bollinger Band Width (20,2)
    panel['bb_ma20'] = panel.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
    panel['bb_std20'] = panel.groupby('code')['close'].transform(lambda x: x.rolling(20).std())
    panel['bb_width'] = (panel['bb_std20'] * 2) / (panel['bb_ma20'] + 1e-8)

    # 3. Turnover Rate CV (coefficient of variation over 20d)
    # Using vol as proxy for turnover
    vol_mean_20 = panel.groupby('code')['vol'].transform(lambda x: x.rolling(20).mean())
    vol_std_20 = panel.groupby('code')['vol'].transform(lambda x: x.rolling(20).std())
    panel['turnover_cv_20'] = vol_std_20 / (vol_mean_20 + 1e-5)

    # 4. VWAP Deviation — (Close - VWAP) / VWAP, VWAP approx = Amount / Vol
    # Since we don't have amount, use (H+L+C)/3 as typical price proxy
    panel['typical_price'] = (panel['high'] + panel['low'] + panel['close']) / 3
    tp_vwap_20 = panel.groupby('code')['typical_price'].transform(lambda x: x.rolling(20).mean())
    panel['vwap_dev'] = (panel['close'] - tp_vwap_20) / (tp_vwap_20 + 1e-8)

    # 5. Price Acceleration (2nd derivative of price: momentum of momentum)
    mom_5 = panel['close'] / panel.groupby('code')['close'].shift(5) - 1
    mom_5_prev = panel.groupby('code').apply(lambda g: (g['close'] / g['close'].shift(5) - 1).shift(5), include_groups=False).reset_index('code', drop=True)
    panel['price_accel'] = mom_5 - mom_5_prev

    # 6. Volume Momentum (vol ratio vs 20d average)
    panel['vol_mom_20'] = panel['vol'] / (vol_mean_20 + 1e-5) - 1

    # Liquidity gate + limit filter
    panel = apply_liquidity_gate(panel, amihud_col='amihud_20', threshold_pct=0.90)
    panel = panel[~panel['is_limit']].copy()

    # Downcast
    float_cols = panel.select_dtypes(include=['float64']).columns
    panel[float_cols] = panel[float_cols].astype(np.float32)

    return panel


def run_wfo(ml_df, feature_cols, epochs, lr, batch_size=2048):
    """Single WFO pass."""
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
        'max_dd': float(q5_dd),
    }


if __name__ == '__main__':
    t0 = time.time()
    panel = load_panel()

    NEW_FACTORS = ['rsi_14', 'bb_width', 'turnover_cv_20', 'vwap_dev', 'price_accel', 'vol_mom_20']

    # ═══════════════════════════════════════════════════════════════
    # Phase A: Ultra-low LR exploration on Baseline features
    # ═══════════════════════════════════════════════════════════════
    phase_a_configs = [
        ("ep3_lr0.0001", BASELINE_FEATURES, 3, 0.0001),
        ("ep5_lr0.0001", BASELINE_FEATURES, 5, 0.0001),
        ("ep3_lr0.0003", BASELINE_FEATURES, 3, 0.0003),
    ]

    # ═══════════════════════════════════════════════════════════════
    # Phase B: New factor ablation (add each new factor individually)
    # Use best known config: ep5_lr0.0003
    # ═══════════════════════════════════════════════════════════════
    phase_b_configs = []
    for factor in NEW_FACTORS:
        features_with_new = BASELINE_FEATURES + [factor]
        phase_b_configs.append((f"baseline+{factor}", features_with_new, 5, 0.0003))

    # Phase C: Best new factors combined (will decide after Phase B)
    all_configs = phase_a_configs + phase_b_configs

    results = {}
    trial_offset = 12  # already used 12 trials

    for i, (name, features, epochs, lr) in enumerate(all_configs):
        trial_num = trial_offset + i + 1
        if trial_num > 30:
            log.error(f"[CRUCIBLE] Trial budget exhausted at trial {trial_num}. Stopping.")
            break

        log.info(f"===== Trial {trial_num}/30: {name} (ep={epochs}, lr={lr}, feat={len(features)}) =====")

        ml_df = panel.dropna(subset=features + ['fwd_ret_real', 'date']).copy()
        if len(ml_df) < 10000:
            log.warning(f"Not enough data for {name}. Skipping.")
            continue

        res = run_wfo(ml_df, features, epochs=epochs, lr=lr)
        if res:
            results[name] = res
            delta = res['sharpe'] - BASELINE_SHARPE
            marker = " ★ NEW BEST" if res['sharpe'] > BASELINE_SHARPE else ""
            log.info(f"[{name}] Spread={res['spread_bps']:.2f} bps | Sharpe={res['sharpe']:.2f} (Δ={delta:+.2f}) | MaxDD={res['max_dd']:.2%}{marker}")
        else:
            log.error(f"[{name}] WFO failed.")
        gc.collect()

    # ── Summary ──
    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info(f"LEVEL 2 EXPLORATION COMPLETE ({elapsed/60:.1f} min)")
    log.info(f"Baseline reference: Sharpe={BASELINE_SHARPE}")
    log.info("=" * 70)

    ranked = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    for rank, (name, r) in enumerate(ranked, 1):
        delta = r['sharpe'] - BASELINE_SHARPE
        log.info(f"  #{rank} {name}: Spread={r['spread_bps']:.2f} | Sharpe={r['sharpe']:.2f} (Δ={delta:+.2f}) | MaxDD={r['max_dd']:.2%}")

    # Identify winning new factors
    log.info("-" * 70)
    log.info("NEW FACTOR CONTRIBUTIONS (vs baseline ep5_lr0.0003 Sharpe=1.83):")
    for factor in NEW_FACTORS:
        key = f"baseline+{factor}"
        if key in results:
            delta = results[key]['sharpe'] - BASELINE_SHARPE
            verdict = "✅ POSITIVE" if delta > 0 else "⛔ NEGATIVE"
            log.info(f"  {factor}: Sharpe={results[key]['sharpe']:.2f} (Δ={delta:+.2f}) {verdict}")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'baseline_sharpe': BASELINE_SHARPE,
        'results': results,
        'trials_used': trial_offset + len(results),
    }
    with open('.quantbot_data/level2_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Results saved to .quantbot_data/level2_results.json")
