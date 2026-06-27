# research/wfo_regime_backtest.py — Phase 9 (2024 Regime Optimization)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ml_engine
import feature_engine
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

TRADING_COST = 0.003
HOLDING_DAYS = 20
BASE_MODEL_TRAIN_END = '2023-12-31'

def main():
    logging.info("1. Loading Data...")
    df = pd.read_parquet('.quantbot_data/ashare_daily.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date'])

    # 1. Basic Returns & Future Returns
    if 'volume' in df.columns:
        df = df.rename(columns={'volume': 'vol'})
    
    df['daily_ret'] = df.groupby('code')['close'].pct_change()
    df['fwd_ret_t1'] = df.groupby('code')['close'].shift(-1) / (df.groupby('code')['open'].shift(-1) + 1e-5) - 1
    
    # 2. Limit Up/Down (Today)
    df['pctChg_float'] = df['pctChg'].astype(float)
    df['is_limit_up'] = (df['high'] == df['close']) & (df['pctChg_float'] > 9.5)
    df['is_limit_down'] = (df['low'] == df['close']) & (df['pctChg_float'] < -9.5)
    
    # 3. Micro-Cap 3D Shield (amount_ma20, float_mv, extremes)
    logging.info("Building 3D Micro-Cap Shield...")
    df['amount_ma20'] = df.groupby('code')['amount'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df['turn_pct'] = df['turn'].astype(float)
    df['float_mv'] = df['amount'] / (df['turn_pct']/100 + 1e-8)
    
    df['extreme_day'] = df['is_limit_up'] | df['is_limit_down']
    df['extreme_count_20d'] = df.groupby('code')['extreme_day'].transform(lambda x: x.rolling(20).sum())
    
    # 4. Volatility
    df['volatility_20d'] = df.groupby('code')['daily_ret'].transform(lambda x: x.rolling(20, min_periods=5).std())
    df['volatility_5d'] = df.groupby('code')['daily_ret'].transform(lambda x: x.rolling(5, min_periods=2).std())
    
    # 5. Macro Stats & Circuit Breaker
    logging.info("Calculating Macro Circuit Breakers...")
    macro_stats = df.groupby('date').agg(
        total_amount=('amount', 'sum'),
        limit_down_count=('is_limit_down', 'sum'),
        mean_vol_20d=('volatility_20d', 'mean'),
        mean_vol_5d=('volatility_5d', 'mean')
    ).reset_index()
    
    macro_stats['total_amount_ma20'] = macro_stats['total_amount'].rolling(20, min_periods=5).mean()
    macro_stats['macro_vol_surge'] = macro_stats['mean_vol_5d'] / (macro_stats['mean_vol_20d'] + 1e-5) > 1.5
    macro_stats['amount_shrink'] = macro_stats['total_amount'] < 0.5 * macro_stats['total_amount_ma20']
    
    macro_stats['cb_pre'] = macro_stats['amount_shrink'] & macro_stats['macro_vol_surge']
    macro_stats['cb_post'] = macro_stats['limit_down_count'] > 500
    macro_stats['cb_trigger'] = macro_stats['cb_pre'] | macro_stats['cb_post']
    
    macro_stats['cb_trigger_t_minus_1'] = macro_stats['cb_trigger'].shift(1).fillna(False)
    
    macro_stats['ld_low_3d'] = (macro_stats['limit_down_count'] < 100).rolling(3).sum() == 3
    macro_stats['amt_recovered'] = macro_stats['total_amount'] > macro_stats['total_amount_ma20']
    macro_stats['cb_recover'] = macro_stats['ld_low_3d'] & macro_stats['amt_recovered']
    macro_stats['cb_recover_t_minus_1'] = macro_stats['cb_recover'].shift(1).fillna(False)
    
    exposure_array = np.ones(len(macro_stats))
    curr_exp = 1.0
    for i in range(1, len(macro_stats)):
        if macro_stats.loc[i, 'cb_trigger_t_minus_1']:
            curr_exp = 0.0
        elif curr_exp < 1.0 and macro_stats.loc[i, 'cb_recover_t_minus_1']:
            if curr_exp == 0.0: curr_exp = 0.3
            elif curr_exp == 0.3: curr_exp = 0.7
            else: curr_exp = 1.0
        exposure_array[i] = curr_exp
    macro_stats['target_exposure'] = exposure_array
    date_to_exposure = dict(zip(macro_stats['date'], macro_stats['target_exposure']))
    
    # 6. Apply 3D Micro-Cap Filter
    logging.info("Applying 3D Liquidity Shield...")
    df['amount_rank_pct'] = df.groupby('date')['amount_ma20'].rank(pct=True)
    bad_liquidity = (df['amount_rank_pct'] <= 0.30) | (df['float_mv'] < 3e9) | (df['extreme_count_20d'] > 5)
    df['is_valid_candidate'] = ~bad_liquidity & (~df['code'].str.contains('ST|\*ST|退', na=False))
    
    logging.info("Building features...")
    df = feature_engine.build_ml_features(df)
    with open('.quantbot_data/prod_pt_meta_t20.json', 'r') as f:
        features = json.load(f)['features']
    df = df.dropna(subset=features)
    
    logging.info("Loading Base PyTorch Model...")
    model = ml_engine.PyTorchDLModel(len(features))
    model.load_model('.quantbot_data/prod_pt_model_t20.pth')
    
    df_eval = df[(df['date'] >= '2015-07-01') & (df['date'] <= BASE_MODEL_TRAIN_END)].copy()
    df_test = df[df['date'] >= '2024-01-01'].copy()
    
    logging.info("Predicting Base Scores...")
    df_eval['base_score'] = model.predict(df_eval, features)
    df_test['base_score'] = model.predict(df_test, features)
    df_eval = df_eval.dropna(subset=['base_score'])
    df_test = df_test.dropna(subset=['base_score'])
    
    # 7. Volatility-Penalized Ranking & Top 20
    logging.info("Applying Cross-Sectional Volatility-Penalized Ranking...")
    for _df in [df_eval, df_test]:
        _df['vol_rank'] = _df.groupby('date')['volatility_20d'].rank(pct=True)
        _df['signal_score'] = _df['base_score'] / (_df['vol_rank'] + 0.1)
        _df['is_valid'] = _df['is_valid_candidate'] & (~_df['is_limit_up'])
        
        valid_df = _df[_df['is_valid']]
        _df['rank'] = valid_df.groupby('date')['signal_score'].rank(ascending=False, method='first')
        
    top20_train = df_eval[df_eval['rank'] <= 20].copy()
    df_test = df_test[df_test['rank'] <= 20].copy()
    
    top20_train['close_t20'] = top20_train.groupby('code')['close'].shift(-20)
    top20_train['actual_ret_t20'] = (top20_train['close_t20'] - top20_train['close']) / top20_train['close']
    top20_train['mkt_ret_t20'] = top20_train.groupby('date')['actual_ret_t20'].transform(lambda x: x.mean(skipna=True))
    top20_train['excess_ret_t20'] = top20_train['actual_ret_t20'] - top20_train['mkt_ret_t20']
    
    top20_train['Target_B'] = np.where(top20_train['excess_ret_t20'] > 0.05, 2,
                                       np.where(top20_train['excess_ret_t20'] > 0, 1, 0))
    
    # 8. Dynamic Meta-Critic WFO
    start_date = pd.to_datetime('2024-01-01')
    end_date = df_test['date'].max()
    months = pd.date_range(start_date, end_date, freq='MS')
    logging.info(f"Walking forward {len(months)} months...")

    df_test['veto'] = False

    for test_month_start in months:
        test_month_end = test_month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        train_start = test_month_start - relativedelta(months=6)
        train_end = test_month_start - pd.Timedelta(days=1)
        safe_train_end = train_end - pd.Timedelta(days=20)

        train_mask = (top20_train['date'] >= train_start) & (top20_train['date'] <= safe_train_end)
        train_df = top20_train[train_mask].dropna(subset=['excess_ret_t20'])

        test_mask = (df_test['date'] >= test_month_start) & (df_test['date'] <= test_month_end)
        test_df = df_test[test_mask]

        if len(train_df) < 50 or len(test_df) == 0:
            continue

        logging.info(f"Training Meta-Critic for {test_month_start.strftime('%Y-%m')}...")
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(train_df[features].dropna(), train_df['Target_B'])

        probs = clf.predict_proba(test_df[features].fillna(0))
        class_0_idx = list(clf.classes_).index(0) if 0 in clf.classes_ else None
        prob_fail = probs[:, class_0_idx] if class_0_idx is not None else np.zeros(len(test_df))
        
        # Dynamic Threshold
        median_fail = np.median(prob_fail)
        dynamic_threshold = np.clip(median_fail, 0.4, 0.6)
        df_test.loc[test_mask, 'veto'] = prob_fail > dynamic_threshold

    # 9. Vectorized Portfolio Simulation
    logging.info("Simulating daily portfolio (Vectorized)...")
    sim_dates = sorted(df_test['date'].unique())
    all_codes = sorted(df['code'].unique())
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    
    sim_df = df[(df['date'] >= sim_dates[0]) & (df['date'] <= sim_dates[-1])].copy()
    R_dict = dict(zip(zip(sim_df['date'], sim_df['code']), sim_df['fwd_ret_t1'].fillna(0.0)))
    L_dict = dict(zip(zip(sim_df['date'], sim_df['code']), sim_df['is_limit_down']))
    
    W = np.zeros(len(all_codes))
    portfolio_returns = []
    dates_record = []
    
    selected = df_test[~df_test['veto']]
    target_matrix = np.zeros((len(sim_dates), len(all_codes)))
    for i, d in enumerate(sim_dates):
        idx_start = max(0, i - HOLDING_DAYS + 1)
        valid_dates = sim_dates[idx_start:i+1]
        sub = selected[selected['date'].isin(valid_dates)]
        counts = sub['code'].value_counts()
        for code, count in counts.items():
            target_matrix[i, code_to_idx[code]] = count
            
    row_sums = target_matrix.sum(axis=1)
    target_matrix = np.where(row_sums[:, None] > 0, target_matrix / row_sums[:, None], 0.0)
    
    for i, d in enumerate(sim_dates):
        exposure = date_to_exposure.get(d, 1.0)
        target_w = target_matrix[i] * exposure
        
        if i > 0:
            prev_d = sim_dates[i-1]
            ret_array = np.array([R_dict.get((prev_d, c), 0.0) for c in all_codes])
            port_ret = np.sum(W * ret_array)
            portfolio_returns.append(port_ret)
            dates_record.append(d)
            
            W_drift = W * (1 + ret_array)
            sum_drift = np.sum(W_drift)
            if sum_drift > 0:
                W_drift /= sum_drift
            
            L_array = np.array([L_dict.get((d, c), False) for c in all_codes])
            can_sell = ~L_array
            
            W_new = np.zeros(len(all_codes))
            W_new[~can_sell] = np.maximum(W_drift[~can_sell], target_w[~can_sell])
            
            budget = exposure - np.sum(W_new[~can_sell])
            if budget > 0:
                target_free = target_w[can_sell]
                target_free_sum = np.sum(target_free)
                if target_free_sum > 0:
                    W_new[can_sell] = target_free * (budget / target_free_sum)
            
            turnover = np.sum(np.abs(W_new - W_drift))
            portfolio_returns[-1] -= turnover * TRADING_COST
            
            W = W_new
        else:
            W = target_w
            
    # 10. Performance
    res = pd.DataFrame({'date': dates_record, 'strat_ret': portfolio_returns})
    res['strat_cum'] = (1 + res['strat_ret']).cumprod()
    
    mkt = sim_df.groupby('date')['fwd_ret_t1'].mean().reset_index()
    mkt = mkt[mkt['date'].isin(dates_record)]
    res['mkt_ew_ret'] = mkt['fwd_ret_t1'].values
    res['mkt_ew_cum'] = (1 + res['mkt_ew_ret']).cumprod()

    days = len(res)
    ann_factor = 252 / days
    
    strat_cagr = res['strat_cum'].iloc[-1] ** ann_factor - 1
    mkt_cagr = res['mkt_ew_cum'].iloc[-1] ** ann_factor - 1
    strat_vol = res['strat_ret'].std() * np.sqrt(252)
    mkt_vol = res['mkt_ew_ret'].std() * np.sqrt(252)
    strat_sharpe = strat_cagr / (strat_vol + 1e-5)
    mkt_sharpe = mkt_cagr / (mkt_vol + 1e-5)
    strat_dd = (res['strat_cum'] / res['strat_cum'].cummax() - 1).min()
    mkt_dd = (res['mkt_ew_cum'] / res['mkt_ew_cum'].cummax() - 1).min()

    logging.info("========================================")
    logging.info("Phase 9 Regime Optimization Backtest (Vectorized)")
    logging.info(f"Strategy: CAGR {strat_cagr:.2%} | MaxDD {strat_dd:.2%} | Sharpe {strat_sharpe:.2f}")
    logging.info(f"Market EW: CAGR {mkt_cagr:.2%} | MaxDD {mkt_dd:.2%} | Sharpe {mkt_sharpe:.2f}")
    logging.info("========================================")

    plt.figure(figsize=(12,6))
    plt.plot(res['date'], res['strat_cum'], label=f'Regime Meta-Critic (DD: {strat_dd:.1%})', color='red')
    plt.plot(res['date'], res['mkt_ew_cum'], label=f'Market EW (DD: {mkt_dd:.1%})', color='gray', alpha=0.7)
    
    exposure_df = macro_stats[macro_stats['date'].isin(dates_record)][['date', 'target_exposure']]
    ax2 = plt.twinx()
    ax2.fill_between(exposure_df['date'], exposure_df['target_exposure'], color='blue', alpha=0.1, label='Target Exposure')
    ax2.set_ylim(0, 1.1)

    plt.title('Phase 9: 2024 Regime Optimization (Vectorized)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig('backtest_regime_phase9.png')
    logging.info("Plot saved to backtest_regime_phase9.png")

if __name__ == '__main__':
    main()
