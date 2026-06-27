# wfo_rolling_xgboost.py — Phase 6 升级版（可交易收益 + 熔断过滤 + 成本扣除）

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import feature_engine
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

TRADING_COST = 0.003   # 双边成本 0.3%

def process_one_month(test_month_start, df, features, target_col):
    test_month_end = test_month_start + relativedelta(months=1) - pd.Timedelta(days=1)
    train_start = test_month_start - relativedelta(months=12)
    train_end = test_month_start - pd.Timedelta(days=1)
    safe_train_end = train_end - pd.Timedelta(days=20)

    train_mask = (df['date'] >= train_start) & (df['date'] <= safe_train_end)
    train_df = df[train_mask].dropna(subset=[target_col])

    test_mask = (df['date'] >= test_month_start) & (df['date'] <= test_month_end)
    test_df = df[test_mask].copy()

    if len(train_df) < 1000 or len(test_df) == 0:
        return test_month_start.strftime('%Y-%m'), None

    # 训练
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                             tree_method='hist', n_jobs=1, random_state=42)
    model.fit(train_df[features].dropna(), train_df[target_col])  # dropna 防假特征

    # 预测
    test_df['pred_score'] = model.predict(test_df[features].fillna(0))

    # ---------- 选股前过滤 ----------
    # 剔除当日涨停股
    test_df['is_limit_up'] = (test_df['high'] == test_df['close']) & (test_df['pct_chg'] > 9.5)
    test_df = test_df[~test_df['is_limit_up']]

    daily_picks = []
    for date, group in test_df.groupby('date'):
        # 宏观 veto 直接空仓
        if group['macro_veto'].iloc[0]:
            daily_picks.append(pd.DataFrame({'date': [date], 'code': ['CASH'], 'fwd_ret_t1': [0.0]}))
            continue

        # 选 Top 20（预测分数最高）
        candidates = group.nlargest(20, 'pred_score')

        # 执行日检查可交易性
        valid_codes = []
        for _, row in candidates.iterrows():
            code = row['code']
            # 次日数据存在且非停牌
            next_day = df[(df['code'] == code) & (df['date'] == date)]
            if next_day.empty:
                continue
            fwd = next_day['fwd_ret_t1'].values[0]
            vol = next_day['vol'].values[0]
            if pd.isna(fwd) or vol == 0:
                continue
            # 跌停无法卖出检查（已在 filter 中处理，但这里简化：若开盘跌停且无量，收益设 -0.10）
            open_p = next_day['open'].values[0]
            low_p = next_day['low'].values[0]
            if open_p == low_p and open_p > 0:
                prev_rows = df[(df['code'] == code) & (df['date'] < date)]
                if len(prev_rows) > 0:
                    prev_close = prev_rows['close'].iloc[-1]
                    if not pd.isna(prev_close) and (open_p - prev_close) / prev_close <= -0.095 and vol < 100:
                        # 跌停封死，用跌停价计算收益
                        row['fwd_ret_t1'] = -0.10
            valid_codes.append(row)

        if len(valid_codes) == 0:
            daily_picks.append(pd.DataFrame({'date': [date], 'code': ['CASH'], 'fwd_ret_t1': [0.0]}))
            continue

        pick_df = pd.DataFrame(valid_codes)
        # 扣除交易成本
        pick_df['fwd_ret_t1'] = pick_df['fwd_ret_t1'] - TRADING_COST
        daily_picks.append(pick_df[['date', 'code', 'fwd_ret_t1']])

    return test_month_start.strftime('%Y-%m'), (daily_picks, train_df.shape[0], test_df.shape[0])

def main():
    logging.info("1. Loading Data...")
    df = pd.read_parquet('.quantbot_data/ashare_daily.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date'])

    # ---------- 可交易收益与宏观 veto ----------
    df['daily_ret'] = df.groupby('code')['close'].pct_change()
    df['fwd_ret_t1'] = df.groupby('code')['close'].shift(-1) / (df.groupby('code')['open'].shift(-1) + 1e-5) - 1

    daily_mkt = df.groupby('date')['daily_ret'].mean()
    mkt_index = (1 + daily_mkt).cumprod()
    sma20 = mkt_index.rolling(20).mean()
    sma60 = mkt_index.rolling(60).mean()
    trend_bearish = sma20 < sma60
    momentum_crash = mkt_index.pct_change(10) < -0.05
    vol_20d = daily_mkt.rolling(20).std()
    vol_252d_p90 = vol_20d.rolling(252, min_periods=60).quantile(0.9)
    vol_spike = vol_20d > vol_252d_p90
    macro_veto_series = trend_bearish | momentum_crash | vol_spike
    df['macro_veto'] = df['date'].map(macro_veto_series).fillna(False)

    # ---------- 标签 ----------
    df['label_close_t20'] = df.groupby('code')['close'].shift(-20)
    df['label_actual_ret_t20'] = (df['label_close_t20'] - df['close']) / df['close']
    df['label_mkt_ret_t20'] = df.groupby('date')['label_actual_ret_t20'].transform(lambda x: x.mean(skipna=True))
    df['label_excess_ret_t20'] = df['label_actual_ret_t20'] - df['label_mkt_ret_t20']

    # 剔除 ST / 退市
    bad_mask = df['code'].str.contains('ST|\*ST|退', na=False)
    df = df[~bad_mask]

    if 'volume' in df.columns:
        df = df.rename(columns={'volume': 'vol'})

    logging.info("Building features...")
    df = feature_engine.build_ml_features(df)

    with open('.quantbot_data/prod_pt_meta_t20.json', 'r') as f:
        features = json.load(f)['features']
    df = df.dropna(subset=features)
    target_col = 'label_excess_ret_t20'

    # ---------- Walk-Forward 月份 ----------
    start_date = pd.to_datetime('2020-01-01')
    end_date = df['date'].max()
    months = pd.date_range(start_date, end_date, freq='MS')

    daily_picks = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_one_month, m, df, features, target_col): m for m in months}
        results_by_month = {}
        for future in as_completed(futures):
            month_str, result = future.result()
            if result is not None:
                month_picks, train_size, test_size = result
                logging.info(f"Completed {month_str}: Train {train_size}, Test {test_size}")
                results_by_month[month_str] = month_picks

    for month_str in sorted(results_by_month.keys()):
        daily_picks.extend(results_by_month[month_str])

    # ---------- 组合收益计算 ----------
    portfolio_df = pd.concat(daily_picks, ignore_index=True)
    # 将 CASH 替换为收益 0
    portfolio_df['fwd_ret_t1'] = portfolio_df['fwd_ret_t1'].astype(float)
    daily_ret = portfolio_df.groupby('date')['fwd_ret_t1'].mean()
    portfolio_daily = daily_ret.reset_index()
    portfolio_daily.columns = ['date', 'fwd_ret_t1']
    portfolio_daily['nav'] = (1 + portfolio_daily['fwd_ret_t1']).cumprod()

    # 市场等权基准（扣除成本）
    market_ret = df.groupby('date')['fwd_ret_t1'].mean() - TRADING_COST
    market_ret = market_ret[market_ret.index >= start_date]
    market_nav = (1 + market_ret).cumprod()

    # 绩效
    ann_ret = portfolio_daily['fwd_ret_t1'].mean() * 252
    ann_vol = portfolio_daily['fwd_ret_t1'].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    mdd = (portfolio_daily['nav'] / portfolio_daily['nav'].cummax() - 1).min()

    logging.info("="*40)
    logging.info("Phase 6 XGBoost WFO (Realistic Execution)")
    logging.info(f"Ann.Ret: {ann_ret:.2%}, Vol: {ann_vol:.2%}, Sharpe: {sharpe:.2f}, MaxDD: {mdd:.2%}")
    logging.info("="*40)

    # 保存和画图
    portfolio_daily.to_csv('.quantbot_data/portfolio_wfo_phase6.csv')
    plt.figure(figsize=(12,6))
    plt.plot(portfolio_daily['date'], portfolio_daily['nav'], label='XGBoost Top20 (Realistic)', color='blue')
    plt.plot(market_nav.index, market_nav.values, label='Market EW (Realistic)', color='gray', linestyle='--')
    plt.title('XGBoost Walk-Forward with Execution Filters & Costs')
    plt.legend()
    plt.grid()
    plt.savefig('backtest_xgb_phase6.png')

if __name__ == '__main__':
    main()
