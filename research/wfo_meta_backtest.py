# wfo_meta_backtest.py — Phase 6 升级版（可交易收益 + 熔断过滤 + 成本扣除）

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

# ═══════════════════════════════════════════════════════════════
# 可调参数
# ═══════════════════════════════════════════════════════════════
TRADING_COST = 0.003                # 双边成本 0.3%（印花税+佣金+滑点）
HOLDING_DAYS = 20                   # 信号有效期（使用过去 HOLDING_DAYS 天的信号等权）
BASE_MODEL_TRAIN_END = '2023-12-31' # 基础模型训练截止日

def is_tradable_next_day(code, date, df_indexed):
    """
    检查股票在指定日期的下一个交易日是否可正常交易。
    返回 (可交易, 是否跌停封死无法卖出)。
    """
    try:
        next_day_data = df_indexed.loc[(date, code)]
    except KeyError:
        return False, False

    # 检查停牌 / 无成交量
    vol = next_day_data.get('vol', 0)
    if pd.isna(vol) or vol == 0:
        return False, False

    # 检查次日收益是否缺失（可能因为停牌）
    fwd_ret = next_day_data.get('fwd_ret_t1', np.nan)
    if pd.isna(fwd_ret):
        return False, False

    # 简化的次日跌停无法卖出检查
    if fwd_ret <= -0.095 and vol < 100:
        return False, True   # 跌停封死，无法卖出
        
    return True, False

def filter_tradable_signals(signal_df, exec_date, df_indexed):
    """
    过滤出在 exec_date 可执行的信号：剔除停牌、剔除已触发跌停卖出（损失已计入）。
    返回：可交易信号的 DataFrame，以及因跌停被剔除的股票列表（收益已单独处理）
    """
    tradable = []
    forced_loss_codes = []
    for _, row in signal_df.iterrows():
        code = row['code']
        ok, is_limit_down = is_tradable_next_day(code, exec_date, df_indexed)
        if ok:
            tradable.append(row)
        elif is_limit_down:
            forced_loss_codes.append(code)
    return pd.DataFrame(tradable), forced_loss_codes

def main():
    logging.info("1. Loading Data...")
    df = pd.read_parquet('.quantbot_data/ashare_daily.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date'])

    # ---------- 基础预处理 ----------
    df['daily_ret'] = df.groupby('code')['close'].pct_change()
    # 可交易收益：T+1 开盘买入 -> 收盘卖出
    df['fwd_ret_t1'] = df.groupby('code')['close'].shift(-1) / (df.groupby('code')['open'].shift(-1) + 1e-5) - 1

    # 标签：T+20 超额收益（用于训练 Meta-Critic）
    df['close_t20'] = df.groupby('code')['close'].shift(-20)
    df['actual_ret_t20'] = (df['close_t20'] - df['close']) / df['close']
    df['mkt_ret_t20'] = df.groupby('date')['actual_ret_t20'].transform(lambda x: x.mean(skipna=True))
    df['excess_ret_t20'] = df['actual_ret_t20'] - df['mkt_ret_t20']

    # 剔除 ST / 退市 / 无法交易股票
    bad_mask = df['code'].str.contains('ST|\*ST|退', na=False)
    df = df[~bad_mask].copy()

    if 'volume' in df.columns:
        df = df.rename(columns={'volume': 'vol'})

    logging.info("Building features...")
    df = feature_engine.build_ml_features(df)

    with open('.quantbot_data/prod_pt_meta_t20.json', 'r') as f:
        features = json.load(f)['features']

    df = df.dropna(subset=features)

    logging.info("Loading Base PyTorch Model...")
    model = ml_engine.PyTorchDLModel(len(features))
    model.load_model('.quantbot_data/prod_pt_model_t20.pth')

    # ---------- 评估区间 ----------
    df_eval = df[(df['date'] >= '2015-07-01') & (df['date'] <= BASE_MODEL_TRAIN_END)].copy()  # 基础模型训练期内部分用于构建信号
    df_test = df[df['date'] >= '2024-01-01'].copy()  # 样本外回测区间

    logging.info("Predicting Base Scores on training period...")
    df_eval['base_score'] = model.predict(df_eval, features)
    df_eval = df_eval.dropna(subset=['base_score'])

    logging.info("Predicting Base Scores on testing period...")
    df_test['base_score'] = model.predict(df_test, features)
    df_test = df_test.dropna(subset=['base_score'])

    # ---------- 生成每日 Top 20 候选（训练期与测试期）----------
    df_eval['rank'] = df_eval.groupby('date')['base_score'].rank(ascending=False, method='first')
    top20_train = df_eval[df_eval['rank'] <= 20].copy()

    df_test['rank'] = df_test.groupby('date')['base_score'].rank(ascending=False, method='first')
    top20_test = df_test[df_test['rank'] <= 20].copy()

    # 涨停过滤：选股日当日涨停的股票不纳入候选
    top20_train['is_limit_up'] = (top20_train['high'] == top20_train['close']) & (top20_train['pct_chg'] > 9.5)
    top20_train = top20_train[~top20_train['is_limit_up']]

    top20_test['is_limit_up'] = (top20_test['high'] == top20_test['close']) & (top20_test['pct_chg'] > 9.5)
    df_test = top20_test[~top20_test['is_limit_up']].copy()

    # 构造 Meta-Critic 目标标签
    top20_train['Target_B'] = np.where(top20_train['excess_ret_t20'] > 0.05, 2,
                                       np.where(top20_train['excess_ret_t20'] > 0, 1, 0))

    # ---------- Walk-Forward Meta-Critic 训练与否决 ----------
    start_date = pd.to_datetime('2024-01-01')
    end_date = df_test['date'].max()
    months = pd.date_range(start_date, end_date, freq='MS')
    logging.info(f"Walking forward {len(months)} months...")

    df_test = df_test.copy()
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
        clf.fit(train_df[features].dropna(), train_df['Target_B'])  # 用 dropna 替换 fillna(0)

        # 预测否决概率
        probs = clf.predict_proba(test_df[features].fillna(0))  # 预测时缺失填0属不得已，但非训练
        class_0_idx = list(clf.classes_).index(0) if 0 in clf.classes_ else None
        prob_fail = probs[:, class_0_idx] if class_0_idx is not None else np.zeros(len(test_df))
        df_test.loc[test_mask, 'veto'] = prob_fail > 0.5

    # ---------- 组合模拟 ----------
    logging.info("Simulating daily portfolio with tradable returns...")
    dates = sorted(df_test['date'].unique())
    portfolio_returns = []
    dates_record = []
    
    # Pre-index for O(1) lookups
    df_indexed = df.set_index(['date', 'code'])

    for d in dates:
        # 过去 HOLDING_DAYS 天内产生的信号（日期在 d 之前且未被否决）
        signal_window = pd.date_range(end=d, periods=HOLDING_DAYS, freq='B')
        active_signals = df_test[(df_test['date'].isin(signal_window)) & (~df_test['veto'])]

        if active_signals.empty:
            portfolio_returns.append(0.0)
            dates_record.append(d)
            continue

        # 过滤可交易性：停牌、跌停等
        tradable_df, forced_loss_codes = filter_tradable_signals(active_signals, d, df_indexed)

        # 计算可交易部分的平均收益（扣除成本）
        if len(tradable_df) > 0:
            # 获取这些代码在次日的 fwd_ret_t1
            try:
                avg_tradable_ret = df_indexed.loc[(d, list(tradable_df['code'])), 'fwd_ret_t1'].mean() - TRADING_COST
            except KeyError:
                avg_tradable_ret = 0.0
            if pd.isna(avg_tradable_ret):
                avg_tradable_ret = 0.0
        else:
            avg_tradable_ret = 0.0

        # 跌停损失单独计算：每只跌停股贡献 -10% 损失
        loss_from_limit = len(forced_loss_codes) * (-0.10)

        # 总信号数（包括不可交易部分），用于平均
        total_signals = len(active_signals)
        if total_signals > 0:
            daily_ret = (avg_tradable_ret * len(tradable_df) + loss_from_limit) / total_signals
        else:
            daily_ret = 0.0

        portfolio_returns.append(daily_ret)
        dates_record.append(d)

    # ---------- 绩效统计 ----------
    res_df = pd.DataFrame({'date': dates_record, 'daily_ret': portfolio_returns})
    res_df['nav'] = (1 + res_df['daily_ret']).cumprod()

    # 简单等权市场基准（同样扣除成本）
    market_ret = df_test.groupby('date')['fwd_ret_t1'].mean().reindex(res_df['date']).fillna(0) - TRADING_COST
    market_nav = (1 + market_ret).cumprod()

    def calc_metrics(nav, rets):
        cagr = (nav.iloc[-1]) ** (252/len(nav)) - 1 if nav.iloc[-1] > 0 else -1
        mdd = (nav / nav.cummax() - 1).min()
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() != 0 else 0
        return cagr, mdd, sharpe

    cagr, mdd, sharpe = calc_metrics(res_df['nav'], res_df['daily_ret'])
    m_cagr, m_mdd, m_sharpe = calc_metrics(market_nav, market_ret)

    logging.info("="*40)
    logging.info("Phase 6 Meta-Critic Backtest (Tradable & Realistic)")
    logging.info(f"Strategy: CAGR {cagr*100:.2f}% | MaxDD {mdd*100:.2f}% | Sharpe {sharpe:.2f}")
    logging.info(f"Market EW: CAGR {m_cagr*100:.2f}% | MaxDD {m_mdd*100:.2f}% | Sharpe {m_sharpe:.2f}")
    logging.info("="*40)

    # 绘图
    plt.figure(figsize=(12,6))
    plt.plot(res_df['date'], res_df['nav'], label='Meta-Critic (Realistic)', color='blue')
    plt.plot(res_df['date'], market_nav, label='Market EW', color='gray', linestyle='--')
    plt.title('Walk-Forward Backtest with Execution Filters & Costs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('backtest_meta_phase6.png')
    logging.info("Plot saved to backtest_meta_phase6.png")

if __name__ == '__main__':
    main()
