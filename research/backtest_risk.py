
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
import json
warnings.filterwarnings('ignore')

def calculate_drawdowns(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1
    max_dd = drawdowns.min()
    return max_dd

def run_backtest():
    print("🚀 正在加载 2021-2026 行情数据进行历史回撤测试...")
    df = pd.read_parquet('.quantbot_data/ashare_daily.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date'])
    
    # 模拟大盘指数（取全市场每天的平均收益作为大盘 Proxy，或者如果有的话用专门的指数数据）
    # 简化处理：以全市场的平均收益近似大盘
    idx_ret = df.groupby('date')['pctChg'].mean() / 100.0
    idx_cum = (1 + idx_ret).cumprod()
    
    # 计算 VIX Proxy (近似)
    vix_proxy = idx_ret.rolling(20).std() * np.sqrt(252) * 100
    ma20 = idx_cum.rolling(20).mean()
    
    veto_mask = (idx_cum < ma20) & (idx_ret < -0.015) & (vix_proxy > 20.0)
    veto_dates = set(idx_ret[veto_mask].index)
    
    print(f"⚠️ 宏观熔断器在过去 5 年中共触发 {len(veto_dates)} 天的绝对空仓指令。")
    
    # 计算均线
    df['ma5'] = df.groupby('code')['close'].rolling(5).mean().reset_index(0, drop=True)
    df['prev_close'] = df.groupby('code')['close'].shift(1)
    df['prev_close_2'] = df.groupby('code')['close'].shift(2)
    df['prev_close_3'] = df.groupby('code')['close'].shift(3)
    
    # 右侧过滤器
    above_ma5 = df['close'] >= df['ma5']
    is_3d_down = (df['prev_close'] < df['prev_close_2']) & (df['prev_close_2'] < df['prev_close_3'])
    right_side_mask = above_ma5 & ~(is_3d_down & (df['close'] <= df['prev_close']))
    
    # 模型得分占位符 (这里用动量作为简化打分)
    df['mom'] = df.groupby('code')['close'].pct_change(20)
    
    df['fwd_ret_t1'] = df.groupby('code')['close'].shift(-1) / df['close'] - 1
    
    # Baseline (Top 20 each day)
    print("⌛ 正在撮合基准策略 (左侧抄底，无防守)...")
    baseline_returns = []
    for date, group in df.dropna(subset=['mom', 'fwd_ret_t1']).groupby('date'):
        top20 = group.nlargest(20, 'mom')
        if len(top20) > 0:
            baseline_returns.append({'date': date, 'ret': top20['fwd_ret_t1'].mean()})
    
    base_df = pd.DataFrame(baseline_returns).set_index('date')
    
    # Upgraded Strategy
    print("⌛ 正在撮合升级策略 (Macro Veto + 右侧确认)...")
    upgraded_returns = []
    filtered_df = df[right_side_mask]
    for date, group in filtered_df.dropna(subset=['mom', 'fwd_ret_t1']).groupby('date'):
        if date in veto_dates:
            upgraded_returns.append({'date': date, 'ret': 0.0}) # 空仓
            continue
        top20 = group.nlargest(20, 'mom')
        if len(top20) > 0:
            upgraded_returns.append({'date': date, 'ret': top20['fwd_ret_t1'].mean()})
            
    upg_df = pd.DataFrame(upgraded_returns).set_index('date')
    
    for name, series in [('基准策略 (无保护)', base_df['ret']), ('升级策略 (熔断+右侧)', upg_df['ret'])]:
        cagr = (1 + series.mean()) ** 252 - 1
        vol = series.std() * np.sqrt(252)
        sharpe = series.mean() / series.std() * np.sqrt(252) if series.std() > 0 else 0
        max_dd = calculate_drawdowns(series)
        win_rate = (series > 0).mean()
        
        print(f"\n[{name}]")
        print(f"  - Sharpe Ratio: {sharpe:.2f}")
        print(f"  - Max Drawdown: {max_dd*100:.2f}%")
        print(f"  - CAGR (年化): {cagr*100:.2f}%")
        print(f"  - Win Rate: {win_rate*100:.2f}%")

if __name__ == '__main__':
    run_backtest()
