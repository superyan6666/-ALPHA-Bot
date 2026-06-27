
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import os
import gc

# ---------------------------------------------------------
# Level 2.3 Factor Cycle Calibration (Standard Anchor Mapping)
# ---------------------------------------------------------

ANCHORS = [5, 10, 20, 60, 120]

def snap_to_anchor(best_window):
    """就近吸附映射机制：寻找绝对距离最小的锚点"""
    return min(ANCHORS, key=lambda x: abs(x - best_window))

def compute_rank_ic(df, factor_col, target_col):
    """计算每日横截面 Spearman Rank IC 均值"""
    valid_df = df[['date', factor_col, target_col]].dropna()
    if valid_df.empty: return 0.0
    
    def daily_ic(g):
        # 样本过少不具备统计意义
        if len(g) < 30: return np.nan
        return g[factor_col].corr(g[target_col], method='spearman')
        
    ic_series = valid_df.groupby('date').apply(daily_ic, include_groups=False).dropna()
    return ic_series.mean()

def run_factor_cycle_calibration(df_path='.quantbot_data/ashare_daily.parquet', target_col='fwd_ret_t20'):
    """
    基于中长期目标(如 T+20)进行因子周期标定
    """
    print("🚀 开始全历史因子时效性寻优 (Rank IC Calibration)...")
    if not os.path.exists(df_path):
        print(f"Error: {df_path} not found.")
        return
        
    df = pd.read_parquet(df_path)
    
    # Calculate target (e.g., T+20) if not present
    if target_col not in df.columns:
        print(f"Calculating target {target_col}...")
        df['shifted_open_t1'] = df.groupby('code')['open'].shift(-1)
        h = int(target_col.split('t')[-1])
        df[f'close_t{h}'] = df.groupby('code')['close'].shift(-(h+1))
        df[target_col] = (df[f'close_t{h}'] - df['shifted_open_t1']) / (df['shifted_open_t1'] + 1e-5)
    
    # Needs pre-calculated columns
    if 'pct_chg' not in df.columns:
        df['prev_close'] = df.groupby('code')['close'].shift(1)
        df['pct_chg'] = (df['close'] / df['prev_close'] - 1) * 100
        
    df['shifted_close'] = df.groupby('code')['close'].shift(1)
    df['shifted_vol'] = df.groupby('code')['volume'].shift(1)
    df['shifted_high'] = df.groupby('code')['high'].shift(1)
    df['shifted_low'] = df.groupby('code')['low'].shift(1)
    df['shifted_pct_chg'] = df.groupby('code')['pctChg'].shift(1)
    
    search_space = range(5, 125, 5) # [5, 10, 15 ..., 120]
    optimal_mappings = []
    
    # ---------------------------
    # Define Factor Evaluators
    # ---------------------------
    evaluators = {
        'rsi': evaluate_rsi,
        'mom': evaluate_mom,
        'bias': evaluate_bias,
        'volatility': evaluate_volatility,
        'amihud': evaluate_amihud,
        'drawdown': evaluate_drawdown,
        'runup': evaluate_runup,
        'atr': evaluate_atr,
        'pv_corr': evaluate_pv_corr
    }
    
    for factor_name, eval_func in evaluators.items():
        print(f"🔍 正在评估因子: {factor_name.upper()}...")
        best_ic, best_w = 0.0, 60 
        
        for w in search_space:
            # 计算临时因子
            df[f'temp_{factor_name}'] = eval_func(df, w)
            
            # 极值处理防污染
            lower = df.groupby('date')[f'temp_{factor_name}'].transform(lambda x: x.quantile(0.01))
            upper = df.groupby('date')[f'temp_{factor_name}'].transform(lambda x: x.quantile(0.99))
            df[f'temp_{factor_name}'] = df[f'temp_{factor_name}'].clip(lower=lower, upper=upper)
            
            # 计算 IC
            ic = compute_rank_ic(df, f'temp_{factor_name}', target_col)
            
            if abs(ic) > abs(best_ic):
                best_ic, best_w = ic, w
                
        # 执行就近锚点吸附
        anchor_w = snap_to_anchor(best_w)
        
        print(f"✅ {factor_name.upper()} 标定完成: 原始极值={best_w}天 (IC={best_ic:.4f}) -> 锚点映射={anchor_w}天")
        optimal_mappings.append({'factor': factor_name, 'anchor_window': anchor_w})
    
    # 结果落盘
    os.makedirs('.quantbot_data', exist_ok=True)
    mapping_df = pd.DataFrame(optimal_mappings)
    mapping_df.to_csv('.quantbot_data/factor_optimal_horizons.csv', index=False)
    print("💾 因子周期动态映射表已保存至 .quantbot_data/factor_optimal_horizons.csv")

# ---------------------------
# Factor Calculation Functions
# ---------------------------
def evaluate_rsi(df, w):
    pct_chg_win = df['shifted_pct_chg'].clip(lower=-20.0, upper=20.0)
    avg_gain = df.groupby('code')['pct_chg'].transform(lambda x: x.shift(1).clip(lower=-20.0, upper=20.0).where(x.shift(1) > 0, 0.0).rolling(w, min_periods=w//2).mean())
    avg_loss = df.groupby('code')['pct_chg'].transform(lambda x: -x.shift(1).clip(lower=-20.0, upper=20.0).where(x.shift(1) < 0, 0.0).rolling(w, min_periods=w//2).mean())
    rs = avg_gain / (avg_loss + 1e-5)
    return 100 - (100 / (1 + rs))

def evaluate_mom(df, w):
    return (df['shifted_close'] / df.groupby('code')['close'].shift(w+1)) - 1

def evaluate_bias(df, w):
    ma = df.groupby('code')['close'].transform(lambda x: x.shift(1).rolling(w, min_periods=w//2).mean())
    return (df['shifted_close'] / (ma + 1e-5)) - 1

def evaluate_volatility(df, w):
    return df.groupby('code')['pct_chg'].transform(lambda x: x.shift(1).rolling(w, min_periods=w//2).std())

def evaluate_amihud(df, w):
    amihud_raw = df['shifted_pct_chg'].abs() / (df['shifted_vol'] * df['shifted_close'] + 1e-5) * 1e6
    is_limit_shifted = df.groupby('code')['is_limit'].shift(1) if 'is_limit' in df.columns else False
    amihud = np.where(is_limit_shifted, 99999.0, amihud_raw)
    df['temp_amihud_base'] = amihud
    return df.groupby('code')['temp_amihud_base'].transform(lambda x: x.rolling(w, min_periods=w//2).mean())

def evaluate_drawdown(df, w):
    roll_max = df.groupby('code')['high'].transform(lambda x: x.shift(1).rolling(w, min_periods=w//2).max())
    return (df['shifted_close'] / (roll_max + 1e-5)) - 1

def evaluate_runup(df, w):
    roll_min = df.groupby('code')['low'].transform(lambda x: x.shift(1).rolling(w, min_periods=w//2).min())
    return (df['shifted_close'] / (roll_min + 1e-5)) - 1

def evaluate_atr(df, w):
    eps = 1e-8
    prev_shifted_close = df.groupby('code')['close'].shift(2)
    tr1 = df['shifted_high'] - df['shifted_low']
    tr2 = (df['shifted_high'] - prev_shifted_close).abs()
    tr3 = (df['shifted_low'] - prev_shifted_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.groupby(df['code']).transform(lambda x: x.rolling(w, min_periods=w//2).mean()) / (df['shifted_close'] + eps)

def evaluate_pv_corr(df, w):
    def calc_pv(g):
        return g['close'].shift(1).rolling(window=w, min_periods=w//2).corr(g['volume'].shift(1))
    res = df.groupby('code', group_keys=False).apply(calc_pv, include_groups=False)
    return res.fillna(0.0)

if __name__ == "__main__":
    run_factor_cycle_calibration()
