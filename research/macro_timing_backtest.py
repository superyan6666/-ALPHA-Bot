import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def get_csi300_price():
    file_path = os.path.join(DATA_DIR, "csi300_price.csv")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

def get_us_vix():
    file_path = os.path.join(DATA_DIR, "us_vix.csv")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

def get_csi300_pe():
    file_path = os.path.join(DATA_DIR, "csi300_pe.csv")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

def get_china_10y_yield():
    file_path = os.path.join(DATA_DIR, "cn_10y_yield.csv")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def extract_events(df, signal_col, left_th, right_th, high_is_panic=True):
    """
    Extract Left-side and Right-side event dates.
    high_is_panic: True for VIX/ERP (higher = extreme), False for RSI (lower = extreme)
    """
    left_events = []
    right_events = []
    
    in_left_extreme = False
    days_since_extreme = 0
    last_event_idx = -999 # Avoid clustering

    for i in range(len(df)):
        val = df[signal_col].iloc[i]
        if pd.isna(val): continue
        
        # Cooling down logic: prevent back-to-back same signals within 10 days
        is_cooling = (i - last_event_idx) < 10

        if high_is_panic:
            if val > left_th:
                in_left_extreme = True
                days_since_extreme = 0
                if not is_cooling:
                    left_events.append(df.index[i])
                    last_event_idx = i
            elif in_left_extreme:
                days_since_extreme += 1
                if val < right_th and days_since_extreme <= 30:
                    if not is_cooling:
                        right_events.append(df.index[i])
                        last_event_idx = i
                    in_left_extreme = False
                elif days_since_extreme > 30:
                    in_left_extreme = False
        else:
            if val < left_th:
                in_left_extreme = True
                days_since_extreme = 0
                if not is_cooling:
                    left_events.append(df.index[i])
                    last_event_idx = i
            elif in_left_extreme:
                days_since_extreme += 1
                if val > right_th and days_since_extreme <= 30:
                    if not is_cooling:
                        right_events.append(df.index[i])
                        last_event_idx = i
                    in_left_extreme = False
                elif days_since_extreme > 30:
                    in_left_extreme = False

    return left_events, right_events

def calc_forward_metrics(df, event_dates, window, tag=""):
    ret_list = []
    mae_list = []
    trap_count = 0
    
    for dt in event_dates:
        locs = np.where(df.index == dt)[0]
        if len(locs) == 0: continue
        idx = locs[0]
        if idx + window >= len(df): continue
        
        entry_price = df['close'].iloc[idx]
        exit_price = df['close'].iloc[idx + window]
        
        period_lows = df['low'].iloc[idx+1:idx+window+1]
        period_closes = df['close'].iloc[idx+1:idx+window+1]
        period_prev_closes = df['close'].iloc[idx:idx+window].values
        period_highs = df['high'].iloc[idx+1:idx+window+1]
        
        ret = (exit_price / entry_price) - 1.0
        mae = (period_lows.min() / entry_price) - 1.0
        
        # Check liquidity traps (e.g. 2016 circuit breakers: return <= -7% or amplitude < 0.5% during crash)
        daily_rets = (period_closes.values / period_prev_closes) - 1.0
        amplitudes = (period_highs.values - period_lows.values) / period_prev_closes
        
        is_trap = any((r <= -0.07) or (r < -0.03 and a < 0.005) for r, a in zip(daily_rets, amplitudes))
        if is_trap: trap_count += 1
        
        ret_list.append(ret)
        mae_list.append(mae)

    if not ret_list:
        return {"Count": 0, "Mean_Ret": 0, "WinRate": 0, "MAE": 0, "Stability": 0, "Traps": 0}

    rets = np.array(ret_list)
    maes = np.array(mae_list)
    
    mean_ret = rets.mean()
    win_rate = (rets > 0).mean()
    mean_mae = maes.mean()
    stability = mean_ret / rets.std() if rets.std() > 0 else 0
    
    return {
        "Count": len(rets),
        "Mean_Ret": mean_ret * 100,
        "WinRate": win_rate * 100,
        "MAE": mean_mae * 100,
        "Stability": stability,
        "Traps": trap_count
    }

def run_backtest_for_period(df, period_name, start_date, end_date):
    sub_df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
    if len(sub_df) < 50: return None
    
    results = []
    
    # 1. VIX-TS
    left_vix, right_vix = extract_events(sub_df, 'vix_ts_pct', 0.95, 0.70, True)
    for name, evs in [("VIX 左侧", left_vix), ("VIX 右侧", right_vix)]:
        m5 = calc_forward_metrics(sub_df, evs, 5)
        m20 = calc_forward_metrics(sub_df, evs, 20)
        results.append({
            "Indicator": name,
            "Count": m5["Count"],
            "T+5_Ret(%)": f"{m5['Mean_Ret']:.2f}",
            "T+5_Win(%)": f"{m5['WinRate']:.1f}",
            "T+5_MAE(%)": f"{m5['MAE']:.2f}",
            "T+5_Stab": f"{m5['Stability']:.2f}",
            "T+20_Ret(%)": f"{m20['Mean_Ret']:.2f}",
            "T+20_Win(%)": f"{m20['WinRate']:.1f}",
            "T+20_MAE(%)": f"{m20['MAE']:.2f}",
            "T+20_Stab": f"{m20['Stability']:.2f}",
            "Traps(d)": f"{m20['Traps']}"
        })

    # 2. ERP
    left_erp, right_erp = extract_events(sub_df, 'erp_pct', 0.95, 0.85, True)
    for name, evs in [("ERP 左侧", left_erp), ("ERP 右侧", right_erp)]:
        m5 = calc_forward_metrics(sub_df, evs, 5)
        m20 = calc_forward_metrics(sub_df, evs, 20)
        results.append({
            "Indicator": name,
            "Count": m5["Count"],
            "T+5_Ret(%)": f"{m5['Mean_Ret']:.2f}",
            "T+5_Win(%)": f"{m5['WinRate']:.1f}",
            "T+5_MAE(%)": f"{m5['MAE']:.2f}",
            "T+5_Stab": f"{m5['Stability']:.2f}",
            "T+20_Ret(%)": f"{m20['Mean_Ret']:.2f}",
            "T+20_Win(%)": f"{m20['WinRate']:.1f}",
            "T+20_MAE(%)": f"{m20['MAE']:.2f}",
            "T+20_Stab": f"{m20['Stability']:.2f}",
            "Traps(d)": f"{m20['Traps']}"
        })
        
    # 3. RSI
    left_rsi, right_rsi = extract_events(sub_df, 'rsi', 20.0, 30.0, False)
    for name, evs in [("RSI 左侧", left_rsi), ("RSI 右侧", right_rsi)]:
        m5 = calc_forward_metrics(sub_df, evs, 5)
        m20 = calc_forward_metrics(sub_df, evs, 20)
        results.append({
            "Indicator": name,
            "Count": m5["Count"],
            "T+5_Ret(%)": f"{m5['Mean_Ret']:.2f}",
            "T+5_Win(%)": f"{m5['WinRate']:.1f}",
            "T+5_MAE(%)": f"{m5['MAE']:.2f}",
            "T+5_Stab": f"{m5['Stability']:.2f}",
            "T+20_Ret(%)": f"{m20['Mean_Ret']:.2f}",
            "T+20_Win(%)": f"{m20['WinRate']:.1f}",
            "T+20_MAE(%)": f"{m20['MAE']:.2f}",
            "T+20_Stab": f"{m20['Stability']:.2f}",
            "Traps(d)": f"{m20['Traps']}"
        })
        
    res_df = pd.DataFrame(results)
    return res_df

def main():
    print("加载并对齐数据...")
    df = get_csi300_price().set_index('Date')
    df_vix = get_us_vix().set_index('Date')
    df_pe = get_csi300_pe()
    df_yield = get_china_10y_yield().set_index('Date')
    
    if df_pe is not None:
        df_pe = df_pe.set_index('Date')
        df = df.join(df_pe, how='left')
    else:
        df['pe'] = 15.0 # Fallback
        
    df = df.join(df_vix, how='left')
    df = df.join(df_yield, how='left')
    df[['vix', 'vix3m', 'pe', 'yield_10y']] = df[['vix', 'vix3m', 'pe', 'yield_10y']].ffill()
    df.dropna(subset=['close'], inplace=True)
    df.index = pd.to_datetime(df.index)

    print("计算宏观因子...")
    
    def rolling_pct(s, window):
        return s.rolling(window).apply(lambda x: (x <= x[-1]).mean(), raw=True)

    # VIX-TS
    vix_ratio = df['vix'] / df['vix3m']
    df['vix_ts_pct'] = rolling_pct(vix_ratio.rolling(5).mean(), 252)
    
    # ERP
    erp = (1 / df['pe']) * 100 - df['yield_10y']
    df['erp_pct'] = rolling_pct(erp.rolling(5).mean(), 252)
    
    # RSI
    df['rsi'] = compute_rsi(df['close'], 14)
    
    # Shift signals to prevent lookahead
    df['vix_ts_pct'] = df['vix_ts_pct'].shift(1)
    df['erp_pct'] = df['erp_pct'].shift(1)
    df['rsi'] = df['rsi'].shift(1)

    periods = {
        "主回测 (2017-01-01 ~ 2024-12-31)": ('2017-01-01', '2024-12-31'),
        "前瞻验证 (2023-02-17 ~ 2026-12-31)": ('2023-02-17', '2026-12-31'),
        "压力测试 (2015-02-09 ~ 2016-12-31)": ('2015-02-09', '2016-12-31')
    }
    
    for p_name, (s_date, e_date) in periods.items():
        print(f"\n{'='*50}\n[{p_name}]\n{'='*50}")
        res_df = run_backtest_for_period(df, p_name, s_date, e_date)
        if res_df is not None:
            # Markdown format for beautiful printing
            print(res_df.to_markdown(index=False))
        else:
            print("No enough data.")

if __name__ == "__main__":
    main()
