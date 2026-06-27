import os
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def get_csi300_price():
    file_path = os.path.join(DATA_DIR, "csi300_price.csv")
    try:
        # 尝试通过 yfinance 获取
        print("Fetching CSI 300 from yfinance...")
        df = yf.download('000300.SS', start='2015-01-01', progress=False)
        if not df.empty:
            # Flatten columns if multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df[['Close']].reset_index()
            df.columns = ['Date', 'close']
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            df.to_csv(file_path, index=False)
            return df
    except Exception as e:
        print(f"yfinance failed: {e}")
    
    # 尝试通过 akshare 获取作为替补
    try:
        print("Fetching CSI 300 from akshare as fallback...")
        df = ak.stock_zh_index_daily(symbol="sh000300")
        if not df.empty:
            df = df[['date', 'close']]
            df.columns = ['Date', 'close']
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            df = df[df['Date'] >= pd.to_datetime('2015-01-01').date()]
            df.to_csv(file_path, index=False)
            return df
    except Exception as e:
        print(f"akshare index daily failed: {e}")

    # 读取缓存
    if os.path.exists(file_path):
        print("Using local cache for CSI 300 price.")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df
    raise ValueError("Failed to fetch CSI 300 data and no cache found.")

def get_us_vix():
    file_path = os.path.join(DATA_DIR, "us_vix.csv")
    try:
        print("Fetching ^VIX and ^VIX3M from yfinance...")
        df = yf.download(['^VIX', '^VIX3M'], start='2015-01-01', progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                # yfinance returns multi-index columns: (Price, Ticker)
                close_vix = df['Close']['^VIX']
                close_vix3m = df['Close']['^VIX3M']
                df_out = pd.DataFrame({'vix': close_vix, 'vix3m': close_vix3m}).reset_index()
            else:
                df_out = df[['Close']].reset_index() # Fallback, might not be multi-index if 1 ticker
            
            df_out.columns = ['Date', 'vix', 'vix3m']
            df_out['Date'] = pd.to_datetime(df_out['Date']).dt.date
            df_out.to_csv(file_path, index=False)
            return df_out
    except Exception as e:
        print(f"yfinance VIX failed: {e}")
    
    if os.path.exists(file_path):
        print("Using local cache for VIX.")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df
    raise ValueError("Failed to fetch VIX data and no cache found.")

def get_csi300_pe():
    file_path = os.path.join(DATA_DIR, "csi300_pe.csv")
    try:
        print("Fetching CSI 300 PE history from akshare (stock_index_pe_lg)...")
        df = ak.stock_index_pe_lg(symbol="沪深300")
        if not df.empty and len(df) > 252:
            # Columns: 日期, 指数, 等权静态市盈率, 静态市盈率, 静态市盈率分位, 等权市盈率, 市盈率(TTM), 市盈率分位
            # We take Date (0) and PE (6)
            df_out = df.iloc[:, [0, 6]].copy()
            df_out.columns = ['Date', 'pe']
            df_out['Date'] = pd.to_datetime(df_out['Date']).dt.date
            df_out = df_out[df_out['Date'] >= pd.to_datetime('2015-01-01').date()]
            df_out.to_csv(file_path, index=False)
            return df_out
        else:
            print(f"akshare PE returned only {len(df) if not df.empty else 0} rows. Falling back to mock data.")
    except Exception as e:
        print(f"akshare PE failed: {e}")
        
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if len(df) > 252:
            print("Using local cache for CSI 300 PE.")
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            return df
            
    raise ValueError("Failed to fetch PE history.")

def get_china_10y_yield():
    file_path = os.path.join(DATA_DIR, "cn_10y_yield.csv")
    try:
        print("Fetching China 10Y Bond Yield from akshare (bond_zh_us_rate)...")
        df = ak.bond_zh_us_rate()
        if not df.empty:
            df_out = df.iloc[:, [0, 3]].copy()
            df_out.columns = ['Date', 'yield_10y']
            df_out['Date'] = pd.to_datetime(df_out['Date']).dt.date
            df_out = df_out[df_out['Date'] >= pd.to_datetime('2015-01-01').date()]
            df_out.to_csv(file_path, index=False)
            return df_out
    except Exception as e:
        print(f"akshare bond_zh_us_rate failed: {e}")
        
    if os.path.exists(file_path):
        print("Using local cache for 10Y yield.")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df
        
    print("Fallback: Mocking China 10Y Yield as 3.0% for backtest demonstration.")
    # Create mock data based on recent CSI300 dates
    dates = pd.date_range(start='2015-01-01', end=pd.Timestamp.today())
    df_out = pd.DataFrame({'Date': dates, 'yield_10y': 3.0})
    df_out['Date'] = df_out['Date'].dt.date
    return df_out

def calculate_ic_metrics(df_merged, signal_col, fwd_cols):
    results = {}
    for fwd in fwd_cols:
        # Drop NaNs for the current pair
        valid_df = df_merged[[signal_col, fwd]].dropna()
        if len(valid_df) < 50:
            continue
            
        # Spearman correlation (Rank IC)
        ic = stats.spearmanr(valid_df[signal_col], valid_df[fwd])[0]
        # We calculate IC on a rolling basis, then aggregate. Or just overall IC.
        # Standard approach: daily cross-sectional IC. Since we have a single asset, 
        # "Time-Series IC" is calculated as correlation over rolling windows or overall correlation.
        # We will compute the overall Spearman correlation between the signal and forward returns.
        
        # Actually, for a single asset, IC is usually the time-series Spearman correlation.
        ic = stats.spearmanr(valid_df[signal_col], valid_df[fwd])[0]
        
        # To get IC IR, we can chunk the data by month to get a series of ICs
        valid_df['Month'] = pd.to_datetime(valid_df.index).to_period('M')
        monthly_ic = valid_df.groupby('Month').apply(lambda x: stats.spearmanr(x[signal_col], x[fwd])[0] if len(x)>5 else np.nan).dropna()
        
        mean_ic = monthly_ic.mean()
        std_ic = monthly_ic.std()
        ic_ir = mean_ic / std_ic if std_ic != 0 else np.nan
        
        # T-stat (sqrt(N) * mean / std)
        t_stat = np.sqrt(len(monthly_ic)) * ic_ir if not np.isnan(ic_ir) else np.nan
        
        # Win rate (IC > 0)
        win_rate = (monthly_ic > 0).mean()
        
        results[fwd] = {
            'Overall IC': ic,
            'Mean Monthly IC': mean_ic,
            'IC IR': ic_ir,
            't-stat': t_stat,
            'Win Rate (IC>0)': win_rate
        }
    return results

def main():
    print("=== Phase 0: VIX-TS & A-Share ERP IC Backtest ===")
    
    # 1. Fetch Data
    df_price = get_csi300_price()
    df_vix = get_us_vix()
    df_pe = get_csi300_pe()
    df_yield = get_china_10y_yield()
    
    # 2. Merge Data on Date
    df = df_price.set_index('Date')
    df = df.join(df_vix.set_index('Date'), how='left')
    df = df.join(df_pe.set_index('Date'), how='left')
    df = df.join(df_yield.set_index('Date'), how='left')
    
    # Forward fill macros within reason (e.g., weekend gap, holiday gap)
    df[['vix', 'vix3m', 'pe', 'yield_10y']] = df[['vix', 'vix3m', 'pe', 'yield_10y']].ffill()
    
    df = df.dropna(subset=['close']).copy() # Only keep A-share trading days
    
    # 3. Construct Signals (Strict Anti-Lookahead)
    # VIX-TS: 5-day smooth
    vix_ratio = df['vix'] / df['vix3m']
    vix_ts_smooth = vix_ratio.rolling(5).mean()
    # ERP: (1 / PE) * 100 - yield_10y
    # Note: pe is e.g. 15.0, yield_10y is e.g. 3.0 (meaning 3.0%)
    erp = (1 / df['pe']) * 100 - df['yield_10y']
    erp_smooth = erp.rolling(5).mean()
    
    # Shift(1) before any ranking to ensure T-day signal applies to T+1 open
    df['vix_ts_signal'] = vix_ts_smooth.shift(1).rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df['erp_signal'] = erp_smooth.shift(1).rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # 4. Forward Returns (Predicting T+N close relative to T close)
    for n in [5, 21, 63]:
        df[f'fwd_ret_{n}d'] = df['close'].shift(-n) / df['close'] - 1.0
        
    # 5. IC Calculation
    print("\nCalculating IC Metrics for VIX-TS...")
    vix_metrics = calculate_ic_metrics(df, 'vix_ts_signal', ['fwd_ret_5d', 'fwd_ret_21d', 'fwd_ret_63d'])
    
    print("Calculating IC Metrics for ERP...")
    erp_metrics = calculate_ic_metrics(df, 'erp_signal', ['fwd_ret_5d', 'fwd_ret_21d', 'fwd_ret_63d'])
    
    # Display Results
    print("\n--- VIX-TS IC Metrics ---")
    vix_df = pd.DataFrame(vix_metrics).T
    print(vix_df.to_string())
    
    print("\n--- A-Share ERP IC Metrics ---")
    erp_df = pd.DataFrame(erp_metrics).T
    print(erp_df.to_string())

    # Plotting
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot Cumulative 21d IC (Rolling 6-month = 126 days correlation)
    valid_plot_df = df[['vix_ts_signal', 'erp_signal', 'fwd_ret_21d']].dropna()
    
    # Calculate rolling Spearman IC (slow but accurate)
    # Since vix_ts_signal and erp_signal are already ranks, we can use standard Pearson corr as an approximation of Spearman
    # but let's apply actual spearmanr for accuracy if it's not too slow, else we just do rolling corr on ranks
    
    def rolling_spearman(x, y, window=126):
        res = []
        for i in range(len(x)):
            if i < window:
                res.append(np.nan)
            else:
                res.append(stats.spearmanr(x[i-window:i], y[i-window:i])[0])
        return res

    valid_plot_df['vix_ts_rolling_ic'] = rolling_spearman(valid_plot_df['vix_ts_signal'].values, valid_plot_df['fwd_ret_21d'].values)
    valid_plot_df['erp_rolling_ic'] = rolling_spearman(valid_plot_df['erp_signal'].values, valid_plot_df['fwd_ret_21d'].values)
    
    plt.figure(figsize=(12, 6))
    plt.plot(valid_plot_df.index, valid_plot_df['vix_ts_rolling_ic'], label='VIX-TS Rolling 126d IC (21d Fwd)', alpha=0.7)
    plt.plot(valid_plot_df.index, valid_plot_df['erp_rolling_ic'], label='ERP Rolling 126d IC (21d Fwd)', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Rolling 126-Day Spearman IC (21-Day Forward Returns)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "rolling_ic.png"))
    print(f"\nPlot saved to {plot_dir}/rolling_ic.png")
    
    # Quantile Spread Plot
    # Group signals into 5 quintiles and calculate average forward 21d return
    df['erp_q'] = pd.qcut(df['erp_signal'], 5, labels=False, duplicates='drop')
    q_ret = df.groupby('erp_q')['fwd_ret_21d'].mean() * 100 # %
    plt.figure(figsize=(8, 5))
    q_ret.plot(kind='bar', color='skyblue')
    plt.title("Average 21-Day Forward Return by ERP Quintile")
    plt.xlabel("ERP Quintile (0=Lowest ERP/Most Expensive, 4=Highest ERP/Cheapest)")
    plt.ylabel("Avg 21d Return (%)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "erp_quintile_returns.png"))
    print(f"Plot saved to {plot_dir}/erp_quintile_returns.png")

if __name__ == "__main__":
    main()
