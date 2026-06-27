import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, r"d:\Antigravity\A-Bot")

from pipeline_manager import PipelineManager
from factor_library import calculate_factors

def verify():
    print("Loading data...")
    manager = PipelineManager(r"d:\Antigravity\A-Bot\research\data\csi300_price.csv")
    df = manager.load_data()
    
    print("Calculating factors...")
    df_factors = calculate_factors(df)
    
    # Calculate target
    target_ret_window = 20
    df_factors['next_open'] = df_factors.groupby('code')['open'].shift(-1)
    df_factors['close_tn'] = df_factors.groupby('code')['close'].shift(-target_ret_window)
    df_factors['fwd_ret'] = df_factors['close_tn'] / (df_factors['next_open'] + 1e-8) - 1.0
    df_factors['target'] = df_factors['fwd_ret']
    
    factor_cols = [c for c in df_factors.columns if c.startswith('F_') and c != 'F_float_cap']
    
    # Match the exact dropna behavior
    screened_df = df_factors.dropna(subset=['target'] + factor_cols).copy()
    print(f"Total rows after dropna: {len(screened_df)} (expected: 757)")
    
    # Month grouping as in InitialScreener for single asset
    screened_df['Month'] = pd.to_datetime(screened_df['date']).dt.to_period('M')
    
    # ERP
    ic_erp = screened_df.groupby('Month').apply(lambda x: x['F_ERP'].corr(x['target'], method='spearman') if len(x) > 5 else np.nan)
    ic_erp = ic_erp.dropna()
    mean_ic_erp = ic_erp.mean()
    ir_erp = mean_ic_erp / (ic_erp.std() + 1e-8)
    t_stat_erp = mean_ic_erp / (ic_erp.std() / np.sqrt(len(ic_erp)) + 1e-8)
    
    print(f"ERP calculated Rank IC Mean: {mean_ic_erp:.6f} (expected: 0.379997)")
    print(f"ERP calculated IR: {ir_erp:.6f} (expected: 0.912834)")
    print(f"ERP calculated t-stat: {t_stat_erp:.6f} (expected: 5.627086)")
    
    # VIX_TS
    ic_vix_ts = screened_df.groupby('Month').apply(lambda x: x['F_VIX_TS'].corr(x['target'], method='spearman') if len(x) > 5 else np.nan)
    ic_vix_ts = ic_vix_ts.dropna()
    mean_ic_vix_ts = ic_vix_ts.mean()
    ir_vix_ts = mean_ic_vix_ts / (ic_vix_ts.std() + 1e-8)
    t_stat_vix_ts = mean_ic_vix_ts / (ic_vix_ts.std() / np.sqrt(len(ic_vix_ts)) + 1e-8)
    
    print(f"VIX_TS calculated Rank IC Mean: {mean_ic_vix_ts:.6f} (expected: 0.178969)")
    print(f"VIX_TS calculated IR: {ir_vix_ts:.6f} (expected: 0.509263)")
    print(f"VIX_TS calculated t-stat: {t_stat_vix_ts:.6f} (expected: 3.139305)")

if __name__ == "__main__":
    verify()
