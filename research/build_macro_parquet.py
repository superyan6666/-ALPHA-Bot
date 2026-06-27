
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import akshare as ak
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_macro_features(start_date='2019-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    logging.info(f"Fetching ChinaBond Data from {start_date} to {end_date}...")
    # 1. Fetch ChinaBond 10Y Yield
    cn_dfs = []
    try:
        # Fetch year by year to avoid timeouts
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        for year in range(start_year, end_year + 1):
            s_dt = max(start_date, f"{year}-01-01")
            e_dt = min(end_date, f"{year}-12-31")
            logging.info(f"Fetching ChinaBond for {year}...")
            year_df = ak.bond_china_yield(
                start_date=pd.to_datetime(s_dt).strftime('%Y%m%d'), 
                end_date=pd.to_datetime(e_dt).strftime('%Y%m%d')
            )
            if not year_df.empty:
                year_df = year_df[year_df['曲线名称'] == '中债国债收益率曲线'].copy()
                cn_dfs.append(year_df)
        
        cn_df = pd.concat(cn_dfs, ignore_index=True)
        # Handle potential column name variations
        col_10y = next((col for col in cn_df.columns if '10' in col), None)
        if col_10y is None:
            raise ValueError("Cannot find 10Y column in ChinaBond data.")
            
        cn_df = cn_df[['日期', col_10y]]
        cn_df.rename(columns={'日期': 'date', col_10y: 'cn_10y'}, inplace=True)
        cn_df['date'] = pd.to_datetime(cn_df['date']).dt.normalize()
        cn_df.set_index('date', inplace=True)
        cn_df = cn_df[~cn_df.index.duplicated(keep='last')] # [B2 Fix] 强制清洗 duplicated 防止静默引发笛卡尔积膨胀
        cn_df['cn_10y'] = cn_df['cn_10y'].astype(float)
    except Exception as e:
        logging.error(f"Failed to fetch ChinaBond from akshare: {e}")
        # Fallback to empty df if it completely fails
        cn_df = pd.DataFrame(columns=['cn_10y'], index=pd.to_datetime([]))
    
    # 2. Load A-share trading dates as the primary alignment spine
    ashare_path = '.quantbot_data/ashare_daily.parquet'
    if os.path.exists(ashare_path):
        df_ashare = pd.read_parquet(ashare_path, columns=['date'])
        trade_dates = sorted(df_ashare['date'].unique())
    else:
        # Fallback if A-share data isn't built yet
        trade_dates = pd.date_range(start_date, end_date, freq='B')
        
    macro_df = pd.DataFrame(index=trade_dates)
    
    # 3. Left Join
    macro_df = macro_df.join(cn_df)
    
    # Forward fill missing values (e.g. if CN holiday but A-share is somehow open, or just missing data)
    macro_df.ffill(inplace=True)
    
    # 4. Compute Features
    # China 10Y Momentum (20 days)
    macro_df['cn_10y_trend'] = macro_df['cn_10y'] / macro_df['cn_10y'].shift(20) - 1
    
    # Save to parquet
    os.makedirs('.quantbot_data', exist_ok=True)
    out_path = '.quantbot_data/macro_daily.parquet'
    macro_df.to_parquet(out_path)
    logging.info(f"Macro features saved to {out_path} with {len(macro_df)} rows.")
    logging.info(f"Sample data:\n{macro_df.tail()}")

if __name__ == "__main__":
    build_macro_features()
